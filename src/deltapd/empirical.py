"""
Validación de Lote Empírico (Laboratorio) - Pipeline Automatizado.

Interconecta `data_loader.py` con las Fases 1-4 para ingestar
datos crudos de laboratorio (.mat, .csv, .h5) y escupir reportes tabulares
JSON/CSV directamente, sin intervención manual de código. Ideal para procesar
carpetas enteras de experimentos para el análisis de resultados de laboratorio.
"""

import glob
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from deltapd.descriptors import extract_delta_t_vector
from deltapd.features import (
    evaluate_descriptors_vs_trackers,
    extract_rolling_descriptors,
)
from deltapd.loader import load_empirical_signal
from deltapd.signal_model import wavelet_denoise_parametric
from deltapd.trackers import apply_delta_t_tracking


def validate_empirical_file(
    file_path: str,
    known_change_time_s: Optional[float] = None,
    best_wavelet: str = "db4",
    best_threshold_mode: str = "soft",
    best_threshold_rule: str = "universal",
    output_dir: str = "./results/",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Toma un archivo de osciloscopio real, lo desempaqueta, limpia,
    busca pulsos y evalúa la efectividad de rastreo y descriptores.
    Exporta todo a JSON / CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if verbose:
        print(f"--- Evaluando Empírico: {base_name} ---")

    # 1. Ingesta Polimórfica (Zero-Mean normalizada)
    try:
        signal, fs = load_empirical_signal(file_path)
    except Exception as e:
        raise ValueError(f"Falla crítica al ingestar {file_path}: {str(e)}")

    # 2. Denoising Fijo Asumido (El usuario debería conocer el óptimo por Phase 1)
    if verbose:
        print(
            f"  > Filtrando WAVELET ({best_wavelet}, {best_threshold_mode}, {best_threshold_rule})..."
        )
    denoised = wavelet_denoise_parametric(
        signal,
        wavelet=best_wavelet,
        threshold_mode=best_threshold_mode,
        threshold_rule=best_threshold_rule,
    )

    # 3. Aislamiento Temporal (Extracción del Vector $\Delta t$)
    if verbose:
        print(r"  > Extracting pulse temporal isolation ($\Delta t$)...")
    try:
        delta_t = extract_delta_t_vector(
            denoised, fs, threshold_sigma=3.0, detection_method="scipy_peaks"
        )
    except ValueError:
        delta_t = np.array([], dtype=np.float64)

    if len(delta_t) < 3:
        raise ValueError(
            f"Falla: Señal {base_name} no produjo suficientes pulsos PD para tracking (encontrados: {len(delta_t)+1}). SNR posiblemente demasiado bajo o ausencia de descargas."
        )

    # 4. Rastreo Asintótico Ciego
    track_summary = apply_delta_t_tracking(delta_t)
    kalman_gain = track_summary.kalman.steady_state_gain
    cusum_alarms = track_summary.cusum.n_alarms

    # 5. Ablación de Descriptores
    if verbose:
        print("  > Calculando descriptores por ventana deslizante estricta...")
    descriptors = extract_rolling_descriptors(denoised, window_size=1024, overlap=512)

    desc_report: Optional[pd.DataFrame] = None
    if known_change_time_s is not None:
        # Calcular el índice asintótico del cambio asumiendo uniformidad global temporal
        change_idx = int(known_change_time_s * fs / (1024 - 512))
        if verbose:
            print(
                f"  > Evento térmico conocido en t={known_change_time_s}s. Evaluando AUC y Latencia de Descriptores..."
            )
        desc_report = evaluate_descriptors_vs_trackers(descriptors, change_idx)

        desc_csv_path = os.path.join(output_dir, f"{base_name}_descriptor_report.csv")
        desc_report.to_csv(desc_csv_path, index=False)

    # 6. Agregación y Serializado
    report_dict = {
        "file_name": base_name,
        "signal_length_samples": len(signal),
        "sampling_frequency_hz": fs,
        "duration_seconds": len(signal) / fs,
        "n_pulses_detected": len(delta_t) + 1,
        "mean_interpulse_time_s": float(np.mean(delta_t)),
        "std_interpulse_time_s": float(np.std(delta_t)),
        "kalman_steady_gain": float(kalman_gain) if kalman_gain else None,
        "cusum_total_alarms": int(cusum_alarms),
    }

    json_path = os.path.join(output_dir, f"{base_name}_summary.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    if verbose:
        print(f"--- Exportación Finalizada: {output_dir} ---")

    return report_dict


def validate_multiple_files(
    directory_path: str,
    pattern: str = "*.*",
    known_change_times: Optional[Dict[str, float]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Función Batch: Ingesta un directorio de pruebas completas y devuelve un
    Dataframe consolidado como conjunto de resultados experimentales.
    """
    archivos = glob.glob(os.path.join(directory_path, pattern))
    valid_extensions = (".csv", ".mat", ".h5", ".hdf5")
    archivos = [a for a in archivos if a.endswith(valid_extensions)]

    if not archivos:
        print(
            f"Advertencia: No se encontraron capturas {valid_extensions} en {directory_path}"
        )
        return pd.DataFrame()

    res_list = []
    known_change_times = known_change_times or {}

    for f in archivos:
        b_name = os.path.splitext(os.path.basename(f))[0]
        # Si el testbed tiene una tabla de tiempos, la introducimos automáticamente.
        change_s = known_change_times.get(b_name, None)

        try:
            r = validate_empirical_file(
                f, known_change_time_s=change_s, verbose=False, **kwargs
            )
            res_list.append(r)
        except Exception as e:
            print(f"[x] Error en {b_name}: {str(e)}")
            res_list.append({"file_name": b_name, "error": str(e)})

    return pd.DataFrame(res_list)
