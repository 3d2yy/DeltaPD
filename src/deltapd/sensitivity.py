"""
Análisis de Sensibilidad SNR (Signal-to-Noise Ratio).

Este módulo ejecuta múltiples realizaciones Monte Carlo evaluando
la degradación del F1-Score y el aumento de la Tasa de Falsas Alarmas (FPR)
a medida que la relación Señal-Ruido (SNR dB) empeora.

Este análisis demuestra la robustez del
vector de extracción del tiempo de arribo (Delta T) aislando las amplitudes,
para su uso analítico general.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from deltapd.descriptors import extract_delta_t_vector
from deltapd.signal_model import (
    compute_rmse,
    generate_uhf_pd_signal_physical,
    wavelet_denoise_parametric,
)
from deltapd.trackers import apply_delta_t_tracking


def run_snr_sensitivity(
    snr_range_db: NDArray[np.floating[Any]] = np.arange(5, 35, 5),
    n_monte_carlo: int = 30,
    n_samples: int = 4096,
    fs: float = 1e9,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Simula la inyección de diferentes niveles de ruido SNR sobre el CUSUM tracking.
    Retorna métricas promedio e IC (Standard Deviation) agregadas por cada decibelio.
    """
    rng = np.random.default_rng(seed)

    # Pre-generar Semillas MC para Reproducibilidad paralela temporal
    mc_seeds = rng.integers(0, 100000, size=n_monte_carlo)

    results = []

    for snr in snr_range_db:
        print(f"Evaluando sensibilidad SNR: {snr} dB ...")

        f1_list, fpr_list, lat_list, rmse_list, npulses_list = [], [], [], [], []

        for i in range(n_monte_carlo):
            current_seed = int(mc_seeds[i])

            # 1. Generación de Señal en el escenario SNR específico
            clean, noisy = generate_uhf_pd_signal_physical(
                n_samples=n_samples,
                fs=fs,
                n_pulses=12,
                snr_db=float(snr),
                seed=current_seed,
            )

            # 2. Denoising Fijo con Óptimo Comprobado (db4, soft, universal)
            denoised = wavelet_denoise_parametric(
                noisy, wavelet="db4", threshold_mode="soft", threshold_rule="universal"
            )
            rmse_val = compute_rmse(clean, denoised)

            # 3. Aislamiento Delta T
            delta_t, _ = extract_delta_t_vector(
                denoised, fs, threshold_sigma=3.0, method="scipy_peaks"
            )
            n_pulses = len(delta_t) + 1

            # Prevenir fallback fatal si el SNR es catastrófico y aniquiló los picos
            if len(delta_t) < 3:
                f1_list.append(0.0)
                fpr_list.append(1.0)  # Penalidad máxima
                lat_list.append(n_samples)  # Penalidad máxima
                npulses_list.append(n_pulses)
                rmse_list.append(rmse_val)
                continue

            # 4. Tracking
            tracking_res = apply_delta_t_tracking(delta_t)

            # 5. Extracción de Métricas de Validacion (Simulación de Anomaly Injection - mitad array)
            dt_anom = np.concatenate([delta_t, delta_t * 0.5])
            gt_change = len(delta_t)

            cusum_detector = tracking_res["CUSUM"]
            # Re-proteger CUSUM track con la anomalia inyectada
            det_anom = cusum_detector.detect(dt_anom)
            alarms = getattr(det_anom, "alarm_indices", [])

            tps = [idx for idx in alarms if idx >= gt_change]
            fps = [idx for idx in alarms if idx < gt_change]

            post_change = np.array([idx for idx in alarms if idx >= gt_change])
            lat = (
                int(post_change[0] - gt_change)
                if len(post_change) > 0
                else len(dt_anom)
            )

            prec = (
                len(tps) / (len(tps) + len(fps)) if (len(tps) + len(fps)) > 0 else 0.0
            )
            rec = (
                len(tps) / len(dt_anom[gt_change:])
                if len(dt_anom[gt_change:]) > 0
                else 0.0
            )

            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = len(fps) / gt_change if gt_change > 0 else 0.0

            f1_list.append(f1)
            fpr_list.append(fpr)
            lat_list.append(lat)
            npulses_list.append(n_pulses)
            rmse_list.append(rmse_val)

        # Agregación Estadística del Bin SNR
        results.append(
            {
                "SNR_dB": snr,
                "N_Pulses_Mean": np.mean(npulses_list),
                "N_Pulses_Std": np.std(npulses_list),
                "F1_Mean": np.mean(f1_list),
                "F1_Std": np.std(f1_list),
                "FPR_Mean": np.mean(fpr_list),
                "FPR_Std": np.std(fpr_list),
                "Latency_Mean": np.mean(lat_list),
                "RMSE_Mean": np.mean(rmse_list),
            }
        )

    df = pd.DataFrame(results)
    return df


def export_sensitivity_to_latex(df: pd.DataFrame) -> str:
    """Exporta el reporte iterativo de SNR a tabla LaTeX Booktabs estructurada."""

    # Formatear Desviaciones Estándar como ± en una columna unificada
    df_tex = pd.DataFrame()
    df_tex["SNR (dB)"] = df["SNR_dB"].apply(lambda x: f"{x:.0f}")
    df_tex["$F_1$ Score"] = df.apply(
        lambda r: f"{r['F1_Mean']:.3f} ± {r['F1_Std']:.3f}", axis=1
    )
    df_tex["FPR"] = df.apply(
        lambda r: f"{r['FPR_Mean']:.3f} ± {r['FPR_Std']:.3f}", axis=1
    )
    df_tex["RMSE"] = df.apply(lambda r: f"{r['RMSE_Mean']:.3e}", axis=1)
    df_tex["Detected Pulses"] = df.apply(lambda r: f"{r['N_Pulses_Mean']:.1f}", axis=1)

    latex_str = df_tex.to_latex(
        index=False,
        column_format="c|cccc",
        caption="Signal-to-Noise Ratio (SNR) Sensitivity Analysis",
        label="tab:snr_sensitivity",
    )

    return (
        latex_str.replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
        .replace("\\midrule", "\\hline")
    )
