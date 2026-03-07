"""
Evaluación de descriptores estadísticos y termodinámicos mediante ventaneo deslizante (rolling window).

Aplica análisis puramente matricial (sin gráficas), calculando 12 métricas
concurrentes vectorizadas usando `numpy.lib.stride_tricks.sliding_window_view`.

Posteriormente ejecuta un estudio de ablación aislando el rendimiento de cada descriptor
sobre un detector estacionario CUSUM.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy import stats

Signal = NDArray[np.floating[Any]]


def extract_rolling_descriptors(
    signal: Signal, window_size: int, overlap: int
) -> Dict[str, Signal]:
    r"""Calcula de forma concurrente doce métricas vectorizadas sobre la señal usando ventaneo deslizante.

    Mathematical Formulations:
    --------------------------
    - **Kurtosis** (Curtosis):
      .. math:: \text{Kurtosis} = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^4}{\sigma^4}

    - **Clearance Factor** (Factor de Margen):
      Relaciona la amplitud pico con el valor medio cuadrático normalizado:
      .. math:: \text{Clearance Factor} = \frac{\max(|x_i|)}{\left( \frac{1}{N} \sum_{i=1}^{N} \sqrt{|x_i|} \right)^2}

    - **Shannon Entropy** (Entropía de Shannon):
      Evaluando el perfil normalizado de densidad de energía de la ventana, donde $p_i = \frac{x_i^2}{\sum x_j^2}$:
      .. math:: H = - \sum_{i=1}^{N} p_i \log_2(p_i + \epsilon)

    Parameters
    ----------
    signal : Signal
        Vector unidimensional (1-D) de voltaje.
    window_size : int
        Número de muestras (longitud de la ventana).
    overlap : int
        Cantidad de muestras superpuestas (debe ser estricatmente menor que window_size).

    Returns
    -------
    dict
        Diccionario conteniendo 12 tensores unidimensionales resultantes, mapeados a sus respectivos nombres.
    """
    step = window_size - overlap
    if step <= 0:
        raise ValueError("El overlap debe ser estrictamente menor que el window_size.")

    # [1] Vetorización matricial del ventaneo usando stride_tricks
    # result shape = (n_windows, window_size)
    windows = sliding_window_view(signal, window_shape=window_size)[::step, :]

    # Tolerancias algebraicas
    eps = 1e-15

    # =========================================================
    # Tensores Base Concurrentes
    # =========================================================
    w_abs = np.abs(windows)
    w_sq = windows**2

    max_abs = np.max(w_abs, axis=1)
    mean_abs = np.mean(w_abs, axis=1)

    # 1. Varianza (Variance)
    var = np.var(windows, axis=1)

    # 2. Curtosis (Kurtosis)
    # math: moment_4 / variance^2
    kurt = stats.kurtosis(windows, axis=1, fisher=False, nan_policy="omit")

    # 3. Asimetría (Skewness)
    skew = stats.skew(windows, axis=1, nan_policy="omit")

    # 4. Valor Cuadrático Medio (RMS)
    rms = np.sqrt(np.mean(w_sq, axis=1))

    # 5. Factor de Forma (Waveform Factor)
    waveform_factor = np.divide(
        rms, mean_abs, out=np.zeros_like(rms), where=mean_abs > eps
    )

    # 6. Factor de Cresta (Crest Factor)
    crest_factor = np.divide(max_abs, rms, out=np.zeros_like(max_abs), where=rms > eps)

    # 7. Factor de Impulso (Impulse Factor)
    impulse_factor = np.divide(
        max_abs, mean_abs, out=np.zeros_like(max_abs), where=mean_abs > eps
    )

    # 8. Factor de Margen (Clearance Factor)
    sq_mean_abs = (np.mean(np.sqrt(w_abs), axis=1)) ** 2
    clearance_factor = np.divide(
        max_abs, sq_mean_abs, out=np.zeros_like(max_abs), where=sq_mean_abs > eps
    )

    # 9. Entropía de Shannon (Shannon Entropy)
    total_energy_per_win = np.sum(w_sq, axis=1, keepdims=True)
    p = np.divide(w_sq, total_energy_per_win + eps)
    shannon_entropy = -np.sum(p * np.log2(p + eps), axis=1)

    # 10. Energía Total de la Ventana (Total Energy)
    total_energy = np.sum(w_sq, axis=1)

    # 11. Tasa de Cruces por Cero (Zero Crossing Rate)
    # Sumatoria de los cambios de signo
    signs = np.sign(windows)
    zcr = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1) / (window_size - 1)

    # 12. Rango Intercuartílico (Interquartile Range - IQR)
    q75, q25 = np.percentile(windows, [75, 25], axis=1)
    iqr = q75 - q25

    return {
        "Variance": var,
        "Kurtosis": kurt,
        "Skewness": skew,
        "RMS": rms,
        "Waveform_Factor": waveform_factor,
        "Crest_Factor": crest_factor,
        "Impulse_Factor": impulse_factor,
        "Clearance_Factor": clearance_factor,
        "Shannon_Entropy": shannon_entropy,
        "Total_Energy": total_energy,
        "Zero_Crossing_Rate": zcr,
        "IQR": iqr,
    }


def compute_correlation_matrix(descriptors: Dict[str, Signal]) -> pd.DataFrame:
    """Calcula y retorna la matriz de correlación de Pearson entre descriptores."""
    df = pd.DataFrame(descriptors)
    corr_matrix = df.corr(method="pearson")

    print("=" * 115)
    print(" " * 35 + "MATRIZ DE CORRELACIÓN DE PEARSON ENTRE DESCRIPTORES")
    print("=" * 115)
    print(corr_matrix.to_string(float_format=lambda x: f"{x:6.3f}"))
    print("=" * 115)
    print("\n")

    return corr_matrix


def evaluate_descriptors_vs_trackers(
    descriptors: Dict[str, Signal],
    ground_truth_change_idx: int,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
) -> pd.DataFrame:
    """Ejecuta un estudio de ablación estadístico aislando descriptores sobre el rastreador CUSUM.

    Parameters
    ----------
    descriptors : dict
        Diccionario generado por extract_rolling_descriptors.
    ground_truth_change_idx : int
        Índice asintótico o experimental de la ventana donde se define un cambio de régimen real
        (falla introducida) en la señal empírica. Se utiliza para medir Clasificación Binaria.
    cusum_threshold, cusum_drift : float
        Hiperparámetros de calibración del CUSUM Tracker.

    Returns
    -------
    pd.DataFrame
        Tabla DataFrame con el ranking del estudio de ablación ordenado por tasa F1-Score.
    """
    from deltapd.trackers import CUSUMDetector

    # Computar Matriz de Colinealidad (side-effect: prints to stdout)
    _corr_matrix = compute_correlation_matrix(descriptors)

    n_windows = len(next(iter(descriptors.values())))
    ablation_results = []

    for name, desc_vector in descriptors.items():
        # Estandarización de contexto para garantizar la convergencia paramétrica
        mu_pre = float(np.mean(desc_vector[:ground_truth_change_idx]))
        std_pre = float(np.std(desc_vector[:ground_truth_change_idx])) + 1e-15

        normalized_vector = (desc_vector - mu_pre) / std_pre

        # Inyección ciega al algoritmo de Page
        detector = CUSUMDetector(threshold=cusum_threshold, drift=cusum_drift)
        res = detector.detect(normalized_vector)

        # Clasificación CUSUM vs Ground Truth
        alarms = np.zeros(n_windows, dtype=bool)
        if len(res.alarm_indices) > 0:
            alarms[np.array(res.alarm_indices, dtype=int)] = True

        fp = int(np.sum(alarms[:ground_truth_change_idx]))
        tp = int(np.sum(alarms[ground_truth_change_idx:]))
        tn = ground_truth_change_idx - fp
        fn = (n_windows - ground_truth_change_idx) - tp

        # Métricas Binarias de Clasificación
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Latencia: tiempo asintótico de primera alarma útil
        post_alarms = np.where(alarms[ground_truth_change_idx:])[0]
        latency = (
            post_alarms[0]
            if len(post_alarms) > 0
            else (n_windows - ground_truth_change_idx)
        )

        ablation_results.append(
            {
                "Descriptor": name,
                "F1_Score": f1,
                "Latency": latency,
                "FPR": fpr,
                "Precision": prec,
                "Recall": rec,
                "Total_Alarms": tp + fp,
            }
        )

    # Ranking Asintótico Tabular (Ablación)
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation = df_ablation.sort_values(by="F1_Score", ascending=False).reset_index(
        drop=True
    )

    return df_ablation
