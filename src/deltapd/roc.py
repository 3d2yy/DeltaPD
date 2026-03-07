"""
Módulo de Análisis ROC (Receiver Operating Characteristic) para Descriptores UHF-PD.

Este módulo computa el Área Bajo la Curva (AUC-ROC) evaluando dinámicamente
el comportamiento de cada descriptor frente a un barrido iterativo de umbrales
del detector de puntos de cambio de Page (CUSUM). El resultado permite
analizar estadísticamente qué variable termodinámica o
estadística maximiza la separabilidad entre el ruido de fondo y los eventos PD.

No utiliza dependencias de trazado gráfico (matplotlib/seaborn) de acuerdo a
las restricciones arquitectónicas del framework.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from deltapd.trackers import CUSUMDetector

Signal = NDArray[np.floating[Any]]


def _zscore_normalize(signal: Signal, baseline_limit: int) -> Signal:
    """Normaliza un descriptor con Z-Score usando solo su segmento de ruido basal."""
    if baseline_limit <= 0 or baseline_limit >= len(signal):
        return signal

    baseline = signal[:baseline_limit]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline)) + 1e-12
    return (signal - mu) / sigma


def compute_roc_per_descriptor(
    descriptors: Dict[str, Signal],
    ground_truth_change_idx: int,
    cusum_threshold_range: NDArray[np.floating[Any]] = np.linspace(0.5, 15.0, 50),
) -> Dict[str, Dict[str, Any]]:
    """
    Computa las métricas de curva ROC iterando sobre los umbrales CUSUM.

    Para cada descriptor provisto:
      1. Se normaliza empleando Z-Score sobre el segmento previo al `ground_truth`.
      2. Se simula una detección CUSUM iterando sobre `cusum_threshold_range`.
      3. Se calcula True Positive Rate (TPR) y False Positive Rate (FPR).
      4. Se computa el AUC usando integración trapezoidal.

    Retorna:
        Dict anidado por nombre de descriptor conteniendo arrays 'fpr', 'tpr', 'thresholds',
        el 'auc' (Area Under Curve) escalar y las métricas óptimas.
    """
    results = {}

    for name, signal in descriptors.items():
        n_samples = len(signal)
        # 1. Normalización controlada
        norm_signal = _zscore_normalize(signal, ground_truth_change_idx)

        tpr_list = []
        fpr_list = []
        best_f1 = 0.0
        opt_thresh = cusum_threshold_range[0]

        # 2. Barrido de Umbrales CUSUM
        for thresh in cusum_threshold_range:
            detector = CUSUMDetector(threshold=float(thresh), drift=0.5)
            # El input al CUSUM es la señal normalizada
            det_res = detector.detect(norm_signal)

            alarms = getattr(det_res, "alarm_indices", [])

            # 3. Cálculo de Clasificación
            # TPs: Alertas válidas que ocurren A PARTIR (>=) del ground_truth
            tps = [idx for idx in alarms if idx >= ground_truth_change_idx]
            # FPs: Alertas prematuras que ocurren ANTES (<) del ground_truth
            fps = [idx for idx in alarms if idx < ground_truth_change_idx]

            # Universo de positivos (la ventana posterior al cambio)
            p_total = n_samples - ground_truth_change_idx
            # Universo de negativos (la ventana anterior al cambio)
            n_total = ground_truth_change_idx

            # Manejo de Edge cases si P_total o N_total son 0
            tpr_val = len(tps) / p_total if p_total > 0 else 0.0
            fpr_val = len(fps) / n_total if n_total > 0 else 0.0

            # Acotamiento estricto a [0, 1]
            tpr_val = min(1.0, tpr_val)
            fpr_val = min(1.0, fpr_val)

            tpr_list.append(tpr_val)
            fpr_list.append(fpr_val)

            # F1 Proxy (usando count_tp vs expected 1, pero aquí avaluamos por ventana)
            # Para curvas ROC estándar de Page Test, TPR refleja la tasa de alarmas correctas.
            prec = (
                len(tps) / (len(tps) + len(fps)) if (len(tps) + len(fps)) > 0 else 0.0
            )
            f1 = (
                2 * (prec * tpr_val) / (prec + tpr_val) if (prec + tpr_val) > 0 else 0.0
            )

            if f1 > best_f1:
                best_f1 = f1
                opt_thresh = thresh

        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)

        # 4. Integración Trapezoidal de Área ROC
        # Es vital que los arrays estén ordenados por FPR ascendente
        sort_idx = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[sort_idx]
        tpr_sorted = tpr_arr[sort_idx]

        # Integración usando trampa (NumPy) manual para evitar dependencia scikit-learn
        auc = np.trapz(tpr_sorted, fpr_sorted)

        # Compensar en caso de auc negativo por orden (raro en ROC pero matemático)
        auc = abs(auc)

        results[name] = {
            "fpr": fpr_sorted,
            "tpr": tpr_sorted,
            "thresholds": cusum_threshold_range[sort_idx],
            "auc": float(auc),
            "optimal_threshold": float(opt_thresh),
            "best_f1": float(best_f1),
        }

    return results


def export_roc_table(roc_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Sintetiza el diccionario de resultados ROC iterativos en un DataFrame tabular
    rankeado por su capacidad discriminante (Área Bajo la Curva).
    """
    records = []
    for desc, metrics in roc_results.items():
        records.append(
            {
                "Descriptor": desc,
                "AUC_ROC": metrics["auc"],
                "Optimal_CUSUM_Threshold": metrics["optimal_threshold"],
                "Best_F1_Score": metrics["best_f1"],
            }
        )

    df = pd.DataFrame(records)
    # Ordenar de mayor a menor AUC
    return df.sort_values(by="AUC_ROC", ascending=False).reset_index(drop=True)


def export_roc_to_latex(roc_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Exporta el reporte tabular a código de marcado LaTeX (Booktabs) para incrustación
    inmediata en documentos técnicos.
    """
    df = export_roc_table(roc_results)

    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        column_format="l|ccc",
        caption="Ablation Study: Receiver Operating Characteristic (ROC) Metrics",
        label="tab:roc_metrics",
    )

    # Inyectar \toprule y \bottomrule estético
    latex_str = (
        latex_str.replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
        .replace("\\midrule", "\\hline")
    )
    return latex_str
