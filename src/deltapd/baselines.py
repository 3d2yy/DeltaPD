"""
Comparación contra detectores "State of the Art" (o baselines clásicos).

Este módulo instrumenta tres baselines comúnmente reportados en diagnóstico
de descargas parciales: Detector de Energía, Detector Zero-Crossing, y
Detector adaptativo de Envolvente por percentiles.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from deltapd.descriptors import extract_delta_t_vector
from deltapd.trackers import CUSUMDetector

Signal = np.ndarray


class EnergyDetector:
    """Detector clásico basado en RMS/Energía local."""

    def __init__(self, window_size: int = 1024, k_sigma: float = 3.0):
        self.window_size = window_size
        self.k_sigma = k_sigma
        self.mu_ref = 0.0
        self.sig_ref = 0.0

    def fit(self, signal_pre_change: Signal):
        # Sliding window de energía sobre la zona pre-incidente (Ruido / Estable)
        energies = []
        for i in range(
            0, len(signal_pre_change) - self.window_size + 1, self.window_size // 2
        ):
            window = signal_pre_change[i : i + self.window_size]
            energies.append(np.mean(window**2))

        if len(energies) > 0:
            self.mu_ref = float(np.mean(energies))
            self.sig_ref = float(np.std(energies))

    def detect(self, signal: Signal) -> List[int]:
        alarm_idx = []
        # Sliding action
        for i in range(0, len(signal) - self.window_size + 1, self.window_size // 2):
            window = signal[i : i + self.window_size]
            e_local = np.mean(window**2)
            if e_local > self.mu_ref + self.k_sigma * self.sig_ref:
                alarm_idx.append(i + self.window_size // 2)
        return alarm_idx


class ZeroCrossingRateDetector:
    """Detector por variación anómala en los cruces por cero (afectado fatalmente por PDs)."""

    def __init__(self, window_size: int = 1024, k_sigma: float = 3.0):
        self.window_size = window_size
        self.k_sigma = k_sigma
        self.mu_ref = 0.0
        self.sig_ref = 0.0

    def _calc_zcr(self, window: Signal) -> float:
        return float(np.sum(np.diff(np.sign(window)) != 0)) / len(window)

    def fit(self, signal_pre_change: Signal):
        zcrs = []
        for i in range(
            0, len(signal_pre_change) - self.window_size + 1, self.window_size // 2
        ):
            zcrs.append(self._calc_zcr(signal_pre_change[i : i + self.window_size]))

        if len(zcrs) > 0:
            self.mu_ref = float(np.mean(zcrs))
            self.sig_ref = float(np.std(zcrs))

    def detect(self, signal: Signal) -> List[int]:
        alarm_idx = []
        for i in range(0, len(signal) - self.window_size + 1, self.window_size // 2):
            window = signal[i : i + self.window_size]
            z_local = self._calc_zcr(window)
            # El ZCR suele CAER o SUBIR abruptamente. Evaluamos anomalía absoluta.
            if abs(z_local - self.mu_ref) > self.k_sigma * self.sig_ref:
                alarm_idx.append(i + self.window_size // 2)
        return alarm_idx


class EnvelopeThresholdDetector:
    """Detector por Envolvente de Hilbert al 99.5 percentil."""

    def __init__(self, percentile: float = 99.5):
        self.percentile = percentile
        self.thresh = 0.0

    def fit(self, signal_pre_change: Signal):
        env = np.abs(hilbert(signal_pre_change))
        self.thresh = float(np.percentile(env, self.percentile))

    def detect(self, signal: Signal) -> List[int]:
        env = np.abs(hilbert(signal))
        peaks = np.where(env > self.thresh)[0]
        return peaks.tolist()


def compare_all_baselines(
    signal: Signal,
    fs: float,
    ground_truth_change_idx: int,
    delta_t: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Simula todos los detectores contra la señal bruta en tiempo continuo y los compara
    contra el sistema CUSUM Delta-T que opera en el dominio del tiempo inter-pulso.
    """
    # Aislar Periodo Pre-Cambio (Ruido o Estado Basal Seguro) para los Baseline (Fit)
    sig_pre = signal[:ground_truth_change_idx]

    baselines = {
        "Energy (SotA Baseline)": EnergyDetector(window_size=1024),
        "ZCR (Frequency Baseline)": ZeroCrossingRateDetector(window_size=1024),
        "Hilbert Env. (99.5th P)": EnvelopeThresholdDetector(),
    }

    records = []

    for name, detector in baselines.items():
        detector.fit(sig_pre)
        alarms = detector.detect(signal)

        # Calcular Matriz de Confusión por Muestras
        tps = [idx for idx in alarms if idx >= ground_truth_change_idx]
        fps = [idx for idx in alarms if idx < ground_truth_change_idx]

        p_total = len(signal) - ground_truth_change_idx
        n_total = ground_truth_change_idx

        prec = len(tps) / (len(tps) + len(fps)) if (len(tps) + len(fps)) > 0 else 0.0
        rec = len(tps) / p_total if p_total > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = len(fps) / n_total if n_total > 0 else 0.0

        post_change = np.array(
            [idx for idx in alarms if idx >= ground_truth_change_idx]
        )
        latency = (
            int(post_change[0] - ground_truth_change_idx)
            if len(post_change) > 0
            else p_total
        )

        records.append(
            {
                "Method": name,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "FPR": fpr,
                "Latency_Samples": latency,
            }
        )

    # Agregando nuestra invención DeltaPD
    if delta_t is None:
        delta_t, _ = extract_delta_t_vector(signal, fs)

    # Extrapolamos el GT de muestras de señal al índice del vector Delta T correspondiente
    # Asumiendo densidad uniforme si no tenemos exactitud de mapeo muestra=>pulso
    time_ratio = ground_truth_change_idx / len(signal)
    gt_pulse_idx = int(time_ratio * len(delta_t))

    if len(delta_t) >= 1:
        # La invención DeltaPD: CUSUM sobre el vector delta t.
        cusum = CUSUMDetector(threshold=2.5, drift=0.5)
        res = cusum.detect(delta_t)
        calarms = getattr(res, "alarm_indices", [])

        tps_c = [idx for idx in calarms if idx >= gt_pulse_idx]
        fps_c = [idx for idx in calarms if idx < gt_pulse_idx]

        # Penalidad si Delta_T fracasa en encontrar pulsos
        p_tot_c = len(delta_t) - gt_pulse_idx
        n_tot_c = gt_pulse_idx

        c_prec = (
            len(tps_c) / (len(tps_c) + len(fps_c))
            if (len(tps_c) + len(fps_c)) > 0
            else 0.0
        )
        c_rec = len(tps_c) / p_tot_c if p_tot_c > 0 else 0.0
        c_f1 = 2 * (c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        c_fpr = len(fps_c) / n_tot_c if n_tot_c > 0 else 0.0

        p_ch_c = np.array([idx for idx in calarms if idx >= gt_pulse_idx])

        # Latency on Delta-T represents pulso-number delay. We convert back to sample-proxy.
        avg_samples_per_pulse = len(signal) / len(delta_t)
        c_latency = (
            int((p_ch_c[0] - gt_pulse_idx) * avg_samples_per_pulse)
            if len(p_ch_c) > 0
            else len(signal) - ground_truth_change_idx
        )

        records.append(
            {
                "Method": r"DeltaPD (CUSUM on $\Delta t$ vector)",
                "Precision": c_prec,
                "Recall": c_rec,
                "F1_Score": c_f1,
                "FPR": c_fpr,
                "Latency_Samples": c_latency,
            }
        )

    df = pd.DataFrame(records)
    return df


def export_comparison_to_latex(df: pd.DataFrame) -> str:
    """Exportar métricas a tabla LaTeX."""
    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Performance Comparison Against Baseline Detectors",
        label="tab:baseline_comparison",
    )
    return (
        latex_str.replace("\\toprule", "\\hline\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
        .replace("\\midrule", "\\hline")
    )
