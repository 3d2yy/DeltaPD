"""Decision layer — operational verdict per time window.

Converts raw pipeline outputs into actionable per-window verdicts
(PD present / absent, confidence, event count, false-alarm-rate estimate)
suitable for deployment reports.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from deltapd.descriptors import extract_delta_t_vector
from deltapd.features import extract_rolling_descriptors
from deltapd.signal_model import wavelet_denoise_parametric
from deltapd.trackers import CUSUMDetector


def evaluate_window(
    signal_chunk: np.ndarray,
    fs: float,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""Run a mini-pipeline on one signal window and return a verdict.

    The decision is based on:
    1. Pulse detection via threshold crossing.
    2. CUSUM change-point detection on the inter-pulse interval vector.
    3. Dominant descriptor identification via maximum kurtosis.

    Confidence is estimated as:

    .. math::

        C = \text{clip}\!\left(\frac{n_{\text{events}}}{n_{\max}} \cdot
            (1 - \text{FAR}_{\text{est}}), \; 0, \; 1\right)

    Parameters
    ----------
    signal_chunk : ndarray
        1-D voltage signal for this window.
    fs : float
        Sampling frequency in Hz.
    config : dict, optional
        Override defaults: ``threshold_sigma``, ``cusum_threshold``,
        ``cusum_drift``, ``wavelet``, ``n_events_max``.

    Returns
    -------
    dict
        Keys: ``pd_present``, ``confidence``, ``n_events``,
        ``far_estimate``, ``dominant_descriptor``.
    """
    cfg = {
        "threshold_sigma": 3.0,
        "cusum_threshold": 5.0,
        "cusum_drift": 0.5,
        "wavelet": "db4",
        "n_events_max": 20,
    }
    if config:
        cfg.update(config)

    chunk = np.asarray(signal_chunk, dtype=np.float64)
    n = len(chunk)

    # Denoise
    denoised = wavelet_denoise_parametric(
        chunk, wavelet=cfg["wavelet"], threshold_mode="soft", threshold_rule="universal"
    )

    # Pulse detection -> delta-t
    try:
        delta_t = extract_delta_t_vector(
            denoised, fs, threshold_sigma=cfg["threshold_sigma"]
        )
        n_events = len(delta_t) + 1
    except ValueError:
        # Fewer than 2 pulses detected
        return {
            "pd_present": False,
            "confidence": 0.0,
            "n_events": 0,
            "far_estimate": 0.0,
            "dominant_descriptor": "none",
        }

    if len(delta_t) < 2:
        return {
            "pd_present": False,
            "confidence": 0.0,
            "n_events": n_events,
            "far_estimate": 0.0,
            "dominant_descriptor": "none",
        }

    # CUSUM on delta-t
    detector = CUSUMDetector(threshold=cfg["cusum_threshold"], drift=cfg["cusum_drift"])
    cusum_res = detector.detect(delta_t)
    n_alarms = cusum_res.n_alarms

    # FAR estimate (alarms in first half / half-length)
    half = len(delta_t) // 2
    early_alarms = int(np.sum(cusum_res.alarms[:half])) if half > 0 else 0
    far_estimate = early_alarms / max(half, 1)

    # Dominant descriptor by max kurtosis
    dominant = "none"
    if n >= 64:
        try:
            from scipy import stats as sp_stats
            import warnings

            descs = extract_rolling_descriptors(
                denoised, min(256, n // 2), min(128, n // 4)
            )
            best_kurt = -np.inf
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for name, values in descs.items():
                    k = float(sp_stats.kurtosis(values, fisher=False, nan_policy="omit"))
                    if k > best_kurt:
                        best_kurt = k
                        dominant = name
        except Exception:
            dominant = "unknown"

    # Confidence
    ratio = min(n_events / cfg["n_events_max"], 1.0)
    confidence = float(np.clip(ratio * (1.0 - far_estimate), 0.0, 1.0))
    pd_present = n_alarms > 0 or n_events >= 3

    return {
        "pd_present": bool(pd_present),
        "confidence": round(confidence, 4),
        "n_events": int(n_events),
        "far_estimate": round(far_estimate, 4),
        "dominant_descriptor": dominant,
    }


def evaluate_campaign(
    signal: np.ndarray,
    fs: float,
    window_size_s: float = 0.05,
    overlap_s: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Segment a signal into windows and evaluate each one.

    Parameters
    ----------
    signal : ndarray
        Full 1-D voltage signal.
    fs : float
        Sampling frequency in Hz.
    window_size_s : float
        Window duration in seconds.
    overlap_s : float
        Overlap between consecutive windows in seconds.
    config : dict, optional
        Passed to :func:`evaluate_window`.

    Returns
    -------
    pd.DataFrame
        One row per window with columns: ``t_start``, ``t_end``,
        ``pd_present``, ``confidence``, ``n_events``,
        ``far_estimate``, ``dominant_descriptor``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    win_samples = max(int(window_size_s * fs), 64)
    step_samples = max(int((window_size_s - overlap_s) * fs), 1)
    n = len(signal)

    rows = []
    start = 0
    while start + win_samples <= n:
        chunk = signal[start : start + win_samples]
        verdict = evaluate_window(chunk, fs, config=config)
        verdict["t_start"] = start / fs
        verdict["t_end"] = (start + win_samples) / fs
        rows.append(verdict)
        start += step_samples

    if not rows:
        return pd.DataFrame(
            columns=[
                "t_start",
                "t_end",
                "pd_present",
                "confidence",
                "n_events",
                "far_estimate",
                "dominant_descriptor",
            ]
        )

    df = pd.DataFrame(rows)
    col_order = [
        "t_start",
        "t_end",
        "pd_present",
        "confidence",
        "n_events",
        "far_estimate",
        "dominant_descriptor",
    ]
    return df[col_order]


def campaign_summary(df_windows: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate window-level verdicts into a campaign-level summary.

    Parameters
    ----------
    df_windows : pd.DataFrame
        Output of :func:`evaluate_campaign`.

    Returns
    -------
    dict
        ``total_windows``, ``pd_present_pct``, ``mean_confidence``,
        ``first_detection_s``, ``false_alarm_rate_per_min``.
    """
    if df_windows.empty:
        return {
            "total_windows": 0,
            "pd_present_pct": 0.0,
            "mean_confidence": 0.0,
            "first_detection_s": None,
            "false_alarm_rate_per_min": 0.0,
        }

    total = len(df_windows)
    pd_present = df_windows["pd_present"].sum()
    pd_pct = float(pd_present / total * 100)
    mean_conf = float(df_windows["confidence"].mean())

    # First detection
    detected = df_windows[df_windows["pd_present"]]
    first_det = float(detected["t_start"].iloc[0]) if not detected.empty else None

    # FAR per minute
    total_duration_s = float(
        df_windows["t_end"].iloc[-1] - df_windows["t_start"].iloc[0]
    )
    total_duration_min = total_duration_s / 60.0 if total_duration_s > 0 else 1e-10
    total_far = float(df_windows["far_estimate"].sum())
    far_per_min = total_far / total_duration_min

    return {
        "total_windows": int(total),
        "pd_present_pct": round(pd_pct, 2),
        "mean_confidence": round(mean_conf, 4),
        "first_detection_s": first_det,
        "false_alarm_rate_per_min": round(far_per_min, 4),
    }
