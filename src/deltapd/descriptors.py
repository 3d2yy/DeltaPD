"""
Módulo de cálculo de descriptores para análisis de descargas parciales.

Phase 2 — Variable Isolation:
    The primary interface of this module is :func:`extract_delta_t_vector`.
    It detects UHF-PD pulses in the (pre-processed) signal and returns a
    **single one-dimensional vector** containing the time differences Δt
    between consecutive pulses.  Amplitude data is **not** propagated.

    Legacy energy / spectral / statistical descriptors are retained in a
    ``_legacy`` namespace for backward compatibility but are **excluded**
    from the current validation pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]


# ===================================================================
# Phase 2 — Pulse detection & Δt extraction  (PRIMARY INTERFACE)
# ===================================================================


def detect_pulses(
    signal_data: Signal,
    fs: float,
    threshold_sigma: float = 3.0,
    min_separation_s: float = 0.0,
    method: str = "threshold",
) -> NDArray[np.intp]:
    """Detect PD pulses in a pre-processed UHF signal.

    Parameters
    ----------
    signal_data : Signal
        Pre-processed (envelope / denoised) signal.
    fs : float
        Sampling frequency in Hz.
    threshold_sigma : float
        Number of standard deviations above the mean used as the peak
        detection threshold (only for ``method='threshold'``).
    min_separation_s : float
        Minimum time separation (in seconds) between consecutive pulses.
        Translated to samples internally.
    method : str
        ``'threshold'`` — simple amplitude threshold on the absolute signal.
        ``'scipy_peaks'`` — ``scipy.signal.find_peaks`` with prominence.

    Returns
    -------
    pulse_indices : ndarray of int
        Sample indices where pulses were detected, sorted ascending.
    """
    data = np.asarray(signal_data, dtype=np.float64)
    abs_data = np.abs(data)

    min_distance: int = max(1, int(min_separation_s * fs))

    if method == "threshold":
        mu = np.mean(abs_data)
        sigma = np.std(abs_data)
        height = mu + threshold_sigma * sigma
        peaks, _ = signal.find_peaks(abs_data, height=height, distance=min_distance)
    elif method == "scipy_peaks":
        # Use prominence-based detection (more robust for UHF-PD)
        prominence = np.std(abs_data) * threshold_sigma * 0.5
        peaks, _ = signal.find_peaks(
            abs_data,
            prominence=prominence,
            distance=min_distance,
        )
    elif method == "cfar":
        peaks = detect_pulses_cfar(signal_data, fs, min_separation_s=min_separation_s)
    else:
        raise ValueError(f"Unknown detection method: {method!r}")

    return np.sort(peaks)


def detect_pulses_cfar(
    signal_data: Signal,
    fs: float,
    cfar_window: int = 64,
    cfar_guard: int = 8,
    pfa: float = 1e-4,
    min_separation_s: float = 0.0,
) -> NDArray[np.intp]:
    r"""Cell-Averaging Constant False Alarm Rate (CA-CFAR) pulse detector.

    For each cell-under-test (CUT) at index *k*, the noise power is estimated
    from ``cfar_window`` training cells on each side, excluding ``cfar_guard``
    guard cells adjacent to the CUT:

    .. math::

        \hat{P}_n(k) = \frac{1}{2W} \sum_{i \in \mathcal{T}(k)} |x_i|^2

    The detection threshold factor is:

    .. math::

        \alpha_{\text{CFAR}} = W \left( P_{fa}^{-1/W} - 1 \right)

    A detection is declared when :math:`|x_k|^2 > \alpha \cdot \hat{P}_n(k)`.

    Parameters
    ----------
    signal_data : Signal
        1-D signal.
    fs : float
        Sampling frequency in Hz.
    cfar_window : int
        Number of training cells on each side (total = 2 * cfar_window).
    cfar_guard : int
        Number of guard cells on each side of the CUT.
    pfa : float
        Design probability of false alarm.
    min_separation_s : float
        Minimum time between detections in seconds.

    Returns
    -------
    pulse_indices : ndarray of int
        Sample indices of detected pulses.
    """
    data = np.asarray(signal_data, dtype=np.float64)
    power = data**2
    n = len(data)

    # CFAR threshold factor
    w = cfar_window
    alpha_cfar = w * (pfa ** (-1.0 / w) - 1.0)

    margin = w + cfar_guard
    detections = np.zeros(n, dtype=bool)

    for k in range(margin, n - margin):
        # Training cells: [k - margin : k - cfar_guard] and [k + cfar_guard + 1 : k + margin + 1]
        left = power[k - margin : k - cfar_guard]
        right = power[k + cfar_guard + 1 : k + margin + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        threshold_val = alpha_cfar * noise_est

        if power[k] > threshold_val:
            detections[k] = True

    # VERSIÓN OPTIMIZADA: Extracción basada en Primer Frente de Onda (ToA)
    detection_idx = np.flatnonzero(detections)
    
    if len(detection_idx) == 0:
        return np.array([], dtype=np.intp)
        
    min_distance = max(1, int(min_separation_s * fs))
    
    # Calcular la distancia en muestras entre detecciones consecutivas
    gaps = np.diff(detection_idx)
    
    # Un nuevo clúster físico comienza solo si la separación supera la 'min_distance'
    # El primer elemento (índice 0) es siempre el inicio de una avalancha
    cluster_starts = np.insert(gaps > min_distance, 0, True)
    
    # Extraer el primer índice de cada clúster (Ignora amplitudes mayores posteriores)
    toa_indices = detection_idx[cluster_starts]
    
    return toa_indices.astype(np.intp)


def compute_delta_t(
    pulse_indices: NDArray[np.intp],
    fs: float,
) -> Signal:
    """Compute Δt — time differences between consecutive detected pulses.

    Parameters
    ----------
    pulse_indices : ndarray of int
        Sorted sample indices of detected pulses.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    delta_t : Signal
        1-D vector of inter-pulse time intervals in **seconds**.
        Length is ``len(pulse_indices) - 1``.

    Raises
    ------
    ValueError
        If fewer than two pulses are provided.
    """
    if len(pulse_indices) < 2:
        raise ValueError(
            f"At least 2 pulses are required to compute Δt "
            f"(got {len(pulse_indices)})."
        )
    idx = np.sort(pulse_indices)
    delta_samples: NDArray[np.intp] = np.diff(idx)
    delta_t: Signal = delta_samples.astype(np.float64) / fs
    return delta_t


def extract_delta_t_vector(
    signal_data: Signal,
    fs: float,
    threshold_sigma: float = 3.0,
    min_separation_s: float = 0.0,
    detection_method: str = "threshold",
) -> Signal:
    """Primary descriptor interface — returns a 1-D Δt vector.

    This is the **single output** prescribed by Phase 2 of the validation
    framework.  Amplitude information is deliberately excluded.

    Parameters
    ----------
    signal_data : Signal
        Pre-processed UHF-PD signal.
    fs : float
        Sampling frequency in Hz.
    threshold_sigma : float
        Detection threshold in units of σ.
    min_separation_s : float
        Minimum inter-pulse gap in seconds.
    detection_method : str
        ``'threshold'`` or ``'scipy_peaks'``.

    Returns
    -------
    delta_t : Signal
        1-D vector ``[Δt₁, Δt₂, …, Δtₙ₋₁]`` in seconds.
    """
    pulse_idx = detect_pulses(
        signal_data,
        fs,
        threshold_sigma=threshold_sigma,
        min_separation_s=min_separation_s,
        method=detection_method,
    )
    return compute_delta_t(pulse_idx, fs)


def extract_pulse_morphology(
    signal_data: Signal,
    pulse_indices: NDArray[np.intp],
    fs: float,
    window_back_ns: float = 10.0,
    window_forward_ns: float = 100.0,
) -> Any:
    """Extract physical shape descriptors (Rise-time, Fall-time, FWHM, Peak) 
    for each detected pulse.

    Parameters
    ----------
    signal_data : Signal
        1-D array of the signal (either raw or denoised).
    pulse_indices : ndarray of int
        Sample indices of the detected Times of Arrival (ToA).
    fs : float
        Sampling frequency in Hz.
    window_back_ns : float
        Nanoseconds to look backward from ToA to construct the isolated pulse window.
    window_forward_ns : float
        Nanoseconds to look forward from ToA to construct the isolated pulse window.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the morphological descriptors for each pulse:
        ['toa_idx', 'v_peak', 't_r_ns', 't_f_ns', 'fwhm_ns']
    """
    import pandas as pd
    
    n_samples = len(signal_data)
    win_back_samples = int(window_back_ns * 1e-9 * fs)
    win_fwd_samples = int(window_forward_ns * 1e-9 * fs)
    
    records = []
    
    for toa in pulse_indices:
        start_idx = max(0, toa - win_back_samples)
        end_idx = min(n_samples, toa + win_fwd_samples)
        
        pulse_window = signal_data[start_idx:end_idx]
        if len(pulse_window) < 5:
            continue
            
        # 1. Peak Amplitude
        # Calculate the absolute max relative to the window base
        # Using abs() as PD pulses can be negative
        abs_window = np.abs(pulse_window)
        local_peak_idx = np.argmax(abs_window)
        v_peak = abs_window[local_peak_idx]
        
        if v_peak <= 0:
            continue
            
        # Levels for T_r, T_f, FWHM
        level_10 = 0.10 * v_peak
        level_50 = 0.50 * v_peak
        level_90 = 0.90 * v_peak
        
        # 2. Rise Time (10% to 90%)
        # Look backwards from the peak to find crossing points
        left_side = abs_window[:local_peak_idx+1]
        
        try:
            # Find the last time it crosses 10% and 90% BEFORE the peak
            idx_10 = np.where(left_side <= level_10)[0][-1] if np.any(left_side <= level_10) else 0
            idx_90 = np.where(left_side >= level_90)[0][0] if np.any(left_side >= level_90) else local_peak_idx
            t_r_samples = max(0, idx_90 - idx_10)
            t_r_ns = (t_r_samples / fs) * 1e9
        except IndexError:
            t_r_ns = 0.0
            
        # 3. Fall Time (90% to 10%)
        # Look forwards from the peak
        right_side = abs_window[local_peak_idx:]
        
        try:
            # Find the first time it crosses 90% and 10% AFTER the peak
            idx_90_fall = np.where(right_side <= level_90)[0][0] if np.any(right_side <= level_90) else 0
            idx_10_fall = np.where(right_side <= level_10)[0][0] if np.any(right_side <= level_10) else len(right_side) - 1
            t_f_samples = max(0, idx_10_fall - idx_90_fall)
            t_f_ns = (t_f_samples / fs) * 1e9
        except IndexError:
            t_f_ns = 0.0
            
        # 4. FWHM (Full Width at Half Maximum)
        try:
            idx_50_rise = np.where(left_side <= level_50)[0][-1] if np.any(left_side <= level_50) else 0
            idx_50_fall = np.where(right_side <= level_50)[0][0] + local_peak_idx if np.any(right_side <= level_50) else end_idx - start_idx - 1
            fwhm_samples = max(0, idx_50_fall - idx_50_rise)
            fwhm_ns = (fwhm_samples / fs) * 1e9
        except IndexError:
            fwhm_ns = 0.0
            
        records.append({
            "toa_idx": int(toa),
            "v_peak": float(v_peak),
            "t_r_ns": float(t_r_ns),
            "t_f_ns": float(t_f_ns),
            "fwhm_ns": float(fwhm_ns)
        })
        
    return pd.DataFrame(records)
