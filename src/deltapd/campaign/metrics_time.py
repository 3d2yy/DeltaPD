from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]


def _noise_window_samples(fs: float, noise_window_ns: float | None = None) -> int:
    if noise_window_ns is None:
        return 64
    return max(16, int((noise_window_ns * 1e-9) * fs))


def compute_time_metrics(signal: Signal, fs: float, noise_window_ns: float | None = None) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Signal must be a non-empty 1-D array.")

    n_noise = min(_noise_window_samples(fs, noise_window_ns), len(x))
    noise = x[:n_noise]
    noise_rms = float(np.sqrt(np.mean(noise**2))) if n_noise else 0.0

    vpp = float(np.max(x) - np.min(x))
    peak_abs = float(np.max(np.abs(x)))
    energy = float(np.sum(x**2) / fs)
    snr_db = float(20.0 * np.log10(peak_abs / noise_rms)) if noise_rms > 0 else float("inf")
    z_peak = float(peak_abs / noise_rms) if noise_rms > 0 else float("inf")

    return {
        "vpp": vpp,
        "peak_abs": peak_abs,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
        "energy": energy,
        "z_peak": z_peak,
    }
