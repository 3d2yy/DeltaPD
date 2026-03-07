from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]


def _bandpower(freqs_hz: NDArray[np.float64], psd: NDArray[np.float64], low_ghz: float, high_ghz: float) -> float:
    low_hz = low_ghz * 1e9
    high_hz = high_ghz * 1e9
    mask = (freqs_hz >= low_hz) & (freqs_hz < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs_hz[mask]))


def compute_spectral_metrics(signal: Signal, fs: float, bands_ghz: list[list[float]]) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(spec) ** 2) / max(n, 1)

    out: dict[str, float] = {}
    total_power = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else float(np.sum(psd))
    out["spectral_total_power"] = total_power

    for low, high in bands_ghz:
        key = f"bandpower_{low:.2f}_{high:.2f}GHz".replace(".", "p")
        p = _bandpower(freqs, psd, low, high)
        out[key] = p
        out[key + "_share"] = (p / total_power) if total_power > 0 else 0.0

    if len(bands_ghz) >= 2:
        low_key = f"bandpower_{bands_ghz[0][0]:.2f}_{bands_ghz[0][1]:.2f}GHz".replace(".", "p")
        high_key = f"bandpower_{bands_ghz[-1][0]:.2f}_{bands_ghz[-1][1]:.2f}GHz".replace(".", "p")
        low_p = out.get(low_key, 0.0)
        high_p = out.get(high_key, 0.0)
        out["ratio_high_low"] = float(high_p / low_p) if low_p > 0 else float("inf")

    centroid = float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
    out["spectral_centroid_hz"] = centroid
    return out
