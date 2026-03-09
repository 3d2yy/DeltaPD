from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deltapd.descriptors import detect_pulses
from deltapd.loader import load_empirical_signal
from deltapd.signal_model import wavelet_denoise_parametric


@dataclass
class EventSeries:
    raw_signal: np.ndarray
    processed_signal: np.ndarray
    fs_hz: float
    absolute_times_s: np.ndarray | None
    ingestion_audit: dict[str, object]
    pulse_indices: np.ndarray
    pulse_toa_s: np.ndarray
    pulse_peaks_v: np.ndarray
    event_toa_s: np.ndarray
    event_delta_t_s: np.ndarray
    event_peaks_v: np.ndarray


def _maybe_denoise(
    signal: np.ndarray,
    *,
    wavelet_denoise: bool,
    is_envelope: bool,
    wavelet: str,
    threshold_mode: str,
    threshold_rule: str,
) -> np.ndarray:
    if not wavelet_denoise or is_envelope:
        return np.asarray(signal, dtype=np.float64)
    return np.asarray(
        wavelet_denoise_parametric(
            signal,
            wavelet=wavelet,
            threshold_mode=threshold_mode,
            threshold_rule=threshold_rule,
        ),
        dtype=np.float64,
    )


def extract_event_series(
    signal: np.ndarray,
    *,
    fs_hz: float,
    absolute_times_s: np.ndarray | None = None,
    threshold_sigma: float = 5.0,
    min_separation_s: float = 20e-9,
    detection_method: str = "threshold",
    wavelet_denoise: bool = False,
    is_envelope: bool = False,
    wavelet: str = "db4",
    threshold_mode: str = "soft",
    threshold_rule: str = "universal",
) -> EventSeries:
    raw_signal = np.asarray(signal, dtype=np.float64)
    processed_signal = _maybe_denoise(
        raw_signal,
        wavelet_denoise=wavelet_denoise,
        is_envelope=is_envelope,
        wavelet=wavelet,
        threshold_mode=threshold_mode,
        threshold_rule=threshold_rule,
    )

    pulse_indices = np.asarray(
        detect_pulses(
            signal_data=processed_signal,
            fs=float(fs_hz),
            threshold_sigma=float(threshold_sigma),
            min_separation_s=float(min_separation_s),
            method=str(detection_method),
        ),
        dtype=int,
    )

    if absolute_times_s is not None:
        pulse_toa_s = np.asarray(absolute_times_s, dtype=np.float64)[pulse_indices]
    else:
        pulse_toa_s = pulse_indices.astype(np.float64) / float(fs_hz)
    pulse_peaks_v = np.abs(processed_signal[pulse_indices]).astype(np.float64, copy=False)

    if len(pulse_toa_s) >= 2:
        event_toa_s = pulse_toa_s[1:]
        event_delta_t_s = np.diff(pulse_toa_s)
        event_peaks_v = pulse_peaks_v[1:]
    else:
        event_toa_s = np.array([], dtype=np.float64)
        event_delta_t_s = np.array([], dtype=np.float64)
        event_peaks_v = np.array([], dtype=np.float64)

    return EventSeries(
        raw_signal=raw_signal,
        processed_signal=processed_signal,
        fs_hz=float(fs_hz),
        absolute_times_s=None if absolute_times_s is None else np.asarray(absolute_times_s, dtype=np.float64),
        ingestion_audit={},
        pulse_indices=pulse_indices,
        pulse_toa_s=pulse_toa_s,
        pulse_peaks_v=pulse_peaks_v,
        event_toa_s=event_toa_s,
        event_delta_t_s=event_delta_t_s,
        event_peaks_v=event_peaks_v,
    )


def load_and_extract_event_series(
    file_path: str,
    *,
    preserve_amplitude: bool = True,
    include_absolute_times: bool = True,
    default_fs: float = 1.0e9,
    threshold_sigma: float = 5.0,
    min_separation_s: float = 20e-9,
    detection_method: str = "threshold",
    wavelet_denoise: bool = False,
    is_envelope: bool = False,
    wavelet: str = "db4",
    threshold_mode: str = "soft",
    threshold_rule: str = "universal",
) -> EventSeries:
    load_result = load_empirical_signal(
        str(file_path),
        default_fs=float(default_fs),
        preserve_amplitude=bool(preserve_amplitude),
        include_absolute_times=bool(include_absolute_times),
        include_diagnostics=True,
    )
    if include_absolute_times:
        raw_signal, fs_hz, absolute_times_s, ingestion_audit = load_result
    else:
        raw_signal, fs_hz, ingestion_audit = load_result
        absolute_times_s = None

    extracted = extract_event_series(
        raw_signal,
        fs_hz=float(fs_hz),
        absolute_times_s=absolute_times_s,
        threshold_sigma=float(threshold_sigma),
        min_separation_s=float(min_separation_s),
        detection_method=str(detection_method),
        wavelet_denoise=bool(wavelet_denoise),
        is_envelope=bool(is_envelope),
        wavelet=str(wavelet),
        threshold_mode=str(threshold_mode),
        threshold_rule=str(threshold_rule),
    )
    extracted.ingestion_audit.update(dict(ingestion_audit or {}))
    return extracted
