"""Tests for deltapd.descriptors — pulse detection and delta-t extraction."""

import numpy as np
import pytest

from deltapd.descriptors import (
    compute_delta_t,
    detect_pulses,
    detect_pulses_cfar,
    extract_delta_t_vector,
)


def _legacy_detect_pulses_cfar(
    signal_data: np.ndarray,
    *,
    fs: float,
    cfar_window: int,
    cfar_guard: int,
    pfa: float,
    min_separation_s: float | None = None,
) -> np.ndarray:
    data = np.asarray(signal_data, dtype=np.float64)
    power = data**2
    n = len(data)
    w = int(cfar_window)
    g = int(cfar_guard)
    margin = w + g
    detections = np.zeros(n, dtype=bool)
    alpha_cfar = w * (pfa ** (-1.0 / w) - 1.0)

    for k in range(margin, n - margin):
        left = power[k - margin : k - g]
        right = power[k + g + 1 : k + margin + 1]
        noise_est = np.mean(np.concatenate([left, right]))
        detections[k] = power[k] > alpha_cfar * noise_est

    detection_idx = np.flatnonzero(detections)
    if len(detection_idx) == 0:
        return np.array([], dtype=np.intp)

    if min_separation_s is None:
        min_distance = 5
    else:
        min_distance = max(1, int(np.ceil(float(min_separation_s) * fs)))
    gaps = np.diff(detection_idx)
    cluster_starts = np.insert(gaps > min_distance, 0, True)
    return detection_idx[cluster_starts].astype(np.intp)


def test_detect_pulses_on_spiky_signal():
    """Pulse detector should find peaks in a signal with clear spikes."""
    signal = np.zeros(1000)
    signal[100] = 10.0
    signal[300] = 8.0
    signal[700] = 12.0

    pulses = detect_pulses(signal, fs=1e6, threshold_sigma=2.0)
    assert len(pulses) >= 3
    assert 100 in pulses
    assert 300 in pulses
    assert 700 in pulses


def test_compute_delta_t_values():
    """Delta-t should be correctly computed from pulse indices."""
    indices = np.array([100, 300, 700])
    fs = 1e6
    dt = compute_delta_t(indices, fs)

    assert len(dt) == 2
    np.testing.assert_allclose(dt[0], 200 / fs)
    np.testing.assert_allclose(dt[1], 400 / fs)


def test_delta_t_all_positive():
    """All delta-t values must be strictly positive."""
    indices = np.array([10, 50, 120, 300, 500])
    dt = compute_delta_t(indices, fs=1.0)
    assert np.all(dt > 0)


def test_extract_delta_t_shape(synthetic_signal):
    """Extracted delta-t from a real signal should have correct shape."""
    _clean, noisy, fs = synthetic_signal
    dt = extract_delta_t_vector(noisy, fs)
    assert dt.ndim == 1
    assert len(dt) >= 1
    assert np.all(dt > 0)


def test_detect_pulses_default_refractory_gap_blocks_adjacent_duplicates():
    """The default detector should not keep sub-refractory duplicate picks."""
    signal = np.zeros(500)
    signal[100] = 10.0
    signal[101] = 9.0
    signal[300] = 11.0

    pulses = detect_pulses(signal, fs=1e9, threshold_sigma=2.0)

    assert len(pulses) == 2
    assert 100 in pulses
    assert 300 in pulses


def test_detect_pulses_warns_and_falls_back_when_zero_refractory_is_requested():
    """Explicit zero refractory should warn and use the automatic floor."""
    signal = np.zeros(500)
    signal[120] = 8.0
    signal[121] = 7.0
    signal[320] = 9.0

    with pytest.warns(RuntimeWarning, match="falling back to the automatic default"):
        pulses = detect_pulses(signal, fs=1e9, threshold_sigma=2.0, min_separation_s=0.0)

    assert len(pulses) == 2


def test_detect_pulses_cfar_matches_legacy_reference_on_simple_signal():
    signal = np.zeros(256, dtype=np.float64)
    signal[40:43] = [5.0, 4.0, 3.0]
    signal[128] = 6.0
    signal[200:202] = [5.5, 4.5]

    expected = _legacy_detect_pulses_cfar(
        signal,
        fs=1e9,
        cfar_window=8,
        cfar_guard=2,
        pfa=1e-3,
    )
    observed = detect_pulses_cfar(
        signal,
        fs=1e9,
        cfar_window=8,
        cfar_guard=2,
        pfa=1e-3,
    )

    np.testing.assert_array_equal(observed, expected)


def test_detect_pulses_cfar_ignores_edges_without_full_training_support():
    signal = np.zeros(80, dtype=np.float64)
    signal[5] = 10.0
    signal[40] = 12.0

    observed = detect_pulses_cfar(
        signal,
        fs=1e9,
        cfar_window=8,
        cfar_guard=2,
        pfa=1e-3,
    )

    assert 40 in observed
    assert 5 not in observed


def test_detect_pulses_cfar_returns_empty_when_signal_is_shorter_than_margin():
    signal = np.zeros(16, dtype=np.float64)
    observed = detect_pulses_cfar(
        signal,
        fs=1e9,
        cfar_window=8,
        cfar_guard=2,
        pfa=1e-3,
    )

    assert observed.size == 0
