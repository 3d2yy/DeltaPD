"""Tests for deltapd.descriptors — pulse detection and delta-t extraction."""

import numpy as np

from deltapd.descriptors import compute_delta_t, detect_pulses, extract_delta_t_vector


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
