"""Tests for deltapd.trackers — Kalman, EWMA, CUSUM."""

import numpy as np

from deltapd.trackers import (
    AdaptiveEWMATracker,
    CUSUMDetector,
    KalmanDeltaTTracker,
    apply_delta_t_tracking,
)


def test_kalman_output_shape(delta_t_vector):
    """Kalman tracker should produce arrays matching input length."""
    tracker = KalmanDeltaTTracker()
    result = tracker.track(delta_t_vector)

    assert result.filtered.shape == delta_t_vector.shape
    assert result.residuals.shape == delta_t_vector.shape
    assert isinstance(result.steady_state_gain, float)


def test_ewma_output_shape(delta_t_vector):
    """EWMA tracker should produce arrays matching input length."""
    tracker = AdaptiveEWMATracker()
    result = tracker.track(delta_t_vector)

    assert result.smoothed.shape == delta_t_vector.shape
    assert result.alpha_sequence.shape == delta_t_vector.shape


def test_cusum_on_stable_signal():
    """CUSUM on a constant signal should produce zero alarms."""
    stable = np.ones(200) * 1e-4
    detector = CUSUMDetector(threshold=5.0, drift=0.5)
    result = detector.detect(stable)

    assert result.n_alarms == 0
    assert len(result.alarm_indices) == 0


def test_cusum_detects_shift():
    """CUSUM should detect a clear mean shift."""
    rng = np.random.default_rng(42)
    stable = rng.normal(1.0, 0.01, 200)
    shifted = rng.normal(3.0, 0.01, 200)
    signal = np.concatenate([stable, shifted])

    detector = CUSUMDetector(threshold=2.0, drift=0.5)
    result = detector.detect(signal)

    assert result.n_alarms > 0
    # First alarm should be near the shift point (sample 200)
    assert result.alarm_indices[0] >= 190


def test_apply_delta_t_tracking_returns_all(delta_t_vector):
    """Unified tracking should return results for all 3 algorithms."""
    result = apply_delta_t_tracking(delta_t_vector)
    assert hasattr(result, "kalman")
    assert hasattr(result, "ewma")
    assert hasattr(result, "cusum")
