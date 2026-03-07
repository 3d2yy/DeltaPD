"""Tests for improvements: CFAR detector, bootstrap CI, §3 operational metrics."""

import numpy as np

from deltapd.descriptors import detect_pulses, detect_pulses_cfar
from deltapd.signal_model import generate_uhf_pd_signal_physical
from deltapd.validation import BigOEstimate, generate_phase4_report

# ----------------------------------------------------------------
# Improvement 1: CA-CFAR detector
# ----------------------------------------------------------------


def test_cfar_returns_array():
    """detect_pulses_cfar should return an ndarray of indices."""
    _, noisy = generate_uhf_pd_signal_physical(n_samples=4096, fs=1e9, seed=42)
    indices = detect_pulses_cfar(noisy, fs=1e9)
    assert isinstance(indices, np.ndarray)
    assert indices.ndim == 1


def test_cfar_via_detect_pulses():
    """detect_pulses with method='cfar' should dispatch to CFAR detector."""
    _, noisy = generate_uhf_pd_signal_physical(n_samples=4096, fs=1e9, seed=42)
    indices = detect_pulses(noisy, fs=1e9, method="cfar")
    assert isinstance(indices, np.ndarray)


def test_cfar_on_clear_pulses():
    """CFAR should detect clear spikes in a quiet signal."""
    signal = np.zeros(2000)
    signal[200] = 10.0
    signal[600] = 8.0
    signal[1200] = 12.0

    indices = detect_pulses_cfar(signal, fs=1e6, cfar_window=32, cfar_guard=4, pfa=1e-3)
    assert len(indices) >= 3


# ----------------------------------------------------------------
# Improvement 2: Bootstrap CI
# ----------------------------------------------------------------


def test_big_o_has_ci_fields():
    """BigOEstimate should have exponent_b_ci_low and exponent_b_ci_high fields."""
    est = BigOEstimate(
        algorithm_name="test",
        exponent_b=1.0,
        coefficient_a=1e-6,
        r_squared=0.99,
        sizes=np.array([256, 512, 1024]),
        wall_times=np.array([0.001, 0.002, 0.004]),
        big_o_label="O(n^1.00)",
        exponent_b_ci_low=0.95,
        exponent_b_ci_high=1.05,
    )
    assert est.exponent_b_ci_low == 0.95
    assert est.exponent_b_ci_high == 1.05


def test_bootstrap_ci_in_label():
    """After real measurement, the label should contain CI95."""
    from deltapd.validation import measure_algorithm_complexity

    est = measure_algorithm_complexity(
        lambda dt: dt.sum(),
        sizes=(128, 256, 512),
        n_repeats=3,
        seed=42,
        algorithm_name="trivial",
    )
    assert "CI95" in est.big_o_label
    assert est.exponent_b_ci_low <= est.exponent_b
    assert est.exponent_b_ci_high >= est.exponent_b


# ----------------------------------------------------------------
# Improvement 3: §3 Operational Metrics in report
# ----------------------------------------------------------------


def test_report_contains_operational_metrics():
    """Phase 4 report should contain §3 OPERATIONAL METRICS section."""
    from deltapd.validation import (
        generate_convergence_confusion_matrix,
        measure_all_tracking_complexities,
    )

    complexity = measure_all_tracking_complexities(
        sizes=(128, 256), n_repeats=2, seed=42
    )
    confusion = generate_convergence_confusion_matrix(
        n_samples=200, n_monte_carlo=3, seed=42
    )
    report = generate_phase4_report(complexity, confusion)

    assert "OPERATIONAL METRICS" in report
    assert "FAR/min" in report
    assert "Avg Latency" in report
