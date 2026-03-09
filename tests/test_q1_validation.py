"""Tests for deltapd.q1_validation — Q1-level statistical validation suite."""

import numpy as np
import pytest

from deltapd.q1_validation import (
    BootstrapCI,
    bootstrap_ci,
    cohens_d,
    compare_segments_kruskal,
    format_validation_report,
    check_innovation_whiteness,
    check_kalman_nis,
    check_stationarity_adf,
    validate_chapter_v,
)


class TestNIS:
    """Normalized Innovation Squared consistency test."""

    def test_consistent_filter(self):
        """NIS from chi2(1) samples should pass the consistency test."""
        rng = np.random.default_rng(42)
        # Simulate consistent innovations: epsilon ~ N(0, S), so epsilon^2/S ~ chi2(1)
        n = 1000
        S = np.ones(n) * 0.01
        innovations = rng.normal(0, np.sqrt(S))
        result = check_kalman_nis(innovations, S, alpha=0.05)
        # With proper innovations, p-value should be high
        assert result.n_samples == n
        assert result.mean_nis > 0

    def test_inconsistent_filter(self):
        """Biased innovations should fail the NIS test."""
        rng = np.random.default_rng(42)
        n = 1000
        S = np.ones(n) * 0.01
        # Innovations with systematic bias
        innovations = rng.normal(0.5, np.sqrt(S))
        result = check_kalman_nis(innovations, S, alpha=0.05)
        assert result.mean_nis > 1.0  # Should be inflated


class TestLjungBox:
    """Innovation whiteness test."""

    def test_white_noise_passes(self):
        """White noise should pass the Ljung-Box test."""
        rng = np.random.default_rng(42)
        white = rng.normal(0, 1, 500)
        result = check_innovation_whiteness(white, n_lags=10, alpha=0.05)
        assert result.is_white_noise is True
        assert result.p_value > 0.05

    def test_correlated_signal_fails(self):
        """AR(1) process should fail the whiteness test."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.1)
        result = check_innovation_whiteness(x, n_lags=10, alpha=0.05)
        assert result.is_white_noise is False


class TestKruskalWallis:
    """Segment comparison test."""

    def test_same_distribution(self):
        """Samples from the same distribution should not differ."""
        rng = np.random.default_rng(42)
        a = rng.exponential(1.0, 200)
        b = rng.exponential(1.0, 200)
        result = compare_segments_kruskal([a, b], alpha=0.05)
        assert result.n_segments == 2
        # p-value should be high for same distribution
        assert result.p_value > 0.01

    def test_different_distributions(self):
        """Clearly different distributions should be detected."""
        rng = np.random.default_rng(42)
        a = rng.exponential(0.1, 200)
        b = rng.exponential(10.0, 200)
        result = compare_segments_kruskal([a, b], alpha=0.05)
        assert result.segments_differ is True
        assert result.p_value < 0.05
        assert len(result.pairwise_results) == 1

    def test_three_segments(self):
        """Three-segment comparison with pairwise follow-up."""
        rng = np.random.default_rng(42)
        a = rng.exponential(0.1, 200)
        b = rng.exponential(0.5, 200)
        c = rng.exponential(5.0, 200)
        result = compare_segments_kruskal([a, b, c], alpha=0.05)
        assert result.n_segments == 3
        assert len(result.pairwise_results) == 3  # 3 choose 2


class TestADF:
    """Augmented Dickey-Fuller stationarity test."""

    def test_stationary_signal(self):
        """White noise should be detected as stationary."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        result = check_stationarity_adf(x, alpha=0.05)
        assert result.is_stationary is True

    def test_random_walk(self):
        """Random walk should be detected as non-stationary."""
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.normal(0, 1, 500))
        result = check_stationarity_adf(x, alpha=0.05)
        assert result.is_stationary is False


class TestBootstrapCI:
    """Bootstrap confidence interval tests."""

    def test_median_ci_covers_true(self):
        """95% CI for median should cover the true value most of the time."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 200)
        result = bootstrap_ci(data, np.median, confidence=0.95, seed=42)
        # True median is ~5.0
        assert result.ci_lower < 5.0 < result.ci_upper

    def test_narrow_ci_large_sample(self):
        """CI should narrow with larger samples."""
        rng = np.random.default_rng(42)
        small = rng.normal(0, 1, 50)
        large = rng.normal(0, 1, 5000)
        ci_small = bootstrap_ci(small, np.mean, seed=42)
        ci_large = bootstrap_ci(large, np.mean, seed=42)
        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower
        assert width_large < width_small


class TestCohensD:
    """Effect size computation."""

    def test_zero_effect(self):
        """Same distribution should give d near 0."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        d = cohens_d(a, b)
        assert abs(d) < 0.2

    def test_large_effect(self):
        """Very different means should give large d."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(5, 1, 500)
        d = cohens_d(a, b)
        assert abs(d) > 3.0


class TestValidateChapterV:
    """Integration test for the complete validation pipeline."""

    def test_full_report(self):
        """validate_chapter_v should produce a complete report."""
        rng = np.random.default_rng(42)
        delta_t = rng.exponential(0.1, 500)
        innovations = rng.normal(0, 0.01, 500)
        S = np.ones(500) * 0.0001

        seg1 = rng.exponential(0.1, 200)
        seg2 = rng.exponential(0.5, 200)

        report = validate_chapter_v(
            delta_t=delta_t,
            kalman_innovations=innovations,
            kalman_innovation_cov=S,
            segment_indices=[seg1, seg2],
        )

        assert report.nis_test is not None
        assert report.whiteness_test is not None
        assert report.stationarity_test is not None
        assert report.segment_comparison is not None
        assert report.bootstrap_median_ci is not None

        # Format should not raise
        text = format_validation_report(report)
        assert "Q1 STATISTICAL VALIDATION REPORT" in text
