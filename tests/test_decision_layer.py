"""Tests for deltapd.decision_layer — window-level PD verdicts."""

import numpy as np

from deltapd.decision_layer import campaign_summary, evaluate_campaign, evaluate_window
from deltapd.signal_model import generate_uhf_pd_signal_physical


def test_evaluate_window_returns_correct_keys():
    """evaluate_window should return dict with required keys."""
    _, noisy = generate_uhf_pd_signal_physical(n_samples=4096, fs=1e9, seed=42)
    result = evaluate_window(noisy, fs=1e9)

    assert isinstance(result, dict)
    assert "pd_present" in result
    assert "confidence" in result
    assert "n_events" in result
    assert "far_estimate" in result
    assert "dominant_descriptor" in result
    assert isinstance(result["pd_present"], bool)
    assert 0.0 <= result["confidence"] <= 1.0


def test_evaluate_window_on_noise_only():
    """Pure noise signal should produce low confidence."""
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 0.01, 4096)
    result = evaluate_window(noise, fs=1e9)

    assert result["n_events"] <= 3
    assert result["confidence"] <= 0.5


def test_evaluate_campaign_returns_dataframe():
    """evaluate_campaign should return a DataFrame with expected columns."""
    _, noisy = generate_uhf_pd_signal_physical(n_samples=8192, fs=1e9, seed=42)
    df = evaluate_campaign(noisy, fs=1e9, window_size_s=4e-6)

    assert len(df) >= 1
    assert "t_start" in df.columns
    assert "t_end" in df.columns
    assert "pd_present" in df.columns
    assert "confidence" in df.columns


def test_campaign_summary_empty():
    """campaign_summary on empty DataFrame should return zero summary."""
    import pandas as pd

    empty_df = pd.DataFrame(
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
    summary = campaign_summary(empty_df)
    assert summary["total_windows"] == 0


def test_campaign_summary_with_data():
    """campaign_summary should produce meaningful stats."""
    _, noisy = generate_uhf_pd_signal_physical(n_samples=8192, fs=1e9, seed=42)
    df = evaluate_campaign(noisy, fs=1e9, window_size_s=4e-6)
    summary = campaign_summary(df)

    assert isinstance(summary, dict)
    assert summary["total_windows"] >= 1
    assert 0.0 <= summary["pd_present_pct"] <= 100.0
    assert isinstance(summary["false_alarm_rate_per_min"], float)
