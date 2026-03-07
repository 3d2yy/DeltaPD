"""Tests for deltapd.loader — data ingestion."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def test_load_csv_valid():
    """Loader should read a simple single-column CSV and return signal + fs."""
    from deltapd.loader import load_empirical_signal

    # Create a temp CSV with known data
    rng = np.random.default_rng(42)
    signal_data = rng.normal(0, 1, 1000)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        pd.DataFrame({"voltage": signal_data}).to_csv(f, index=False)
        f_path = f.name

    try:
        signal, fs = load_empirical_signal(f_path)
        assert isinstance(signal, np.ndarray)
        assert signal.ndim == 1
        assert len(signal) == 1000
        assert fs > 0
    finally:
        os.unlink(f_path)


def test_load_nonexistent_file_raises():
    """Loader should raise on missing file."""
    from deltapd.loader import load_empirical_signal

    with pytest.raises(Exception):
        load_empirical_signal("nonexistent_file.csv")


def test_normalized_signal_zero_mean():
    """After loading, signal should be approximately zero-mean."""
    from deltapd.loader import _normalize_amplitude

    signal = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    normed = _normalize_amplitude(signal)
    assert abs(np.mean(normed)) < 1e-10
    assert np.max(np.abs(normed)) <= 1.0 + 1e-10
