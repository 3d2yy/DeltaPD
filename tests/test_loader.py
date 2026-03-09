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


def test_load_keysight_segmented_csv_uses_metadata_xinc_and_absolute_times():
    from deltapd.loader import load_empirical_signal

    lines = [
        ",Channel 3,Channel 3",
        "Points:,4,4",
        "XInc:,1.00000000E-09,1.00000000E-09",
        "XOrg:,-1.00000000E-09,-1.00000000E-09",
        "Date:,19 NOV 2025,19 NOV 2025",
        "Time:,14:23:42,14:23:42",
        "Time Since Seg 1:,0.0E+00,2.0E-01",
        "Time Tags (Channel 3),Channel 3,Channel 3",
        "-1.0E-09,0.1,1.99999999E-01,0.5",
        "0.0E+00,0.2,2.00000000E-01,0.6",
        "1.0E-09,0.3,2.00000001E-01,0.7",
        "2.0E-09,0.4,2.00000002E-01,0.8",
    ]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f_path = f.name

    try:
        signal, fs, times = load_empirical_signal(
            f_path,
            preserve_amplitude=True,
            include_absolute_times=True,
        )
        assert len(signal) == 8
        assert np.isclose(fs, 1e9)
        assert np.isclose(times[0], -1e-9)
        assert np.isclose(times[4], 0.199999999)
        assert np.isclose(np.diff(times[:4]).mean(), 1e-9)
    finally:
        os.unlink(f_path)


def test_load_keysight_segmented_csv_reconstructs_segment_offsets_when_time_columns_are_local():
    from deltapd.loader import load_empirical_signal

    lines = [
        ",Channel 3,Channel 3",
        "Points:,4,4",
        "XInc:,1.00000000E-09,1.00000000E-09",
        "XOrg:,-1.00000000E-09,-1.00000000E-09",
        "Date:,19 NOV 2025,19 NOV 2025",
        "Time:,14:23:42,14:23:42",
        "Time Since Seg 1:,0.0E+00,2.0E-01",
        "Time Tags (Channel 3),Channel 3,Channel 3",
        "-1.0E-09,0.1,-1.0E-09,0.5",
        "0.0E+00,0.2,0.0E+00,0.6",
        "1.0E-09,0.3,1.0E-09,0.7",
        "2.0E-09,0.4,2.0E-09,0.8",
    ]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f_path = f.name

    try:
        signal, fs, times = load_empirical_signal(
            f_path,
            preserve_amplitude=True,
            include_absolute_times=True,
        )
        assert len(signal) == 8
        assert np.isclose(fs, 1e9)
        assert times[4] > 0.19
        assert np.isclose(times[4], 0.199999999)
    finally:
        os.unlink(f_path)
