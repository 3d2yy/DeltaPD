from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

"""Shared fixtures for DeltaPD tests."""

import pytest

from deltapd.signal_model import generate_uhf_pd_signal_physical

SEED = 42
N_SAMPLES = 4096
FS = 1e9


@pytest.fixture(scope="session")
def synthetic_signal():
    """Generate a deterministic synthetic UHF-PD signal pair."""
    clean, noisy = generate_uhf_pd_signal_physical(
        n_samples=N_SAMPLES, fs=FS, n_pulses=12, snr_db=20.0, seed=SEED
    )
    return clean, noisy, FS


@pytest.fixture(scope="session")
def delta_t_vector(synthetic_signal):
    """Extract delta-t from the synthetic signal."""
    from deltapd.descriptors import extract_delta_t_vector
    from deltapd.signal_model import wavelet_denoise_parametric

    _clean, noisy, fs = synthetic_signal
    denoised = wavelet_denoise_parametric(
        noisy, wavelet="db4", threshold_mode="soft", threshold_rule="universal"
    )
    delta_t = extract_delta_t_vector(denoised, fs)
    return delta_t
