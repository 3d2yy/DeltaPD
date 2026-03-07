"""Generate a deterministic synthetic UHF-PD signal (seed=42).

Usage::

    python examples/generate_synthetic.py

Writes ``outputs/synthetic_signal.npz`` containing:
- ``clean``: noise-free signal
- ``noisy``: signal with AWGN + NBI + corona noise
- ``fs``: sampling frequency (float)
"""

import os

import numpy as np

from deltapd.signal_model import generate_uhf_pd_signal_physical


def main():
    seed = 42
    n_samples = 4096
    fs = 1e9

    print(f"Generating synthetic UHF-PD signal (seed={seed}, n={n_samples}) ...")
    clean, noisy = generate_uhf_pd_signal_physical(
        n_samples=n_samples, fs=fs, n_pulses=12, snr_db=20.0, seed=seed
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "synthetic_signal.npz")
    np.savez(out_path, clean=clean, noisy=noisy, fs=np.array(fs))
    print(f"Saved to {out_path}")
    print(f"  clean shape: {clean.shape}, noisy shape: {noisy.shape}")


if __name__ == "__main__":
    main()
