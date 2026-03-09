"""Run the full DeltaPD pipeline end-to-end with deterministic outputs.

Usage::

    python examples/run_pipeline.py

Produces always-identical results (seed=42). Outputs written to ``outputs/``:
- ``metrics.json``: quantitative results from all 4 phases
- ``delta_t.csv``: the extracted inter-pulse interval vector
"""

import json
import os

import numpy as np
import pandas as pd

from deltapd.pipeline import run_phase1, run_phase2, run_phase3, run_phase4


def main():
    seed = 42
    n_samples = 4096
    fs = 1e9
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("DeltaPD — Deterministic Pipeline Demo (seed=42)")
    print("=" * 60)

    # Phase 1
    print("\n[Phase 1] Wavelet optimisation ...")
    mc_result, clean, noisy = run_phase1(
        n_samples=n_samples, fs=fs, n_iterations=50, seed=seed, verbose=False
    )
    print(
        f"  Best: {mc_result.best_wavelet}, "
        f"E[RMSE]={mc_result.best_rmse_mean:.6f}, "
        f"converged={mc_result.converged}"
    )

    # Phase 2
    print("\n[Phase 2] Delta-t extraction ...")
    delta_t, denoised = run_phase2(noisy, fs=fs, mc_result=mc_result, verbose=False)
    print(f"  Extracted {len(delta_t)} intervals")
    print(f"  mean={np.mean(delta_t):.3e} s, std={np.std(delta_t):.3e} s")

    # Save delta_t
    dt_path = os.path.join("outputs", "delta_t.csv")
    pd.DataFrame({"delta_t_seconds": delta_t}).to_csv(dt_path, index=False)
    print(f"  Saved to {dt_path}")

    # Phase 3
    print("\n[Phase 3] Tracking (Kalman / EWMA / CUSUM) ...")
    tracking = run_phase3(delta_t, verbose=False)
    print(f"  Kalman gain: {tracking.kalman.steady_state_gain:.6f}")
    print(f"  CUSUM alarms: {tracking.cusum.n_alarms}")

    # Phase 4
    print("\n[Phase 4] Big-O complexity + confusion matrix ...")
    complexity, confusion, report = run_phase4(
        sizes=(256, 512, 1024, 2048, 4096),
        n_repeats=3,
        seed=seed,
        verbose=False,
    )

    # Build metrics dict
    metrics = {
        "seed": seed,
        "n_samples": n_samples,
        "phase1": {
            "best_wavelet": mc_result.best_wavelet,
            "best_threshold_mode": mc_result.best_threshold_mode,
            "best_rmse_mean": mc_result.best_rmse_mean,
            "converged": mc_result.converged,
            "feasibility_rate": mc_result.feasibility_rate,
        },
        "phase2": {
            "n_intervals": len(delta_t),
            "mean_dt_s": float(np.mean(delta_t)),
            "std_dt_s": float(np.std(delta_t)),
        },
        "phase3": {
            "kalman_gain": float(tracking.kalman.steady_state_gain),
            "cusum_alarms": int(tracking.cusum.n_alarms),
        },
        "phase4": {
            algo: {
                "exponent": est.exponent_b,
                "r_squared": est.r_squared,
                "label": est.big_o_label,
            }
            for algo, est in complexity.items()
        },
    }

    metrics_path = os.path.join("outputs", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete. All outputs in outputs/")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
