"""Run the DeltaPD pipeline on a real empirical UHF-PD signal.

Usage::

    python examples/run_empirical.py "<path_to_csv>"

"""

import argparse
import sys
import numpy as np

from deltapd.loader import load_empirical_signal
from deltapd.pipeline import run_phase1, run_phase2, run_phase3, run_phase4
from deltapd.decision_layer import evaluate_campaign, campaign_summary


def main():
    parser = argparse.ArgumentParser(description="Run DeltaPD on empirical CSV/MAT data.")
    parser.add_argument("file_path", help="Path to empirical data file (e.g., .csv)")
    parser.add_argument("--fs", type=float, default=1e9, help="Fallback sampling frequency (Hz)")
    parser.add_argument("--window", type=float, default=0.01, help="Campaign evaluation window size (s)")
    args = parser.parse_args()

    print(f"Loading empirical data from: {args.file_path} ...")
    try:
        signal, fs = load_empirical_signal(args.file_path, default_fs=args.fs)
        print(f"  Loaded {len(signal)} samples at fs = {fs:.2e} Hz")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # For empirical data, we might not have a clean reference for Phase 1. 
    # But the framework's Phase 1 uses a generic synthetic reference to find 
    # the best wavelet configuration independently of the empirical signal.
    # So we can still run Phase 1 to get `mc_result`, then apply it to the empirical signal.
    print("\nRunning stochastic optimization (Phase 1) to find optimal wavelet parameters...")
    mc_result, _, _ = run_phase1(n_samples=4096, fs=fs, n_iterations=30, verbose=False)
    print(f"  Optimal config chosen: {mc_result.best_wavelet}, {mc_result.best_threshold_mode}, {mc_result.best_threshold_rule}")

    # Phase 2 — Extract Δt
    # Using the empirical signal
    delta_t, denoised = run_phase2(
        signal,
        fs=fs,
        mc_result=mc_result,
        threshold_sigma=3.5,  # Higher threshold for real data
        verbose=True
    )

    if len(delta_t) < 3:
        print("\n⚠ Not enough pulses detected in this empirical signal. Exiting.")
        sys.exit(0)

    # Phase 3 — Tracking
    tracking = run_phase3(delta_t, verbose=True)

    # Phase 4 — Quantification (on synthetic sizes for Big-O, but we can print the report)
    # The Big-O and Confusion Matrix are algorithm attributes, independent of the empirical signal,
    # but we run it for completeness.
    _, _, report = run_phase4(verbose=False)
    print("\n" + report)

    # Campaign Evaluation
    print("\n" + "=" * 70)
    print("EMPIRICAL CAMPAIGN SUMMARY")
    print("=" * 70)
    campaign_df = evaluate_campaign(denoised, fs, window_size_s=args.window)
    summary = campaign_summary(campaign_df)
    
    for key, val in summary.items():
        print(f"  {key}: {val}")

    print("\nEmpirical execution finished successfully.")

if __name__ == "__main__":
    main()
