"""Run the DeltaPD pipeline and visualize the results using Matplotlib.

Usage::

    python examples/plot_empirical.py "<path_to_csv>"

"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from deltapd.loader import load_empirical_signal
from deltapd.pipeline import run_phase1, run_phase2, run_phase3
from deltapd.decision_layer import evaluate_campaign


def main():
    parser = argparse.ArgumentParser(description="Visualize DeltaPD results.")
    parser.add_argument("file_path", help="Path to empirical data file (e.g., .csv)")
    parser.add_argument("--fs", type=float, default=1e9, help="Fallback sampling frequency")
    parser.add_argument("--window", type=float, default=0.5, help="Campaign evaluation window size (s)")
    args = parser.parse_args()

    print(f"Loading empirical data from: {args.file_path} ...")
    try:
        signal, fs = load_empirical_signal(args.file_path, default_fs=args.fs)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print("Running stochastic optimization (Phase 1)...")
    mc_result, _, _ = run_phase1(n_samples=4096, fs=fs, n_iterations=10, verbose=False)

    print("Extracting Delta-t (Phase 2)...")
    delta_t, denoised = run_phase2(
        signal, fs=fs, mc_result=mc_result, threshold_sigma=3.5, verbose=False
    )

    if len(delta_t) < 3:
        print("Not enough pulses detected to visualize tracking.")
        sys.exit(0)

    print("Applying Tracking Algorithms (Phase 3)...")
    tracking = run_phase3(delta_t, verbose=False)
    
    # Run the windowed campaign to show confidence score
    print("Evaluating Campaign Density...")
    campaign_df = evaluate_campaign(denoised, fs, window_size_s=args.window)

    # --- Plotting ---
    print("Generating Plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"DeltaPD Analysis: {args.file_path}", fontsize=14)

    # 1. Raw vs Denoised Signal (first 10,000 pts for visibility)
    ax1 = axes[0]
    time_pts = min(10000, len(signal))
    t_axis = np.arange(time_pts) / fs
    ax1.plot(t_axis * 1e6, signal[:time_pts], color='lightgray', label='Raw Signal')
    ax1.plot(t_axis * 1e6, denoised[:time_pts], color='blue', alpha=0.7, label='Denoised Signal')
    ax1.set_title("Time-Domain Signal (first 10k samples)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (µs)")
    ax1.legend()

    # 2. Delta-T Vector with Tracking Alarms
    ax2 = axes[1]
    ax2.plot(delta_t, marker='o', linestyle='-', color='black', alpha=0.5, markersize=3, label='Delta-T Interval')
    
    cusum_alarms = getattr(tracking.cusum, "alarm_indices", [])
    if isinstance(cusum_alarms, (list, np.ndarray)) and len(cusum_alarms) > 0:
        alarm_idx = np.array(cusum_alarms, dtype=int)
        valid_alarms = alarm_idx[alarm_idx < len(delta_t)]
        ax2.scatter(valid_alarms, delta_t[valid_alarms], color='red', s=50, label='CUSUM Alarm')
    
    ax2.set_title("Phase 3: Inter-Pulse Interval (Delta-T) Tracking")
    ax2.set_ylabel("Time between pulses (s)")
    ax2.set_xlabel("Pulse Index")
    ax2.set_yscale('log')
    ax2.legend()

    # 3. Campaign Confidence Score
    ax3 = axes[2]
    windows = np.arange(len(campaign_df)) * args.window
    ax3.plot(windows, campaign_df['confidence'], color='purple', linewidth=2, label='PD Confidence Score')
    ax3.fill_between(windows, campaign_df['confidence'], alpha=0.3, color='purple')
    
    ax3.set_title("Decision Layer: Anomalous PD Probability over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Confidence [0, 1]")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()

    plt.tight_layout()
    print("Plot generated. Close the Matplotlib window to exit the script.")
    plt.show(block=True)

if __name__ == "__main__":
    main()
