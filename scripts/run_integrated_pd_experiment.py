from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deltapd.blind_prpd import calibrate_grid_frequency, reconstruct_blind_prpd
from deltapd.descriptors import detect_pulses
from deltapd.loader import load_empirical_signal
from deltapd.statistics import compute_burstiness_index, compute_fano_factor


DEFAULT_DATASETS = {
    "P1": "Prueba 1 - Internas",
    "P2": "Prueba 2 - Superficiales",
    "P3": "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas",
}


def _phase_cluster_metrics(phases_deg: np.ndarray, peaks_v: np.ndarray) -> dict[str, float]:
    if len(phases_deg) == 0:
        return {
            "phase_center_pos_deg": float("nan"),
            "phase_center_neg_deg": float("nan"),
            "phase_separation_deg": float("nan"),
            "phase_width_pos_deg": float("nan"),
            "phase_width_neg_deg": float("nan"),
            "phase_spread_deg": float("nan"),
            "inlier_ratio": float("nan"),
            "amplitude_balance_ratio": float("nan"),
            "phase_entropy_global": float("nan"),
        }

    theta2 = np.deg2rad(phases_deg) * 2.0
    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    center1 = np.rad2deg(mean_angle) % 360.0
    center2 = (center1 + 180.0) % 360.0

    d1 = np.minimum(np.abs(phases_deg - center1), 360.0 - np.abs(phases_deg - center1))
    d2 = np.minimum(np.abs(phases_deg - center2), 360.0 - np.abs(phases_deg - center2))
    d_min = np.minimum(d1, d2)

    median_d = np.median(d_min)
    mad = np.median(np.abs(d_min - median_d))
    threshold = median_d + 2.5 * max(mad * 1.4826, 5.0)
    inlier_mask = d_min <= threshold

    pos_mask = phases_deg <= 180.0
    neg_mask = ~pos_mask
    pos_phases = phases_deg[pos_mask]
    neg_phases = phases_deg[neg_mask]
    pos_amp = float(np.sum(np.abs(peaks_v[pos_mask])))
    neg_amp = float(np.sum(np.abs(peaks_v[neg_mask])))
    amp_balance = pos_amp / neg_amp if neg_amp > 0 else float("nan")

    pos_width = (
        float(np.percentile(pos_phases, 90) - np.percentile(pos_phases, 10))
        if len(pos_phases) >= 5
        else float("nan")
    )
    neg_width = (
        float(np.percentile(neg_phases, 90) - np.percentile(neg_phases, 10))
        if len(neg_phases) >= 5
        else float("nan")
    )

    hist, _ = np.histogram(phases_deg, bins=np.linspace(0.0, 360.0, 37))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(36) if len(p) else float("nan")

    pos_center = center1 if center1 <= 180.0 else center2
    neg_center = center2 if center1 <= 180.0 else center1
    separation = (neg_center - pos_center) % 360.0

    return {
        "phase_center_pos_deg": float(pos_center),
        "phase_center_neg_deg": float(neg_center),
        "phase_separation_deg": float(separation),
        "phase_width_pos_deg": pos_width,
        "phase_width_neg_deg": neg_width,
        "phase_spread_deg": float(median_d),
        "inlier_ratio": float(np.mean(inlier_mask)),
        "amplitude_balance_ratio": float(amp_balance),
        "phase_entropy_global": float(entropy),
    }


def _signed_amplitude(phases_deg: np.ndarray, peaks_v: np.ndarray) -> np.ndarray:
    return np.where((phases_deg >= 0.0) & (phases_deg <= 180.0), peaks_v, -peaks_v)


def _compute_case_metrics(
    file_path: Path,
    dataset_key: str,
    dataset_label: str,
    threshold_sigma: float,
    min_separation_s: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    signal, fs_hz, times_abs_s = load_empirical_signal(
        str(file_path),
        preserve_amplitude=True,
        include_absolute_times=True,
    )

    pulse_idx = detect_pulses(
        signal,
        fs_hz,
        threshold_sigma=threshold_sigma,
        min_separation_s=min_separation_s,
        method="threshold",
    )
    toa_s = times_abs_s[pulse_idx]
    peaks_v = np.abs(signal[pulse_idx])

    if len(toa_s) < 10:
        raise ValueError(f"Not enough pulses in {file_path}")

    delta_t_s = np.diff(toa_s)
    blind_freq_hz = calibrate_grid_frequency(toa_s, base_freq=50.0)
    phases_deg, peaks_out = reconstruct_blind_prpd(
        toa_s,
        peaks_v,
        freq_hz=50.0,
        auto_calibrate=True,
    )

    burstiness_series = compute_burstiness_index(
        delta_t_s,
        window=min(100, max(20, len(delta_t_s) // 10)),
        min_periods=10,
    )
    _, fano_vals = compute_fano_factor(toa_s, bin_duration_s=0.1, window_bins=20, min_bins=5)
    phase_metrics = _phase_cluster_metrics(phases_deg, peaks_out)

    result = {
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "file_path": str(file_path),
        "fs_hz": float(fs_hz),
        "n_pulses": int(len(toa_s)),
        "n_delta_t": int(len(delta_t_s)),
        "median_dt_s": float(np.median(delta_t_s)),
        "iqr_dt_s": float(np.percentile(delta_t_s, 75) - np.percentile(delta_t_s, 25)),
        "cv_dt": float(np.std(delta_t_s) / np.mean(delta_t_s)),
        "burstiness_global": float(np.nanmedian(burstiness_series)),
        "fano_global": float(np.nanmedian(fano_vals)) if len(fano_vals) else float("nan"),
        "blind_freq_hz": float(blind_freq_hz),
        "peak_mean_v": float(np.mean(peaks_out)),
        "peak_p90_v": float(np.percentile(peaks_out, 90)),
    }
    result.update(phase_metrics)

    df_case = pd.DataFrame(
        {
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "toa_s": toa_s,
            "peak_v": peaks_out,
            "prpd_phase_deg": phases_deg,
            "signed_peak_v": _signed_amplitude(phases_deg, peaks_out),
        }
    )
    return result, df_case


def _plot_prpd_comparison(df_all: pd.DataFrame, out_png: Path) -> None:
    keys = list(df_all["dataset_key"].unique())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5.5), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        df_case = df_all[df_all["dataset_key"] == key].copy()
        x = df_case["prpd_phase_deg"].to_numpy()
        y = df_case["signed_peak_v"].to_numpy()
        ax.scatter(x, y, s=10, alpha=0.65, edgecolors="none")
        max_amp = np.max(np.abs(y)) * 1.05
        t_sin = np.linspace(0.0, 360.0, 360)
        ax.plot(t_sin, max_amp * np.sin(np.radians(t_sin)), color="red", alpha=0.45, linewidth=1.4)
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks(np.arange(0.0, 361.0, 45.0))
        ax.set_title(df_case["dataset_label"].iloc[0])
        ax.set_xlabel("Fase (grados)")
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Amplitud aparente firmada (V)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    base_dir: Path,
    out_dir: Path,
    channel: str,
    threshold_sigma: float,
    min_separation_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    prpd_frames: list[pd.DataFrame] = []

    for dataset_key, dataset_label in DEFAULT_DATASETS.items():
        file_path = base_dir / dataset_label / f"{channel}.csv"
        row, df_case = _compute_case_metrics(
            file_path=file_path,
            dataset_key=dataset_key,
            dataset_label=dataset_label,
            threshold_sigma=threshold_sigma,
            min_separation_s=min_separation_s,
        )
        rows.append(row)
        prpd_frames.append(df_case)

    metrics_df = pd.DataFrame(rows)
    prpd_df = pd.concat(prpd_frames, ignore_index=True)

    metrics_df.to_csv(out_dir / "integrated_pd_metrics_p1_p2_p3.csv", index=False, encoding="utf-8-sig")
    prpd_df.to_csv(out_dir / "integrated_pd_prpd_points_p1_p2_p3.csv", index=False, encoding="utf-8-sig")
    _plot_prpd_comparison(prpd_df, out_dir / "integrated_pd_prpd_comparison_p1_p2_p3.png")

    return metrics_df, prpd_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrated Delta t + blind PRPD experiment for P1/P2/P3."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory containing P1, P2 and P3 CSV folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/integrated_pd_experiment"),
        help="Output directory for metrics and figures.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="CH3",
        help="Channel to analyze across P1/P2/P3.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=5.0,
        help="Detection threshold in sigma units.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=20e-9,
        help="Minimum pulse separation in seconds.",
    )
    args = parser.parse_args()

    metrics_df, _ = run_experiment(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        channel=args.channel,
        threshold_sigma=args.threshold_sigma,
        min_separation_s=args.min_separation_s,
    )
    print(metrics_df.to_string(index=False))
    print(f"CSV={args.out_dir / 'integrated_pd_metrics_p1_p2_p3.csv'}")
    print(f"PNG={args.out_dir / 'integrated_pd_prpd_comparison_p1_p2_p3.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

