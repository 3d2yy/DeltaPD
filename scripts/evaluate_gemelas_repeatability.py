from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from deltapd.blind_prpd import calibrate_grid_frequency
from deltapd.descriptors import detect_pulses
from deltapd.loader import load_empirical_signal


GEMELAS_DATASETS = {
    "G1": "Prueba 1 - Internas (Gemelas)",
    "G2": "Prueba 2 - Superficiales (Gemelas)",
    "G3": "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas (Gemelas)",
}

CHANNEL_MAP = {
    "CH2": "Gemela Vivaldi antipodal propuesta",
    "CH3": "Vivaldi antipodal propuesta",
    "CH4": "Bioinspirada",
}


def _center_phases(phases_deg: np.ndarray, target_deg: float = 70.0) -> np.ndarray:
    theta = np.deg2rad(phases_deg) * 2.0
    avg_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))) / 2.0
    avg_deg = np.rad2deg(avg_theta)
    shift_deg = target_deg - avg_deg
    return np.mod(phases_deg + shift_deg, 360.0)


def _phase_metrics(phases_deg: np.ndarray, peaks_v: np.ndarray) -> dict[str, float]:
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
    inlier_ratio = float(np.mean(d_min <= threshold))

    pos_mask = phases_deg <= 180.0
    neg_mask = ~pos_mask
    pos_phases = phases_deg[pos_mask]
    neg_phases = phases_deg[neg_mask]

    hist, _ = np.histogram(phases_deg, bins=np.linspace(0.0, 360.0, 37))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(36) if len(p) else float("nan")

    pos_amp = float(np.sum(np.abs(peaks_v[pos_mask])))
    neg_amp = float(np.sum(np.abs(peaks_v[neg_mask])))

    return {
        "phase_entropy_global": float(entropy),
        "phase_spread_deg": float(median_d),
        "inlier_ratio": inlier_ratio,
        "phase_width_pos_deg": float(np.percentile(pos_phases, 90) - np.percentile(pos_phases, 10)),
        "phase_width_neg_deg": float(np.percentile(neg_phases, 90) - np.percentile(neg_phases, 10)),
        "amplitude_balance_ratio": float(pos_amp / neg_amp) if neg_amp > 0 else float("nan"),
    }


def _signed_amplitude(phases_deg: np.ndarray, peaks_v: np.ndarray) -> np.ndarray:
    return np.where((phases_deg >= 0.0) & (phases_deg <= 180.0), peaks_v, -peaks_v)


def _process_case(file_path: Path, threshold_sigma: float, min_separation_s: float) -> tuple[dict[str, float], pd.DataFrame]:
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

    blind_freq_hz = calibrate_grid_frequency(toa_s, base_freq=50.0, search_width=0.2, steps=50000)
    period_s = 1.0 / blind_freq_hz
    phases_deg = np.mod(toa_s, period_s) / period_s * 360.0
    phases_deg = _center_phases(phases_deg)

    row = {
        "n_pulses": int(len(toa_s)),
        "blind_freq_hz": float(blind_freq_hz),
        "peak_mean_v": float(np.mean(peaks_v)),
        "peak_p90_v": float(np.percentile(peaks_v, 90)),
    }
    row.update(_phase_metrics(phases_deg, peaks_v))

    df_points = pd.DataFrame(
        {
            "toa_s": toa_s,
            "peak_v": peaks_v,
            "prpd_phase_deg": phases_deg,
            "signed_peak_v": _signed_amplitude(phases_deg, peaks_v),
        }
    )
    return row, df_points


def _plot_case(df_case: pd.DataFrame, out_png: Path, title: str) -> None:
    grouped = list(df_case.groupby("antenna"))
    fig, axes = plt.subplots(1, len(grouped), figsize=(6 * len(grouped), 5.5), sharey=True)
    if len(grouped) == 1:
        axes = [axes]

    for ax, (antenna, df_ant) in zip(axes, grouped):
        x = df_ant["prpd_phase_deg"].to_numpy()
        y = df_ant["signed_peak_v"].to_numpy()
        weights = np.abs(y)
        kde = gaussian_kde(np.vstack([x, y]), bw_method=0.15, weights=weights)
        z = kde(np.vstack([x, y]))
        order = z.argsort()
        sizes = 6 + 24 * (np.abs(y[order]) / np.max(np.abs(y[order])))
        ax.scatter(x[order], y[order], c=z[order], cmap="turbo", s=sizes, alpha=0.9, edgecolors="none")
        t_sin = np.linspace(0.0, 360.0, 360)
        max_amp = np.max(np.abs(y)) * 1.05
        ax.plot(t_sin, max_amp * np.sin(np.radians(t_sin)), color="red", alpha=0.45, linewidth=1.3)
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks(np.arange(0.0, 361.0, 45.0))
        ax.set_xlabel("Fase (grados)")
        ax.set_title(antenna)
        ax.grid(True, linestyle=":", alpha=0.35)
    axes[0].set_ylabel("Amplitud aparente firmada (V)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_repeatability(
    base_dir: Path,
    out_dir: Path,
    threshold_sigma: float,
    min_separation_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    point_frames: list[pd.DataFrame] = []

    for dataset_key, folder in GEMELAS_DATASETS.items():
        case_frames: list[pd.DataFrame] = []
        for channel, antenna in CHANNEL_MAP.items():
            file_path = base_dir / folder / f"{channel}.csv"
            row, df_points = _process_case(file_path, threshold_sigma=threshold_sigma, min_separation_s=min_separation_s)
            row.update(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": folder,
                    "channel": channel,
                    "antenna": antenna,
                    "file_path": str(file_path),
                }
            )
            rows.append(row)

            df_points.insert(0, "dataset_key", dataset_key)
            df_points.insert(1, "dataset_label", folder)
            df_points.insert(2, "channel", channel)
            df_points.insert(3, "antenna", antenna)
            point_frames.append(df_points)
            case_frames.append(df_points)

        _plot_case(
            pd.concat(case_frames, ignore_index=True),
            out_dir / f"{dataset_key.lower()}_gemelas_prpd_comparison.png",
            title=folder,
        )

    metrics_df = pd.DataFrame(rows)
    points_df = pd.concat(point_frames, ignore_index=True)

    diff_rows = []
    for dataset_key, df_case in metrics_df.groupby("dataset_key"):
        if len(df_case) < 2:
            continue
        df_case = df_case.set_index("antenna")
        antenna_names = list(df_case.index)
        for idx_a, ant_a in enumerate(antenna_names):
            for ant_b in antenna_names[idx_a + 1 :]:
                row_a = df_case.loc[ant_a]
                row_b = df_case.loc[ant_b]
                diff_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset_label": row_a["dataset_label"],
                        "antenna_a": ant_a,
                        "antenna_b": ant_b,
                        "freq_diff_hz": float(abs(row_a["blind_freq_hz"] - row_b["blind_freq_hz"])),
                        "entropy_diff": float(abs(row_a["phase_entropy_global"] - row_b["phase_entropy_global"])),
                        "spread_diff_deg": float(abs(row_a["phase_spread_deg"] - row_b["phase_spread_deg"])),
                        "width_pos_diff_deg": float(abs(row_a["phase_width_pos_deg"] - row_b["phase_width_pos_deg"])),
                        "width_neg_diff_deg": float(abs(row_a["phase_width_neg_deg"] - row_b["phase_width_neg_deg"])),
                        "peak_mean_ratio_a_over_b": float(row_a["peak_mean_v"] / row_b["peak_mean_v"]) if row_b["peak_mean_v"] > 0 else float("nan"),
                    }
                )

    diff_df = pd.DataFrame(diff_rows)
    metrics_df.to_csv(out_dir / "gemelas_repeatability_metrics.csv", index=False, encoding="utf-8-sig")
    points_df.to_csv(out_dir / "gemelas_repeatability_points.csv", index=False, encoding="utf-8-sig")
    diff_df.to_csv(out_dir / "gemelas_repeatability_differences.csv", index=False, encoding="utf-8-sig")
    return metrics_df, diff_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate blind PRPD repeatability on gemelas datasets.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory containing Gemelas waveform folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/gemelas_repeatability"),
        help="Output folder for repeatability tables and figures.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=5.0,
        help="Threshold sigma for pulse detection.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=20e-9,
        help="Minimum separation between pulses in seconds.",
    )
    args = parser.parse_args()

    metrics_df, diff_df = run_repeatability(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        threshold_sigma=args.threshold_sigma,
        min_separation_s=args.min_separation_s,
    )
    print(metrics_df.to_string(index=False))
    print(diff_df.to_string(index=False))
    print(f"CSV={args.out_dir / 'gemelas_repeatability_metrics.csv'}")
    print(f"CSV_DIFF={args.out_dir / 'gemelas_repeatability_differences.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

