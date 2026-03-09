from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deltapd.blind_prpd import calibrate_grid_frequency
from deltapd.descriptors import detect_pulses
from deltapd.loader import load_empirical_signal


DATASETS = {
    "P1": "Prueba 1 - Internas",
    "P2": "Prueba 2 - Superficiales",
    "P3": "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas",
}


def _center_phases(phases_deg: np.ndarray, target_deg: float = 70.0) -> np.ndarray:
    theta = np.deg2rad(phases_deg) * 2.0
    avg_theta = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))) / 2.0
    avg_deg = np.rad2deg(avg_theta)
    shift_deg = target_deg - avg_deg
    return np.mod(phases_deg + shift_deg, 360.0)


def _phases_from_frequency(toa_s: np.ndarray, freq_hz: float) -> np.ndarray:
    period_s = 1.0 / freq_hz
    phases_deg = np.mod(toa_s, period_s) / period_s * 360.0
    return _center_phases(phases_deg)


def _phase_metrics(phases_deg: np.ndarray) -> dict[str, float]:
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

    hist, _ = np.histogram(phases_deg, bins=np.linspace(0.0, 360.0, 37))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(36) if len(p) else float("nan")

    return {
        "phase_entropy_global": float(entropy),
        "phase_spread_deg": float(median_d),
        "inlier_ratio": inlier_ratio,
    }


def _kuramoto_r_curve(toa_s: np.ndarray, freq_grid: np.ndarray) -> np.ndarray:
    toa_work = toa_s[: min(10000, len(toa_s))]
    phase_matrix = 4.0 * np.pi * np.outer(freq_grid, toa_work)
    z = np.exp(1j * phase_matrix)
    return np.abs(np.mean(z, axis=1))


def _segment_frequencies(toa_s: np.ndarray, base_freq: float, gap_threshold_s: float, min_pulses: int) -> pd.DataFrame:
    if len(toa_s) < min_pulses:
        return pd.DataFrame(columns=["segment_id", "n_pulses", "t_start_s", "t_end_s", "duration_s", "local_freq_hz"])

    gaps = np.diff(toa_s)
    split_idx = np.where(gaps > gap_threshold_s)[0] + 1
    segments = np.split(toa_s, split_idx)

    rows = []
    for idx, seg in enumerate(segments):
        if len(seg) < min_pulses:
            continue
        local_freq = calibrate_grid_frequency(seg, base_freq=base_freq, search_width=0.2, steps=50000)
        rows.append(
            {
                "segment_id": idx,
                "n_pulses": int(len(seg)),
                "t_start_s": float(seg[0]),
                "t_end_s": float(seg[-1]),
                "duration_s": float(seg[-1] - seg[0]),
                "local_freq_hz": float(local_freq),
            }
        )
    return pd.DataFrame(rows)


def _plot_frequency_vs_threshold(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for dataset_key, df_case in df.groupby("dataset_key"):
        axes[0].plot(df_case["threshold_sigma"], df_case["blind_freq_hz"], marker="o", label=dataset_key)
        axes[1].plot(df_case["threshold_sigma"], df_case["phase_entropy_global"], marker="o", label=dataset_key)
        axes[2].plot(df_case["threshold_sigma"], df_case["phase_spread_deg"], marker="o", label=dataset_key)

    axes[0].set_ylabel("Frecuencia ciega (Hz)")
    axes[1].set_ylabel("Entropía de fase")
    axes[2].set_ylabel("Spread de fase (deg)")
    axes[2].set_xlabel("threshold_sigma")
    axes[0].set_title("Sensibilidad a umbral: frecuencia y fase")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_segment_frequencies(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for dataset_key, df_case in df.groupby("dataset_key"):
        ax.plot(df_case["segment_id"], df_case["local_freq_hz"], marker="o", label=dataset_key)
    ax.set_xlabel("Segmento")
    ax.set_ylabel("Frecuencia local (Hz)")
    ax.set_title("Frecuencia ciega por segmento")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_frequency_phase_curve(df: pd.DataFrame, out_png: Path, dataset_key: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(df["freq_hz"], df["kuramoto_r"], color="tab:blue")
    axes[1].plot(df["freq_hz"], df["phase_entropy_global"], color="tab:orange")
    axes[2].plot(df["freq_hz"], df["phase_spread_deg"], color="tab:green")

    axes[0].set_ylabel("Kuramoto R")
    axes[1].set_ylabel("Entropía de fase")
    axes[2].set_ylabel("Spread de fase (deg)")
    axes[2].set_xlabel("Frecuencia (Hz)")
    axes[0].set_title(f"Curva frecuencia-fase: {dataset_key}")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(
    base_dir: Path,
    out_dir: Path,
    channel: str,
    thresholds: list[float],
    min_separation_s: float,
    gap_threshold_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold_rows: list[dict[str, float | str]] = []
    segment_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []

    for dataset_key, dataset_label in DATASETS.items():
        file_path = base_dir / dataset_label / f"{channel}.csv"
        signal, fs_hz, times_abs_s = load_empirical_signal(
            str(file_path),
            preserve_amplitude=True,
            include_absolute_times=True,
        )

        toa_by_threshold: dict[float, np.ndarray] = {}
        best_freq_ref = float("nan")
        for threshold_sigma in thresholds:
            pulse_idx = detect_pulses(
                signal,
                fs_hz,
                threshold_sigma=threshold_sigma,
                min_separation_s=min_separation_s,
                method="threshold",
            )
            toa_s = times_abs_s[pulse_idx]
            toa_by_threshold[threshold_sigma] = toa_s
            blind_freq_hz = calibrate_grid_frequency(toa_s, base_freq=50.0, search_width=0.2, steps=50000)
            phases_deg = _phases_from_frequency(toa_s, blind_freq_hz)
            metrics = _phase_metrics(phases_deg)
            threshold_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "threshold_sigma": float(threshold_sigma),
                    "n_pulses": int(len(toa_s)),
                    "blind_freq_hz": float(blind_freq_hz),
                    **metrics,
                }
            )
            if abs(threshold_sigma - 5.0) < 1e-9:
                best_freq_ref = float(blind_freq_hz)

        toa_ref = toa_by_threshold[5.0]
        seg_df = _segment_frequencies(toa_ref, base_freq=50.0, gap_threshold_s=gap_threshold_s, min_pulses=25)
        if not seg_df.empty:
            seg_df.insert(0, "dataset_key", dataset_key)
            seg_df.insert(1, "dataset_label", dataset_label)
            segment_frames.append(seg_df)

        freq_grid = np.linspace(best_freq_ref - 0.08, best_freq_ref + 0.08, 161)
        r_vals = _kuramoto_r_curve(toa_ref, freq_grid)
        curve_rows = []
        for freq_hz, r_val in zip(freq_grid, r_vals):
            phases_deg = _phases_from_frequency(toa_ref, float(freq_hz))
            metrics = _phase_metrics(phases_deg)
            curve_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "freq_hz": float(freq_hz),
                    "kuramoto_r": float(r_val),
                    **metrics,
                }
            )
        curve_df = pd.DataFrame(curve_rows)
        curve_frames.append(curve_df)
        _plot_frequency_phase_curve(curve_df, out_dir / f"{dataset_key.lower()}_frequency_phase_curve.png", dataset_key)

    threshold_df = pd.DataFrame(threshold_rows)
    segment_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    curve_df = pd.concat(curve_frames, ignore_index=True)

    threshold_df.to_csv(out_dir / "blind_prpd_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    if not segment_df.empty:
        segment_df.to_csv(out_dir / "blind_prpd_segment_frequencies.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv(out_dir / "blind_prpd_frequency_phase_curves.csv", index=False, encoding="utf-8-sig")

    _plot_frequency_vs_threshold(threshold_df, out_dir / "blind_prpd_threshold_sensitivity.png")
    if not segment_df.empty:
        _plot_segment_frequencies(segment_df, out_dir / "blind_prpd_segment_frequencies.png")

    return threshold_df, segment_df, curve_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate blind PRPD frequency and phase stability.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory containing P1/P2/P3 waveform folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/blind_prpd_frequency_phase_eval"),
        help="Output folder for tables and figures.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="CH3",
        help="Channel to evaluate.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[4.0, 4.5, 5.0, 5.5, 6.0],
        help="Threshold sigma values to evaluate.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=20e-9,
        help="Minimum separation between pulses in seconds.",
    )
    parser.add_argument(
        "--gap-threshold-s",
        type=float,
        default=0.1,
        help="Gap threshold used to define local segments.",
    )
    args = parser.parse_args()

    threshold_df, segment_df, curve_df = run_evaluation(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        channel=args.channel,
        thresholds=args.thresholds,
        min_separation_s=args.min_separation_s,
        gap_threshold_s=args.gap_threshold_s,
    )
    print(threshold_df.to_string(index=False))
    if not segment_df.empty:
        print(segment_df.to_string(index=False))
    print(curve_df.groupby("dataset_key")[["kuramoto_r", "phase_entropy_global", "phase_spread_deg"]].agg(["min", "max"]).to_string())
    print(f"CSV_THRESH={args.out_dir / 'blind_prpd_threshold_sensitivity.csv'}")
    print(f"CSV_SEG={args.out_dir / 'blind_prpd_segment_frequencies.csv'}")
    print(f"CSV_CURVES={args.out_dir / 'blind_prpd_frequency_phase_curves.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
