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
    inlier_mask = d_min <= threshold

    pos_mask = phases_deg <= 180.0
    neg_mask = ~pos_mask
    pos_phases = phases_deg[pos_mask]
    neg_phases = phases_deg[neg_mask]
    hist, _ = np.histogram(phases_deg, bins=np.linspace(0.0, 360.0, 37))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(36) if len(p) else float("nan")

    return {
        "phase_entropy_global": float(entropy),
        "phase_spread_deg": float(median_d),
        "inlier_ratio": float(np.mean(inlier_mask)),
        "phase_width_pos_deg": float(np.percentile(pos_phases, 90) - np.percentile(pos_phases, 10)),
        "phase_width_neg_deg": float(np.percentile(neg_phases, 90) - np.percentile(neg_phases, 10)),
        "peak_mean_v": float(np.mean(peaks_v)),
    }


def _signed_amplitude(phases_deg: np.ndarray, peaks_v: np.ndarray) -> np.ndarray:
    return np.where((phases_deg >= 0.0) & (phases_deg <= 180.0), peaks_v, -peaks_v)


def _fixed_frequency_phases(toa_s: np.ndarray, freq_hz: float) -> np.ndarray:
    period_s = 1.0 / freq_hz
    phases_deg = np.mod(toa_s, period_s) / period_s * 360.0
    return phases_deg


def _penalized_frequency(toa_s: np.ndarray, base_freq: float, sigma_hz: float) -> float:
    f_test = np.linspace(base_freq - 0.2, base_freq + 0.2, 50000)
    toa_work = toa_s[: min(10000, len(toa_s))]
    z = np.exp(1j * 4.0 * np.pi * np.outer(f_test, toa_work))
    r = np.abs(np.mean(z, axis=1))
    penalty = np.exp(-((f_test - base_freq) ** 2) / (2.0 * sigma_hz**2))
    return float(f_test[np.argmax(r * penalty)])


def _segmented_frequency_phases(toa_s: np.ndarray, base_freq: float, gap_threshold_s: float = 0.1) -> tuple[np.ndarray, float]:
    if len(toa_s) == 0:
        return np.array([]), float("nan")

    gaps = np.diff(toa_s)
    split_idx = np.where(gaps > gap_threshold_s)[0] + 1
    segments = np.split(toa_s, split_idx)

    all_phases: list[np.ndarray] = []
    freqs: list[float] = []
    for seg in segments:
        if len(seg) < 10:
            phases = _fixed_frequency_phases(seg, base_freq)
            freqs.append(base_freq)
        else:
            local_freq = calibrate_grid_frequency(seg, base_freq=base_freq, search_width=0.2, steps=50000)
            phases = _fixed_frequency_phases(seg, local_freq)
            freqs.append(local_freq)
        all_phases.append(phases)

    phases_deg = np.concatenate(all_phases)
    phases_deg = _center_phases(phases_deg)
    return phases_deg, float(np.mean(freqs))


def _render_density(ax: plt.Axes, phases_deg: np.ndarray, signed_peaks: np.ndarray, mode: str) -> None:
    if mode == "base_kde":
        kde = gaussian_kde(np.vstack([phases_deg, signed_peaks]), bw_method=0.15)
        z = kde(np.vstack([phases_deg, signed_peaks]))
        order = z.argsort()
        ax.scatter(phases_deg[order], signed_peaks[order], c=z[order], cmap="turbo", s=10, alpha=0.85, edgecolors="none")
        ax.set_title("KDE Base")
        return

    if mode == "weighted_kde":
        weights = np.abs(signed_peaks)
        kde = gaussian_kde(np.vstack([phases_deg, signed_peaks]), bw_method=0.15, weights=weights)
        z = kde(np.vstack([phases_deg, signed_peaks]))
        order = z.argsort()
        sizes = 6 + 24 * (np.abs(signed_peaks[order]) / np.max(np.abs(signed_peaks[order])))
        ax.scatter(phases_deg[order], signed_peaks[order], c=z[order], cmap="turbo", s=sizes, alpha=0.9, edgecolors="none")
        ax.set_title("KDE Ponderado")
        return

    if mode == "hexbin":
        ax.hexbin(phases_deg, signed_peaks, gridsize=55, cmap="turbo", mincnt=1, alpha=0.9)
        ax.scatter(phases_deg, signed_peaks, c="black", s=2, alpha=0.18, edgecolors="none")
        ax.set_title("Hexbin")
        return

    raise ValueError(f"Unknown render mode: {mode}")


def _plot_render_benchmark(phases_deg: np.ndarray, peaks_v: np.ndarray, out_png: Path) -> None:
    signed_peaks = _signed_amplitude(phases_deg, peaks_v)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, mode in zip(axes, ["base_kde", "weighted_kde", "hexbin"]):
        _render_density(ax, phases_deg, signed_peaks, mode)
        t_sin = np.linspace(0.0, 360.0, 360)
        max_amp = np.max(np.abs(signed_peaks)) * 1.05
        ax.plot(t_sin, max_amp * np.sin(np.radians(t_sin)), color="red", alpha=0.45, linewidth=1.3)
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks(np.arange(0.0, 361.0, 45.0))
        ax.set_xlabel("Fase (grados)")
        ax.grid(True, linestyle=":", alpha=0.35)
    axes[0].set_ylabel("Amplitud aparente firmada (V)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_algorithm_benchmark(phases_by_method: dict[str, np.ndarray], peaks_v: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    for ax, (label, phases_deg) in zip(axes, phases_by_method.items()):
        signed_peaks = _signed_amplitude(phases_deg, peaks_v)
        weights = np.abs(signed_peaks)
        kde = gaussian_kde(np.vstack([phases_deg, signed_peaks]), bw_method=0.15, weights=weights)
        z = kde(np.vstack([phases_deg, signed_peaks]))
        order = z.argsort()
        sizes = 6 + 24 * (np.abs(signed_peaks[order]) / np.max(np.abs(signed_peaks[order])))
        ax.scatter(phases_deg[order], signed_peaks[order], c=z[order], cmap="turbo", s=sizes, alpha=0.9, edgecolors="none")
        t_sin = np.linspace(0.0, 360.0, 360)
        max_amp = np.max(np.abs(signed_peaks)) * 1.05
        ax.plot(t_sin, max_amp * np.sin(np.radians(t_sin)), color="red", alpha=0.45, linewidth=1.3)
        ax.set_xlim(0.0, 360.0)
        ax.set_xticks(np.arange(0.0, 361.0, 45.0))
        ax.set_xlabel("Fase (grados)")
        ax.set_title(label)
        ax.grid(True, linestyle=":", alpha=0.35)
    axes[0].set_ylabel("Amplitud aparente firmada (V)")
    axes[2].set_ylabel("Amplitud aparente firmada (V)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_comparison(file_path: Path, out_dir: Path, threshold_sigma: float, min_separation_s: float) -> pd.DataFrame:
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

    phases_fixed = _fixed_frequency_phases(toa_s, 50.0)
    phases_fixed = _center_phases(phases_fixed)
    freq_global = calibrate_grid_frequency(toa_s, base_freq=50.0, search_width=0.2, steps=50000)
    phases_global = _fixed_frequency_phases(toa_s, freq_global)
    phases_global = _center_phases(phases_global)
    freq_pen = _penalized_frequency(toa_s, base_freq=50.0, sigma_hz=0.02)
    phases_pen = _fixed_frequency_phases(toa_s, freq_pen)
    phases_pen = _center_phases(phases_pen)
    phases_seg, freq_seg = _segmented_frequency_phases(toa_s, base_freq=50.0, gap_threshold_s=0.1)

    phases_by_method = {
        "50 Hz fijo": phases_fixed,
        f"Global {freq_global:.4f} Hz": phases_global,
        f"Penalizado {freq_pen:.4f} Hz": phases_pen,
        f"Segmentado {freq_seg:.4f} Hz": phases_seg,
    }

    rows = []
    for label, phases_deg in phases_by_method.items():
        row = {
            "method": label,
            "n_pulses": int(len(toa_s)),
        }
        row.update(_phase_metrics(phases_deg, peaks_v))
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "blind_prpd_method_metrics.csv", index=False, encoding="utf-8-sig")
    _plot_algorithm_benchmark(phases_by_method, peaks_v, out_dir / "blind_prpd_algorithm_comparison.png")
    _plot_render_benchmark(phases_global, peaks_v, out_dir / "blind_prpd_render_comparison.png")
    return metrics_df


def run_multi_dataset_comparison(
    base_dir: Path,
    out_dir: Path,
    channel: str,
    threshold_sigma: float,
    min_separation_s: float,
) -> pd.DataFrame:
    datasets = {
        "P1": "Prueba 1 - Internas",
        "P2": "Prueba 2 - Superficiales",
        "P3": "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas",
    }
    frames: list[pd.DataFrame] = []
    for key, folder in datasets.items():
        case_out_dir = out_dir / key
        file_path = base_dir / folder / f"{channel}.csv"
        df_case = run_comparison(
            file_path=file_path,
            out_dir=case_out_dir,
            threshold_sigma=threshold_sigma,
            min_separation_s=min_separation_s,
        ).copy()
        df_case.insert(0, "dataset_key", key)
        df_case.insert(1, "dataset_label", folder)
        frames.append(df_case)

    summary_df = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "blind_prpd_method_metrics_p1_p2_p3.csv", index=False, encoding="utf-8-sig")
    return summary_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare blind PRPD rendering and algorithm variants.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas/Prueba 2 - Superficiales/CH3.csv"),
        help="Input CSV waveform path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/blind_prpd_variants_p2"),
        help="Output folder for figures and metrics.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory for multi-dataset comparison.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="CH3",
        help="Channel for multi-dataset comparison.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run the comparison on P1, P2 and P3.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=5.0,
        help="Pulse detection threshold in sigma.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=20e-9,
        help="Minimum separation between pulses in seconds.",
    )
    args = parser.parse_args()

    if args.all_datasets:
        metrics_df = run_multi_dataset_comparison(
            base_dir=args.base_dir,
            out_dir=args.out_dir,
            channel=args.channel,
            threshold_sigma=args.threshold_sigma,
            min_separation_s=args.min_separation_s,
        )
        print(metrics_df.to_string(index=False))
        print(f"CSV={args.out_dir / 'blind_prpd_method_metrics_p1_p2_p3.csv'}")
    else:
        metrics_df = run_comparison(
            file_path=args.csv,
            out_dir=args.out_dir,
            threshold_sigma=args.threshold_sigma,
            min_separation_s=args.min_separation_s,
        )
        print(metrics_df.to_string(index=False))
        print(f"CSV={args.out_dir / 'blind_prpd_method_metrics.csv'}")
        print(f"PNG_RENDER={args.out_dir / 'blind_prpd_render_comparison.png'}")
        print(f"PNG_ALGO={args.out_dir / 'blind_prpd_algorithm_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
