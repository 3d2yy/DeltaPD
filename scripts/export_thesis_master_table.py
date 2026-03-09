from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deltapd.blind_prpd import calibrate_grid_frequency_details, reconstruct_blind_prpd
from deltapd.campaign.config import load_config
from deltapd.campaign.event_pipeline import load_and_extract_event_series
from deltapd.statistics import compute_burstiness_index, compute_fano_factor

FREQ_SEARCH_WIDTH_HZ = 0.25
FREQ_SEARCH_STEPS = 4001


def _resolve_event_pipeline_config(
    cfg: dict[str, Any],
    *,
    threshold_sigma_override: float | None,
    min_separation_s_override: float | None,
) -> dict[str, Any]:
    analysis = cfg.get("analysis_params", {})
    pipeline_cfg = dict(analysis.get("master_event_pipeline", {}))
    denoise_cfg = analysis.get("denoise", {})

    threshold_sigma = threshold_sigma_override
    if threshold_sigma is None:
        threshold_sigma = pipeline_cfg.get("threshold_sigma", analysis.get("delta_t_threshold_sigma", 5.0))

    min_separation_s = min_separation_s_override
    if min_separation_s is None:
        if "min_separation_s" in pipeline_cfg:
            min_separation_s = pipeline_cfg["min_separation_s"]
        elif "refractory_ns" in pipeline_cfg:
            min_separation_s = float(pipeline_cfg["refractory_ns"]) * 1e-9
        else:
            min_separation_s = 20e-9

    return {
        "preserve_amplitude": bool(pipeline_cfg.get("preserve_amplitude", analysis.get("preserve_amplitude", True))),
        "default_fs": float(analysis.get("default_fs", 1.0e9)),
        "threshold_sigma": float(threshold_sigma),
        "min_separation_s": float(min_separation_s),
        "detection_method": str(pipeline_cfg.get("detection_method", "threshold")),
        "wavelet_denoise": bool(pipeline_cfg.get("wavelet_denoise", denoise_cfg.get("enabled", False))),
        "is_envelope": bool(pipeline_cfg.get("is_envelope", False)),
        "wavelet": str(pipeline_cfg.get("wavelet", denoise_cfg.get("wavelet", "db4"))),
        "threshold_mode": str(pipeline_cfg.get("threshold_mode", denoise_cfg.get("threshold_mode", "soft"))),
        "threshold_rule": str(pipeline_cfg.get("threshold_rule", denoise_cfg.get("threshold_rule", "universal"))),
    }


def _iter_channel_files(dataset_dir: Path, channel_name: str) -> list[Path]:
    exact = sorted(dataset_dir.rglob(f"{channel_name}.csv"))
    if exact:
        return exact
    generic = sorted(dataset_dir.rglob(f"*{channel_name}*.csv"))
    return generic


def _phase_cluster_metrics(phases_deg: np.ndarray, peaks_v: np.ndarray) -> dict[str, float]:
    if len(phases_deg) == 0:
        return {
            "phase_entropy_global": float("nan"),
            "phase_spread_deg": float("nan"),
            "inlier_ratio": float("nan"),
            "phase_width_pos_deg": float("nan"),
            "phase_width_neg_deg": float("nan"),
            "amplitude_balance_ratio": float("nan"),
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

    hist, _ = np.histogram(phases_deg, bins=np.linspace(0.0, 360.0, 37))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(36) if len(p) else float("nan")

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

    return {
        "phase_entropy_global": float(entropy),
        "phase_spread_deg": float(median_d),
        "inlier_ratio": float(np.mean(inlier_mask)),
        "phase_width_pos_deg": pos_width,
        "phase_width_neg_deg": neg_width,
        "amplitude_balance_ratio": float(pos_amp / neg_amp) if neg_amp > 0 else float("nan"),
    }


def _signed_amplitude(phases_deg: np.ndarray, peaks_v: np.ndarray) -> np.ndarray:
    return np.where((phases_deg >= 0.0) & (phases_deg <= 180.0), peaks_v, -peaks_v)


def _compute_master_row(
    file_path: Path,
    dataset_key: str,
    dataset_label: str,
    group_family: str,
    channel: str,
    antenna_label: str,
    event_pipeline: dict[str, Any],
    blind_prpd_method: str,
    blind_prpd_harmonics: int,
    blind_prpd_bootstrap_iterations: int,
    blind_prpd_bootstrap_sample_fraction: float,
    blind_prpd_bootstrap_seed: int | None,
    blind_prpd_local_window_size_events: int,
    blind_prpd_local_window_step_events: int,
    blind_prpd_local_min_events_per_window: int,
    blind_prpd_local_min_window_count: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    extracted = load_and_extract_event_series(
        str(file_path),
        preserve_amplitude=bool(event_pipeline.get("preserve_amplitude", True)),
        include_absolute_times=True,
        default_fs=float(event_pipeline.get("default_fs", 1.0e9)),
        threshold_sigma=float(event_pipeline["threshold_sigma"]),
        min_separation_s=float(event_pipeline["min_separation_s"]),
        detection_method=str(event_pipeline.get("detection_method", "threshold")),
        wavelet_denoise=bool(event_pipeline.get("wavelet_denoise", False)),
        is_envelope=bool(event_pipeline.get("is_envelope", False)),
        wavelet=str(event_pipeline.get("wavelet", "db4")),
        threshold_mode=str(event_pipeline.get("threshold_mode", "soft")),
        threshold_rule=str(event_pipeline.get("threshold_rule", "universal")),
    )
    fs_hz = extracted.fs_hz
    toa_s = extracted.event_toa_s
    delta_t_s = extracted.event_delta_t_s
    peaks_v = extracted.event_peaks_v

    row: dict[str, Any] = {
        "group_family": group_family,
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "channel": channel,
        "antenna_label": antenna_label,
        "source_file": str(file_path),
        "fs_hz": float(fs_hz),
        "threshold_sigma": float(event_pipeline["threshold_sigma"]),
        "min_separation_s": float(event_pipeline["min_separation_s"]),
        "event_pipeline_detection_method": str(event_pipeline.get("detection_method", "threshold")),
        "event_pipeline_wavelet_denoise": bool(event_pipeline.get("wavelet_denoise", False)),
        "event_pipeline_wavelet": str(event_pipeline.get("wavelet", "db4")),
        "pulse_count": int(len(toa_s)),
        "blind_freq_hz": float("nan"),
        "blind_prpd_method": blind_prpd_method,
        "blind_prpd_selected_method": "",
        "blind_prpd_score": float("nan"),
        "blind_prpd_coherence": float("nan"),
        "blind_prpd_axial_entropy_score": float("nan"),
        "blind_prpd_sharpness": float("nan"),
        "blind_prpd_half_height_width_hz": float("nan"),
        "blind_prpd_score_prominence": float("nan"),
        "blind_prpd_common_axial_peak_freq_hz": float("nan"),
        "blind_prpd_common_axial_peak_offset_hz": float("nan"),
        "blind_prpd_common_axial_sharpness": float("nan"),
        "blind_prpd_common_axial_half_height_width_hz": float("nan"),
        "blind_prpd_common_axial_prominence": float("nan"),
        "blind_prpd_common_axial_width_ratio": float("nan"),
        "blind_prpd_common_axial_confidence": float("nan"),
        "blind_prpd_bootstrap_iterations": 0,
        "blind_prpd_bootstrap_sample_fraction": float("nan"),
        "blind_prpd_bootstrap_freq_mean_hz": float("nan"),
        "blind_prpd_bootstrap_freq_std_hz": float("nan"),
        "blind_prpd_bootstrap_ci_low_hz": float("nan"),
        "blind_prpd_bootstrap_ci_high_hz": float("nan"),
        "blind_prpd_bootstrap_ci_width_hz": float("nan"),
        "blind_prpd_bootstrap_method_agreement": float("nan"),
        "blind_prpd_local_window_count": 0,
        "blind_prpd_local_window_size_events": 0,
        "blind_prpd_local_window_step_events": 0,
        "blind_prpd_local_freq_mean_hz": float("nan"),
        "blind_prpd_local_freq_std_hz": float("nan"),
        "blind_prpd_local_freq_min_hz": float("nan"),
        "blind_prpd_local_freq_max_hz": float("nan"),
        "blind_prpd_local_freq_span_hz": float("nan"),
        "blind_prpd_local_method_agreement": float("nan"),
        "blind_prpd_local_common_confidence_mean": float("nan"),
        "blind_prpd_local_dominant_method": "",
        "blind_prpd_candidate_spread_hz": float("nan"),
        "blind_prpd_winner_margin": float("nan"),
        "mean_peak_v": float("nan"),
        "std_peak_v": float("nan"),
        "peak_p90_v": float("nan"),
        "phase_entropy_global": float("nan"),
        "phase_spread_deg": float("nan"),
        "phase_width_pos_deg": float("nan"),
        "phase_width_neg_deg": float("nan"),
        "inlier_ratio": float("nan"),
        "amplitude_balance_ratio": float("nan"),
        "median_dt_s": float("nan"),
        "iqr_dt_s": float("nan"),
        "cv_dt": float("nan"),
        "burstiness_mean": float("nan"),
        "fano_global": float("nan"),
        "notes": "",
    }

    df_points = pd.DataFrame(
        columns=[
            "group_family",
            "dataset_key",
            "dataset_label",
            "channel",
            "antenna_label",
            "toa_s",
            "peak_v",
            "prpd_phase_deg",
            "signed_peak_v",
        ]
    )

    if len(toa_s) < 10:
        row["notes"] = "too_few_pulses_for_blind_prpd"
        return row, df_points

    blind_details = calibrate_grid_frequency_details(
        toa_s,
        base_freq=50.0,
        search_width=FREQ_SEARCH_WIDTH_HZ,
        coarse_steps=FREQ_SEARCH_STEPS,
        peak_weights=peaks_v,
        method=blind_prpd_method,
        n_harmonics=blind_prpd_harmonics,
        bootstrap_iterations=blind_prpd_bootstrap_iterations,
        bootstrap_sample_fraction=blind_prpd_bootstrap_sample_fraction,
        bootstrap_seed=blind_prpd_bootstrap_seed,
        local_window_size_events=blind_prpd_local_window_size_events,
        local_window_step_events=blind_prpd_local_window_step_events,
        local_min_events_per_window=blind_prpd_local_min_events_per_window,
        local_min_window_count=blind_prpd_local_min_window_count,
    )
    blind_freq_hz = blind_details.freq_hz
    blind_score = blind_details.score
    phases_deg, peaks_out = reconstruct_blind_prpd(
        toa_s,
        peaks_v,
        freq_hz=blind_freq_hz,
        auto_calibrate=False,
        calibration_method=blind_prpd_method,
        n_harmonics=blind_prpd_harmonics,
        local_window_size_events=blind_prpd_local_window_size_events,
        local_window_step_events=blind_prpd_local_window_step_events,
        local_min_events_per_window=blind_prpd_local_min_events_per_window,
        local_min_window_count=blind_prpd_local_min_window_count,
    )
    burstiness_series = compute_burstiness_index(
        delta_t_s,
        window=min(100, max(20, len(delta_t_s) // 10)),
        min_periods=10,
    )
    _, fano_vals = compute_fano_factor(toa_s, bin_duration_s=0.1, window_bins=20, min_bins=5)

    row.update(
        {
            "blind_freq_hz": float(blind_freq_hz),
            "blind_prpd_score": float(blind_score),
            "blind_prpd_selected_method": str(blind_details.selected_method),
            "blind_prpd_coherence": float(blind_details.coherence),
            "blind_prpd_axial_entropy_score": float(blind_details.axial_entropy_score),
            "blind_prpd_sharpness": float(blind_details.sharpness),
            "blind_prpd_half_height_width_hz": float(blind_details.half_height_width_hz),
            "blind_prpd_score_prominence": float(blind_details.score_prominence),
            "blind_prpd_common_axial_peak_freq_hz": float(blind_details.common_axial_peak_freq_hz),
            "blind_prpd_common_axial_peak_offset_hz": float(blind_details.common_axial_peak_offset_hz),
            "blind_prpd_common_axial_sharpness": float(blind_details.common_axial_sharpness),
            "blind_prpd_common_axial_half_height_width_hz": float(blind_details.common_axial_half_height_width_hz),
            "blind_prpd_common_axial_prominence": float(blind_details.common_axial_prominence),
            "blind_prpd_common_axial_width_ratio": float(blind_details.common_axial_width_ratio),
            "blind_prpd_common_axial_confidence": float(blind_details.common_axial_confidence),
            "blind_prpd_bootstrap_iterations": int(blind_details.bootstrap_iterations),
            "blind_prpd_bootstrap_sample_fraction": float(blind_details.bootstrap_sample_fraction),
            "blind_prpd_bootstrap_freq_mean_hz": float(blind_details.bootstrap_freq_mean_hz),
            "blind_prpd_bootstrap_freq_std_hz": float(blind_details.bootstrap_freq_std_hz),
            "blind_prpd_bootstrap_ci_low_hz": float(blind_details.bootstrap_ci_low_hz),
            "blind_prpd_bootstrap_ci_high_hz": float(blind_details.bootstrap_ci_high_hz),
            "blind_prpd_bootstrap_ci_width_hz": float(blind_details.bootstrap_ci_width_hz),
            "blind_prpd_bootstrap_method_agreement": float(blind_details.bootstrap_method_agreement),
            "blind_prpd_local_window_count": int(blind_details.local_window_count),
            "blind_prpd_local_window_size_events": int(blind_details.local_window_size_events),
            "blind_prpd_local_window_step_events": int(blind_details.local_window_step_events),
            "blind_prpd_local_freq_mean_hz": float(blind_details.local_freq_mean_hz),
            "blind_prpd_local_freq_std_hz": float(blind_details.local_freq_std_hz),
            "blind_prpd_local_freq_min_hz": float(blind_details.local_freq_min_hz),
            "blind_prpd_local_freq_max_hz": float(blind_details.local_freq_max_hz),
            "blind_prpd_local_freq_span_hz": float(blind_details.local_freq_span_hz),
            "blind_prpd_local_method_agreement": float(blind_details.local_method_agreement),
            "blind_prpd_local_common_confidence_mean": float(blind_details.local_common_confidence_mean),
            "blind_prpd_local_dominant_method": str(blind_details.local_dominant_method),
            "blind_prpd_candidate_spread_hz": float(blind_details.candidate_spread_hz),
            "blind_prpd_winner_margin": float(blind_details.winner_margin),
            "mean_peak_v": float(np.mean(peaks_out)),
            "std_peak_v": float(np.std(peaks_out)),
            "peak_p90_v": float(np.percentile(peaks_out, 90)),
            "median_dt_s": float(np.median(delta_t_s)) if len(delta_t_s) else float("nan"),
            "iqr_dt_s": float(np.percentile(delta_t_s, 75) - np.percentile(delta_t_s, 25))
            if len(delta_t_s)
            else float("nan"),
            "cv_dt": float(np.std(delta_t_s) / np.mean(delta_t_s))
            if len(delta_t_s) and np.mean(delta_t_s) > 0
            else float("nan"),
            "burstiness_mean": float(np.nanmedian(burstiness_series))
            if len(burstiness_series)
            else float("nan"),
            "fano_global": float(np.nanmedian(fano_vals)) if len(fano_vals) else float("nan"),
        }
    )
    row.update(_phase_cluster_metrics(phases_deg, peaks_out))

    df_points = pd.DataFrame(
        {
            "group_family": group_family,
            "dataset_key": dataset_key,
            "dataset_label": dataset_label,
            "channel": channel,
            "antenna_label": antenna_label,
            "toa_s": toa_s,
            "delta_t_s": delta_t_s,
            "peak_v": peaks_out,
            "prpd_phase_deg": phases_deg,
            "signed_peak_v": _signed_amplitude(phases_deg, peaks_out),
        }
    )
    return row, df_points


def export_master_tables(
    config_path: Path,
    base_dir: Path,
    out_dir: Path,
    threshold_sigma: float | None,
    min_separation_s: float | None,
    blind_prpd_method: str = "auto",
    blind_prpd_harmonics: int = 4,
    blind_prpd_bootstrap_iterations: int = 0,
    blind_prpd_bootstrap_sample_fraction: float = 0.75,
    blind_prpd_bootstrap_seed: int | None = None,
    blind_prpd_local_window_size_events: int = 0,
    blind_prpd_local_window_step_events: int = 0,
    blind_prpd_local_min_events_per_window: int = 128,
    blind_prpd_local_min_window_count: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    datasets = cfg.get("datasets", {})
    event_pipeline = _resolve_event_pipeline_config(
        cfg,
        threshold_sigma_override=threshold_sigma,
        min_separation_s_override=min_separation_s,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []

    for dataset_key, dataset_cfg in datasets.items():
        dataset_dir = base_dir / dataset_cfg["folder"]
        dataset_label = dataset_cfg.get("label", dataset_cfg["folder"])
        group_family = dataset_cfg.get("mode", "benchmark")
        channel_map = dataset_cfg.get("channel_map", {})

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

        for channel, antenna_label in channel_map.items():
            files = _iter_channel_files(dataset_dir, channel)
            if not files:
                raise FileNotFoundError(f"No CSV file found for {dataset_key}/{channel} in {dataset_dir}")
            if len(files) > 1:
                raise RuntimeError(f"Multiple files found for {dataset_key}/{channel}: {files}")

            row, df_points = _compute_master_row(
                file_path=files[0],
                dataset_key=dataset_key,
                dataset_label=dataset_label,
                group_family=group_family,
                channel=channel,
                antenna_label=antenna_label,
                event_pipeline=event_pipeline,
                blind_prpd_method=blind_prpd_method,
                blind_prpd_harmonics=blind_prpd_harmonics,
                blind_prpd_bootstrap_iterations=blind_prpd_bootstrap_iterations,
                blind_prpd_bootstrap_sample_fraction=blind_prpd_bootstrap_sample_fraction,
                blind_prpd_bootstrap_seed=blind_prpd_bootstrap_seed,
                blind_prpd_local_window_size_events=blind_prpd_local_window_size_events,
                blind_prpd_local_window_step_events=blind_prpd_local_window_step_events,
                blind_prpd_local_min_events_per_window=blind_prpd_local_min_events_per_window,
                blind_prpd_local_min_window_count=blind_prpd_local_min_window_count,
            )
            rows.append(row)
            if not df_points.empty:
                point_frames.append(df_points)

    metrics_df = pd.DataFrame(rows).sort_values(["group_family", "dataset_key", "channel"]).reset_index(drop=True)
    points_df = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()

    pair_rows: list[dict[str, Any]] = []
    for (group_family, dataset_key), df_case in metrics_df.groupby(["group_family", "dataset_key"], dropna=False):
        dataset_label = df_case["dataset_label"].iloc[0]
        for idx_a, idx_b in combinations(df_case.index.tolist(), 2):
            row_a = metrics_df.loc[idx_a]
            row_b = metrics_df.loc[idx_b]
            pair_rows.append(
                {
                    "group_family": group_family,
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                    "antenna_a": row_a["antenna_label"],
                    "antenna_b": row_b["antenna_label"],
                    "channel_a": row_a["channel"],
                    "channel_b": row_b["channel"],
                    "freq_diff_hz": float(abs(row_a["blind_freq_hz"] - row_b["blind_freq_hz"])),
                    "mean_peak_diff_v": float(abs(row_a["mean_peak_v"] - row_b["mean_peak_v"])),
                    "entropy_diff": float(abs(row_a["phase_entropy_global"] - row_b["phase_entropy_global"])),
                    "phase_width_pos_diff_deg": float(
                        abs(row_a["phase_width_pos_deg"] - row_b["phase_width_pos_deg"])
                    ),
                    "phase_width_neg_diff_deg": float(
                        abs(row_a["phase_width_neg_deg"] - row_b["phase_width_neg_deg"])
                    ),
                    "pulse_ratio": float(row_a["pulse_count"] / row_b["pulse_count"])
                    if row_b["pulse_count"] > 0
                    else float("nan"),
                }
            )
    pairs_df = pd.DataFrame(pair_rows)

    metrics_df.to_csv(out_dir / "thesis_master_metrics.csv", index=False, encoding="utf-8-sig")
    pairs_df.to_csv(out_dir / "thesis_master_pairwise_differences.csv", index=False, encoding="utf-8-sig")
    if not points_df.empty:
        points_df.to_csv(out_dir / "thesis_master_prpd_points.csv", index=False, encoding="utf-8-sig")

    return metrics_df, pairs_df, points_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a unified thesis master table for P1-P3 and G1-G3.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_thesis.yaml"),
        help="YAML config containing dataset definitions and channel labels.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory containing the waveform folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/thesis_master"),
        help="Output directory for the unified thesis tables.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=None,
        help="Detection threshold in sigma units. Defaults to config analysis_params.master_event_pipeline.threshold_sigma.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=None,
        help="Minimum pulse separation in seconds. Defaults to config analysis_params.master_event_pipeline.min_separation_s or refractory_ns.",
    )
    parser.add_argument(
        "--blind-prpd-method",
        type=str,
        default="auto",
    help="Blind PRPD calibration method: coherence, harmonic_power, epoch_folding, h_test, pdm, gregory_loredo, phase_distance_correlation, auto.",
    )
    parser.add_argument(
        "--blind-prpd-harmonics",
        type=int,
        default=4,
        help="Number of harmonics used by harmonic-power calibration.",
    )
    parser.add_argument(
        "--blind-prpd-bootstrap-iterations",
        type=int,
        default=6,
        help="Bootstrap iterations for blind PRPD stability estimation.",
    )
    parser.add_argument(
        "--blind-prpd-bootstrap-sample-fraction",
        type=float,
        default=0.75,
        help="Fraction of events sampled in each blind PRPD bootstrap replicate.",
    )
    parser.add_argument(
        "--blind-prpd-bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for blind PRPD bootstrap stability.",
    )
    parser.add_argument(
        "--blind-prpd-local-window-size-events",
        type=int,
        default=256,
        help="Contiguous window size in events for local blind PRPD stability.",
    )
    parser.add_argument(
        "--blind-prpd-local-window-step-events",
        type=int,
        default=128,
        help="Step between contiguous windows for local blind PRPD stability.",
    )
    parser.add_argument(
        "--blind-prpd-local-min-events-per-window",
        type=int,
        default=128,
        help="Minimum events required in each local blind PRPD window.",
    )
    parser.add_argument(
        "--blind-prpd-local-min-window-count",
        type=int,
        default=3,
        help="Minimum number of contiguous windows required to report local stability.",
    )
    args = parser.parse_args()

    metrics_df, pairs_df, _ = export_master_tables(
        config_path=args.config,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        threshold_sigma=args.threshold_sigma,
        min_separation_s=args.min_separation_s,
        blind_prpd_method=args.blind_prpd_method,
        blind_prpd_harmonics=args.blind_prpd_harmonics,
        blind_prpd_bootstrap_iterations=args.blind_prpd_bootstrap_iterations,
        blind_prpd_bootstrap_sample_fraction=args.blind_prpd_bootstrap_sample_fraction,
        blind_prpd_bootstrap_seed=args.blind_prpd_bootstrap_seed,
        blind_prpd_local_window_size_events=args.blind_prpd_local_window_size_events,
        blind_prpd_local_window_step_events=args.blind_prpd_local_window_step_events,
        blind_prpd_local_min_events_per_window=args.blind_prpd_local_min_events_per_window,
        blind_prpd_local_min_window_count=args.blind_prpd_local_min_window_count,
    )
    print(metrics_df.to_string(index=False))
    print(pairs_df.to_string(index=False))
    print(f"CSV={args.out_dir / 'thesis_master_metrics.csv'}")
    print(f"CSV_PAIRS={args.out_dir / 'thesis_master_pairwise_differences.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
