from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.io import loadmat

from deltapd.campaign.pdf_reports import build_mat_series_pdf
from deltapd.descriptors import detect_pulses

DEFAULT_DESCRIPTOR_FEATURES = [
    "energy_v2",
    "rms_v",
    "p95_abs_v",
    "peak_abs_v",
    "crest_factor",
    "skewness",
    "kurtosis",
    "active_ratio",
    "pulse_count",
    "pulse_rate_hz",
    "mean_pulse_amp_v",
    "median_dt_s",
    "iqr_dt_s",
    "cv_dt",
    "burstiness",
    "event_center_frac",
    "event_width_frac",
]

DEFAULT_ACTIVITY_FEATURES = [
    "energy_v2",
    "p95_abs_v",
    "peak_abs_v",
    "active_ratio",
    "pulse_count",
]

DEFAULT_CHANGE_FEATURES = [
    "energy_v2",
    "p95_abs_v",
    "peak_abs_v",
    "active_ratio",
    "pulse_count",
    "median_dt_s",
    "cv_dt",
    "kurtosis",
    "event_width_frac",
]


def _repo_root_from_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    return resolved.parents[1] if len(resolved.parents) >= 2 else resolved.parent


def _resolve_path(config_path: str | Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _repo_root_from_config(config_path) / candidate


def _load_mat_series(
    mat_path: Path,
    *,
    signal_key: str,
    time_key: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    mat = loadmat(mat_path)
    if signal_key not in mat:
        raise KeyError(f"Signal key {signal_key!r} not found in {mat_path}.")

    matrix = np.asarray(mat[signal_key], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Signal key {signal_key!r} must be a 2-D matrix.")

    time_axis: np.ndarray | None = None
    if time_key and time_key in mat:
        raw_time = np.squeeze(np.asarray(mat[time_key], dtype=np.float64))
        if raw_time.ndim == 1:
            if raw_time.size == matrix.shape[0]:
                matrix = matrix.T
                time_axis = raw_time
            elif raw_time.size == matrix.shape[1]:
                time_axis = raw_time

    return matrix, time_axis


def _display_axis(time_axis: np.ndarray | None, n_samples: int) -> tuple[np.ndarray, str]:
    if time_axis is None or len(time_axis) != n_samples:
        return np.arange(n_samples, dtype=np.float64), "sample_index"

    diffs = np.diff(time_axis)
    if np.sum(diffs < 0) > 0:
        return np.arange(n_samples, dtype=np.float64), "sample_index"
    if np.sum(diffs > 0) < max(10, int(0.5 * n_samples)):
        return np.arange(n_samples, dtype=np.float64), "sample_index"
    return time_axis, "time_axis"


def _robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = float(numeric.median()) if numeric.notna().any() else 0.0
    mad = float(np.median(np.abs(numeric.dropna().to_numpy() - median))) if numeric.notna().any() else 0.0
    scale = max(mad * 1.4826, 1e-12)
    return (numeric - median) / scale


def _feature_quality_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "present": False,
                    "non_null": 0,
                    "missing_rate": 1.0,
                    "nunique": 0,
                    "eligible": False,
                    "reason": "missing_column",
                }
            )
            continue

        values = pd.to_numeric(df[feature], errors="coerce")
        non_null = int(values.notna().sum())
        nunique = int(values.nunique(dropna=True))
        eligible = non_null > 0 and nunique >= 2
        if non_null == 0:
            reason = "all_nan"
        elif nunique < 2:
            reason = "constant"
        else:
            reason = "ok"
        rows.append(
            {
                "feature": feature,
                "present": True,
                "non_null": non_null,
                "missing_rate": float(1.0 - (non_null / max(len(df), 1))),
                "nunique": nunique,
                "eligible": eligible,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _contiguous_blocks(mask: np.ndarray, row_ids: np.ndarray) -> pd.DataFrame:
    if len(mask) == 0 or not np.any(mask):
        return pd.DataFrame(
            columns=["block_id", "row_start", "row_end", "length_rows"]
        )

    rows = []
    idx = np.flatnonzero(mask)
    start = prev = int(idx[0])
    block_id = 1
    for pos in idx[1:]:
        pos = int(pos)
        if pos == prev + 1:
            prev = pos
            continue
        rows.append(
            {
                "block_id": block_id,
                "row_start": int(row_ids[start]),
                "row_end": int(row_ids[prev]),
                "length_rows": int(prev - start + 1),
            }
        )
        block_id += 1
        start = prev = pos
    rows.append(
        {
            "block_id": block_id,
            "row_start": int(row_ids[start]),
            "row_end": int(row_ids[prev]),
            "length_rows": int(prev - start + 1),
        }
    )
    return pd.DataFrame(rows)


def compute_row_descriptors(
    matrix: np.ndarray,
    *,
    fs_hz: float,
    threshold_sigma: float,
    min_separation_ns: float,
    center_rows: bool = True,
) -> pd.DataFrame:
    rows = []
    min_separation_s = float(min_separation_ns) * 1e-9

    for row_idx, raw_row in enumerate(matrix):
        x = np.asarray(raw_row, dtype=np.float64)
        if center_rows:
            x = x - np.mean(x)

        absx = np.abs(x)
        energy = float(np.mean(x**2))
        rms = float(np.sqrt(energy))
        p95_abs = float(np.percentile(absx, 95))
        peak_abs = float(np.max(absx))
        crest_factor = float(peak_abs / rms) if rms > 1e-12 else float("nan")
        skewness = float(stats.skew(x, bias=False)) if rms > 1e-12 else float("nan")
        kurtosis = float(stats.kurtosis(x, fisher=True, bias=False)) if rms > 1e-12 else float("nan")

        median_abs = float(np.median(absx))
        mad_abs = float(np.median(np.abs(absx - median_abs)))
        active_threshold = median_abs + 6.0 * max(mad_abs, 1e-12)
        active_ratio = float(np.mean(absx >= active_threshold))

        try:
            pulse_indices = detect_pulses(
                signal_data=x,
                fs=fs_hz,
                threshold_sigma=threshold_sigma,
                min_separation_s=min_separation_s,
                method="threshold",
            )
        except Exception:
            pulse_indices = np.array([], dtype=np.int64)

        pulse_count = int(len(pulse_indices))
        row_duration_s = len(x) / max(fs_hz, 1e-12)
        pulse_rate_hz = float(pulse_count / row_duration_s) if row_duration_s > 0 else float("nan")
        mean_pulse_amp_v = (
            float(np.mean(absx[pulse_indices])) if pulse_count > 0 else float("nan")
        )

        median_dt_s = float("nan")
        iqr_dt_s = float("nan")
        cv_dt = float("nan")
        burstiness = float("nan")
        if pulse_count >= 2:
            dt = np.diff(pulse_indices) / max(fs_hz, 1e-12)
            mean_dt = float(np.mean(dt))
            std_dt = float(np.std(dt))
            median_dt_s = float(np.median(dt))
            iqr_dt_s = float(np.percentile(dt, 75) - np.percentile(dt, 25))
            cv_dt = float(std_dt / mean_dt) if mean_dt > 0 else float("nan")
            burstiness = (
                float((std_dt - mean_dt) / (std_dt + mean_dt))
                if (std_dt + mean_dt) > 0
                else float("nan")
            )

        energy_weights = absx**2
        if np.sum(energy_weights) > 0:
            cdf = np.cumsum(energy_weights) / np.sum(energy_weights)
            center_idx = int(np.searchsorted(cdf, 0.5))
            q10_idx = int(np.searchsorted(cdf, 0.1))
            q90_idx = int(np.searchsorted(cdf, 0.9))
            event_center_frac = float(center_idx / max(len(x) - 1, 1))
            event_width_frac = float((q90_idx - q10_idx) / max(len(x) - 1, 1))
        else:
            event_center_frac = float("nan")
            event_width_frac = float("nan")

        rows.append(
            {
                "row_idx": row_idx,
                "energy_v2": energy,
                "rms_v": rms,
                "p95_abs_v": p95_abs,
                "peak_abs_v": peak_abs,
                "crest_factor": crest_factor,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "active_ratio": active_ratio,
                "pulse_count": pulse_count,
                "pulse_rate_hz": pulse_rate_hz,
                "mean_pulse_amp_v": mean_pulse_amp_v,
                "median_dt_s": median_dt_s,
                "iqr_dt_s": iqr_dt_s,
                "cv_dt": cv_dt,
                "burstiness": burstiness,
                "event_center_frac": event_center_frac,
                "event_width_frac": event_width_frac,
            }
        )

    return pd.DataFrame(rows)


def _build_activity_score(df: pd.DataFrame, features: list[str]) -> pd.Series:
    z_cols = []
    for feature in features:
        if feature not in df.columns:
            continue
        z = _robust_z(df[feature])
        z_cols.append(z.rename(feature))
    if not z_cols:
        return pd.Series(np.nan, index=df.index, name="activity_score")
    score = pd.concat(z_cols, axis=1).mean(axis=1, skipna=True)
    return score.rename("activity_score")


def _build_change_candidates(
    df: pd.DataFrame,
    features: list[str],
    *,
    top_k: int,
    min_row_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.sort_values("row_idx").reset_index(drop=True).copy()

    usable_features = []
    for feature in features:
        if feature not in working.columns:
            continue
        values = pd.to_numeric(working[feature], errors="coerce")
        if values.notna().sum() == 0 or values.nunique(dropna=True) < 2:
            continue
        usable_features.append(feature)

    if not usable_features or len(working) < 2:
        return pd.DataFrame(), pd.DataFrame()

    delta_columns: list[str] = []
    for feature in usable_features:
        z = _robust_z(working[feature]).fillna(0.0)
        delta_col = f"{feature}_delta_z"
        working[delta_col] = z.diff().abs()
        delta_columns.append(delta_col)

    working["change_score"] = working[delta_columns].mean(axis=1, skipna=True)
    delta_matrix = working[delta_columns].to_numpy(dtype=np.float64, copy=True)
    finite_mask = np.isfinite(delta_matrix)
    safe_delta_matrix = np.where(finite_mask, delta_matrix, -np.inf)
    dominant_idx = np.argmax(safe_delta_matrix, axis=1)
    working["dominant_feature"] = [
        usable_features[idx] if finite_mask[row_idx].any() else ""
        for row_idx, idx in enumerate(dominant_idx)
    ]
    dominant_delta = safe_delta_matrix[np.arange(len(working)), dominant_idx].astype(np.float64)
    dominant_delta[~finite_mask.any(axis=1)] = np.nan
    working["dominant_delta_z"] = dominant_delta

    score_series = working.dropna(subset=["change_score"]).copy()
    if score_series.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_rows: list[pd.Series] = []
    taken_rows: list[int] = []
    for _, row in score_series.sort_values("change_score", ascending=False).iterrows():
        row_id = int(row["row_idx"])
        if any(abs(row_id - taken) <= min_row_gap for taken in taken_rows):
            continue
        selected_rows.append(row)
        taken_rows.append(row_id)
        if len(selected_rows) >= top_k:
            break

    candidates = pd.DataFrame(selected_rows).reset_index(drop=True)
    if not candidates.empty:
        candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))
    return score_series, candidates


def _build_descriptor_behavior(
    df: pd.DataFrame,
    features: list[str],
    *,
    activity_score: pd.Series,
    change_score: pd.Series,
    primary_block: dict[str, int] | None,
) -> pd.DataFrame:
    rows = []
    inside_mask = None
    if primary_block is not None:
        inside_mask = (df["row_idx"] >= primary_block["row_start"]) & (df["row_idx"] <= primary_block["row_end"])

    for feature in features:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() == 0:
            continue

        activity_corr = float(values.corr(activity_score)) if values.nunique(dropna=True) >= 2 else float("nan")
        feature_diff = _robust_z(values).diff().abs()
        change_corr = (
            float(feature_diff.corr(change_score))
            if feature_diff.notna().sum() > 1 and change_score.notna().sum() > 1
            else float("nan")
        )
        block_shift_z = float("nan")
        median_inside = float("nan")
        median_outside = float("nan")
        if inside_mask is not None and inside_mask.any() and (~inside_mask).any():
            z_values = _robust_z(values)
            block_shift_z = float(z_values[inside_mask].mean() - z_values[~inside_mask].mean())
            median_inside = float(values[inside_mask].median())
            median_outside = float(values[~inside_mask].median())

        rows.append(
            {
                "feature": feature,
                "median": float(values.median()),
                "q90": float(values.quantile(0.90)),
                "q99": float(values.quantile(0.99)),
                "activity_corr": activity_corr,
                "change_corr": change_corr,
                "block_shift_z": block_shift_z,
                "median_inside_block": median_inside,
                "median_outside_block": median_outside,
            }
        )

    behavior_df = pd.DataFrame(rows)
    if behavior_df.empty:
        return behavior_df
    return behavior_df.sort_values(
        by=["block_shift_z", "activity_corr", "change_corr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _plot_series_matrix(
    matrix: np.ndarray,
    *,
    primary_block: dict[str, int] | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    vmin, vmax = np.quantile(matrix, [0.01, 0.99])
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    if primary_block is not None:
        ax.axhspan(
            primary_block["row_start"],
            primary_block["row_end"],
            color="gold",
            alpha=0.18,
        )
    ax.set_title("Series Matrix Overview")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Capture row")
    fig.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_descriptor_trends(
    df: pd.DataFrame,
    *,
    primary_block: dict[str, int] | None,
    change_candidates: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    x = df["row_idx"]

    panels = [
        ("Activity and Change", [("activity_score", "activity_score"), ("change_score", "change_score")]),
        ("Amplitude Descriptors", [("energy_v2", "energy_v2"), ("p95_abs_v", "p95_abs_v"), ("peak_abs_v", "peak_abs_v")]),
        ("Occupancy and Count", [("active_ratio", "active_ratio"), ("pulse_count", "pulse_count"), ("pulse_rate_hz", "pulse_rate_hz")]),
        ("Temporal and Shape", [("median_dt_s", "median_dt_s"), ("cv_dt", "cv_dt"), ("kurtosis", "kurtosis")]),
    ]

    for ax, (title, series_list) in zip(axes, panels):
        for label, column in series_list:
            if column in df.columns:
                ax.plot(x, df[column], label=label, linewidth=1.2)
        if primary_block is not None:
            ax.axvspan(
                primary_block["row_start"],
                primary_block["row_end"],
                color="gold",
                alpha=0.18,
            )
        if not change_candidates.empty:
            for row_id in change_candidates["row_idx"].head(8):
                ax.axvline(float(row_id), color="crimson", alpha=0.15, linewidth=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Capture row")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_descriptor_heatmap(
    df: pd.DataFrame,
    *,
    features: list[str],
    primary_block: dict[str, int] | None,
    output_path: Path,
) -> None:
    if not features:
        return

    z_rows = []
    labels = []
    for feature in features:
        if feature not in df.columns:
            continue
        z = _robust_z(df[feature]).to_numpy(dtype=np.float64, copy=True)
        if np.all(~np.isfinite(z)):
            continue
        z = np.where(np.isfinite(z), z, 0.0)
        z_rows.append(z)
        labels.append(feature)

    if not z_rows:
        return

    matrix = np.vstack(z_rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Capture row")
    ax.set_title("Standardized Descriptor Heatmap")
    if primary_block is not None:
        ax.axvspan(
            primary_block["row_start"],
            primary_block["row_end"],
            color="gold",
            alpha=0.12,
        )
    fig.colorbar(im, ax=ax, label="Robust z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_representative_waveforms(
    matrix: np.ndarray,
    *,
    axis_values: np.ndarray,
    axis_label: str,
    representative_rows: dict[str, int],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(representative_rows), 1, figsize=(12, 8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (label, row_idx) in zip(axes, representative_rows.items()):
        ax.plot(axis_values, matrix[row_idx], linewidth=1.0)
        ax.set_title(f"{label}: row {row_idx}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(axis_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _select_representative_rows(
    df: pd.DataFrame,
    *,
    primary_block: dict[str, int] | None,
) -> dict[str, int]:
    peak_row = int(df.loc[df["activity_score"].idxmax(), "row_idx"])
    baseline_df = df.copy()
    if primary_block is not None:
        baseline_df = baseline_df[baseline_df["row_idx"] < primary_block["row_start"]]
    if baseline_df.empty:
        baseline_df = df
    baseline_row = int(
        baseline_df.iloc[(baseline_df["activity_score"] - baseline_df["activity_score"].median()).abs().argmin()]["row_idx"]
    )

    post_df = df.copy()
    if primary_block is not None:
        post_df = post_df[post_df["row_idx"] > primary_block["row_end"]]
    if post_df.empty:
        post_df = df[df["row_idx"] > peak_row]
    if post_df.empty:
        post_df = df
    post_row = int(
        post_df.iloc[(post_df["change_score"] - post_df["change_score"].median()).abs().argmin()]["row_idx"]
    )

    return {
        "Baseline waveform": baseline_row,
        "Peak activity waveform": peak_row,
        "Post-event waveform": post_row,
    }


def _format_report(
    *,
    mat_path: Path,
    matrix_shape: tuple[int, int],
    axis_label: str,
    primary_block: dict[str, int] | None,
    activity_blocks_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    change_candidates_df: pd.DataFrame,
) -> str:
    lines = [
        "# MAT Series Study",
        "",
        f"Source file: {mat_path}",
        f"Matrix shape: {matrix_shape[0]} captures x {matrix_shape[1]} samples",
        f"Horizontal axis used in waveform plots: {axis_label}",
        "",
    ]

    if primary_block is not None:
        lines.extend(
            [
                "## Primary Activity Block",
                "",
                f"Rows {primary_block['row_start']} to {primary_block['row_end']} "
                f"({primary_block['length_rows']} captures) define the dominant activity block.",
                "",
            ]
        )

    if not activity_blocks_df.empty:
        lines.append("## Activity Blocks")
        lines.append("")
        for _, row in activity_blocks_df.head(8).iterrows():
            lines.append(
                f"- block {int(row['block_id'])}: rows {int(row['row_start'])}-{int(row['row_end'])} "
                f"({int(row['length_rows'])} captures)"
            )
        lines.append("")

    if not behavior_df.empty:
        lines.append("## Descriptor Behavior")
        lines.append("")
        for _, row in behavior_df.head(8).iterrows():
            lines.append(
                f"- {row['feature']}: block_shift_z={row['block_shift_z']:.3f}, "
                f"activity_corr={row['activity_corr']:.3f}, change_corr={row['change_corr']:.3f}"
            )
        lines.append("")

    if not change_candidates_df.empty:
        lines.append("## Candidate Transition Rows")
        lines.append("")
        for _, row in change_candidates_df.head(10).iterrows():
            lines.append(
                f"- rank {int(row['candidate_rank'])}: row {int(row['row_idx'])}, "
                f"score={float(row['change_score']):.4f}, dominant_feature={row['dominant_feature']}"
            )
        lines.append("")

    return "\n".join(lines).strip()


def run_mat_series_study(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_path = Path(config_path)
    input_cfg = cfg.get("input", {})
    analysis_cfg = cfg.get("analysis", {})
    plot_cfg = cfg.get("plots", {})

    mat_path = _resolve_path(config_path, input_cfg["mat_path"])
    output_dir = _resolve_path(config_path, cfg.get("output_dir", "outputs/mat_series_study"))
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix, time_axis = _load_mat_series(
        mat_path,
        signal_key=str(input_cfg.get("signal_key", "serie")),
        time_key=input_cfg.get("time_key", "time"),
    )
    max_rows = analysis_cfg.get("max_rows")
    if max_rows is not None:
        matrix = matrix[: int(max_rows)]

    axis_values, axis_label = _display_axis(time_axis, matrix.shape[1])
    fs_hz = float(input_cfg.get("fs_hz", 1e9))

    df = compute_row_descriptors(
        matrix,
        fs_hz=fs_hz,
        threshold_sigma=float(analysis_cfg.get("threshold_sigma", 4.0)),
        min_separation_ns=float(analysis_cfg.get("min_separation_ns", 20.0)),
        center_rows=bool(analysis_cfg.get("center_rows", True)),
    )

    descriptor_features = cfg.get("descriptors", {}).get("features", DEFAULT_DESCRIPTOR_FEATURES)
    feature_quality_df = _feature_quality_table(df, descriptor_features)
    eligible_features = feature_quality_df[feature_quality_df["eligible"]]["feature"].tolist()
    if not eligible_features:
        raise ValueError("No eligible descriptor features found for MAT series study.")

    activity_features_cfg = cfg.get("descriptors", {}).get("activity_features", DEFAULT_ACTIVITY_FEATURES)
    change_features_cfg = cfg.get("descriptors", {}).get("change_features", DEFAULT_CHANGE_FEATURES)
    activity_features = [feature for feature in activity_features_cfg if feature in eligible_features]
    change_features = [feature for feature in change_features_cfg if feature in eligible_features]
    if not activity_features:
        activity_features = eligible_features[: min(5, len(eligible_features))]
    if not change_features:
        change_features = eligible_features[: min(8, len(eligible_features))]

    df["activity_score"] = _build_activity_score(df, activity_features)
    change_series_df, change_candidates_df = _build_change_candidates(
        df,
        change_features,
        top_k=int(analysis_cfg.get("change_top_k", 10)),
        min_row_gap=int(analysis_cfg.get("change_min_row_gap", 25)),
    )
    if not change_series_df.empty:
        df = df.merge(
            change_series_df[["row_idx", "change_score", "dominant_feature", "dominant_delta_z"]],
            on="row_idx",
            how="left",
        )
    else:
        df["change_score"] = np.nan
        df["dominant_feature"] = ""
        df["dominant_delta_z"] = np.nan

    activity_threshold = float(df["activity_score"].quantile(float(analysis_cfg.get("active_quantile", 0.99))))
    activity_blocks_df = _contiguous_blocks(
        (df["activity_score"] >= activity_threshold).to_numpy(dtype=bool),
        df["row_idx"].to_numpy(dtype=np.int64),
    )
    primary_block = None
    if not activity_blocks_df.empty:
        primary_block = activity_blocks_df.sort_values(
            by=["length_rows", "row_start"],
            ascending=[False, True],
        ).iloc[0].to_dict()

    corr_features = [feature for feature in eligible_features if feature in df.columns]
    corr_df = df[corr_features].corr(method="spearman")
    behavior_df = _build_descriptor_behavior(
        df,
        corr_features,
        activity_score=df["activity_score"],
        change_score=df["change_score"],
        primary_block=primary_block,
    )

    representative_rows = _select_representative_rows(df, primary_block=primary_block)

    df.to_csv(output_dir / "row_descriptors.csv", index=False, encoding="utf-8-sig")
    feature_quality_df.to_csv(output_dir / "feature_quality.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(output_dir / "feature_correlation_spearman.csv", encoding="utf-8-sig")
    activity_blocks_df.to_csv(output_dir / "activity_blocks.csv", index=False, encoding="utf-8-sig")
    behavior_df.to_csv(output_dir / "descriptor_behavior.csv", index=False, encoding="utf-8-sig")
    if not change_candidates_df.empty:
        change_candidates_df.to_csv(output_dir / "change_candidates.csv", index=False, encoding="utf-8-sig")

    _plot_series_matrix(
        matrix,
        primary_block=primary_block,
        output_path=output_dir / "series_matrix_overview.png",
    )
    _plot_descriptor_trends(
        df,
        primary_block=primary_block,
        change_candidates=change_candidates_df,
        output_path=output_dir / "descriptor_trends.png",
    )
    _plot_descriptor_heatmap(
        df,
        features=behavior_df["feature"].head(8).tolist(),
        primary_block=primary_block,
        output_path=output_dir / "descriptor_heatmap.png",
    )
    _plot_representative_waveforms(
        matrix,
        axis_values=axis_values,
        axis_label=axis_label,
        representative_rows=representative_rows,
        output_path=output_dir / "representative_waveforms.png",
    )

    manifest = {
        "config_path": str(config_path.resolve()),
        "source_file": str(mat_path.resolve()),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "axis_label": axis_label,
        "fs_hz": fs_hz,
        "activity_features": activity_features,
        "change_features": change_features,
        "primary_block": primary_block,
        "representative_rows": representative_rows,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report_text = _format_report(
        mat_path=mat_path.resolve(),
        matrix_shape=matrix.shape,
        axis_label=axis_label,
        primary_block=primary_block,
        activity_blocks_df=activity_blocks_df,
        behavior_df=behavior_df,
        change_candidates_df=change_candidates_df,
    )
    with (output_dir / "study_report.md").open("w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    report_cfg = cfg.get("report", {})
    pdf_path = None
    if bool(report_cfg.get("export_pdf", False)):
        pdf_path = build_mat_series_pdf(
            output_dir,
            pdf_filename=str(report_cfg.get("pdf_filename", "mat_series_study_report.pdf")),
        )

    return {
        "row_descriptors": df,
        "feature_quality": feature_quality_df,
        "correlation": corr_df,
        "activity_blocks": activity_blocks_df,
        "behavior": behavior_df,
        "change_candidates": change_candidates_df,
        "output_dir": output_dir,
        "pdf_path": pdf_path,
    }
