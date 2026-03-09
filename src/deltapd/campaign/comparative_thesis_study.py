from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from deltapd.campaign.descriptor_study import (
    _ensure_task_labels,
    _recommend_subset,
    _task_primary_metric,
    evaluate_univariate_descriptors,
    exhaustive_feature_search,
    forward_feature_selection,
    run_descriptor_study,
)
from deltapd.campaign.pdf_reports import build_descriptor_study_pdf


DATASET_TYPE_MAP = {
    "P1": "internal",
    "P2": "superficial",
    "P3": "multiple",
    "G1": "internal",
    "G2": "superficial",
    "G3": "multiple",
}

DATASET_VARIANT_MAP = {
    "P1": "benchmark",
    "P2": "benchmark",
    "P3": "benchmark",
    "G1": "gemela",
    "G2": "gemela",
    "G3": "gemela",
}

TRANSITION_METHOD_ORDER = [
    "coherence",
    "harmonic_power",
    "epoch_folding",
    "h_test",
    "pdm",
    "gregory_loredo",
    "phase_distance_correlation",
]

COMPARATIVE_BLOCK_ORDER = ["temporal", "phase_prpd", "fused"]
COMPARATIVE_BLOCK_LABELS = {
    "temporal": "Temporal only",
    "phase_prpd": "Phase/PRPD only",
    "fused": "Temporal + Phase",
}
SEMICYCLE_BLOCK_ORDER = ["negative", "positive", "both"]
SEMICYCLE_BLOCK_LABELS = {
    "negative": "Negative semicycle",
    "positive": "Positive semicycle",
    "both": "Both semicycles",
}
TEMPORAL_FEATURE_NAMES = {
    "median_dt_s",
    "iqr_dt_s",
    "p90_dt_s",
    "cv_dt",
    "cv2_dt",
    "local_variation",
    "weibull_beta",
    "burstiness",
    "fano_factor",
}
PHASE_PRPD_FEATURE_NAMES = {
    "phase_inlier_ratio",
    "amplitude_balance_ratio",
}
SEMICYCLE_NEGATIVE_FEATURES = [
    "phase_neg_min_deg",
    "phase_neg_q25_deg",
    "phase_neg_median_deg",
    "phase_neg_q75_deg",
    "phase_neg_max_deg",
    "phase_neg_mean_deg",
    "phase_neg_skewness",
    "phase_neg_kurtosis",
    "phase_neg_count",
]
SEMICYCLE_POSITIVE_FEATURES = [
    "phase_pos_min_deg",
    "phase_pos_q25_deg",
    "phase_pos_median_deg",
    "phase_pos_q75_deg",
    "phase_pos_max_deg",
    "phase_pos_mean_deg",
    "phase_pos_skewness",
    "phase_pos_kurtosis",
    "phase_pos_count",
]


def _repo_root_from_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    return resolved.parents[1] if len(resolved.parents) >= 2 else resolved.parent


def _resolve_path(config_path: str | Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _repo_root_from_config(config_path) / candidate


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _format_score(value: Any) -> str:
    number = _safe_float(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def _format_feature_list(features: list[str]) -> str:
    return ", ".join(features) if features else "(none)"


def _format_counts(counts: dict[str, Any]) -> str:
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{label}={count}" for label, count in ordered)


def _is_temporal_feature(feature: str) -> bool:
    return feature in TEMPORAL_FEATURE_NAMES or feature.endswith("_dt_s")


def _is_phase_prpd_feature(feature: str) -> bool:
    return feature.startswith("phase_") or feature in PHASE_PRPD_FEATURE_NAMES


def _has_numeric_variation(df: pd.DataFrame, feature: str) -> bool:
    if feature not in df.columns:
        return False
    numeric = pd.to_numeric(df[feature], errors="coerce").dropna()
    return not numeric.empty and int(numeric.nunique()) > 1


def _partition_comparative_feature_blocks(feature_bank: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    temporal: list[str] = []
    phase_prpd: list[str] = []
    auxiliary: list[str] = []
    for feature in feature_bank:
        if _is_phase_prpd_feature(feature):
            phase_prpd.append(feature)
        elif _is_temporal_feature(feature):
            temporal.append(feature)
        else:
            auxiliary.append(feature)
    return {
        "temporal": temporal,
        "phase_prpd": phase_prpd,
        "fused": temporal + phase_prpd,
    }, auxiliary


def _run_feature_block_ablation(
    df_windows: pd.DataFrame,
    *,
    tasks_cfg: dict[str, dict[str, Any]],
    feature_bank: list[str],
    n_splits: int,
    seed: int,
    top_k_features: int,
    max_combo_size: int,
    forward_max_features: int,
    tolerance: float,
) -> dict[str, Any]:
    block_features, auxiliary_features = _partition_comparative_feature_blocks(feature_bank)
    rows: list[dict[str, Any]] = []

    for task_name, task_cfg in tasks_cfg.items():
        task_type = str(task_cfg.get("type", "multiclass"))
        task_df, label_column, positive_values, label_note = _ensure_task_labels(
            df_windows,
            task_name,
            task_cfg,
        )
        for block_name in COMPARATIVE_BLOCK_ORDER:
            available_features = [
                feature
                for feature in block_features.get(block_name, [])
                if _has_numeric_variation(task_df, feature)
            ]
            base_row = {
                "task_name": task_name,
                "task_type": task_type,
                "label_column": label_column,
                "label_note": label_note,
                "block_name": block_name,
                "block_label": COMPARATIVE_BLOCK_LABELS[block_name],
                "available_features": available_features,
                "candidate_features": [],
                "n_available_features": int(len(available_features)),
                "primary_metric": _task_primary_metric(task_type),
                "strategy": "",
                "features": [],
                "subset_size": 0,
                "primary_score": float("nan"),
                "balanced_accuracy": float("nan"),
            }
            if not available_features:
                rows.append(base_row)
                continue

            univariate_df = evaluate_univariate_descriptors(
                task_df,
                available_features,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
            )
            ranked_features = univariate_df["feature"].tolist()
            candidate_features = ranked_features[: min(top_k_features, len(ranked_features))]
            exhaustive_df = exhaustive_feature_search(
                task_df,
                candidate_features,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
                max_combo_size=max_combo_size,
            )
            forward_df = forward_feature_selection(
                task_df,
                candidate_features,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
                max_features=forward_max_features,
            )
            recommendation = _recommend_subset(
                univariate_df,
                exhaustive_df,
                forward_df,
                task_type=task_type,
                tolerance=tolerance,
            )
            row = base_row.copy()
            row["candidate_features"] = candidate_features
            if recommendation:
                row.update(
                    {
                        "strategy": str(recommendation.get("strategy", "")),
                        "features": list(recommendation.get("features", [])),
                        "subset_size": int(recommendation.get("subset_size", 0) or 0),
                        "primary_score": float(recommendation.get("primary_score", float("nan"))),
                        "balanced_accuracy": float(recommendation.get("balanced_accuracy", float("nan"))),
                    }
                )
            rows.append(row)

    ablation_df = pd.DataFrame(rows)
    return {
        "results": ablation_df,
        "block_features": block_features,
        "auxiliary_features": auxiliary_features,
    }


def _semicycle_feature_blocks() -> dict[str, list[str]]:
    return {
        "negative": list(SEMICYCLE_NEGATIVE_FEATURES),
        "positive": list(SEMICYCLE_POSITIVE_FEATURES),
        "both": list(SEMICYCLE_NEGATIVE_FEATURES) + list(SEMICYCLE_POSITIVE_FEATURES),
    }


def _run_semicycle_phase_ablation(
    df_windows: pd.DataFrame,
    *,
    tasks_cfg: dict[str, dict[str, Any]],
    n_splits: int,
    seed: int,
    top_k_features: int,
    max_combo_size: int,
    forward_max_features: int,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    block_features = _semicycle_feature_blocks()

    for task_name, task_cfg in tasks_cfg.items():
        task_type = str(task_cfg.get("type", "multiclass"))
        task_df, label_column, positive_values, label_note = _ensure_task_labels(
            df_windows,
            task_name,
            task_cfg,
        )
        for block_name in SEMICYCLE_BLOCK_ORDER:
            available_features = [
                feature
                for feature in block_features.get(block_name, [])
                if _has_numeric_variation(task_df, feature)
            ]
            base_row = {
                "task_name": task_name,
                "task_type": task_type,
                "label_column": label_column,
                "label_note": label_note,
                "block_name": block_name,
                "block_label": SEMICYCLE_BLOCK_LABELS[block_name],
                "available_features": available_features,
                "candidate_features": [],
                "n_available_features": int(len(available_features)),
                "primary_metric": _task_primary_metric(task_type),
                "strategy": "",
                "features": [],
                "subset_size": 0,
                "primary_score": float("nan"),
                "balanced_accuracy": float("nan"),
            }
            if not available_features:
                rows.append(base_row)
                continue

            univariate_df = evaluate_univariate_descriptors(
                task_df,
                available_features,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
            )
            ranked_features = univariate_df["feature"].tolist()
            candidate_features = ranked_features[: min(top_k_features, len(ranked_features))]
            exhaustive_df = exhaustive_feature_search(
                task_df,
                candidate_features,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
                max_combo_size=max_combo_size,
            )
            forward_df = forward_feature_selection(
                task_df,
                candidate_features,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
                max_features=forward_max_features,
            )
            recommendation = _recommend_subset(
                univariate_df,
                exhaustive_df,
                forward_df,
                task_type=task_type,
                tolerance=tolerance,
            )
            row = base_row.copy()
            row["candidate_features"] = candidate_features
            if recommendation:
                row.update(
                    {
                        "strategy": str(recommendation.get("strategy", "")),
                        "features": list(recommendation.get("features", [])),
                        "subset_size": int(recommendation.get("subset_size", 0) or 0),
                        "primary_score": float(recommendation.get("primary_score", float("nan"))),
                        "balanced_accuracy": float(recommendation.get("balanced_accuracy", float("nan"))),
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def build_comparative_event_table(
    points_csv: str | Path,
    *,
    channel: str,
    dataset_keys: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(points_csv)
    required = {"dataset_key", "channel", "toa_s", "peak_v", "prpd_phase_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Points CSV missing columns: {sorted(missing)}")

    df = df[df["channel"] == channel].copy()
    df = df[df["dataset_key"].isin(dataset_keys)].copy()
    if df.empty:
        raise ValueError("No rows remained after filtering channel/dataset_keys.")

    rows: list[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        df_g = df[df["dataset_key"] == dataset_key].copy()
        if df_g.empty:
            continue
        df_g = df_g.sort_values("toa_s").reset_index(drop=True)
        df_g["event_idx"] = range(1, len(df_g) + 1)
        if "delta_t_s" in df_g.columns and pd.to_numeric(df_g["delta_t_s"], errors="coerce").notna().all():
            df_g["delta_t_s"] = pd.to_numeric(df_g["delta_t_s"], errors="coerce")
        else:
            df_g["delta_t_s"] = df_g["toa_s"].diff()
        df_g["log10_dt"] = np.log10(df_g["delta_t_s"].clip(lower=1e-12))
        df_g["pulse_rate_hz"] = 1.0 / df_g["delta_t_s"].clip(lower=1e-12)
        df_g["is_outlier"] = False
        df_g["mean_peak_v"] = df_g["peak_v"].mean()
        df_g["discharge_type"] = DATASET_TYPE_MAP.get(dataset_key, "unknown")
        df_g["acquisition_variant"] = DATASET_VARIANT_MAP.get(dataset_key, "unknown")
        rows.append(df_g)

    out = pd.concat(rows, ignore_index=True)
    out["dataset_rank"] = out["dataset_key"].map({key: idx for idx, key in enumerate(dataset_keys, start=1)})
    return out[
        [
            "event_idx",
            "toa_s",
            "delta_t_s",
            "log10_dt",
            "pulse_rate_hz",
            "peak_v",
            "prpd_phase_deg",
            "is_outlier",
            "dataset_key",
            "dataset_label",
            "group_family",
            "antenna_label",
            "signed_peak_v",
            "discharge_type",
            "acquisition_variant",
            "dataset_rank",
        ]
    ]


def _plot_feature_boxplots(
    df: pd.DataFrame,
    *,
    label_column: str,
    features: list[str],
    title: str,
    output_path: Path,
) -> Path | None:
    usable_features = [feature for feature in features if feature in df.columns]
    if not usable_features:
        return None

    labels = [str(label) for label in pd.Series(df[label_column]).dropna().unique().tolist()]
    if len(labels) < 2:
        return None

    fig, axes = plt.subplots(1, len(usable_features), figsize=(4.2 * len(usable_features), 4.6), squeeze=False)
    axes_row = axes[0]
    palette = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#2a9d8f"]

    for ax, feature in zip(axes_row, usable_features):
        groups = []
        group_labels = []
        for label in labels:
            values = pd.to_numeric(df.loc[df[label_column] == label, feature], errors="coerce").dropna()
            if values.empty:
                continue
            groups.append(values.to_numpy())
            group_labels.append(label)
        if not groups:
            ax.set_axis_off()
            continue
        box = ax.boxplot(groups, patch_artist=True, tick_labels=group_labels)
        for idx, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[idx % len(palette)])
            patch.set_alpha(0.75)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_dataset_heatmap(
    df: pd.DataFrame,
    *,
    dataset_keys: list[str],
    features: list[str],
    output_path: Path,
) -> Path | None:
    usable_features = [feature for feature in features if feature in df.columns]
    if not usable_features:
        return None

    matrix = (
        df.groupby("dataset_key")[usable_features]
        .mean(numeric_only=True)
        .reindex(dataset_keys)
        .dropna(how="all")
    )
    if matrix.empty:
        return None

    z_matrix = matrix.copy()
    for feature in usable_features:
        col = pd.to_numeric(matrix[feature], errors="coerce")
        std = float(col.std(ddof=0))
        if std <= 1e-12 or not np.isfinite(std):
            z_matrix[feature] = 0.0
        else:
            z_matrix[feature] = (col - float(col.mean())) / std

    fig, ax = plt.subplots(figsize=(1.3 * len(usable_features) + 2.0, 0.55 * len(z_matrix.index) + 2.4))
    im = ax.imshow(z_matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(usable_features)))
    ax.set_xticklabels(usable_features, rotation=30, ha="right")
    ax.set_yticks(range(len(z_matrix.index)))
    ax.set_yticklabels(z_matrix.index.tolist())
    ax.set_title("Dataset-level standardized descriptor means")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_window_counts(df: pd.DataFrame, *, output_path: Path) -> Path:
    counts = df["dataset_key"].value_counts().sort_index()
    colors = ["#355070" if key.startswith("P") else "#2a9d8f" for key in counts.index]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(counts.index.tolist(), counts.to_numpy(), color=colors)
    for idx, value in enumerate(counts.to_numpy()):
        ax.text(idx, value + max(counts.max() * 0.015, 1.0), f"{int(value)}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Windows")
    ax.set_title("Comparative window count by dataset")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_transition_case_summary(
    config_path: str | Path,
    *,
    channel: str,
    dataset_keys: list[str],
    state_alarm_root: str | Path | None,
) -> pd.DataFrame | None:
    raw_root = state_alarm_root or f"outputs/state_alarm_{channel.lower()}"
    batch_root = _resolve_path(config_path, raw_root)
    summary_path = batch_root / "transition_overlap_case_summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if "dataset_key" not in df.columns:
        return None
    df = df[df["dataset_key"].isin(dataset_keys)].copy()
    if df.empty:
        return None
    return df


def _ordered_transition_count_columns(df: pd.DataFrame) -> list[str]:
    columns = [col for col in df.columns if col.startswith("transition_count_")]
    priority = {
        f"transition_count_{method}": idx for idx, method in enumerate(TRANSITION_METHOD_ORDER)
    }
    return sorted(columns, key=lambda col: (priority.get(col, len(priority)), col))


def _augment_transition_case_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    count_cols = _ordered_transition_count_columns(work)
    numeric_cols = [
        "n_transition_windows",
        "n_ranked_transition_candidates",
        "n_duplicate_candidate_matches",
        "n_unique_local_methods",
        "max_abs_local_freq_offset_hz",
        "mean_abs_local_freq_offset_hz",
        "mean_local_common_axial_confidence",
        "local_freq_offset_std_hz",
        "local_common_axial_confidence_std",
        "local_method_switch_count",
        "local_method_switch_rate",
        "local_regime_run_count",
        "local_regime_mean_run_length",
        "local_regime_max_run_length",
        "local_regime_persistence_ratio",
        "local_regime_transition_entropy",
        "local_offset_sign_switch_count",
        "local_offset_sign_switch_rate",
        "local_abs_offset_step_mean_hz",
        "local_abs_offset_step_max_hz",
        "local_abs_confidence_step_mean",
        "local_abs_confidence_step_max",
        "bocpd_max_change_prob",
        "bocpd_mean_change_prob",
        "bocpd_run_length_mean",
        "bocpd_run_length_min",
        "bocpd_change_count",
        "bocpd_surprise_score",
        "hmm_high_state_share",
        "hmm_high_state_prob_mean",
        "hmm_state_switch_count",
        "hmm_state_switch_rate",
        "hmm_state_mean_run_length",
        "hmm_state_persistence_ratio",
        "hmm_state_entropy",
        "semi_markov_high_state_share",
        "semi_markov_state_switch_count",
        "semi_markov_state_switch_rate",
        "semi_markov_state_mean_run_length",
        "semi_markov_state_persistence_ratio",
        "semi_markov_state_entropy",
        "semi_markov_segment_count",
        "semi_markov_short_run_count",
        "semi_markov_duration_surprise_mean",
        "transition_method_entropy",
        "transition_dominant_method_share",
        "tfa_wavelet_entropy_mean",
        "tfa_wavelet_entropy_max",
        "tfa_wavelet_dominant_band_unique_count",
        "tfa_wavelet_dominant_band_entropy",
        "tfa_wavelet_dominant_band_switch_count",
        "tfa_wavelet_dominant_band_switch_rate",
        "tfa_wavelet_detail_entropy_mean",
        "tfa_wavelet_detail_entropy_max",
        "tfa_wavelet_detail_dominant_band_unique_count",
        "tfa_wavelet_detail_dominant_band_entropy",
        "tfa_wavelet_detail_dominant_band_switch_count",
        "tfa_wavelet_detail_dominant_band_switch_rate",
    ] + count_cols
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if not count_cols:
        return work

    fallback_total = work[count_cols].sum(axis=1, numeric_only=True)
    total_windows = work.get("n_transition_windows", fallback_total)
    total_windows = pd.to_numeric(total_windows, errors="coerce")
    total_windows = total_windows.where(total_windows > 0, fallback_total)
    work["transition_window_total"] = total_windows

    share_cols: list[str] = []
    for count_col in count_cols:
        share_col = count_col.replace("transition_count_", "transition_share_")
        denom = work["transition_window_total"]
        work[share_col] = np.where(denom > 0, work[count_col] / denom, np.nan)
        share_cols.append(share_col)

    share_matrix = work[share_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    share_matrix = np.where(np.isfinite(share_matrix), share_matrix, 0.0)
    share_sums = share_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        share_matrix,
        share_sums,
        out=np.zeros_like(share_matrix),
        where=share_sums > 0,
    )
    positive = normalized > 0
    entropy = -np.sum(
        np.where(positive, normalized * np.log2(np.clip(normalized, 1e-12, None)), 0.0),
        axis=1,
    )
    valid_rows = share_sums[:, 0] > 0
    entropy_out = np.full(len(work), np.nan, dtype=np.float64)
    dominant_share_out = np.full(len(work), np.nan, dtype=np.float64)
    entropy_denom = np.log2(max(len(share_cols), 2))
    if entropy_denom > 0:
        entropy_out[valid_rows] = entropy[valid_rows] / entropy_denom
    dominant_share_out[valid_rows] = np.max(normalized[valid_rows], axis=1)
    work["transition_method_entropy"] = entropy_out
    work["transition_dominant_method_share"] = dominant_share_out
    return work


def _default_transition_features(df: pd.DataFrame) -> list[str]:
    def _has_variation(series: pd.Series) -> bool:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return not numeric.empty and int(numeric.nunique()) > 1

    features = [
        "max_abs_local_freq_offset_hz",
        "local_freq_offset_std_hz",
        "mean_local_common_axial_confidence",
        "local_common_axial_confidence_std",
        "transition_method_entropy",
        "local_regime_transition_entropy",
        "semi_markov_high_state_share",
        "semi_markov_state_mean_run_length",
        "semi_markov_state_switch_rate",
        "semi_markov_state_entropy",
        "hmm_high_state_share",
        "hmm_state_mean_run_length",
        "bocpd_max_change_prob",
        "bocpd_run_length_mean",
        "bocpd_surprise_score",
        "local_method_switch_rate",
        "local_regime_mean_run_length",
        "local_offset_sign_switch_rate",
        "transition_dominant_method_share",
    ]
    selected = [
        feature
        for feature in features
        if feature in df.columns and _has_variation(df[feature])
    ]
    return selected[:8]


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    if len(labels) == 0:
        return 0.0
    scores = []
    for label in labels:
        scores.append(_binary_confusion(y_true == label, y_pred == label)["f1"])
    return float(np.mean(scores))


def _balanced_accuracy_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    if len(labels) == 0:
        return 0.0
    recalls = []
    for label in labels:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)) if np.any(mask) else 0.0)
    return float(np.mean(recalls))


def _loo_nearest_centroid_case_level(
    df: pd.DataFrame,
    *,
    features: list[str],
    label_column: str,
) -> dict[str, Any]:
    usable_features = [feature for feature in features if feature in df.columns]
    if label_column not in df.columns or len(usable_features) == 0 or len(df) < 4:
        return {}
    work = df.dropna(subset=[label_column]).copy()
    if len(work) < 4:
        return {}
    x_all = work[usable_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y_all = work[label_column].astype(str).to_numpy()
    valid_rows = np.isfinite(x_all).all(axis=1)
    x_all = x_all[valid_rows]
    y_all = y_all[valid_rows]
    labels_present = np.unique(y_all)
    if len(labels_present) < 2 or len(x_all) < len(labels_present) + 1:
        return {}

    preds: list[str] = []
    truth: list[str] = []
    for idx in range(len(x_all)):
        x_train = np.delete(x_all, idx, axis=0)
        y_train = np.delete(y_all, idx, axis=0)
        x_test = x_all[idx : idx + 1]
        train_labels = np.unique(y_train)
        if len(train_labels) < len(labels_present):
            continue
        means = np.mean(x_train, axis=0)
        stds = np.std(x_train, axis=0)
        stds = np.where(stds > 1e-12, stds, 1.0)
        x_train_n = (x_train - means) / stds
        x_test_n = (x_test - means) / stds
        centroids = np.vstack([np.mean(x_train_n[y_train == label], axis=0) for label in train_labels])
        dists = np.sum((x_test_n[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        pred = str(train_labels[int(np.argmin(dists, axis=1)[0])])
        preds.append(pred)
        truth.append(str(y_all[idx]))

    if len(preds) < 3:
        return {}
    y_true = np.asarray(truth)
    y_pred = np.asarray(preds)
    result = {
        "features": usable_features,
        "n_cases": int(len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        metrics = _binary_confusion(y_true == np.unique(y_true)[-1], y_pred == np.unique(y_true)[-1])
        result.update(metrics)
    else:
        result["macro_f1"] = _macro_f1(y_true, y_pred)
        result["balanced_accuracy"] = _balanced_accuracy_multiclass(y_true, y_pred)
    return result


def _plot_transition_case_heatmap(
    df: pd.DataFrame,
    *,
    dataset_keys: list[str],
    features: list[str],
    output_path: Path,
) -> Path | None:
    usable_features = [feature for feature in features if feature in df.columns]
    if not usable_features:
        return None
    matrix = df.set_index("dataset_key")[usable_features].reindex(dataset_keys).dropna(how="all")
    if matrix.empty:
        return None
    z_matrix = matrix.copy()
    for feature in usable_features:
        col = pd.to_numeric(matrix[feature], errors="coerce")
        std = float(col.std(ddof=0))
        z_matrix[feature] = 0.0 if std <= 1e-12 or not np.isfinite(std) else (col - float(col.mean())) / std
    fig, ax = plt.subplots(figsize=(1.2 * len(usable_features) + 2.0, 0.55 * len(z_matrix.index) + 2.2))
    im = ax.imshow(z_matrix.to_numpy(dtype=float), aspect="auto", cmap="PuOr", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(usable_features)))
    ax.set_xticklabels(usable_features, rotation=30, ha="right")
    ax.set_yticks(range(len(z_matrix.index)))
    ax.set_yticklabels(z_matrix.index.tolist())
    ax.set_title("Case-level transition metrics heatmap")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="z-score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_transition_case_scatter(df: pd.DataFrame, *, output_path: Path) -> Path | None:
    required = {"max_abs_local_freq_offset_hz", "mean_local_common_axial_confidence", "dataset_key", "discharge_type"}
    if not required.issubset(df.columns):
        return None
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    palette = {"internal": "#355070", "superficial": "#c26d2d", "multiple": "#6d597a"}
    for _, row in df.iterrows():
        x = _safe_float(row.get("max_abs_local_freq_offset_hz"))
        y = _safe_float(row.get("mean_local_common_axial_confidence"))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        color = palette.get(str(row.get("discharge_type", "")), "#555555")
        ax.scatter(x, y, s=95, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(str(row.get("dataset_key", "")), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Max local abs offset (Hz)")
    ax.set_ylabel("Mean local common axial confidence")
    ax.set_title("Case-level transition offsets vs axial confidence")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_block_ablation_heatmap(
    df: pd.DataFrame,
    *,
    output_path: Path,
) -> Path | None:
    if df.empty:
        return None
    work = df.copy()
    if "balanced_accuracy" not in work.columns:
        return None
    matrix = (
        work.pivot(index="task_name", columns="block_label", values="balanced_accuracy")
        .reindex(index=["dataset6", "type3", "variant2"], columns=[COMPARATIVE_BLOCK_LABELS[name] for name in COMPARATIVE_BLOCK_ORDER])
        .dropna(how="all")
    )
    if matrix.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 2.8 + 0.55 * len(matrix.index)))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Strict block ablation by balanced accuracy")
    for row_idx, task_name in enumerate(matrix.index.tolist()):
        for col_idx, block_label in enumerate(matrix.columns.tolist()):
            value = matrix.iloc[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=9, color="#102a43")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="balanced_accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_semicycle_ablation_heatmap(
    df: pd.DataFrame,
    *,
    output_path: Path,
) -> Path | None:
    if df.empty or "balanced_accuracy" not in df.columns:
        return None
    matrix = (
        df.pivot(index="task_name", columns="block_label", values="balanced_accuracy")
        .reindex(
            index=["dataset6", "type3", "variant2"],
            columns=[SEMICYCLE_BLOCK_LABELS[name] for name in SEMICYCLE_BLOCK_ORDER],
        )
        .dropna(how="all")
    )
    if matrix.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 2.8 + 0.55 * len(matrix.index)))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="YlOrBr", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Semicycle PRPD ablation by balanced accuracy")
    for row_idx in range(len(matrix.index)):
        for col_idx in range(len(matrix.columns)):
            value = matrix.iloc[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=9, color="#40210f")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="balanced_accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_block_ablation_section(
    block_ablation_df: pd.DataFrame | None,
    *,
    auxiliary_features: list[str] | None = None,
) -> str:
    if block_ablation_df is None or block_ablation_df.empty:
        return ""

    lines = [
        "## 4. Strict block ablation",
        "",
        "This section re-runs the same comparative evaluator under three constrained feature banks:",
        "",
        "- `Temporal only`: inter-pulse timing statistics only.",
        "- `Phase/PRPD only`: blind phase concentration, width, and symmetry descriptors only.",
        "- `Temporal + Phase`: union of the two previous banks.",
        "",
    ]
    if auxiliary_features:
        lines.append(
            "Reserve-only auxiliary features excluded from this strict comparison: "
            + ", ".join(f"`{feature}`" for feature in auxiliary_features)
            + "."
        )
        lines.append("")

    lines.extend(
        [
            "| Task | Block | Features | Primary metric | Primary score | Balanced acc. | Strategy |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    task_order = ["dataset6", "type3", "variant2"]
    block_order = {name: idx for idx, name in enumerate(COMPARATIVE_BLOCK_ORDER)}
    work = block_ablation_df.copy()
    work["block_order"] = work["block_name"].map(block_order).fillna(len(block_order))
    work = work.sort_values(by=["task_name", "block_order"]).reset_index(drop=True)
    for _, row in work.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task_name", "")),
                    str(row.get("block_label", "")),
                    _format_feature_list(list(row.get("features", []))),
                    str(row.get("primary_metric", "")),
                    _format_score(row.get("primary_score")),
                    _format_score(row.get("balanced_accuracy")),
                    str(row.get("strategy", "")),
                ]
            )
            + " |"
        )
    lines.append("")

    for task_name in task_order:
        task_rows = work[work["task_name"] == task_name].copy()
        if task_rows.empty:
            continue
        task_rows["primary_score_num"] = pd.to_numeric(task_rows["primary_score"], errors="coerce")
        task_rows["balanced_accuracy_num"] = pd.to_numeric(task_rows["balanced_accuracy"], errors="coerce")
        task_rows = task_rows.sort_values(
            by=["primary_score_num", "balanced_accuracy_num", "block_order"],
            ascending=[False, False, True],
        )
        best_row = task_rows.iloc[0]
        lines.append(
            f"- `{task_name}`: best strict block = `{best_row['block_label']}` with "
            f"`{best_row['primary_metric']} = {_format_score(best_row['primary_score'])}` and "
            f"`balanced_accuracy = {_format_score(best_row['balanced_accuracy'])}` using "
            f"`{_format_feature_list(list(best_row.get('features', [])))}`."
        )
    lines.append("")
    return "\n".join(lines)


def _build_semicycle_ablation_section(semicycle_ablation_df: pd.DataFrame | None) -> str:
    if semicycle_ablation_df is None or semicycle_ablation_df.empty:
        return ""

    lines = [
        "## 4b. Semicycle PRPD baseline",
        "",
        "This benchmark isolates simple PRPD statistics by semicycle, motivated by recent UHF-conditioned PRPD literature.",
        "The feature bank is intentionally simple and interpretable: `min`, `q25`, `median`, `q75`, `max`, `mean`, `skewness`, `kurtosis`, and `count` per semicycle.",
        "",
        "| Task | Block | Features | Primary metric | Primary score | Balanced acc. | Strategy |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    work = semicycle_ablation_df.copy()
    order = {name: idx for idx, name in enumerate(SEMICYCLE_BLOCK_ORDER)}
    work["block_order"] = work["block_name"].map(order).fillna(len(order))
    work = work.sort_values(by=["task_name", "block_order"]).reset_index(drop=True)
    for _, row in work.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task_name", "")),
                    str(row.get("block_label", "")),
                    _format_feature_list(list(row.get("features", []))),
                    str(row.get("primary_metric", "")),
                    _format_score(row.get("primary_score")),
                    _format_score(row.get("balanced_accuracy")),
                    str(row.get("strategy", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    for task_name in ["dataset6", "type3", "variant2"]:
        task_rows = work[work["task_name"] == task_name].copy()
        if task_rows.empty:
            continue
        task_rows["primary_score_num"] = pd.to_numeric(task_rows["primary_score"], errors="coerce")
        task_rows["balanced_accuracy_num"] = pd.to_numeric(task_rows["balanced_accuracy"], errors="coerce")
        task_rows = task_rows.sort_values(
            by=["primary_score_num", "balanced_accuracy_num", "block_order"],
            ascending=[False, False, True],
        )
        best_row = task_rows.iloc[0]
        lines.append(
            f"- `{task_name}`: best semicycle block = `{best_row['block_label']}` with "
            f"`{best_row['primary_metric']} = {_format_score(best_row['primary_score'])}` and "
            f"`balanced_accuracy = {_format_score(best_row['balanced_accuracy'])}` using "
            f"`{_format_feature_list(list(best_row.get('features', [])))}`."
        )
    lines.append("")
    return "\n".join(lines)


def _build_comparative_markdown(
    *,
    output_dir: Path,
    channel: str,
    dataset_keys: list[str],
    recommendations: dict[str, Any],
    df_events: pd.DataFrame,
    df_windows: pd.DataFrame,
    blind_metrics_df: pd.DataFrame | None = None,
    block_ablation_df: pd.DataFrame | None = None,
    block_auxiliary_features: list[str] | None = None,
    semicycle_ablation_df: pd.DataFrame | None = None,
    transition_case_df: pd.DataFrame | None = None,
    transition_eval: dict[str, Any] | None = None,
) -> Path:
    dataset_counts = df_windows["dataset_key"].value_counts().reindex(dataset_keys).dropna().astype(int)
    event_counts = df_events["dataset_key"].value_counts().reindex(dataset_keys).dropna().astype(int)
    type_counts = recommendations.get("type3", {}).get("class_counts", {})
    variant_counts = recommendations.get("variant2", {}).get("class_counts", {})

    dataset_rec = recommendations.get("dataset6", {}).get("recommendation", {})
    type_rec = recommendations.get("type3", {}).get("recommendation", {})
    variant_rec = recommendations.get("variant2", {}).get("recommendation", {})

    blind_section = ""
    if blind_metrics_df is not None and not blind_metrics_df.empty:
        rows = []
        for _, row in blind_metrics_df.iterrows():
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("dataset_key", "")),
                        str(row.get("blind_prpd_method", "")),
                        str(row.get("blind_prpd_selected_method", "")),
                        _format_score(row.get("blind_freq_hz")),
                        _format_score(row.get("blind_prpd_coherence")),
                        _format_score(row.get("blind_prpd_common_axial_confidence")),
                        _format_score(row.get("blind_prpd_common_axial_peak_offset_hz")),
                        _format_score(row.get("blind_prpd_bootstrap_freq_std_hz")),
                        _format_score(row.get("blind_prpd_bootstrap_method_agreement")),
                        _format_score(row.get("blind_prpd_local_freq_std_hz")),
                        _format_score(row.get("blind_prpd_local_method_agreement")),
                    ]
                )
                + " |"
            )
        blind_section = (
            "## 3. Blind PRPD calibration used to build the comparative table\n\n"
            "| Dataset | Requested | Selected | Freq (Hz) | Coherence | Common conf. | Peak offset (Hz) | Boot std (Hz) | Boot agree. | Local std (Hz) | Local agree. |\n"
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            + "\n".join(rows)
            + "\n\n"
        )

    block_ablation_section = _build_block_ablation_section(
        block_ablation_df,
        auxiliary_features=block_auxiliary_features,
    )
    block_ablation_figure_section = ""
    if block_ablation_section:
        block_ablation_figure_section = "### Strict block ablation\n\n![](comparative_block_ablation.png)\n\n"
    semicycle_ablation_section = _build_semicycle_ablation_section(semicycle_ablation_df)
    semicycle_ablation_figure_section = ""
    if semicycle_ablation_section:
        semicycle_ablation_figure_section = "### Semicycle PRPD baseline\n\n![](comparative_semicycle_ablation.png)\n\n"

    transition_section = ""
    if transition_case_df is not None and not transition_case_df.empty:
        rows = []
        for _, row in transition_case_df.iterrows():
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("dataset_key", "")),
                        str(row.get("discharge_type", "")),
                        str(int(row.get("n_transition_windows", 0) or 0)),
                        str(int(row.get("n_ranked_transition_candidates", 0) or 0)),
                        str(int(row.get("n_unique_local_methods", 0) or 0)),
                        _format_score(row.get("transition_method_entropy")),
                        _format_score(row.get("local_method_switch_rate")),
                        _format_score(row.get("local_regime_transition_entropy")),
                        _format_score(row.get("local_regime_mean_run_length")),
                        _format_score(row.get("bocpd_max_change_prob")),
                        _format_score(row.get("bocpd_run_length_mean")),
                        _format_score(row.get("bocpd_surprise_score")),
                        _format_score(row.get("semi_markov_high_state_share")),
                        _format_score(row.get("semi_markov_state_switch_rate")),
                        _format_score(row.get("semi_markov_state_mean_run_length")),
                        _format_score(row.get("hmm_high_state_share")),
                        _format_score(row.get("hmm_state_switch_rate")),
                        _format_score(row.get("hmm_state_mean_run_length")),
                        _format_score(row.get("local_offset_sign_switch_rate")),
                        _format_score(row.get("tfa_wavelet_entropy_mean")),
                        str(int(row.get("tfa_wavelet_dominant_band_unique_count", 0) or 0)),
                        _format_score(row.get("tfa_wavelet_detail_entropy_mean")),
                        str(int(row.get("tfa_wavelet_detail_dominant_band_unique_count", 0) or 0)),
                        str(row.get("dominant_local_method", "")),
                        _format_score(row.get("max_abs_local_freq_offset_hz")),
                        _format_score(row.get("mean_local_common_axial_confidence")),
                        _format_score(row.get("state_primary_score")),
                        _format_score(row.get("alarm_primary_score")),
                    ]
                )
                + " |"
            )
        transition_section = (
            "### Exploratory case-level transition metrics\n\n"
            "This block uses one summary row per dataset from the within-test `state/alarm` pipeline. "
            "It is exploratory because `n=6`, so it should not be sold as the primary result. "
            "Transition-family counts below are deduplicated by matched local blind-PRPD window; "
            "`ranked candidates` is shown separately because several nearby candidates can land on the same local regime.\n\n"
            "| Dataset | Type | Local windows | Ranked candidates | Unique local methods | Method entropy | Switch rate | Seq. entropy | Mean run | BOCPD max change | BOCPD mean run | BOCPD surprise | SMM high share | SMM switch | SMM mean run | HMM high share | HMM switch | HMM mean run | Sign-switch rate | Wavelet H mean | Wavelet band diversity | Wavelet detail H | Wavelet detail diversity | Dominant local method | Max abs offset (Hz) | Mean local conf. | State score | Alarm score |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n"
            + "\n".join(rows)
            + "\n\n"
        )
        transition_section += (
            "The exploratory nearest-centroid check below uses normalized method shares and method-mix entropy, "
            "plus sequence-stability metrics such as switch rate, mean run length, sequence entropy, BOCPD change evidence, HMM and semi-Markov state persistence, and local dispersion, "
            "not raw transition counts, so that the comparison is less sensitive to case length.\n\n"
        )
        if transition_eval:
            type_eval = transition_eval.get("type3", {})
            variant_eval = transition_eval.get("variant2", {})
            if type_eval:
                transition_section += (
                    f"- Exploratory LOO nearest-centroid on normalized transition metrics for `type3`: "
                    f"`macro_f1 = {_format_score(type_eval.get('macro_f1'))}`, "
                    f"`balanced_accuracy = {_format_score(type_eval.get('balanced_accuracy'))}` using "
                    f"`{_format_feature_list(type_eval.get('features', []))}`.\n"
                )
            if variant_eval:
                transition_section += (
                    f"- Exploratory LOO nearest-centroid on normalized transition metrics for `variant2`: "
                    f"`balanced_accuracy = {_format_score(variant_eval.get('balanced_accuracy'))}`, "
                    f"`f1 = {_format_score(variant_eval.get('f1'))}` using "
                    f"`{_format_feature_list(variant_eval.get('features', []))}`.\n"
                )
            transition_section += "\n"

    text = f"""# Comparative CH3 descriptor study

## 1. What this study answers

This study separates three different questions using the same descriptor bank on channel `{channel}`:

- `dataset6`: whether each individual acquisition (`P1`, `P2`, `P3`, `G1`, `G2`, `G3`) has its own signature;
- `type3`: whether the descriptors separate `internal`, `superficial`, and `multiple` discharge regimes;
- `variant2`: whether the same descriptor bank can distinguish `benchmark` from `gemela`.

This avoids the mistake of reading a state study as if it were a discharge-type study.

## 2. Experimental support used here

- Datasets included: {", ".join(dataset_keys)}.
- Events available by dataset: {_format_counts(event_counts.to_dict())}.
- Descriptor windows available by dataset: {_format_counts(dataset_counts.to_dict())}.
- `type3` class balance: {_format_counts(type_counts)}.
- `variant2` class balance: {_format_counts(variant_counts)}.

Important reading rule:

- `delta t` is physical inter-pulse time **inside each acquisition**;
- this comparative study uses distributions of windows across datasets, not a single global timeline joining all six experiments.

{blind_section}{block_ablation_section}{semicycle_ablation_section}## 5. Main findings

### Type separation (`type3`)

- Best subset: `{_format_feature_list(type_rec.get("features", []))}`.
- Strategy: `{type_rec.get("strategy", "unknown")}`.
- `macro_f1 = {_format_score(type_rec.get("primary_score"))}`.
- `balanced_accuracy = {_format_score(type_rec.get("balanced_accuracy"))}`.

Technical reading:

- `p90_dt_s` captures the long tail of inter-pulse spacing;
- `local_variation` captures short-scale irregularity from pulse to pulse;
- `phase_kuramoto_r` captures how concentrated or diffuse the blind phase structure is.

This is the strongest result of the current study because it separates the three physical regimes with very high discrimination using only three descriptors.

### Six-dataset separation (`dataset6`)

- Best subset: `{_format_feature_list(dataset_rec.get("features", []))}`.
- Strategy: `{dataset_rec.get("strategy", "unknown")}`.
- `macro_f1 = {_format_score(dataset_rec.get("primary_score"))}`.
- `balanced_accuracy = {_format_score(dataset_rec.get("balanced_accuracy"))}`.

Technical reading:

- `cv2_dt` and `local_variation` represent fast temporal irregularity and are almost redundant;
- `mean_peak_v` adds amplitude scale;
- `phase_kuramoto_r` adds blind phase concentration;
- `median_dt_s` stabilizes the central timing level.

This result says each dataset is not only different by discharge type; each run also keeps its own acquisition fingerprint.

{transition_section}### Benchmark vs gemela (`variant2`)

- Best subset: `{_format_feature_list(variant_rec.get("features", []))}`.
- Strategy: `{variant_rec.get("strategy", "unknown")}`.
- `auroc = {_format_score(variant_rec.get("primary_score"))}`.
- `balanced_accuracy = {_format_score(variant_rec.get("balanced_accuracy"))}`.

Technical reading:

- `mean_peak_v` dominates this task, so the twin/benchmark difference is not purely temporal;
- `weibull_beta` and `p90_dt_s` indicate that the tail and shape of `delta t` still contribute once amplitude is included.

## 6. What descriptors are doing physically

- `p90_dt_s`: alerts when long gaps between pulses begin to stretch or compress.
- `median_dt_s`: tracks the central pulse cadence.
- `cv2_dt` and `local_variation`: show whether pulse generation becomes bursty or locally unstable.
- `weibull_beta`: summarizes the shape of the `delta t` distribution more compactly than raw percentiles.
- `phase_kuramoto_r`: measures how strongly the blind PRPD clusters around preferred phase sectors.
- `mean_peak_v`: adds pulse severity, not just pulse timing.

## 7. What not to conclude yet

- This comparative study is strong evidence for descriptor usefulness, but it is not yet a final alarm model.
- The best `type3` subset proves discriminatory power across experiments; it does not by itself prove chronological damage progression inside one single specimen.
- For alarm/state publication, the next study must remain separate and be run within each long-duration acquisition.
- The case-level transition block is only exploratory because there are just six datasets.

## 8. Immediate thesis direction

- Use `p90_dt_s + local_variation + phase_kuramoto_r` as the comparative baseline for discharge type.
- Use the within-test state pipeline separately for alarm and regime-change detection.
- Keep `wavelet` available as preprocessing for raw 1-D traces, but do not force it into every comparative experiment if the event table is already stable.

## 9. Figures

### Dataset count overview

![](comparative_window_counts.png)

{block_ablation_figure_section}{semicycle_ablation_figure_section}### Type-separation descriptor distributions

![](comparative_type3_boxplots.png)

### Variant-separation descriptor distributions

![](comparative_variant2_boxplots.png)

### Dataset-level descriptor heatmap

![](comparative_dataset6_heatmap.png)
"""
    report_path = output_dir / "comparative_summary.md"
    report_path.write_text(text, encoding="utf-8")
    return report_path


def _build_comparative_artifacts(
    *,
    output_dir: Path,
    channel: str,
    dataset_keys: list[str],
    blind_metrics_df: pd.DataFrame | None = None,
    block_ablation_df: pd.DataFrame | None = None,
    block_auxiliary_features: list[str] | None = None,
    semicycle_ablation_df: pd.DataFrame | None = None,
    transition_case_df: pd.DataFrame | None = None,
    transition_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recommendations = json.loads((output_dir / "study_recommendations.json").read_text(encoding="utf-8"))
    df_events = pd.read_csv(output_dir / "comparative_event_table.csv")
    df_windows = pd.read_csv(output_dir / "descriptor_windows.csv")
    if transition_case_df is not None and not transition_case_df.empty:
        transition_case_df = _augment_transition_case_metrics(transition_case_df)

    type_features = recommendations.get("type3", {}).get("recommendation", {}).get("features", [])
    variant_features = recommendations.get("variant2", {}).get("recommendation", {}).get("features", [])
    dataset_features = recommendations.get("dataset6", {}).get("recommendation", {}).get("features", [])
    transition_features = _default_transition_features(transition_case_df) if transition_case_df is not None else []

    window_counts_png = _plot_window_counts(df_windows, output_path=output_dir / "comparative_window_counts.png")
    type_boxplot_png = _plot_feature_boxplots(
        df_windows,
        label_column="discharge_type",
        features=type_features,
        title="Type separation using recommended descriptors",
        output_path=output_dir / "comparative_type3_boxplots.png",
    )
    variant_boxplot_png = _plot_feature_boxplots(
        df_windows,
        label_column="acquisition_variant",
        features=variant_features,
        title="Benchmark vs gemela using recommended descriptors",
        output_path=output_dir / "comparative_variant2_boxplots.png",
    )
    heatmap_png = _plot_dataset_heatmap(
        df_windows,
        dataset_keys=dataset_keys,
        features=dataset_features,
        output_path=output_dir / "comparative_dataset6_heatmap.png",
    )
    block_ablation_png = _plot_block_ablation_heatmap(
        block_ablation_df if block_ablation_df is not None else pd.DataFrame(),
        output_path=output_dir / "comparative_block_ablation.png",
    )
    semicycle_ablation_png = _plot_semicycle_ablation_heatmap(
        semicycle_ablation_df if semicycle_ablation_df is not None else pd.DataFrame(),
        output_path=output_dir / "comparative_semicycle_ablation.png",
    )
    transition_heatmap_png = None
    transition_scatter_png = None
    if transition_case_df is not None and not transition_case_df.empty:
        transition_heatmap_png = _plot_transition_case_heatmap(
            transition_case_df,
            dataset_keys=dataset_keys,
            features=transition_features,
            output_path=output_dir / "comparative_transition_case_heatmap.png",
        )
        transition_scatter_png = _plot_transition_case_scatter(
            transition_case_df,
            output_path=output_dir / "comparative_transition_case_scatter.png",
        )
    markdown_path = _build_comparative_markdown(
        output_dir=output_dir,
        channel=channel,
        dataset_keys=dataset_keys,
        recommendations=recommendations,
        df_events=df_events,
        df_windows=df_windows,
        blind_metrics_df=blind_metrics_df,
        block_ablation_df=block_ablation_df,
        block_auxiliary_features=block_auxiliary_features,
        semicycle_ablation_df=semicycle_ablation_df,
        transition_case_df=transition_case_df,
        transition_eval=transition_eval,
    )
    extra_images = [
        (window_counts_png, "Dataset count overview"),
        (block_ablation_png, "Strict block ablation"),
        (semicycle_ablation_png, "Semicycle PRPD baseline"),
        (type_boxplot_png, "Type-separation descriptor distributions"),
        (variant_boxplot_png, "Benchmark-vs-gemela descriptor distributions"),
        (heatmap_png, "Dataset-level descriptor heatmap"),
        (transition_heatmap_png, "Case-level transition metric heatmap"),
        (transition_scatter_png, "Case-level transition offsets vs axial confidence"),
    ]
    return {
        "markdown_path": markdown_path,
        "extra_images": [(path, title) for path, title in extra_images if path is not None],
    }


def run_comparative_thesis_study(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_path = Path(config_path)
    output_dir = _resolve_path(config_path, cfg.get("output_dir", "outputs/comparative_thesis_ch3"))
    output_dir.mkdir(parents=True, exist_ok=True)

    points_csv = _resolve_path(
        config_path,
        cfg.get("input", {}).get("points_csv", "outputs/thesis_master/thesis_master_prpd_points.csv"),
    )
    metrics_csv = points_csv.with_name("thesis_master_metrics.csv")
    channel = str(cfg.get("input", {}).get("channel", "CH3"))
    dataset_keys = list(cfg.get("input", {}).get("dataset_keys", ["P1", "P2", "P3", "G1", "G2", "G3"]))
    blind_metrics_df = None
    transition_case_df = _load_transition_case_summary(
        config_path,
        channel=channel,
        dataset_keys=dataset_keys,
        state_alarm_root=cfg.get("state_alarm_root"),
    )
    if transition_case_df is not None and not transition_case_df.empty:
        transition_case_df = _augment_transition_case_metrics(transition_case_df)
    if metrics_csv.exists():
        candidate_metrics = pd.read_csv(metrics_csv)
        if {"dataset_key", "channel"}.issubset(candidate_metrics.columns):
            blind_metrics_df = candidate_metrics[
                (candidate_metrics["channel"] == channel)
                & (candidate_metrics["dataset_key"].isin(dataset_keys))
            ].copy()
            blind_metrics_df.to_csv(
                output_dir / "blind_prpd_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
    if transition_case_df is not None and not transition_case_df.empty:
        transition_case_df.to_csv(
            output_dir / "comparative_transition_case_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    df_events = build_comparative_event_table(
        points_csv,
        channel=channel,
        dataset_keys=dataset_keys,
    )
    event_csv = output_dir / "comparative_event_table.csv"
    df_events.to_csv(event_csv, index=False, encoding="utf-8-sig")

    report_cfg = cfg.get("report", {})
    descriptor_cfg = {
        "study_name": cfg.get("study_name", "Comparative Thesis CH3 Descriptor Study"),
        "input": {
            "event_csv": str(event_csv),
            "group_columns": ["dataset_key"],
        },
        "windowing": cfg.get(
            "windowing",
            {
                "window_events": 64,
                "step_events": 16,
                "min_valid_events": 32,
                "max_valid_dt_s": 1.0,
                "fano_bin_count": 8,
            },
        ),
        "descriptors": cfg.get("descriptors", {}),
        "tasks": cfg.get(
            "tasks",
            {
                "dataset6": {
                    "type": "multiclass",
                    "label_column": "dataset_key",
                },
                "type3": {
                    "type": "multiclass",
                    "label_column": "discharge_type",
                },
                "variant2": {
                    "type": "binary",
                    "label_column": "acquisition_variant",
                    "positive_values": ["gemela"],
                },
            },
        ),
        "search": cfg.get("search", {}),
        "change_detection": {"enabled": False},
        "report": {"export_pdf": False},
        "output_dir": str(output_dir),
    }

    generated_cfg_path = output_dir / "generated_descriptor_study.yaml"
    with generated_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(descriptor_cfg, f, sort_keys=False, allow_unicode=False)

    outputs = run_descriptor_study(generated_cfg_path)
    df_windows = outputs.get("windows", pd.DataFrame()).copy()
    search_cfg = descriptor_cfg.get("search", {})
    descriptor_bank = list(dict.fromkeys(
        list(descriptor_cfg.get("descriptors", {}).get("search_features", []))
        + list(descriptor_cfg.get("descriptors", {}).get("reserve_features", []))
    ))
    block_ablation_payload = _run_feature_block_ablation(
        df_windows,
        tasks_cfg=descriptor_cfg.get("tasks", {}),
        feature_bank=descriptor_bank,
        n_splits=int(search_cfg.get("n_splits", 5)),
        seed=int(search_cfg.get("random_seed", 42)),
        top_k_features=int(search_cfg.get("top_k_features", 8)),
        max_combo_size=int(search_cfg.get("max_combo_size", 3)),
        forward_max_features=int(search_cfg.get("forward_selection_max_features", 5)),
        tolerance=float(search_cfg.get("score_tolerance", 0.01)),
    )
    block_ablation_df = block_ablation_payload["results"]
    if not block_ablation_df.empty:
        block_ablation_export = block_ablation_df.copy()
        for col in ["available_features", "candidate_features", "features"]:
            if col in block_ablation_export.columns:
                block_ablation_export[col] = block_ablation_export[col].apply(
                    lambda values: ",".join(values) if isinstance(values, list) else values
                )
        block_ablation_export.to_csv(
            output_dir / "comparative_block_ablation.csv",
            index=False,
            encoding="utf-8-sig",
        )
    with (output_dir / "comparative_block_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "blocks": block_ablation_payload["block_features"],
                "auxiliary_features": block_ablation_payload["auxiliary_features"],
                "results": json.loads(block_ablation_df.to_json(orient="records")),
            },
            f,
            indent=2,
        )
    semicycle_ablation_df = _run_semicycle_phase_ablation(
        df_windows,
        tasks_cfg=descriptor_cfg.get("tasks", {}),
        n_splits=int(search_cfg.get("n_splits", 5)),
        seed=int(search_cfg.get("random_seed", 42)),
        top_k_features=int(search_cfg.get("top_k_features", 8)),
        max_combo_size=int(search_cfg.get("max_combo_size", 3)),
        forward_max_features=int(search_cfg.get("forward_selection_max_features", 5)),
        tolerance=float(search_cfg.get("score_tolerance", 0.01)),
    )
    if not semicycle_ablation_df.empty:
        semicycle_export = semicycle_ablation_df.copy()
        for col in ["available_features", "candidate_features", "features"]:
            if col in semicycle_export.columns:
                semicycle_export[col] = semicycle_export[col].apply(
                    lambda values: ",".join(values) if isinstance(values, list) else values
                )
        semicycle_export.to_csv(
            output_dir / "comparative_semicycle_ablation.csv",
            index=False,
            encoding="utf-8-sig",
        )
    with (output_dir / "comparative_semicycle_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "blocks": _semicycle_feature_blocks(),
                "results": json.loads(semicycle_ablation_df.to_json(orient="records")),
            },
            f,
            indent=2,
        )
    transition_eval = {}
    if transition_case_df is not None and not transition_case_df.empty:
        transition_features = _default_transition_features(transition_case_df)
        transition_eval = {
            "type3": _loo_nearest_centroid_case_level(
                transition_case_df,
                features=transition_features,
                label_column="discharge_type",
            ),
            "variant2": _loo_nearest_centroid_case_level(
                transition_case_df,
                features=transition_features,
                label_column="variant",
            ),
        }
        with (output_dir / "comparative_transition_case_eval.json").open("w", encoding="utf-8") as f:
            json.dump(transition_eval, f, indent=2)
    artifact_outputs = _build_comparative_artifacts(
        output_dir=output_dir,
        channel=channel,
        dataset_keys=dataset_keys,
        blind_metrics_df=blind_metrics_df,
        block_ablation_df=block_ablation_df,
        block_auxiliary_features=block_ablation_payload["auxiliary_features"],
        semicycle_ablation_df=semicycle_ablation_df,
        transition_case_df=transition_case_df,
        transition_eval=transition_eval,
    )

    if bool(report_cfg.get("export_pdf", True)):
        pdf_path = build_descriptor_study_pdf(
            output_dir,
            title=str(cfg.get("study_name", "Comparative Thesis CH3 Descriptor Study")),
            pdf_filename=str(report_cfg.get("pdf_filename", "comparative_ch3_descriptor_report.pdf")),
            narrative_markdown_path=artifact_outputs["markdown_path"],
            extra_images=artifact_outputs["extra_images"],
        )
        outputs["pdf_path"] = pdf_path

    outputs["event_csv"] = event_csv
    outputs["generated_config"] = generated_cfg_path
    outputs["comparative_summary_md"] = artifact_outputs["markdown_path"]
    outputs["comparative_images"] = [Path(path) for path, _ in artifact_outputs["extra_images"]]
    outputs["blind_metrics_df"] = blind_metrics_df
    outputs["block_ablation_df"] = block_ablation_df
    outputs["block_ablation_auxiliary_features"] = block_ablation_payload["auxiliary_features"]
    outputs["semicycle_ablation_df"] = semicycle_ablation_df
    outputs["transition_case_df"] = transition_case_df
    outputs["transition_eval"] = transition_eval
    return outputs
