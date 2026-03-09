from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, weibull_min

from deltapd.campaign.config import load_config
from deltapd.campaign.material_state import run_material_state
from deltapd.campaign.pdf_reports import build_descriptor_study_pdf
from deltapd.q1_validation import cohens_d, compare_segments_kruskal

PRIMARY_DESCRIPTOR_BANK = [
    "median_dt_s",
    "iqr_dt_s",
    "p90_dt_s",
    "cv_dt",
    "cv2_dt",
    "local_variation",
    "weibull_beta",
    "burstiness",
    "fano_factor",
    "phase_entropy",
    "phase_kuramoto_r",
    "phase_width_pos_deg",
    "phase_width_neg_deg",
]

RESERVE_DESCRIPTOR_BANK = [
    "phase_inlier_ratio",
    "amplitude_balance_ratio",
    "mean_peak_v",
    "n_events",
]


def _repo_root_from_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    return resolved.parents[1] if len(resolved.parents) >= 2 else resolved.parent


def _resolve_path(config_path: str | Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _repo_root_from_config(config_path) / candidate


def _material_output_event_csv(material_config_path: str | Path) -> Path:
    cfg = load_config(material_config_path)
    output_dir = _material_output_dir(material_config_path)
    return output_dir / "delta_t_series_master.csv"


def _material_output_dir(material_config_path: str | Path) -> Path:
    cfg = load_config(material_config_path)
    output_dir = _resolve_path(
        material_config_path,
        cfg.get("output_dir", "outputs/material_state_outputs"),
    )
    return output_dir


def _window_majority_label(series: pd.Series) -> Any:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    counts = valid.value_counts(dropna=True)
    return counts.index[0]


def _compute_fano_factor(toa_s: np.ndarray, n_bins: int) -> float:
    if len(toa_s) < 8:
        return float("nan")
    t0 = float(np.min(toa_s))
    t1 = float(np.max(toa_s))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return float("nan")
    edges = np.linspace(t0, t1, n_bins + 1)
    counts, _ = np.histogram(toa_s, bins=edges)
    mu = float(np.mean(counts))
    if mu <= 0:
        return float("nan")
    return float(np.var(counts) / mu)


def _phase_cluster_metrics(phases_deg: np.ndarray, peaks_v: np.ndarray) -> dict[str, float]:
    metrics = {
        "phase_entropy": float("nan"),
        "phase_kuramoto_r": float("nan"),
        "phase_width_pos_deg": float("nan"),
        "phase_width_neg_deg": float("nan"),
        "phase_inlier_ratio": float("nan"),
        "amplitude_balance_ratio": float("nan"),
    }
    if len(phases_deg) < 8:
        return metrics

    phases = np.mod(np.asarray(phases_deg, dtype=np.float64), 360.0)
    peaks = np.abs(np.asarray(peaks_v, dtype=np.float64))

    theta2 = np.deg2rad(phases) * 2.0
    metrics["phase_kuramoto_r"] = float(np.abs(np.mean(np.exp(1j * theta2))))

    hist, _ = np.histogram(phases, bins=np.linspace(0.0, 360.0, 37))
    p = hist / max(np.sum(hist), 1)
    p = p[p > 0]
    metrics["phase_entropy"] = (
        float(-np.sum(p * np.log2(p)) / np.log2(36)) if len(p) else float("nan")
    )

    mean_angle = np.arctan2(np.mean(np.sin(theta2)), np.mean(np.cos(theta2))) / 2.0
    center1 = np.rad2deg(mean_angle) % 360.0
    center2 = (center1 + 180.0) % 360.0

    d1 = np.minimum(np.abs(phases - center1), 360.0 - np.abs(phases - center1))
    d2 = np.minimum(np.abs(phases - center2), 360.0 - np.abs(phases - center2))
    d_min = np.minimum(d1, d2)
    median_d = float(np.median(d_min))
    mad = float(np.median(np.abs(d_min - median_d)))
    threshold = median_d + 2.5 * max(mad * 1.4826, 5.0)
    metrics["phase_inlier_ratio"] = float(np.mean(d_min <= threshold))

    pos_mask = phases <= 180.0
    neg_mask = ~pos_mask
    pos_phases = phases[pos_mask]
    neg_phases = phases[neg_mask]
    if len(pos_phases) >= 5:
        metrics["phase_width_pos_deg"] = float(
            np.percentile(pos_phases, 90) - np.percentile(pos_phases, 10)
        )
    if len(neg_phases) >= 5:
        metrics["phase_width_neg_deg"] = float(
            np.percentile(neg_phases, 90) - np.percentile(neg_phases, 10)
        )

    pos_amp = float(np.sum(peaks[pos_mask]))
    neg_amp = float(np.sum(peaks[neg_mask]))
    if neg_amp > 0:
        metrics["amplitude_balance_ratio"] = float(pos_amp / neg_amp)

    return metrics


def compute_window_descriptors(
    df_window: pd.DataFrame,
    *,
    max_valid_dt_s: float = 1.0,
    fano_bin_count: int = 8,
) -> dict[str, float]:
    dt = df_window["delta_t_s"].to_numpy(dtype=np.float64, copy=True)
    valid_mask = np.isfinite(dt) & (dt > 0)
    if "is_outlier" in df_window.columns:
        valid_mask &= ~df_window["is_outlier"].to_numpy(dtype=bool, copy=False)
    else:
        valid_mask &= dt <= max_valid_dt_s

    dt_valid = dt[valid_mask]
    toa = df_window["toa_s"].to_numpy(dtype=np.float64, copy=False)
    peaks = (
        df_window["peak_v"].to_numpy(dtype=np.float64, copy=False)
        if "peak_v" in df_window.columns
        else np.full(len(df_window), np.nan)
    )
    phases = (
        df_window["prpd_phase_deg"].to_numpy(dtype=np.float64, copy=False)
        if "prpd_phase_deg" in df_window.columns
        else np.full(len(df_window), np.nan)
    )
    descriptors = {
        "n_events": int(len(df_window)),
        "n_valid_events": int(len(dt_valid)),
        "valid_event_ratio": float(len(dt_valid) / max(len(df_window), 1)),
        "mean_peak_v": float(np.nanmean(peaks)) if np.any(np.isfinite(peaks)) else float("nan"),
        "median_dt_s": float("nan"),
        "iqr_dt_s": float("nan"),
        "p90_dt_s": float("nan"),
        "cv_dt": float("nan"),
        "cv2_dt": float("nan"),
        "local_variation": float("nan"),
        "weibull_beta": float("nan"),
        "burstiness": float("nan"),
        "fano_factor": float("nan"),
    }

    if len(dt_valid) > 0:
        mean_dt = float(np.mean(dt_valid))
        std_dt = float(np.std(dt_valid))
        descriptors["median_dt_s"] = float(np.median(dt_valid))
        descriptors["iqr_dt_s"] = float(
            np.percentile(dt_valid, 75) - np.percentile(dt_valid, 25)
        )
        descriptors["p90_dt_s"] = float(np.percentile(dt_valid, 90))
        if mean_dt > 0:
            descriptors["cv_dt"] = float(std_dt / mean_dt)
            descriptors["burstiness"] = float((std_dt - mean_dt) / (std_dt + mean_dt))

    if len(dt_valid) > 1:
        prev_dt = dt_valid[:-1]
        next_dt = dt_valid[1:]
        denom = np.maximum(prev_dt + next_dt, 1e-30)
        descriptors["cv2_dt"] = float(np.mean(2.0 * np.abs(next_dt - prev_dt) / denom))
        descriptors["local_variation"] = float(
            3.0 * np.mean(((next_dt - prev_dt) / denom) ** 2)
        )

    if len(dt_valid) >= 8:
        try:
            params = weibull_min.fit(dt_valid, floc=0)
            descriptors["weibull_beta"] = float(params[0])
        except Exception:
            pass

    descriptors["fano_factor"] = _compute_fano_factor(toa[np.isfinite(toa)], fano_bin_count)

    phase_peak_mask = np.isfinite(phases) & np.isfinite(peaks)
    descriptors.update(
        _phase_cluster_metrics(
            phases_deg=phases[phase_peak_mask],
            peaks_v=peaks[phase_peak_mask],
        )
    )
    return descriptors


def build_feature_windows(
    df_events: pd.DataFrame,
    *,
    window_events: int = 64,
    step_events: int = 16,
    min_valid_events: int = 32,
    max_valid_dt_s: float = 1.0,
    fano_bin_count: int = 8,
    group_columns: list[str] | None = None,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    if window_events <= 1:
        raise ValueError("window_events must be greater than 1.")
    if step_events <= 0:
        raise ValueError("step_events must be greater than 0.")

    group_columns = [col for col in (group_columns or []) if col in df_events.columns]
    if label_columns is None:
        label_columns = [
            col for col in ["stage", "dataset_key", "pred_category"] if col in df_events.columns
        ]
    else:
        label_columns = [col for col in label_columns if col in df_events.columns]

    if group_columns:
        grouped = df_events.groupby(group_columns, sort=False, dropna=False)
    else:
        grouped = [(("__all__",), df_events)]

    rows: list[dict[str, Any]] = []
    global_window_index = 0

    for group_key, df_group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        sort_cols = [col for col in ["toa_s", "event_idx"] if col in df_group.columns]
        df_group = df_group.sort_values(sort_cols).reset_index(drop=True)
        if len(df_group) < window_events:
            continue

        for start in range(0, len(df_group) - window_events + 1, step_events):
            df_window = df_group.iloc[start : start + window_events].copy()
            descriptors = compute_window_descriptors(
                df_window,
                max_valid_dt_s=max_valid_dt_s,
                fano_bin_count=fano_bin_count,
            )
            if descriptors["n_valid_events"] < min_valid_events:
                continue

            row: dict[str, Any] = {
                "window_index": global_window_index,
                "event_start_idx": int(df_window.index[0]),
                "event_end_idx": int(df_window.index[-1]),
                "toa_start_s": float(df_window["toa_s"].iloc[0]) if "toa_s" in df_window.columns else float(start),
                "toa_end_s": float(df_window["toa_s"].iloc[-1]) if "toa_s" in df_window.columns else float(start + window_events - 1),
            }
            for col_name, col_value in zip(group_columns, group_key):
                row[col_name] = col_value
            for label_col in label_columns:
                row[label_col] = _window_majority_label(df_window[label_col])
            row.update(descriptors)
            rows.append(row)
            global_window_index += 1

    return pd.DataFrame(rows)


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    scores = []
    for label in labels:
        yt = y_true == label
        yp = y_pred == label
        scores.append(_binary_confusion(yt, yp)["f1"])
    return float(np.mean(scores)) if scores else 0.0


def _balanced_accuracy_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    recalls = []
    for label in labels:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)) if np.any(mask) else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    pos_scores = scores[y_true]
    neg_scores = scores[~y_true]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    ranked = rankdata(np.concatenate([pos_scores, neg_scores]))
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    rank_sum_pos = float(np.sum(ranked[:n_pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _normalize_split(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_test = np.where(np.isfinite(x_test), x_test, medians)

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)

    return (x_train - means) / stds, (x_test - means) / stds


def _fit_centroids(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.unique(y_train)
    centroids = np.vstack([np.mean(x_train[y_train == label], axis=0) for label in labels])
    return labels, centroids


def _predict_nearest_centroid(
    labels: np.ndarray,
    centroids: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    dists = np.sum((x_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    pred = labels[np.argmin(dists, axis=1)]
    if len(labels) == 2:
        score = dists[:, 0] - dists[:, 1]
        return pred, score
    return pred, None


def _make_binary_labels(y: pd.Series, positive_values: list[Any] | None) -> np.ndarray:
    if positive_values:
        positives = set(positive_values)
    else:
        numeric = pd.to_numeric(y, errors="coerce")
        if np.all(np.isfinite(numeric.to_numpy())):
            positives = {float(np.nanmax(numeric.to_numpy()))}
        else:
            positives = {y.dropna().astype(str).sort_values().iloc[-1]}
    return y.apply(lambda value: value in positives if pd.notna(value) else np.nan).to_numpy()


def _effective_n_splits(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    if len(counts) == 0:
        return 0
    return int(max(1, min(requested, int(np.min(counts)))))


def _stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for fold_id, part in enumerate(np.array_split(idx, n_splits)):
            folds[fold_id].extend(part.tolist())
    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


def evaluate_feature_subset(
    df_windows: pd.DataFrame,
    features: list[str],
    *,
    label_column: str,
    task_type: str,
    positive_values: list[Any] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    data = df_windows.dropna(subset=[label_column]).copy()
    if data.empty:
        raise ValueError(f"No valid labels found in column {label_column!r}.")

    x = data[features].to_numpy(dtype=np.float64)
    y_raw = data[label_column]

    if task_type == "binary":
        y = _make_binary_labels(y_raw, positive_values)
        mask = pd.notna(y)
        x = x[mask]
        y = y[mask].astype(bool)
    else:
        y = y_raw.to_numpy()

    n_samples = len(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return {
            "n_samples": int(n_samples),
            "n_classes": int(len(classes)),
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "precision": float("nan"),
            "recall": float("nan"),
            "auroc": float("nan"),
            "evaluation_mode": "degenerate",
        }

    effective_splits = _effective_n_splits(y, n_splits)
    oof_pred = np.empty(n_samples, dtype=object)
    oof_score = np.full(n_samples, np.nan)

    if effective_splits >= 2:
        folds = _stratified_folds(y, effective_splits, seed)
        eval_mode = f"cv{effective_splits}"
        for fold in folds:
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[fold] = False
            x_train, x_test = _normalize_split(x[train_mask], x[fold])
            y_train = y[train_mask]
            labels, centroids = _fit_centroids(x_train, y_train)
            pred, score = _predict_nearest_centroid(labels, centroids, x_test)
            oof_pred[fold] = pred
            if score is not None:
                oof_score[fold] = score
    else:
        eval_mode = "resubstitution"
        x_train, x_test = _normalize_split(x, x)
        labels, centroids = _fit_centroids(x_train, y)
        pred, score = _predict_nearest_centroid(labels, centroids, x_test)
        oof_pred = pred.astype(object)
        if score is not None:
            oof_score = score

    if task_type == "binary":
        y_true = y.astype(bool)
        y_pred = np.asarray(oof_pred, dtype=bool)
        confusion = _binary_confusion(y_true, y_pred)
        return {
            "n_samples": int(n_samples),
            "n_classes": 2,
            "macro_f1": float(confusion["f1"]),
            "balanced_accuracy": float(confusion["balanced_accuracy"]),
            "precision": float(confusion["precision"]),
            "recall": float(confusion["recall"]),
            "auroc": _binary_auc(y_true, oof_score),
            "evaluation_mode": eval_mode,
        }

    y_pred = np.asarray(oof_pred)
    return {
        "n_samples": int(n_samples),
        "n_classes": int(len(classes)),
        "macro_f1": _macro_f1(y, y_pred),
        "balanced_accuracy": _balanced_accuracy_multiclass(y, y_pred),
        "precision": float("nan"),
        "recall": float("nan"),
        "auroc": float("nan"),
        "evaluation_mode": eval_mode,
    }


def _task_primary_metric(task_type: str) -> str:
    return "auroc" if task_type == "binary" else "macro_f1"


def _univariate_stats(
    df_windows: pd.DataFrame,
    feature: str,
    *,
    label_column: str,
    task_type: str,
    positive_values: list[Any] | None,
) -> dict[str, Any]:
    values = df_windows[feature]
    missing_rate = float(np.mean(~np.isfinite(values.to_numpy(dtype=np.float64))))
    stats_row: dict[str, Any] = {
        "missing_rate": missing_rate,
        "effect_size": float("nan"),
        "p_value": float("nan"),
    }

    valid = df_windows[[feature, label_column]].dropna()
    if valid.empty:
        return stats_row

    if task_type == "binary":
        y_bin = _make_binary_labels(valid[label_column], positive_values)
        mask = pd.notna(y_bin)
        x = valid.loc[mask, feature].to_numpy(dtype=np.float64)
        y = y_bin[mask].astype(bool)
        if np.unique(x[np.isfinite(x)]).size < 2:
            return stats_row
        if np.sum(y) > 1 and np.sum(~y) > 1:
            stats_row["effect_size"] = float(cohens_d(x[y], x[~y]))
            _, p_value = stats.mannwhitneyu(x[y], x[~y], alternative="two-sided")
            stats_row["p_value"] = float(p_value)
        return stats_row

    grouped = [
        grp[feature].to_numpy(dtype=np.float64)
        for _, grp in valid.groupby(label_column)
        if len(grp) >= 2
    ]
    if len(grouped) >= 2:
        pooled = np.concatenate(grouped)
        if np.unique(pooled[np.isfinite(pooled)]).size < 2:
            return stats_row
        try:
            result = compare_segments_kruskal(grouped, alpha=0.05)
        except ValueError:
            return stats_row
        stats_row["effect_size"] = float(result.effect_size_eta_squared)
        stats_row["p_value"] = float(result.p_value)
    return stats_row


def evaluate_univariate_descriptors(
    df_windows: pd.DataFrame,
    features: list[str],
    *,
    label_column: str,
    task_type: str,
    positive_values: list[Any] | None,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        metrics = evaluate_feature_subset(
            df_windows,
            [feature],
            label_column=label_column,
            task_type=task_type,
            positive_values=positive_values,
            n_splits=n_splits,
            seed=seed,
        )
        stats_row = _univariate_stats(
            df_windows,
            feature,
            label_column=label_column,
            task_type=task_type,
            positive_values=positive_values,
        )
        rows.append(
            {
                "feature": feature,
                "subset_size": 1,
                "features": feature,
                **metrics,
                **stats_row,
            }
        )

    df = pd.DataFrame(rows)
    primary_metric = _task_primary_metric(task_type)
    return df.sort_values(
        by=[primary_metric, "balanced_accuracy", "effect_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def exhaustive_feature_search(
    df_windows: pd.DataFrame,
    candidate_features: list[str],
    *,
    label_column: str,
    task_type: str,
    positive_values: list[Any] | None,
    n_splits: int,
    seed: int,
    max_combo_size: int,
) -> pd.DataFrame:
    rows = []
    for combo_size in range(2, max_combo_size + 1):
        for combo in itertools.combinations(candidate_features, combo_size):
            metrics = evaluate_feature_subset(
                df_windows,
                list(combo),
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
            )
            rows.append(
                {
                    "subset_size": combo_size,
                    "features": ",".join(combo),
                    **metrics,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "subset_size",
                "features",
                "n_samples",
                "n_classes",
                "macro_f1",
                "balanced_accuracy",
                "precision",
                "recall",
                "auroc",
                "evaluation_mode",
            ]
        )
    primary_metric = _task_primary_metric(task_type)
    return pd.DataFrame(rows).sort_values(
        by=[primary_metric, "balanced_accuracy", "subset_size"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def forward_feature_selection(
    df_windows: pd.DataFrame,
    candidate_features: list[str],
    *,
    label_column: str,
    task_type: str,
    positive_values: list[Any] | None,
    n_splits: int,
    seed: int,
    max_features: int,
) -> pd.DataFrame:
    if not candidate_features:
        return pd.DataFrame()

    primary_metric = _task_primary_metric(task_type)
    selected: list[str] = []
    remaining = list(candidate_features)
    rows = []

    for step in range(1, min(max_features, len(candidate_features)) + 1):
        best_feature = None
        best_metrics = None
        best_score = -np.inf

        for feature in remaining:
            subset = selected + [feature]
            metrics = evaluate_feature_subset(
                df_windows,
                subset,
                label_column=label_column,
                task_type=task_type,
                positive_values=positive_values,
                n_splits=n_splits,
                seed=seed,
            )
            score = metrics.get(primary_metric)
            score = float(score) if score is not None and np.isfinite(score) else -np.inf
            if score > best_score:
                best_score = score
                best_feature = feature
                best_metrics = metrics

        if best_feature is None or best_metrics is None:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        rows.append(
            {
                "step": step,
                "added_feature": best_feature,
                "subset_size": len(selected),
                "features": ",".join(selected),
                **best_metrics,
            }
        )

    return pd.DataFrame(rows)


def _recommend_subset(
    univariate_df: pd.DataFrame,
    exhaustive_df: pd.DataFrame,
    forward_df: pd.DataFrame,
    *,
    task_type: str,
    tolerance: float,
) -> dict[str, Any]:
    primary_metric = _task_primary_metric(task_type)
    frames = []
    for strategy, df in [
        ("univariate", univariate_df),
        ("exhaustive", exhaustive_df),
        ("forward", forward_df),
    ]:
        if df is None or df.empty:
            continue
        current = df.copy()
        current["strategy"] = strategy
        frames.append(current)
    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[np.isfinite(combined[primary_metric])]
    if combined.empty:
        return {}

    best_score = float(combined[primary_metric].max())
    pool = combined[combined[primary_metric] >= (best_score - tolerance)].copy()
    pool = pool.sort_values(
        by=["subset_size", primary_metric, "balanced_accuracy"],
        ascending=[True, False, False],
    )
    best_row = pool.iloc[0]
    return {
        "strategy": str(best_row["strategy"]),
        "features": str(best_row["features"]).split(","),
        "subset_size": int(best_row["subset_size"]),
        "primary_metric": primary_metric,
        "primary_score": float(best_row[primary_metric]),
        "balanced_accuracy": float(best_row["balanced_accuracy"]),
    }


def _find_redundant_features(
    redundancy_df: pd.DataFrame,
    ranked_features: list[str],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    positions = {feature: idx for idx, feature in enumerate(ranked_features)}
    redundant = []
    for feature_a, feature_b in itertools.combinations(ranked_features, 2):
        if feature_a not in redundancy_df.index or feature_b not in redundancy_df.columns:
            continue
        rho = redundancy_df.loc[feature_a, feature_b]
        if not np.isfinite(rho) or abs(float(rho)) < threshold:
            continue
        keep, drop = (
            (feature_a, feature_b)
            if positions.get(feature_a, 10**6) <= positions.get(feature_b, 10**6)
            else (feature_b, feature_a)
        )
        redundant.append(
            {
                "keep": keep,
                "drop": drop,
                "spearman_rho": float(rho),
            }
        )
    return redundant


def _default_tasks(df_events: pd.DataFrame) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    preferred_labels: list[tuple[str, pd.Series]] = []
    for label_column in ["stage", "segment"]:
        if label_column not in df_events.columns:
            continue
        valid = pd.to_numeric(df_events[label_column], errors="coerce").dropna()
        if not valid.empty:
            preferred_labels.append((label_column, valid))

    if not preferred_labels:
        return tasks

    informative = [item for item in preferred_labels if item[1].nunique() >= 2]
    label_column, valid_levels = informative[0] if informative else preferred_labels[0]
    max_level = int(valid_levels.max())
    tasks["state"] = {
        "type": "multiclass",
        "label_column": label_column,
        "fallback_time_segments": 3,
    }
    tasks["alarm"] = {
        "type": "binary",
        "label_column": label_column,
        "positive_values": [max_level],
        "fallback_time_segments": 3,
    }
    return tasks


def _ensure_task_labels(
    df_windows: pd.DataFrame,
    task_name: str,
    task_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, str, list[Any] | None, str | None]:
    label_column = task_cfg["label_column"]
    positive_values = task_cfg.get("positive_values")
    if label_column in df_windows.columns and df_windows[label_column].dropna().nunique() >= 2:
        return df_windows, label_column, positive_values, None

    fallback_segments = int(task_cfg.get("fallback_time_segments", 0))
    if fallback_segments < 2:
        return df_windows, label_column, positive_values, None

    derived_column = f"__auto_{task_name}"
    df_with_labels = df_windows.copy()
    ordered_index = df_with_labels.sort_values(["toa_start_s", "window_index"]).index.to_numpy()
    derived_labels = np.full(len(df_with_labels), np.nan)
    for segment_id, indices in enumerate(np.array_split(ordered_index, fallback_segments), start=1):
        derived_labels[indices] = segment_id
    df_with_labels[derived_column] = derived_labels.astype(int)

    if task_cfg.get("type", "multiclass") == "binary" and not positive_values:
        positive_values = [fallback_segments]

    note = (
        f"Task {task_name} used automatic time segmentation into {fallback_segments} states "
        f"because {label_column!r} had fewer than two classes."
    )
    return df_with_labels, derived_column, positive_values, note


def _format_task_report(
    task_name: str,
    label_note: str | None,
    recommendation: dict[str, Any],
    univariate_df: pd.DataFrame,
    exhaustive_df: pd.DataFrame,
    forward_df: pd.DataFrame,
) -> str:
    lines = [f"## Task: {task_name}", ""]
    if label_note:
        lines.append(label_note)
        lines.append("")
    if recommendation:
        lines.append(
            f"Recommended subset: {', '.join(recommendation['features'])} "
            f"({recommendation['strategy']}, {recommendation['primary_metric']}="
            f"{recommendation['primary_score']:.4f}, "
            f"balanced_accuracy={recommendation['balanced_accuracy']:.4f})"
        )
    else:
        lines.append("No valid recommendation generated.")
    lines.append("")

    if not univariate_df.empty:
        lines.append("Top univariate descriptors:")
        for _, row in univariate_df.head(5).iterrows():
            lines.append(
                f"- {row['feature']}: macro_f1={row['macro_f1']:.4f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
                f"auroc={row['auroc']:.4f}"
            )
        lines.append("")

    if not exhaustive_df.empty:
        lines.append("Top exhaustive combinations:")
        for _, row in exhaustive_df.head(5).iterrows():
            lines.append(
                f"- {row['features']}: macro_f1={row['macro_f1']:.4f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
                f"auroc={row['auroc']:.4f}"
            )
        lines.append("")

    if not forward_df.empty:
        lines.append("Forward-selection path:")
        for _, row in forward_df.iterrows():
            lines.append(
                f"- step {int(row['step'])}: {row['features']} "
                f"(macro_f1={row['macro_f1']:.4f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
                f"auroc={row['auroc']:.4f})"
            )
        lines.append("")

    return "\n".join(lines).strip()


def _feature_quality_table(
    df_windows: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df_windows.columns:
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

        values = pd.to_numeric(df_windows[feature], errors="coerce")
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
                "missing_rate": float(1.0 - (non_null / max(len(df_windows), 1))),
                "nunique": nunique,
                "eligible": eligible,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _recommended_change_features(
    recommendations: dict[str, Any],
    fallback_features: list[str],
) -> list[str]:
    ordered: list[str] = []
    for task_payload in recommendations.values():
        recommendation = task_payload.get("recommendation", {})
        for feature in recommendation.get("features", []):
            if feature not in ordered:
                ordered.append(feature)
    for feature in fallback_features:
        if feature not in ordered:
            ordered.append(feature)
    return ordered


def _build_blind_transition_overlap(
    change_candidates_df: pd.DataFrame,
    blind_trace_df: pd.DataFrame,
) -> pd.DataFrame:
    if change_candidates_df.empty or blind_trace_df.empty:
        return pd.DataFrame()

    candidates = change_candidates_df.copy()
    trace = blind_trace_df.copy()
    for col in [
        "toa_start_s",
        "toa_end_s",
        "toa_center_s",
        "freq_hz",
        "freq_offset_from_global_hz",
        "common_axial_confidence",
        "coherence",
    ]:
        if col in trace.columns:
            trace[col] = pd.to_numeric(trace[col], errors="coerce")
    for col in ["toa_start_s", "toa_end_s", "change_score", "dominant_delta_z"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.sort_values("candidate_rank").iterrows():
        cand_start = float(candidate["toa_start_s"])
        cand_end = float(candidate["toa_end_s"])
        cand_center = 0.5 * (cand_start + cand_end)
        trace_eval = trace.copy()
        overlap_start = np.maximum(trace_eval["toa_start_s"].to_numpy(dtype=np.float64), cand_start)
        overlap_end = np.minimum(trace_eval["toa_end_s"].to_numpy(dtype=np.float64), cand_end)
        overlap_duration = np.maximum(0.0, overlap_end - overlap_start)
        trace_eval["overlap_duration_s"] = overlap_duration
        trace_eval["center_distance_s"] = np.abs(
            trace_eval["toa_center_s"].to_numpy(dtype=np.float64) - cand_center
        )

        overlapping = trace_eval[trace_eval["overlap_duration_s"] > 0].copy()
        if not overlapping.empty:
            selected = overlapping.sort_values(
                ["overlap_duration_s", "common_axial_confidence", "center_distance_s"],
                ascending=[False, False, True],
            ).iloc[0]
            match_mode = "overlap"
        else:
            selected = trace_eval.sort_values(
                ["center_distance_s", "common_axial_confidence"],
                ascending=[True, False],
            ).iloc[0]
            match_mode = "nearest"

        rows.append(
            {
                "candidate_rank": int(candidate["candidate_rank"]),
                "candidate_toa_start_s": cand_start,
                "candidate_toa_end_s": cand_end,
                "candidate_toa_center_s": cand_center,
                "change_score": float(candidate["change_score"]),
                "dominant_feature": str(candidate.get("dominant_feature", "")),
                "dominant_delta_z": float(candidate.get("dominant_delta_z", float("nan"))),
                "match_mode": match_mode,
                "local_window_index": int(selected["local_window_index"]),
                "local_toa_start_s": float(selected["toa_start_s"]),
                "local_toa_end_s": float(selected["toa_end_s"]),
                "local_toa_center_s": float(selected["toa_center_s"]),
                "local_selected_method": str(selected.get("selected_method", "")),
                "local_freq_hz": float(selected.get("freq_hz", float("nan"))),
                "local_freq_offset_from_global_hz": float(
                    selected.get("freq_offset_from_global_hz", float("nan"))
                ),
                "local_common_axial_confidence": float(
                    selected.get("common_axial_confidence", float("nan"))
                ),
                "local_coherence": float(selected.get("coherence", float("nan"))),
                "local_n_events": int(selected.get("n_events", 0)),
                "center_distance_s": float(selected["center_distance_s"]),
                "overlap_duration_s": float(selected["overlap_duration_s"]),
            }
        )

    overlap_df = pd.DataFrame(rows)
    if overlap_df.empty:
        return overlap_df

    overlap_df["candidate_rank"] = pd.to_numeric(
        overlap_df["candidate_rank"],
        errors="coerce",
    ).astype("Int64")
    overlap_df["local_window_index"] = pd.to_numeric(
        overlap_df["local_window_index"],
        errors="coerce",
    ).astype("Int64")

    overlap_df = overlap_df.sort_values(
        ["local_window_index", "candidate_rank", "center_distance_s"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    overlap_df["local_window_match_rank"] = (
        overlap_df.groupby("local_window_index", dropna=False).cumcount() + 1
    )
    overlap_df["local_window_candidate_count"] = overlap_df.groupby(
        "local_window_index",
        dropna=False,
    )["candidate_rank"].transform("size")
    overlap_df["is_primary_local_match"] = overlap_df["local_window_match_rank"].eq(1)
    return overlap_df.sort_values("candidate_rank").reset_index(drop=True)


def _plot_blind_transition_map(
    blind_trace_df: pd.DataFrame,
    change_candidates_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    *,
    out_png: str | Path,
    title: str = "Blind PRPD local stability vs transition windows",
) -> Path | None:
    if blind_trace_df.empty or change_candidates_df.empty or overlap_df.empty:
        return None

    trace = blind_trace_df.copy().sort_values("toa_center_s").reset_index(drop=True)
    candidates = change_candidates_df.copy().sort_values("candidate_rank").reset_index(drop=True)
    overlap = overlap_df.copy().sort_values("candidate_rank").reset_index(drop=True)
    out_png = Path(out_png)

    for col in [
        "toa_start_s",
        "toa_end_s",
        "toa_center_s",
        "freq_offset_from_global_hz",
        "common_axial_confidence",
    ]:
        if col in trace.columns:
            trace[col] = pd.to_numeric(trace[col], errors="coerce")
    for col in ["toa_start_s", "toa_end_s", "toa_center_s"]:
        if col in overlap.columns:
            overlap[col] = pd.to_numeric(overlap[col], errors="coerce")
    for col in ["toa_start_s", "toa_end_s", "change_score"]:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")

    method_palette = {
        "coherence": "#2d6a8a",
        "harmonic_power": "#c26d2d",
        "epoch_folding": "#6d597a",
        "gregory_loredo": "#5d8f52",
        "phase_distance_correlation": "#9b2226",
    }
    methods = [str(m) for m in trace["selected_method"].dropna().astype(str).unique().tolist()]
    if not methods:
        methods = ["unknown"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0, 0.85]},
    )
    ax_offset, ax_conf, ax_method = axes

    for _, row in candidates.iterrows():
        start = float(row["toa_start_s"])
        end = float(row["toa_end_s"])
        center = 0.5 * (start + end)
        for ax in axes:
            ax.axvspan(start, end, color="#e9c7ad", alpha=0.18, zorder=0)
            ax.axvline(center, color="#b5653b", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)
        ax_offset.text(
            center,
            1.02,
            f"#{int(row['candidate_rank'])}",
            transform=ax_offset.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#7a3e1d",
        )

    ax_offset.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
    ax_offset.plot(
        trace["toa_center_s"],
        trace["freq_offset_from_global_hz"],
        color="#8796a5",
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
    )
    for method in methods:
        mask = trace["selected_method"].astype(str) == method
        ax_offset.scatter(
            trace.loc[mask, "toa_center_s"],
            trace.loc[mask, "freq_offset_from_global_hz"],
            s=40,
            color=method_palette.get(method, "#555555"),
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=method,
        )
    for _, row in overlap.iterrows():
        ax_offset.annotate(
            str(int(row["candidate_rank"])),
            (
                float(row["local_toa_center_s"]),
                float(row["local_freq_offset_from_global_hz"]),
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#222222",
        )
    ax_offset.set_ylabel("Local offset (Hz)")
    ax_offset.set_title(title)
    ax_offset.grid(True, linestyle="--", alpha=0.25)
    ax_offset.legend(loc="upper right", ncol=max(1, min(4, len(methods))), frameon=True)

    ax_conf.plot(
        trace["toa_center_s"],
        trace["common_axial_confidence"],
        color="#355070",
        linewidth=1.3,
        alpha=0.85,
        zorder=2,
    )
    ax_conf.scatter(
        trace["toa_center_s"],
        trace["common_axial_confidence"],
        c=[method_palette.get(str(method), "#555555") for method in trace["selected_method"]],
        s=36,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax_conf.set_ylabel("Axial conf.")
    ax_conf.set_ylim(bottom=max(-0.02, np.nanmin(trace["common_axial_confidence"]) - 0.05), top=1.05)
    ax_conf.grid(True, linestyle="--", alpha=0.25)

    method_index = {method: idx for idx, method in enumerate(methods)}
    method_y = trace["selected_method"].astype(str).map(method_index).to_numpy(dtype=np.float64)
    ax_method.step(
        trace["toa_center_s"],
        method_y,
        where="mid",
        color="#444444",
        linewidth=1.1,
        alpha=0.7,
    )
    ax_method.scatter(
        trace["toa_center_s"],
        method_y,
        c=[method_palette.get(str(method), "#555555") for method in trace["selected_method"]],
        s=44,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    ax_method.set_yticks(list(method_index.values()))
    ax_method.set_yticklabels(list(method_index.keys()))
    ax_method.set_ylabel("Local winner")
    ax_method.set_xlabel("Experiment time (s)")
    ax_method.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _build_change_candidates(
    df_windows: pd.DataFrame,
    features: list[str],
    *,
    top_k: int,
    min_window_gap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_windows.empty:
        return pd.DataFrame(), pd.DataFrame()

    ordered = df_windows.sort_values(["toa_start_s", "window_index"]).reset_index(drop=True).copy()
    usable_features = []
    for feature in features:
        if feature not in ordered.columns:
            continue
        values = pd.to_numeric(ordered[feature], errors="coerce")
        if values.notna().sum() == 0 or values.nunique(dropna=True) < 2:
            continue
        usable_features.append(feature)

    if not usable_features or len(ordered) < 2:
        return pd.DataFrame(), pd.DataFrame()

    delta_columns: list[str] = []
    for feature in usable_features:
        values = pd.to_numeric(ordered[feature], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        z_values = (filled - float(filled.mean())) / std
        delta_col = f"{feature}_delta_z"
        ordered[delta_col] = z_values.diff().abs()
        delta_columns.append(delta_col)

    ordered["change_score"] = ordered[delta_columns].mean(axis=1, skipna=True)
    delta_matrix = ordered[delta_columns].to_numpy(dtype=np.float64, copy=True)
    finite_mask = np.isfinite(delta_matrix)
    safe_delta_matrix = np.where(finite_mask, delta_matrix, -np.inf)
    dominant_idx = np.argmax(safe_delta_matrix, axis=1)
    ordered["dominant_feature"] = [
        usable_features[idx] if finite_mask[row_idx].any() else ""
        for row_idx, idx in enumerate(dominant_idx)
    ]
    dominant_delta = safe_delta_matrix[np.arange(len(ordered)), dominant_idx].astype(np.float64)
    dominant_delta[~finite_mask.any(axis=1)] = np.nan
    ordered["dominant_delta_z"] = dominant_delta

    score_series = ordered.dropna(subset=["change_score"]).copy()
    if score_series.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_rows: list[pd.Series] = []
    taken_positions: list[int] = []
    for _, row in score_series.sort_values("change_score", ascending=False).iterrows():
        pos = int(row.name)
        if any(abs(pos - taken) <= min_window_gap for taken in taken_positions):
            continue
        selected_rows.append(row)
        taken_positions.append(pos)
        if len(selected_rows) >= top_k:
            break

    candidates = pd.DataFrame(selected_rows).reset_index(drop=True)
    if not candidates.empty:
        candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))

    return score_series, candidates


def run_descriptor_study(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    config_path = Path(config_path)

    material_config = cfg.get("material_config")
    run_material_pipeline = bool(cfg.get("run_material_pipeline", False))
    if material_config is not None:
        material_config = _resolve_path(config_path, material_config)
        if run_material_pipeline:
            run_material_state(material_config)

    input_cfg = cfg.get("input", {})
    event_csv_raw = input_cfg.get("event_csv")
    if event_csv_raw is not None:
        event_csv = _resolve_path(config_path, event_csv_raw)
    elif material_config is not None:
        event_csv = _material_output_event_csv(material_config)
    else:
        raise ValueError("Descriptor study requires either input.event_csv or material_config.")

    if not event_csv.exists():
        raise FileNotFoundError(f"Event CSV not found: {event_csv}")

    output_dir = _resolve_path(config_path, cfg.get("output_dir", "outputs/descriptor_study"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df_events = pd.read_csv(event_csv)
    tasks_cfg = cfg.get("tasks") or _default_tasks(df_events)
    if not tasks_cfg:
        raise ValueError(
            "No tasks configured and no default task could be inferred from the input event table."
        )

    window_cfg = cfg.get("windowing", {})
    label_columns = sorted(
        {
            task_cfg["label_column"]
            for task_cfg in tasks_cfg.values()
            if "label_column" in task_cfg
        }
    )
    group_columns = input_cfg.get("group_columns")
    if group_columns is None:
        group_columns = [
            col for col in ["source_file", "dataset_key"] if col in df_events.columns
        ]

    df_windows = build_feature_windows(
        df_events,
        window_events=int(window_cfg.get("window_events", 64)),
        step_events=int(window_cfg.get("step_events", 16)),
        min_valid_events=int(window_cfg.get("min_valid_events", 32)),
        max_valid_dt_s=float(window_cfg.get("max_valid_dt_s", 1.0)),
        fano_bin_count=int(window_cfg.get("fano_bin_count", 8)),
        group_columns=group_columns,
        label_columns=label_columns,
    )
    if df_windows.empty:
        raise ValueError(
            "Window feature matrix is empty. Relax the window settings or check event data."
        )

    descriptor_cfg = cfg.get("descriptors", {})
    search_features = descriptor_cfg.get("search_features", PRIMARY_DESCRIPTOR_BANK)
    reserve_features = descriptor_cfg.get("reserve_features", RESERVE_DESCRIPTOR_BANK)
    requested_features = list(dict.fromkeys(search_features + reserve_features))
    feature_quality_df = _feature_quality_table(df_windows, requested_features)
    feature_quality_df.to_csv(
        output_dir / "feature_quality.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligible_features = feature_quality_df[feature_quality_df["eligible"]]["feature"].tolist()
    search_features = [feature for feature in search_features if feature in eligible_features]
    reserve_features = [feature for feature in reserve_features if feature in eligible_features]
    feature_bank = list(dict.fromkeys(search_features + reserve_features))
    if not feature_bank:
        raise ValueError("No eligible descriptor features remained after quality screening.")

    redundancy_df = df_windows[feature_bank].corr(method="spearman")
    redundancy_df.to_csv(
        output_dir / "descriptor_redundancy_spearman.csv",
        encoding="utf-8-sig",
    )

    search_cfg = cfg.get("search", {})
    seed = int(search_cfg.get("random_seed", 42))
    n_splits = int(search_cfg.get("n_splits", 5))
    top_k_features = int(search_cfg.get("top_k_features", 8))
    max_combo_size = int(search_cfg.get("max_combo_size", 3))
    forward_max_features = int(search_cfg.get("forward_selection_max_features", 5))
    tolerance = float(search_cfg.get("score_tolerance", 0.01))
    redundancy_threshold = float(search_cfg.get("redundancy_threshold", 0.85))
    change_cfg = cfg.get("change_detection", {})
    change_enabled = bool(change_cfg.get("enabled", True))
    change_top_k = int(change_cfg.get("top_k", 8))
    change_min_gap = int(change_cfg.get("min_window_gap", 3))

    protocol = {
        "config_path": str(config_path.resolve()),
        "event_csv": str(event_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "windowing": {
            "window_events": int(window_cfg.get("window_events", 64)),
            "step_events": int(window_cfg.get("step_events", 16)),
            "min_valid_events": int(window_cfg.get("min_valid_events", 32)),
            "max_valid_dt_s": float(window_cfg.get("max_valid_dt_s", 1.0)),
            "fano_bin_count": int(window_cfg.get("fano_bin_count", 8)),
        },
        "search_features": search_features,
        "reserve_features": reserve_features,
        "excluded_features": feature_quality_df[
            ~feature_quality_df["eligible"]
        ].to_dict(orient="records"),
        "tasks": tasks_cfg,
    }
    with (output_dir / "study_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    task_reports: list[str] = ["# DeltaPD Descriptor Study", ""]
    blind_trace_df = pd.DataFrame()
    if material_config is not None:
        material_output_dir = _material_output_dir(material_config)
        manifest_path = material_output_dir / "run_manifest.json"
        blind_trace_path = material_output_dir / "blind_prpd_local_trace.csv"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            blind = dict(manifest.get("blind_prpd", {}))
            if blind:
                task_reports.extend(
                    [
                        "## Blind PRPD Calibration",
                        "",
                        f"- requested_method: {blind.get('requested_method', blind.get('method', ''))}",
                        f"- selected_method: {blind.get('selected_method', blind.get('method', ''))}",
                        f"- calibrated_freq_hz: {float(blind.get('calibrated_freq_hz', float('nan'))):.6f}",
                        f"- coherence: {float(blind.get('coherence', float('nan'))):.6f}",
                        f"- axial_entropy_score: {float(blind.get('axial_entropy_score', float('nan'))):.6f}",
                        f"- sharpness: {float(blind.get('sharpness', float('nan'))):.6f}",
                        f"- half_height_width_hz: {float(blind.get('half_height_width_hz', float('nan'))):.6f}",
                        f"- common_axial_confidence: {float(blind.get('common_axial_confidence', float('nan'))):.6f}",
                        f"- common_axial_peak_offset_hz: {float(blind.get('common_axial_peak_offset_hz', float('nan'))):.6f}",
                        f"- bootstrap_iterations: {int(blind.get('bootstrap_iterations', 0) or 0)}",
                        f"- bootstrap_freq_std_hz: {float(blind.get('bootstrap_freq_std_hz', float('nan'))):.6f}",
                        f"- bootstrap_ci_width_hz: {float(blind.get('bootstrap_ci_width_hz', float('nan'))):.6f}",
                        f"- bootstrap_method_agreement: {float(blind.get('bootstrap_method_agreement', float('nan'))):.6f}",
                        f"- local_window_count: {int(blind.get('local_window_count', 0) or 0)}",
                        f"- local_freq_std_hz: {float(blind.get('local_freq_std_hz', float('nan'))):.6f}",
                        f"- local_freq_span_hz: {float(blind.get('local_freq_span_hz', float('nan'))):.6f}",
                        f"- local_method_agreement: {float(blind.get('local_method_agreement', float('nan'))):.6f}",
                        f"- candidate_spread_hz: {float(blind.get('candidate_spread_hz', float('nan'))):.6f}",
                        f"- winner_margin: {float(blind.get('winner_margin', float('nan'))):.6f}",
                        "",
                    ]
                )
        if blind_trace_path.exists():
            blind_trace_df = pd.read_csv(blind_trace_path)
    recommendations: dict[str, Any] = {}
    outputs: dict[str, Any] = {
        "windows": df_windows,
        "redundancy": redundancy_df,
        "feature_quality": feature_quality_df,
    }

    for task_name, task_cfg in tasks_cfg.items():
        task_type = str(task_cfg.get("type", "multiclass"))
        task_df, label_column, positive_values, label_note = _ensure_task_labels(
            df_windows,
            task_name,
            task_cfg,
        )
        if label_column not in task_df.columns:
            raise ValueError(
                f"Task {task_name!r} could not resolve label column {label_column!r}."
            )
        if label_column not in df_windows.columns and label_column in task_df.columns:
            df_windows[label_column] = task_df[label_column].to_numpy()

        univariate_df = evaluate_univariate_descriptors(
            task_df,
            feature_bank,
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
        redundant_pairs = _find_redundant_features(
            redundancy_df,
            ranked_features,
            threshold=redundancy_threshold,
        )

        task_payload = {
            "configured_label_column": task_cfg["label_column"],
            "effective_label_column": label_column,
            "positive_values": positive_values,
            "label_note": label_note,
            "class_counts": {
                str(label): int(count)
                for label, count in task_df[label_column].value_counts(dropna=False).items()
            },
            "recommendation": recommendation,
            "redundant_pairs": redundant_pairs,
        }
        recommendations[task_name] = task_payload

        univariate_df.to_csv(
            output_dir / f"{task_name}_univariate.csv",
            index=False,
            encoding="utf-8-sig",
        )
        exhaustive_df.to_csv(
            output_dir / f"{task_name}_exhaustive.csv",
            index=False,
            encoding="utf-8-sig",
        )
        forward_df.to_csv(
            output_dir / f"{task_name}_forward_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

        outputs[f"{task_name}_univariate"] = univariate_df
        outputs[f"{task_name}_exhaustive"] = exhaustive_df
        outputs[f"{task_name}_forward"] = forward_df
        outputs[f"{task_name}_summary"] = task_payload
        task_reports.append(
            _format_task_report(
                task_name,
                label_note,
                recommendation,
                univariate_df,
                exhaustive_df,
                forward_df,
            )
        )
        task_reports.append("")

    df_windows.to_csv(output_dir / "descriptor_windows.csv", index=False, encoding="utf-8-sig")
    change_score_df = pd.DataFrame()
    change_candidates_df = pd.DataFrame()
    blind_transition_overlap_df = pd.DataFrame()
    blind_transition_plot_path: Path | None = None
    if change_enabled:
        change_features = _recommended_change_features(
            recommendations,
            fallback_features=search_features[:3],
        )
        change_score_df, change_candidates_df = _build_change_candidates(
            df_windows,
            change_features,
            top_k=change_top_k,
            min_window_gap=change_min_gap,
        )
        if not change_score_df.empty:
            change_score_df.to_csv(
                output_dir / "change_score_series.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if not change_candidates_df.empty:
            change_candidates_df.to_csv(
                output_dir / "change_candidates.csv",
                index=False,
                encoding="utf-8-sig",
            )
            task_reports.append("## Candidate Transition Windows")
            task_reports.append("")
            for _, row in change_candidates_df.head(8).iterrows():
                task_reports.append(
                    f"- rank {int(row['candidate_rank'])}: "
                    f"t={float(row['toa_start_s']):.6f}-{float(row['toa_end_s']):.6f} s, "
                    f"score={float(row['change_score']):.4f}, "
                    f"dominant_feature={row['dominant_feature']}"
                )
            task_reports.append("")
        if not change_candidates_df.empty and not blind_trace_df.empty:
            blind_transition_overlap_df = _build_blind_transition_overlap(
                change_candidates_df,
                blind_trace_df,
            )
            if not blind_transition_overlap_df.empty:
                blind_transition_overlap_df.to_csv(
                    output_dir / "blind_prpd_transition_overlap.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                task_reports.append("## Blind PRPD / Transition Overlap")
                task_reports.append("")
                for _, row in blind_transition_overlap_df.head(8).iterrows():
                    task_reports.append(
                        f"- rank {int(row['candidate_rank'])}: "
                        f"blind window={int(row['local_window_index'])} "
                        f"({row['match_mode']}), "
                        f"local_method={row['local_selected_method']}, "
                        f"local_freq={float(row['local_freq_hz']):.6f} Hz, "
                        f"offset={float(row['local_freq_offset_from_global_hz']):+.6f} Hz, "
                        f"confidence={float(row['local_common_axial_confidence']):.4f}"
                    )
                task_reports.append("")
                blind_transition_plot_path = _plot_blind_transition_map(
                    blind_trace_df,
                    change_candidates_df,
                    blind_transition_overlap_df,
                    out_png=output_dir / "blind_prpd_transition_map.png",
                    title="Blind PRPD local stability aligned with transition windows",
                )
    with (output_dir / "study_recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2)
    with (output_dir / "study_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(task_reports).strip() + "\n")

    report_cfg = cfg.get("report", {})
    pdf_path = None
    if bool(report_cfg.get("export_pdf", False)):
        material_output_dir = None
        if material_config is not None:
            material_output_dir = _material_output_dir(material_config)
        extra_images = []
        if blind_transition_plot_path is not None and blind_transition_plot_path.exists():
            extra_images.append(
                (blind_transition_plot_path, "Blind PRPD local stability vs transition windows")
            )
        pdf_path = build_descriptor_study_pdf(
            output_dir,
            material_output_dir=material_output_dir,
            title=str(cfg.get("study_name", "Descriptor Study Report")),
            pdf_filename=str(report_cfg.get("pdf_filename", "descriptor_study_report.pdf")),
            extra_images=extra_images or None,
        )

    outputs["recommendations"] = recommendations
    outputs["change_score_series"] = change_score_df
    outputs["change_candidates"] = change_candidates_df
    outputs["blind_transition_overlap"] = blind_transition_overlap_df
    outputs["blind_transition_plot_path"] = blind_transition_plot_path
    outputs["output_dir"] = output_dir
    outputs["pdf_path"] = pdf_path
    return outputs
