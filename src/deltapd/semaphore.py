from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SEMAPHORE_COMPONENTS = [
    ("max_abs_local_freq_offset_hz", "high"),
    ("local_freq_offset_std_hz", "high"),
    ("local_common_axial_confidence_std", "high"),
    ("transition_method_entropy", "high"),
    ("local_regime_transition_entropy", "high"),
    ("local_method_switch_rate", "high"),
    ("bocpd_max_change_prob", "high"),
    ("bocpd_surprise_score", "high"),
    ("hmm_high_state_share", "high"),
    ("hmm_state_switch_rate", "high"),
    ("hmm_state_entropy", "high"),
    ("semi_markov_high_state_share", "high"),
    ("semi_markov_state_switch_rate", "high"),
    ("semi_markov_state_entropy", "high"),
    ("mean_local_common_axial_confidence", "low"),
    ("local_regime_mean_run_length", "low"),
    ("bocpd_run_length_mean", "low"),
    ("hmm_state_mean_run_length", "low"),
    ("semi_markov_state_mean_run_length", "low"),
]
SHORT_SEQUENCE_SENSITIVE_COMPONENTS = {
    "transition_method_entropy",
    "local_regime_transition_entropy",
    "local_method_switch_rate",
    "bocpd_max_change_prob",
    "bocpd_surprise_score",
    "bocpd_run_length_mean",
    "hmm_high_state_share",
    "hmm_state_switch_rate",
    "hmm_state_entropy",
    "hmm_state_mean_run_length",
    "semi_markov_high_state_share",
    "semi_markov_state_switch_rate",
    "semi_markov_state_entropy",
    "semi_markov_state_mean_run_length",
}
GRAY_INGESTION_THRESHOLD = 0.55
GREEN_RISK_THRESHOLD = 0.36
RED_RISK_THRESHOLD = 0.55


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _transition_evidence_weight(df: pd.DataFrame) -> pd.Series:
    window_counts = _numeric_series(df, "n_transition_windows")
    if window_counts.isna().all():
        return pd.Series(1.0, index=df.index, dtype=float)
    evidence = ((window_counts - 2.0) / 4.0).clip(lower=0.0, upper=1.0)
    return evidence.fillna(0.0).astype(float)


def ingestion_audit_confidence(audit: dict[str, Any]) -> tuple[float, list[str]]:
    if not audit:
        return 0.0, ["no_ingestion_audit"]

    score = 1.0
    flags: list[str] = []
    fs_source = str(audit.get("fs_source", "")).strip().lower()
    sample_value = _safe_float(audit.get("final_sample_count"))
    if np.isnan(sample_value):
        sample_value = _safe_float(audit.get("sample_count"))
    numeric_value = _safe_float(audit.get("numeric_row_count"))
    sample_count = int(sample_value) if np.isfinite(sample_value) else 0
    numeric_rows = int(numeric_value) if np.isfinite(numeric_value) else 0
    signal_column = str(audit.get("signal_column_label", "")).strip()
    time_column = str(audit.get("time_column_label", "")).strip()
    loader_mode = str(audit.get("loader_mode", "")).strip().lower()

    if fs_source in {"", "default_fs"}:
        score -= 0.35
        flags.append("default_fs")
    if not signal_column:
        score -= 0.20
        flags.append("missing_signal_column")
    if sample_count < 64:
        score -= 0.25
        flags.append("short_trace")
    elif sample_count < 256:
        score -= 0.10
        flags.append("limited_trace")
    if numeric_rows and numeric_rows < 64:
        score -= 0.10
        flags.append("few_numeric_rows")
    if loader_mode == "generic_csv" and not time_column and fs_source != "metadata_sample_rate":
        score -= 0.15
        flags.append("implicit_time_axis")

    return float(np.clip(score, 0.0, 1.0)), flags


def build_semaphore_df(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not summary_rows:
        return pd.DataFrame()
    df = pd.DataFrame(summary_rows).copy()
    if "dataset_key" not in df.columns:
        return pd.DataFrame()
    for column, _ in SEMAPHORE_COMPONENTS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["ingestion_confidence"] = _numeric_series(df, "ingestion_confidence").fillna(0.0)
    evidence_weight = _transition_evidence_weight(df)
    df["semaphore_transition_evidence"] = evidence_weight
    component_scores: list[pd.Series] = []
    available_components: list[str] = []
    for column, direction in SEMAPHORE_COMPONENTS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty or int(valid.nunique()) <= 1:
            continue
        ranked = series.rank(method="average", pct=True)
        if direction == "low":
            ranked = 1.0 - ranked
        if column in SHORT_SEQUENCE_SENSITIVE_COMPONENTS:
            ranked = ranked * evidence_weight
        component_scores.append(ranked)
        available_components.append(column)
    if not component_scores:
        return pd.DataFrame()

    component_frame = pd.concat(component_scores, axis=1)
    component_frame.columns = available_components
    df["semaphore_component_count"] = component_frame.notna().sum(axis=1)
    df["semaphore_risk_score"] = component_frame.mean(axis=1, skipna=True)
    df["semaphore_confidence_score"] = (
        _numeric_series(df, "mean_local_common_axial_confidence").fillna(0.0) * 0.45
        + _numeric_series(df, "blind_local_method_agreement").fillna(0.0) * 0.25
        + _numeric_series(df, "ingestion_confidence").fillna(0.0) * 0.30
    )

    bands: list[str] = []
    colors: list[str] = []
    drivers: list[str] = []
    for idx, row in df.iterrows():
        ingestion_confidence = _safe_float(row.get("ingestion_confidence"))
        if not np.isfinite(ingestion_confidence) or ingestion_confidence < GRAY_INGESTION_THRESHOLD:
            bands.append("gray")
            colors.append("#a0a7b0")
        elif int(row.get("semaphore_component_count", 0) or 0) < 4:
            bands.append("gray")
            colors.append("#a0a7b0")
        else:
            risk = _safe_float(row.get("semaphore_risk_score"))
            if risk >= RED_RISK_THRESHOLD:
                bands.append("red")
                colors.append("#9b2226")
            elif risk >= GREEN_RISK_THRESHOLD:
                bands.append("yellow")
                colors.append("#c26d2d")
            else:
                bands.append("green")
                colors.append("#5d8f52")
        component_values = component_frame.iloc[idx].dropna().sort_values(ascending=False)
        top_components = component_values.index.tolist()[:3]
        if not np.isfinite(ingestion_confidence) or ingestion_confidence < GRAY_INGESTION_THRESHOLD:
            top_components = ["low_ingestion_confidence"] + top_components[:2]
        elif float(row.get("semaphore_transition_evidence", 1.0) or 0.0) < 0.5:
            top_components = top_components + ["short_transition_sequence"]
            top_components = top_components[:3]
        drivers.append(", ".join(top_components))
    df["semaphore_band"] = bands
    df["semaphore_color"] = colors
    df["semaphore_top_drivers"] = drivers
    return df
