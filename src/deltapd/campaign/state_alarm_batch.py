from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from deltapd.campaign.descriptor_study import run_descriptor_study

DEFAULT_DATASETS = [
    {"dataset_key": "P1", "folder": "Prueba 1 - Internas", "discharge_type": "internal", "variant": "benchmark"},
    {"dataset_key": "P2", "folder": "Prueba 2 - Superficiales", "discharge_type": "superficial", "variant": "benchmark"},
    {"dataset_key": "P3", "folder": "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas", "discharge_type": "multiple", "variant": "benchmark"},
    {"dataset_key": "G1", "folder": "Prueba 1 - Internas (Gemelas)", "discharge_type": "internal", "variant": "gemela"},
    {"dataset_key": "G2", "folder": "Prueba 2 - Superficiales (Gemelas)", "discharge_type": "superficial", "variant": "gemela"},
    {"dataset_key": "G3", "folder": "Prueba 3 - Ensayo de Fuentes Múltiples Simultáneas (Gemelas)", "discharge_type": "multiple", "variant": "gemela"},
]

DEFAULT_MATERIAL_DEFAULTS = {
    "antenna_name": "Vivaldi antipodal propuesta",
    "preprocess": {
        "preserve_amplitude": True,
        "centering": True,
        "normalization": "none",
        "wavelet_denoise": True,
    },
    "detection": {
        "k_sigma": 5.0,
        "refractory_ns": 20,
        "noise_window_ns": 40,
    },
    "analysis": {
        "export_sensitivity_report": True,
        "stage_aware": True,
        "stage_boundaries_s": "auto",
        "max_valid_dt_s": 1.0,
        "rolling_window_events": 50,
        "ewma_alpha": 0.2,
        "cusum_k": 0.5,
        "cusum_h": 5.0,
        "kalman_q": 1e-4,
        "kalman_r": 1.0,
        "bands_ghz": [[0.0, 0.5], [0.5, 1.5], [1.5, 2.4]],
        "blind_prpd": {
            "calibration_method": "auto",
            "n_harmonics": 4,
        },
    },
    "plots": {
        "show_raw_with_detections": True,
        "show_delta_t_series": True,
        "show_delta_t_hist": False,
        "show_rate_series": True,
        "show_rolling_stats": True,
        "show_ewma_cusum": True,
        "show_advanced_stats": True,
        "show_blind_prpd": True,
    },
}

DEFAULT_STUDY_DEFAULTS = {
    "windowing": {
        "window_events": 64,
        "step_events": 16,
        "min_valid_events": 32,
        "max_valid_dt_s": 1.0,
        "fano_bin_count": 8,
    },
    "descriptors": {
        "search_features": [
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
        ],
        "reserve_features": [
            "phase_inlier_ratio",
            "amplitude_balance_ratio",
            "mean_peak_v",
            "n_events",
        ],
    },
    "tasks": {
        "state": {
            "type": "multiclass",
            "label_column": "segment",
            "fallback_time_segments": 3,
        },
        "alarm": {
            "type": "binary",
            "label_column": "segment",
            "positive_values": [3],
            "fallback_time_segments": 3,
        },
    },
    "search": {
        "n_splits": 5,
        "random_seed": 42,
        "top_k_features": 8,
        "max_combo_size": 3,
        "forward_selection_max_features": 5,
        "score_tolerance": 0.01,
        "redundancy_threshold": 0.85,
    },
    "report": {
        "export_pdf": True,
    },
}


def _repo_root_from_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    return resolved.parents[1] if len(resolved.parents) >= 2 else resolved.parent


def _resolve_path(config_path: str | Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _repo_root_from_config(config_path) / candidate


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_features(features: list[str]) -> str:
    return ", ".join(features) if features else ""


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    xy = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(xy) < 3:
        return float("nan")
    return float(xy["x"].corr(xy["y"], method="spearman"))


def _case_material_config(
    *,
    base_dir: str,
    case: dict[str, Any],
    material_defaults: dict[str, Any],
    channel: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "campaign_name": f"{case['dataset_key']}_{channel}_Material_Study",
        "base_dir": base_dir,
        "output_dir": output_dir.as_posix(),
        "dataset": {
            "folder": case["folder"],
            "channel": channel,
            "antenna_name": material_defaults.get("antenna_name", "unknown"),
        },
        "preprocess": material_defaults.get("preprocess", {}),
        "detection": material_defaults.get("detection", {}),
        "analysis": material_defaults.get("analysis", {}),
        "plots": material_defaults.get("plots", {}),
    }


def _case_study_config(
    *,
    case: dict[str, Any],
    material_config_path: Path,
    study_defaults: dict[str, Any],
    channel: str,
    output_dir: Path,
) -> dict[str, Any]:
    report_cfg = dict(study_defaults.get("report", {}))
    if "pdf_filename" not in report_cfg:
        report_cfg["pdf_filename"] = f"{case['dataset_key'].lower()}_{channel.lower()}_state_alarm_report.pdf"
    return {
        "study_name": f"{case['dataset_key']}_{channel}_State_Alarm_Study",
        "material_config": material_config_path.as_posix(),
        "run_material_pipeline": True,
        "windowing": study_defaults.get("windowing", {}),
        "descriptors": study_defaults.get("descriptors", {}),
        "tasks": study_defaults.get("tasks", {}),
        "search": study_defaults.get("search", {}),
        "report": report_cfg,
        "output_dir": output_dir.as_posix(),
    }


def _feature_frequency(summary_rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in summary_rows:
        features = [part.strip() for part in str(row.get(field, "")).split(",") if part.strip()]
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _read_case_summary(case: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(outputs["output_dir"])
    recommendations = outputs["recommendations"]
    material_output_dir = output_dir.parent / "material"
    duration_s = float("nan")
    total_events = None
    blind_prpd: dict[str, Any] = {}
    manifest_path = material_output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_events = manifest.get("total_events")
        blind_prpd = dict(manifest.get("blind_prpd", {}))
    delta_csv = material_output_dir / "delta_t_series_master.csv"
    if delta_csv.exists():
        df_delta = pd.read_csv(delta_csv, usecols=["toa_s"])
        if not df_delta.empty:
            duration_s = float(df_delta["toa_s"].max())

    top_transition = {}
    change_candidates = outputs.get("change_candidates")
    if isinstance(change_candidates, pd.DataFrame) and not change_candidates.empty:
        top = change_candidates.iloc[0]
        top_transition = {
            "transition_start_s": _safe_float(top.get("toa_start_s")),
            "transition_end_s": _safe_float(top.get("toa_end_s")),
            "transition_feature": str(top.get("dominant_feature", "")),
        }

    state_rec = recommendations.get("state", {}).get("recommendation", {})
    alarm_rec = recommendations.get("alarm", {}).get("recommendation", {})
    return {
        "dataset_key": case["dataset_key"],
        "folder": case["folder"],
        "discharge_type": case.get("discharge_type", ""),
        "variant": case.get("variant", ""),
        "duration_s": duration_s,
        "total_events": total_events,
        "state_features": _format_features(state_rec.get("features", [])),
        "state_strategy": state_rec.get("strategy", ""),
        "state_primary_metric": state_rec.get("primary_metric", ""),
        "state_primary_score": _safe_float(state_rec.get("primary_score")),
        "state_balanced_accuracy": _safe_float(state_rec.get("balanced_accuracy")),
        "alarm_features": _format_features(alarm_rec.get("features", [])),
        "alarm_strategy": alarm_rec.get("strategy", ""),
        "alarm_primary_metric": alarm_rec.get("primary_metric", ""),
        "alarm_primary_score": _safe_float(alarm_rec.get("primary_score")),
        "alarm_balanced_accuracy": _safe_float(alarm_rec.get("balanced_accuracy")),
        "blind_requested_method": str(blind_prpd.get("requested_method", blind_prpd.get("method", ""))),
        "blind_selected_method": str(blind_prpd.get("selected_method", blind_prpd.get("method", ""))),
        "blind_freq_hz": _safe_float(blind_prpd.get("calibrated_freq_hz")),
        "blind_coherence": _safe_float(blind_prpd.get("coherence")),
        "blind_axial_entropy": _safe_float(blind_prpd.get("axial_entropy_score")),
        "blind_sharpness": _safe_float(blind_prpd.get("sharpness")),
        "blind_half_height_width_hz": _safe_float(blind_prpd.get("half_height_width_hz")),
        "blind_common_axial_confidence": _safe_float(blind_prpd.get("common_axial_confidence")),
        "blind_common_axial_peak_offset_hz": _safe_float(blind_prpd.get("common_axial_peak_offset_hz")),
        "blind_bootstrap_iterations": int(blind_prpd.get("bootstrap_iterations", 0) or 0),
        "blind_bootstrap_freq_std_hz": _safe_float(blind_prpd.get("bootstrap_freq_std_hz")),
        "blind_bootstrap_ci_width_hz": _safe_float(blind_prpd.get("bootstrap_ci_width_hz")),
        "blind_bootstrap_method_agreement": _safe_float(blind_prpd.get("bootstrap_method_agreement")),
        "blind_local_window_count": int(blind_prpd.get("local_window_count", 0) or 0),
        "blind_local_freq_std_hz": _safe_float(blind_prpd.get("local_freq_std_hz")),
        "blind_local_freq_span_hz": _safe_float(blind_prpd.get("local_freq_span_hz")),
        "blind_local_method_agreement": _safe_float(blind_prpd.get("local_method_agreement")),
        "blind_candidate_spread_hz": _safe_float(blind_prpd.get("candidate_spread_hz")),
        "blind_winner_margin": _safe_float(blind_prpd.get("winner_margin")),
        "pdf_path": str(outputs.get("pdf_path") or ""),
        "study_output_dir": str(output_dir),
        **top_transition,
    }


def _write_batch_summary_markdown(
    output_root: Path,
    *,
    channel: str,
    summary_rows: list[dict[str, Any]],
    state_feature_counts: dict[str, int],
    alarm_feature_counts: dict[str, int],
    transition_case_summary_df: pd.DataFrame | None = None,
    transition_method_summary_df: pd.DataFrame | None = None,
) -> Path:
    lines = [
        f"# {channel} state/alarm batch study",
        "",
        "## 1. What this batch does",
        "",
        f"- Runs a separate `state` and `alarm` study inside each individual {channel} acquisition.",
        "- Keeps the within-test objective separate from the cross-test discharge-type study.",
        "- Exports per-case CSVs, plots and PDFs plus this master summary.",
        "",
        "## 2. Recurrent descriptors across cases",
        "",
        f"- State feature frequency: {', '.join(f'{key}={value}' for key, value in state_feature_counts.items()) or 'none'}.",
        f"- Alarm feature frequency: {', '.join(f'{key}={value}' for key, value in alarm_feature_counts.items()) or 'none'}.",
        "",
        "## 3. Blind PRPD calibration snapshot",
        "",
    ]
    selected_methods = {}
    blind_freqs = []
    for row in summary_rows:
        method = str(row.get("blind_selected_method", "")).strip()
        if method:
            selected_methods[method] = selected_methods.get(method, 0) + 1
        blind_freq = _safe_float(row.get("blind_freq_hz"))
        if pd.notna(blind_freq):
            blind_freqs.append(blind_freq)
    lines.extend(
        [
            f"- Selected blind methods: {', '.join(f'{key}={value}' for key, value in sorted(selected_methods.items())) or 'none'}.",
            (
                f"- Blind frequency range: {min(blind_freqs):.6f} to {max(blind_freqs):.6f} Hz."
                if blind_freqs
                else "- Blind frequency range: none."
            ),
            "",
            "## 4. Per-case summary",
            "",
            "| Dataset | Type | Variant | Duration (s) | Events | Blind PRPD | Best state subset | State score | Best alarm subset | Alarm score | Top transition |",
            "| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in summary_rows:
        transition_text = ""
        if pd.notna(row.get("transition_start_s")) and pd.notna(row.get("transition_end_s")):
            transition_text = (
                f"{row['transition_start_s']:.3f}-{row['transition_end_s']:.3f} s "
                f"({row.get('transition_feature', '')})"
            )
        blind_text = ""
        blind_method = str(row.get("blind_selected_method", "")).strip()
        blind_freq = _safe_float(row.get("blind_freq_hz"))
        blind_confidence = _safe_float(row.get("blind_common_axial_confidence"))
        blind_offset = _safe_float(row.get("blind_common_axial_peak_offset_hz"))
        blind_boot_std = _safe_float(row.get("blind_bootstrap_freq_std_hz"))
        blind_boot_agreement = _safe_float(row.get("blind_bootstrap_method_agreement"))
        blind_local_std = _safe_float(row.get("blind_local_freq_std_hz"))
        blind_local_agreement = _safe_float(row.get("blind_local_method_agreement"))
        if blind_method:
            blind_parts = [blind_method]
            if pd.notna(blind_freq):
                blind_parts.append(f"{blind_freq:.4f} Hz")
            if pd.notna(blind_confidence):
                blind_parts.append(f"conf={blind_confidence:.3f}")
            if pd.notna(blind_offset):
                blind_parts.append(f"off={blind_offset:.4f} Hz")
            if pd.notna(blind_boot_std):
                blind_parts.append(f"boot_sd={blind_boot_std:.4f}")
            if pd.notna(blind_boot_agreement):
                blind_parts.append(f"boot_agree={blind_boot_agreement:.2f}")
            if pd.notna(blind_local_std):
                blind_parts.append(f"local_sd={blind_local_std:.4f}")
            if pd.notna(blind_local_agreement):
                blind_parts.append(f"local_agree={blind_local_agreement:.2f}")
            blind_text = ", ".join(blind_parts)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset_key", "")),
                    str(row.get("discharge_type", "")),
                    str(row.get("variant", "")),
                    f"{_safe_float(row.get('duration_s')):.3f}",
                    str(row.get("total_events", "")),
                    blind_text,
                    str(row.get("state_features", "")),
                    f"{_safe_float(row.get('state_primary_score')):.4f}",
                    str(row.get("alarm_features", "")),
                    f"{_safe_float(row.get('alarm_primary_score')):.4f}",
                    transition_text,
                ]
            )
            + " |"
        )
    if transition_case_summary_df is not None and not transition_case_summary_df.empty:
        lines.extend(
            [
                "",
                "## 5. Transition overlap summary",
                "",
                "Counts below use distinct matched blind-PRPD local windows. `Ranked candidates` is shown separately because nearby top candidates can collapse onto the same local regime.",
                "",
                "| Dataset | Local windows | Ranked candidates | Unique local methods | Dominant local method | Max abs offset (Hz) | Mean abs offset (Hz) | Mean local conf. | State score | Alarm score |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        ordered = transition_case_summary_df.sort_values("dataset_key").reset_index(drop=True)
        for _, row in ordered.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("dataset_key", "")),
                        str(int(row.get("n_transition_windows", 0) or 0)),
                        str(int(row.get("n_ranked_transition_candidates", 0) or 0)),
                        str(int(row.get("n_unique_local_methods", 0) or 0)),
                        str(row.get("dominant_local_method", "")),
                        f"{_safe_float(row.get('max_abs_local_freq_offset_hz')):.6f}",
                        f"{_safe_float(row.get('mean_abs_local_freq_offset_hz')):.6f}",
                        f"{_safe_float(row.get('mean_local_common_axial_confidence')):.4f}",
                        f"{_safe_float(row.get('state_primary_score')):.4f}",
                        f"{_safe_float(row.get('alarm_primary_score')):.4f}",
                    ]
                )
                + " |"
            )
        if transition_method_summary_df is not None and not transition_method_summary_df.empty:
            totals = (
                transition_method_summary_df.groupby("local_selected_method", dropna=False)["transition_count"]
                .sum()
                .sort_values(ascending=False)
            )
            lines.extend(
                [
                    "",
                    f"- Transition-local method totals (distinct local windows): {', '.join(f'{key}={int(value)}' for key, value in totals.items())}.",
                ]
            )
        duplicate_counts = ordered[ordered["n_duplicate_candidate_matches"] > 0][
            ["dataset_key", "n_duplicate_candidate_matches"]
        ]
        if not duplicate_counts.empty:
            lines.append(
                "- Duplicate candidate matches collapsed during summary: "
                + ", ".join(
                    f"{row['dataset_key']}={int(row['n_duplicate_candidate_matches'])}"
                    for _, row in duplicate_counts.iterrows()
                )
                + "."
            )
        offset_df = ordered.copy()
        rho_state = _safe_spearman(offset_df["max_abs_local_freq_offset_hz"], offset_df["state_primary_score"])
        rho_alarm = _safe_spearman(offset_df["max_abs_local_freq_offset_hz"], offset_df["alarm_primary_score"])
        rho_mix_alarm = _safe_spearman(offset_df["n_unique_local_methods"], offset_df["alarm_primary_score"])
        lines.extend(
            [
                f"- Exploratory Spearman rho: max |offset| vs state score = {rho_state:.3f}.",
                f"- Exploratory Spearman rho: max |offset| vs alarm score = {rho_alarm:.3f}.",
                f"- Exploratory Spearman rho: local-method diversity vs alarm score = {rho_mix_alarm:.3f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## 6. Reading rule",
            "",
            "- `state` here means internal temporal regime inside one test.",
            "- `alarm` here means the latest segment of the same test against the earlier segments.",
            "- `Blind PRPD` summarizes the phase reconstruction quality feeding the descriptors.",
            "- Transition-family counts in this summary are deduplicated by matched local blind-PRPD window.",
            "- These outputs should feed the future visual environment; they do not replace the comparative `type` study.",
            "",
        ]
    )
    summary_path = output_root / "state_alarm_batch_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _build_transition_overlap_master(
    summary_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        study_output_dir = Path(str(row.get("study_output_dir", "")))
        overlap_path = study_output_dir / "blind_prpd_transition_overlap.csv"
        if not overlap_path.exists():
            continue
        overlap_df = pd.read_csv(overlap_path)
        if overlap_df.empty:
            continue
        for record in overlap_df.to_dict("records"):
            record.update(
                {
                    "dataset_key": row.get("dataset_key", ""),
                    "discharge_type": row.get("discharge_type", ""),
                    "variant": row.get("variant", ""),
                    "state_primary_score": _safe_float(row.get("state_primary_score")),
                    "alarm_primary_score": _safe_float(row.get("alarm_primary_score")),
                    "blind_selected_method_global": row.get("blind_selected_method", ""),
                    "blind_local_freq_std_hz": _safe_float(row.get("blind_local_freq_std_hz")),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _select_primary_transition_matches(overlap_df: pd.DataFrame) -> pd.DataFrame:
    if overlap_df.empty:
        return overlap_df.copy()

    work = overlap_df.copy()
    if "local_window_index" not in work.columns:
        return work.reset_index(drop=True)

    work["local_window_index"] = pd.to_numeric(work["local_window_index"], errors="coerce")
    if not work["local_window_index"].notna().any():
        return work.reset_index(drop=True)

    if "candidate_rank" in work.columns:
        work["candidate_rank"] = pd.to_numeric(work["candidate_rank"], errors="coerce")

    if "is_primary_local_match" in work.columns:
        primary_mask = (
            work["is_primary_local_match"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes"})
        )
        if primary_mask.any():
            return work.loc[primary_mask].reset_index(drop=True)

    sort_cols = ["local_window_index"]
    ascending = [True]
    dedupe_subset = ["local_window_index"]
    if "dataset_key" in work.columns:
        sort_cols.insert(0, "dataset_key")
        ascending.insert(0, True)
        dedupe_subset.insert(0, "dataset_key")
    if "candidate_rank" in work.columns:
        sort_cols.append("candidate_rank")
        ascending.append(True)
    return (
        work.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(subset=dedupe_subset, keep="first")
        .reset_index(drop=True)
    )


def _dominant_transition_method(method_counts: dict[str, int]) -> str:
    valid_counts = {
        str(method).strip(): int(count)
        for method, count in method_counts.items()
        if str(method).strip()
    }
    if not valid_counts:
        return ""
    max_count = max(valid_counts.values())
    winners = sorted(
        method for method, count in valid_counts.items() if int(count) == int(max_count)
    )
    return winners[0] if len(winners) == 1 else f"tie: {', '.join(winners)}"


def _summarize_transition_overlap(
    overlap_master_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if overlap_master_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    overlap = overlap_master_df.copy()
    overlap["abs_local_freq_offset_hz"] = pd.to_numeric(
        overlap["local_freq_offset_from_global_hz"],
        errors="coerce",
    ).abs()
    primary_overlap = _select_primary_transition_matches(overlap)
    transition_count_col = (
        "local_window_index" if "local_window_index" in primary_overlap.columns else "candidate_rank"
    )
    method_candidate_counts_df = (
        overlap.groupby(
            ["dataset_key", "discharge_type", "variant", "local_selected_method"],
            dropna=False,
        )
        .agg(candidate_match_count=("candidate_rank", "count"))
        .reset_index()
    )
    method_summary_df = (
        primary_overlap.groupby(
            ["dataset_key", "discharge_type", "variant", "local_selected_method"],
            dropna=False,
        )
        .agg(
            transition_count=(transition_count_col, "count"),
            mean_abs_local_freq_offset_hz=("abs_local_freq_offset_hz", "mean"),
            max_abs_local_freq_offset_hz=("abs_local_freq_offset_hz", "max"),
            mean_local_common_axial_confidence=("local_common_axial_confidence", "mean"),
        )
        .reset_index()
    )
    if not method_summary_df.empty and not method_candidate_counts_df.empty:
        method_summary_df = method_summary_df.merge(
            method_candidate_counts_df,
            on=["dataset_key", "discharge_type", "variant", "local_selected_method"],
            how="left",
        )
    elif method_summary_df.empty:
        method_summary_df = method_candidate_counts_df.copy()

    case_rows: list[dict[str, Any]] = []
    known_methods = [
        "coherence",
        "harmonic_power",
        "epoch_folding",
        "gregory_loredo",
        "phase_distance_correlation",
    ]
    for dataset_key, df_case_all in overlap.groupby("dataset_key", dropna=False):
        df_case = primary_overlap[primary_overlap["dataset_key"] == dataset_key].copy()
        if df_case.empty:
            df_case = df_case_all.copy()

        method_counts = (
            df_case["local_selected_method"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )
        candidate_method_counts = (
            df_case_all["local_selected_method"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )
        dominant_method = _dominant_transition_method(method_counts)
        row = {
            "dataset_key": dataset_key,
            "discharge_type": str(df_case["discharge_type"].iloc[0]),
            "variant": str(df_case["variant"].iloc[0]),
            "n_ranked_transition_candidates": int(len(df_case_all)),
            "n_transition_windows": int(len(df_case)),
            "n_duplicate_candidate_matches": int(len(df_case_all) - len(df_case)),
            "n_unique_local_methods": int(len(method_counts)),
            "dominant_local_method": dominant_method,
            "max_abs_local_freq_offset_hz": float(df_case["abs_local_freq_offset_hz"].max()),
            "mean_abs_local_freq_offset_hz": float(df_case["abs_local_freq_offset_hz"].mean()),
            "mean_local_common_axial_confidence": float(
                pd.to_numeric(df_case["local_common_axial_confidence"], errors="coerce").mean()
            ),
        }
        for method in known_methods:
            row[f"transition_count_{method}"] = int(method_counts.get(method, 0))
            row[f"candidate_match_count_{method}"] = int(candidate_method_counts.get(method, 0))
        case_rows.append(row)

    case_summary_df = pd.DataFrame(case_rows)
    if not summary_df.empty and not case_summary_df.empty:
        merge_cols = [
            "dataset_key",
            "state_primary_score",
            "alarm_primary_score",
            "blind_selected_method",
            "blind_local_freq_std_hz",
            "blind_local_method_agreement",
        ]
        available_cols = [col for col in merge_cols if col in summary_df.columns]
        case_summary_df = case_summary_df.merge(
            summary_df[available_cols],
            on="dataset_key",
            how="left",
        )
    return case_summary_df.sort_values("dataset_key").reset_index(drop=True), method_summary_df


def _plot_transition_method_mix(case_summary_df: pd.DataFrame, *, out_png: Path) -> Path | None:
    if case_summary_df.empty:
        return None
    method_cols = [col for col in case_summary_df.columns if col.startswith("transition_count_")]
    if not method_cols:
        return None
    method_labels = [col.replace("transition_count_", "") for col in method_cols]
    palette = {
        "coherence": "#2d6a8a",
        "harmonic_power": "#c26d2d",
        "epoch_folding": "#6d597a",
        "gregory_loredo": "#5d8f52",
        "phase_distance_correlation": "#9b2226",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(case_summary_df))
    bottom = np.zeros(len(case_summary_df), dtype=np.float64)
    for method_label, col in zip(method_labels, method_cols):
        values = pd.to_numeric(case_summary_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.72,
            label=method_label,
            color=palette.get(method_label, "#777777"),
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(case_summary_df["dataset_key"].tolist())
    ax.set_ylabel("Distinct local windows")
    ax.set_xlabel("Dataset")
    ax.set_title("Transition-local method mix by dataset (deduplicated)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper center", ncol=max(1, min(5, len(method_cols))))
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_transition_offset_vs_scores(case_summary_df: pd.DataFrame, *, out_png: Path) -> Path | None:
    if case_summary_df.empty:
        return None
    df = case_summary_df.copy()
    color_map = {
        "internal": "#2d6a8a",
        "superficial": "#c26d2d",
        "multiple": "#6d597a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    for ax, score_col, title in [
        (axes[0], "state_primary_score", "State score vs max local |offset|"),
        (axes[1], "alarm_primary_score", "Alarm score vs max local |offset|"),
    ]:
        for _, row in df.iterrows():
            x = _safe_float(row.get("max_abs_local_freq_offset_hz"))
            y = _safe_float(row.get(score_col))
            if pd.isna(x) or pd.isna(y):
                continue
            color = color_map.get(str(row.get("discharge_type", "")), "#555555")
            ax.scatter(
                x,
                y,
                s=90,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.annotate(
                str(row.get("dataset_key", "")),
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_ylim(0.0, 1.02)
    axes[0].set_xlabel("Max local |offset| (Hz)")
    axes[1].set_xlabel("Max local |offset| (Hz)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_state_alarm_batch(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    channel = str(cfg.get("channel", "CH3"))
    output_root = _resolve_path(config_path, cfg.get("output_root", f"outputs/state_alarm_{channel.lower()}"))
    output_root.mkdir(parents=True, exist_ok=True)
    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    base_dir = str(cfg.get("base_dir", "E:/Carpeta definitiva de Tesis/programas"))
    material_defaults = _merge_dict(DEFAULT_MATERIAL_DEFAULTS, cfg.get("material_defaults", {}))
    study_defaults = _merge_dict(DEFAULT_STUDY_DEFAULTS, cfg.get("study_defaults", {}))
    cases = list(cfg.get("datasets", DEFAULT_DATASETS))

    summary_rows: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []

    for case in cases:
        dataset_key = str(case["dataset_key"])
        case_root = output_root / dataset_key
        material_output_dir = case_root / "material"
        study_output_dir = case_root / "study"
        case_root.mkdir(parents=True, exist_ok=True)

        material_cfg = _case_material_config(
            base_dir=base_dir,
            case=case,
            material_defaults=material_defaults,
            channel=channel,
            output_dir=material_output_dir,
        )
        material_cfg_path = generated_dir / f"{dataset_key.lower()}_material.yaml"
        material_cfg_path.write_text(
            yaml.safe_dump(material_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

        study_cfg = _case_study_config(
            case=case,
            material_config_path=material_cfg_path,
            study_defaults=study_defaults,
            channel=channel,
            output_dir=study_output_dir,
        )
        study_cfg_path = generated_dir / f"{dataset_key.lower()}_study.yaml"
        study_cfg_path.write_text(
            yaml.safe_dump(study_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

        outputs = run_descriptor_study(study_cfg_path)
        case_summary = _read_case_summary(case, outputs)
        summary_rows.append(case_summary)
        manifest_cases.append(
            {
                "dataset_key": dataset_key,
                "folder": case["folder"],
                "material_config": str(material_cfg_path),
                "study_config": str(study_cfg_path),
                "material_output_dir": str(material_output_dir),
                "study_output_dir": str(study_output_dir),
                "pdf_path": case_summary.get("pdf_path", ""),
            }
        )

    summary_rows.sort(key=lambda row: row["dataset_key"])
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_root / "state_alarm_batch_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    overlap_master_df = _build_transition_overlap_master(summary_rows)
    overlap_master_csv_path = output_root / "transition_overlap_master.csv"
    if not overlap_master_df.empty:
        overlap_master_df.to_csv(overlap_master_csv_path, index=False, encoding="utf-8-sig")
    case_transition_summary_df, method_transition_summary_df = _summarize_transition_overlap(
        overlap_master_df,
        summary_df,
    )
    case_transition_csv_path = output_root / "transition_overlap_case_summary.csv"
    method_transition_csv_path = output_root / "transition_overlap_method_summary.csv"
    if not case_transition_summary_df.empty:
        case_transition_summary_df.to_csv(case_transition_csv_path, index=False, encoding="utf-8-sig")
    if not method_transition_summary_df.empty:
        method_transition_summary_df.to_csv(method_transition_csv_path, index=False, encoding="utf-8-sig")

    transition_method_mix_png = _plot_transition_method_mix(
        case_transition_summary_df,
        out_png=output_root / "transition_method_mix.png",
    )
    transition_offset_score_png = _plot_transition_offset_vs_scores(
        case_transition_summary_df,
        out_png=output_root / "transition_offset_vs_scores.png",
    )

    state_feature_counts = _feature_frequency(summary_rows, "state_features")
    alarm_feature_counts = _feature_frequency(summary_rows, "alarm_features")
    summary_md_path = _write_batch_summary_markdown(
        output_root,
        channel=channel,
        summary_rows=summary_rows,
        state_feature_counts=state_feature_counts,
        alarm_feature_counts=alarm_feature_counts,
        transition_case_summary_df=case_transition_summary_df,
        transition_method_summary_df=method_transition_summary_df,
    )

    manifest = {
        "base_dir": base_dir,
        "channel": channel,
        "output_root": str(output_root),
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
        "state_feature_counts": state_feature_counts,
        "alarm_feature_counts": alarm_feature_counts,
        "transition_overlap_master_csv": str(overlap_master_csv_path) if overlap_master_df is not None and not overlap_master_df.empty else "",
        "transition_overlap_case_summary_csv": str(case_transition_csv_path) if not case_transition_summary_df.empty else "",
        "transition_overlap_method_summary_csv": str(method_transition_csv_path) if not method_transition_summary_df.empty else "",
        "transition_method_mix_png": str(transition_method_mix_png) if transition_method_mix_png is not None else "",
        "transition_offset_vs_scores_png": str(transition_offset_score_png) if transition_offset_score_png is not None else "",
        "cases": manifest_cases,
    }
    manifest_path = output_root / "state_alarm_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "summary_df": summary_df,
        "summary_csv": summary_csv_path,
        "summary_md": summary_md_path,
        "manifest_path": manifest_path,
        "output_root": output_root,
    }
