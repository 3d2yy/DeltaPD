from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import yaml
from matplotlib.lines import Line2D

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
            "search_width_hz": 0.25,
            "coarse_steps": 4001,
            "refine_half_width_hz": 0.02,
            "max_events": 20000,
            "robust_refine": True,
            "bootstrap_iterations": 6,
            "bootstrap_sample_fraction": 0.75,
            "bootstrap_seed": 42,
            "local_window_size_events": 256,
            "local_window_step_events": 128,
            "local_min_events_per_window": 128,
            "local_min_window_count": 3,
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

TRANSITION_METHOD_ORDER = [
    "coherence",
    "harmonic_power",
    "epoch_folding",
    "h_test",
    "pdm",
    "gregory_loredo",
    "phase_distance_correlation",
]


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
    dataset_cfg = {
        "folder": case["folder"],
        "channel": channel,
        "antenna_name": material_defaults.get("antenna_name", "unknown"),
    }
    if case.get("file_path"):
        dataset_cfg["file_path"] = str(case["file_path"])
    return {
        "campaign_name": f"{case['dataset_key']}_{channel}_Material_Study",
        "base_dir": base_dir,
        "output_dir": output_dir.as_posix(),
        "dataset": dataset_cfg,
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
                "| Dataset | Local windows | Ranked candidates | Unique local methods | Method entropy | Switch rate | Dominant local method | Max abs offset (Hz) | Mean abs offset (Hz) | Mean local conf. | State score | Alarm score |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
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
                        f"{_safe_float(row.get('transition_method_entropy')):.4f}",
                        f"{_safe_float(row.get('local_method_switch_rate')):.4f}",
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
        entropy_top = ordered.sort_values("transition_method_entropy", ascending=False)[
            ["dataset_key", "transition_method_entropy"]
        ].head(3)
        switch_top = ordered.sort_values("local_method_switch_rate", ascending=False)[
            ["dataset_key", "local_method_switch_rate"]
        ].head(3)
        lines.extend(
            [
                "- Highest transition-method entropy: "
                + ", ".join(
                    f"{row['dataset_key']}={_safe_float(row['transition_method_entropy']):.4f}"
                    for _, row in entropy_top.iterrows()
                )
                + ".",
                "- Highest local method switch rate: "
                + ", ".join(
                    f"{row['dataset_key']}={_safe_float(row['local_method_switch_rate']):.4f}"
                    for _, row in switch_top.iterrows()
                )
                + ".",
            ]
        )
        if {"tfa_wavelet_entropy_mean", "tfa_wavelet_dominant_band_unique_count"}.issubset(ordered.columns):
            tfa_entropy_top = ordered.sort_values("tfa_wavelet_entropy_mean", ascending=False)[
                ["dataset_key", "tfa_wavelet_entropy_mean"]
            ].head(3)
            tfa_band_top = ordered.sort_values("tfa_wavelet_dominant_band_unique_count", ascending=False)[
                ["dataset_key", "tfa_wavelet_dominant_band_unique_count"]
            ].head(3)
            lines.extend(
                [
                    "- Highest local wavelet entropy mean: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['tfa_wavelet_entropy_mean']):.4f}"
                        for _, row in tfa_entropy_top.iterrows()
                    )
                    + ".",
                    "- Highest local wavelet-band diversity: "
                    + ", ".join(
                        f"{row['dataset_key']}={int(row['tfa_wavelet_dominant_band_unique_count'])}"
                        for _, row in tfa_band_top.iterrows()
                    )
                    + ".",
                ]
            )
        if {"tfa_wavelet_detail_entropy_mean", "tfa_wavelet_detail_dominant_band_unique_count"}.issubset(ordered.columns):
            tfa_detail_entropy_top = ordered.sort_values("tfa_wavelet_detail_entropy_mean", ascending=False)[
                ["dataset_key", "tfa_wavelet_detail_entropy_mean"]
            ].head(3)
            tfa_detail_band_top = ordered.sort_values("tfa_wavelet_detail_dominant_band_unique_count", ascending=False)[
                ["dataset_key", "tfa_wavelet_detail_dominant_band_unique_count"]
            ].head(3)
            lines.extend(
                [
                    "- Highest local wavelet detail entropy mean: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['tfa_wavelet_detail_entropy_mean']):.4f}"
                        for _, row in tfa_detail_entropy_top.iterrows()
                    )
                    + ".",
                    "- Highest local wavelet detail-band diversity: "
                    + ", ".join(
                        f"{row['dataset_key']}={int(row['tfa_wavelet_detail_dominant_band_unique_count'])}"
                        for _, row in tfa_detail_band_top.iterrows()
                    )
                    + ".",
                ]
            )
        if {
            "local_regime_transition_entropy",
            "local_regime_mean_run_length",
            "local_offset_sign_switch_rate",
        }.issubset(ordered.columns):
            regime_entropy_top = ordered.sort_values("local_regime_transition_entropy", ascending=False)[
                ["dataset_key", "local_regime_transition_entropy"]
            ].head(3)
            run_length_top = ordered.sort_values("local_regime_mean_run_length", ascending=False)[
                ["dataset_key", "local_regime_mean_run_length"]
            ].head(3)
            sign_switch_top = ordered.sort_values("local_offset_sign_switch_rate", ascending=False)[
                ["dataset_key", "local_offset_sign_switch_rate"]
            ].head(3)
            lines.extend(
                [
                    "- Highest local regime-transition entropy: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['local_regime_transition_entropy']):.4f}"
                        for _, row in regime_entropy_top.iterrows()
                    )
                    + ".",
                    "- Longest local mean regime run length: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['local_regime_mean_run_length']):.4f}"
                        for _, row in run_length_top.iterrows()
                    )
                    + ".",
                    "- Highest local offset-sign switch rate: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['local_offset_sign_switch_rate']):.4f}"
                        for _, row in sign_switch_top.iterrows()
                    )
                    + ".",
                ]
            )
        if {
            "bocpd_max_change_prob",
            "bocpd_run_length_mean",
            "bocpd_surprise_score",
        }.issubset(ordered.columns):
            bocpd_change_top = ordered.sort_values("bocpd_max_change_prob", ascending=False)[
                ["dataset_key", "bocpd_max_change_prob"]
            ].head(3)
            bocpd_run_top = ordered.sort_values("bocpd_run_length_mean", ascending=True)[
                ["dataset_key", "bocpd_run_length_mean"]
            ].head(3)
            bocpd_surprise_top = ordered.sort_values("bocpd_surprise_score", ascending=False)[
                ["dataset_key", "bocpd_surprise_score"]
            ].head(3)
            lines.extend(
                [
                    "- Highest BOCPD max change probability: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['bocpd_max_change_prob']):.4f}"
                        for _, row in bocpd_change_top.iterrows()
                    )
                    + ".",
                    "- Shortest BOCPD mean expected run length: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['bocpd_run_length_mean']):.4f}"
                        for _, row in bocpd_run_top.iterrows()
                    )
                    + ".",
                    "- Highest BOCPD surprise score: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['bocpd_surprise_score']):.4f}"
                        for _, row in bocpd_surprise_top.iterrows()
                    )
                    + ".",
                ]
            )
        if {
            "hmm_high_state_share",
            "hmm_state_switch_rate",
            "hmm_state_mean_run_length",
        }.issubset(ordered.columns):
            hmm_share_top = ordered.sort_values("hmm_high_state_share", ascending=False)[
                ["dataset_key", "hmm_high_state_share"]
            ].head(3)
            hmm_switch_top = ordered.sort_values("hmm_state_switch_rate", ascending=False)[
                ["dataset_key", "hmm_state_switch_rate"]
            ].head(3)
            hmm_run_top = ordered.sort_values("hmm_state_mean_run_length", ascending=True)[
                ["dataset_key", "hmm_state_mean_run_length"]
            ].head(3)
            lines.extend(
                [
                    "- Highest HMM high-state share: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['hmm_high_state_share']):.4f}"
                        for _, row in hmm_share_top.iterrows()
                    )
                    + ".",
                    "- Highest HMM switch rate: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['hmm_state_switch_rate']):.4f}"
                        for _, row in hmm_switch_top.iterrows()
                    )
                    + ".",
                    "- Shortest HMM mean run length: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['hmm_state_mean_run_length']):.4f}"
                        for _, row in hmm_run_top.iterrows()
                    )
                    + ".",
                ]
            )
        if {
            "semi_markov_high_state_share",
            "semi_markov_state_switch_rate",
            "semi_markov_state_mean_run_length",
        }.issubset(ordered.columns):
            semi_markov_share_top = ordered.sort_values("semi_markov_high_state_share", ascending=False)[
                ["dataset_key", "semi_markov_high_state_share"]
            ].head(3)
            semi_markov_switch_top = ordered.sort_values("semi_markov_state_switch_rate", ascending=False)[
                ["dataset_key", "semi_markov_state_switch_rate"]
            ].head(3)
            semi_markov_run_top = ordered.sort_values("semi_markov_state_mean_run_length", ascending=True)[
                ["dataset_key", "semi_markov_state_mean_run_length"]
            ].head(3)
            lines.extend(
                [
                    "- Highest semi-Markov high-state share: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['semi_markov_high_state_share']):.4f}"
                        for _, row in semi_markov_share_top.iterrows()
                    )
                    + ".",
                    "- Highest semi-Markov switch rate: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['semi_markov_state_switch_rate']):.4f}"
                        for _, row in semi_markov_switch_top.iterrows()
                    )
                    + ".",
                    "- Shortest semi-Markov mean run length: "
                    + ", ".join(
                        f"{row['dataset_key']}={_safe_float(row['semi_markov_state_mean_run_length']):.4f}"
                        for _, row in semi_markov_run_top.iterrows()
                    )
                    + ".",
                ]
            )
        offset_df = ordered.copy()
        rho_state = _safe_spearman(offset_df["max_abs_local_freq_offset_hz"], offset_df["state_primary_score"])
        rho_alarm = _safe_spearman(offset_df["max_abs_local_freq_offset_hz"], offset_df["alarm_primary_score"])
        rho_mix_alarm = _safe_spearman(offset_df["n_unique_local_methods"], offset_df["alarm_primary_score"])
        rho_entropy_alarm = _safe_spearman(offset_df["transition_method_entropy"], offset_df["alarm_primary_score"])
        rho_switch_alarm = _safe_spearman(offset_df["local_method_switch_rate"], offset_df["alarm_primary_score"])
        lines.extend(
            [
                f"- Exploratory Spearman rho: max |offset| vs state score = {rho_state:.3f}.",
                f"- Exploratory Spearman rho: max |offset| vs alarm score = {rho_alarm:.3f}.",
                f"- Exploratory Spearman rho: local-method diversity vs alarm score = {rho_mix_alarm:.3f}.",
                f"- Exploratory Spearman rho: transition-method entropy vs alarm score = {rho_entropy_alarm:.3f}.",
                f"- Exploratory Spearman rho: local switch rate vs alarm score = {rho_switch_alarm:.3f}.",
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


def _normalized_transition_entropy(
    method_counts: dict[str, int],
    *,
    known_methods: list[str],
) -> tuple[float, float]:
    counts = np.asarray([float(method_counts.get(method, 0)) for method in known_methods], dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return float("nan"), float("nan")
    probs = counts / total
    positive = probs > 0
    entropy = -np.sum(np.where(positive, probs * np.log2(np.clip(probs, 1e-12, None)), 0.0))
    denom = np.log2(max(len(known_methods), 2))
    normalized_entropy = float(entropy / denom) if denom > 0 else float("nan")
    dominant_share = float(np.max(probs))
    return normalized_entropy, dominant_share


def _transition_switch_metrics(df_case: pd.DataFrame) -> tuple[int, float]:
    if df_case.empty or "local_selected_method" not in df_case.columns:
        return 0, 0.0

    work = df_case.copy()
    sort_cols: list[str] = []
    if "local_window_index" in work.columns:
        work["local_window_index"] = pd.to_numeric(work["local_window_index"], errors="coerce")
        sort_cols.append("local_window_index")
    if "candidate_rank" in work.columns:
        work["candidate_rank"] = pd.to_numeric(work["candidate_rank"], errors="coerce")
        sort_cols.append("candidate_rank")
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")

    methods = (
        work["local_selected_method"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )
    if len(methods) < 2:
        return 0, 0.0
    switch_count = int(sum(curr != prev for prev, curr in zip(methods, methods[1:])))
    return switch_count, float(switch_count / max(len(methods) - 1, 1))


def _order_transition_sequence_df(df_case: pd.DataFrame) -> pd.DataFrame:
    if df_case.empty:
        return df_case.copy()

    work = df_case.copy()
    sort_cols: list[str] = []
    if "local_window_index" in work.columns:
        work["local_window_index"] = pd.to_numeric(work["local_window_index"], errors="coerce")
        sort_cols.append("local_window_index")
    if "candidate_rank" in work.columns:
        work["candidate_rank"] = pd.to_numeric(work["candidate_rank"], errors="coerce")
        sort_cols.append("candidate_rank")
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")
    return work.reset_index(drop=True)


def _offset_sign_label(value: Any) -> str:
    number = _safe_float(value)
    if not np.isfinite(number):
        return ""
    if number > 1e-12:
        return "positive"
    if number < -1e-12:
        return "negative"
    return "zero"


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(arr), dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    valid = arr[finite]
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center)))
    scale = mad * 1.4826
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(valid, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        out[finite] = 0.0
        return out
    out[finite] = (valid - center) / scale
    return out


def _method_code_series(methods: list[str]) -> np.ndarray:
    lookup = {method: idx for idx, method in enumerate(TRANSITION_METHOD_ORDER)}
    fallback = len(lookup)
    return np.asarray([float(lookup.get(str(method).strip(), fallback)) for method in methods], dtype=np.float64)


def _build_bocpd_signal(
    offsets_hz: np.ndarray,
    confidences: np.ndarray,
    methods: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    offset_component = _robust_standardize(offsets_hz)
    confidence_component = _robust_standardize(-np.asarray(confidences, dtype=np.float64))
    method_codes = _robust_standardize(_method_code_series(methods))
    method_change = np.zeros(len(methods), dtype=np.float64)
    if len(methods) >= 2:
        method_change[1:] = np.asarray(
            [1.0 if methods[idx] != methods[idx - 1] else 0.0 for idx in range(1, len(methods))],
            dtype=np.float64,
        )
    signal = 0.45 * offset_component + 0.30 * confidence_component + 0.15 * method_codes + 0.10 * method_change
    return signal.astype(np.float64), method_change


def _bocpd_gaussian_change_profile(
    signal: np.ndarray,
    *,
    hazard: float = 0.20,
    prior_mean: float = 0.0,
    prior_kappa: float = 1.0,
    obs_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    x = np.asarray(signal, dtype=np.float64)
    if len(x) == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "change_prob": empty,
            "expected_run_length": empty,
            "log_surprise": empty,
        }

    hazard = float(np.clip(hazard, 1e-4, 0.999))
    obs_var = max(float(obs_scale) ** 2, 1e-6)
    run_probs = np.array([1.0], dtype=np.float64)
    mu_params = np.array([float(prior_mean)], dtype=np.float64)
    kappa_params = np.array([max(float(prior_kappa), 1e-6)], dtype=np.float64)
    change_probs: list[float] = []
    expected_runs: list[float] = []
    log_surprise: list[float] = []

    for value in x:
        pred_var = obs_var * (1.0 + 1.0 / np.maximum(kappa_params, 1e-12))
        pred_std = np.sqrt(pred_var)
        z = (value - mu_params) / pred_std
        pred = np.exp(-0.5 * z * z) / np.maximum(np.sqrt(2.0 * np.pi) * pred_std, 1e-30)
        pred = np.clip(pred, 1e-300, None)
        prior_pred_std = np.sqrt(obs_var * (1.0 + 1.0 / max(prior_kappa, 1e-12)))
        prior_z = (value - prior_mean) / prior_pred_std
        prior_pred = float(
            np.exp(-0.5 * prior_z * prior_z)
            / max(np.sqrt(2.0 * np.pi) * prior_pred_std, 1e-30)
        )
        prior_pred = max(prior_pred, 1e-300)

        new_run_probs = np.empty(len(run_probs) + 1, dtype=np.float64)
        new_run_probs[0] = float(prior_pred * np.sum(run_probs * hazard))
        new_run_probs[1:] = run_probs * pred * (1.0 - hazard)
        evidence = float(np.sum(new_run_probs))
        if not np.isfinite(evidence) or evidence <= 1e-300:
            new_run_probs = np.zeros_like(new_run_probs)
            new_run_probs[0] = 1.0
            evidence = 1e-300
        else:
            new_run_probs /= evidence

        change_probs.append(float(new_run_probs[0]))
        expected_runs.append(float(np.sum(np.arange(len(new_run_probs), dtype=np.float64) * new_run_probs)))
        log_surprise.append(float(-np.log(evidence)))

        updated_prior_mu = (prior_kappa * prior_mean + value) / (prior_kappa + 1.0)
        updated_prior_kappa = prior_kappa + 1.0
        updated_growth_mu = (kappa_params * mu_params + value) / (kappa_params + 1.0)
        updated_growth_kappa = kappa_params + 1.0
        mu_params = np.concatenate(([updated_prior_mu], updated_growth_mu))
        kappa_params = np.concatenate(([updated_prior_kappa], updated_growth_kappa))
        run_probs = new_run_probs

    return {
        "change_prob": np.asarray(change_probs, dtype=np.float64),
        "expected_run_length": np.asarray(expected_runs, dtype=np.float64),
        "log_surprise": np.asarray(log_surprise, dtype=np.float64),
    }


def _gaussian_hmm_2state_profile(
    signal: np.ndarray,
    *,
    stay_prob: float = 0.84,
    n_iter: int = 6,
) -> dict[str, Any]:
    x = np.asarray(signal, dtype=np.float64)
    if len(x) == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "state_path": np.array([], dtype=int),
            "high_state_prob": empty,
            "posterior": np.empty((0, 2), dtype=np.float64),
            "means": empty,
            "variances": empty,
            "state_entropy": float("nan"),
        }

    finite = np.isfinite(x)
    fill_value = float(np.median(x[finite])) if np.any(finite) else 0.0
    x = np.where(finite, x, fill_value)
    if len(x) == 1:
        return {
            "state_path": np.array([0], dtype=int),
            "high_state_prob": np.array([0.5], dtype=np.float64),
            "posterior": np.asarray([[0.5, 0.5]], dtype=np.float64),
            "means": np.asarray([float(x[0]), float(x[0])], dtype=np.float64),
            "variances": np.asarray([1.0, 1.0], dtype=np.float64),
            "state_entropy": 1.0,
        }

    base_var = max(float(np.var(x, ddof=0)), 1e-3)
    q25, q75 = np.percentile(x, [25.0, 75.0])
    if abs(float(q75) - float(q25)) <= 1e-6:
        spread = max(float(np.sqrt(base_var)), 0.25)
        q25 = float(np.median(x)) - 0.5 * spread
        q75 = float(np.median(x)) + 0.5 * spread
    means = np.asarray([float(q25), float(q75)], dtype=np.float64)
    variances = np.asarray([base_var, base_var], dtype=np.float64)
    initial = np.asarray([0.5, 0.5], dtype=np.float64)
    stay_prob = float(np.clip(stay_prob, 0.55, 0.995))
    transition = np.asarray(
        [[stay_prob, 1.0 - stay_prob], [1.0 - stay_prob, stay_prob]],
        dtype=np.float64,
    )

    def _emission_matrix(curr_means: np.ndarray, curr_vars: np.ndarray) -> np.ndarray:
        matrix = np.empty((len(x), 2), dtype=np.float64)
        for state_idx in range(2):
            variance = max(float(curr_vars[state_idx]), 1e-6)
            std = np.sqrt(variance)
            z = (x - float(curr_means[state_idx])) / std
            pdf = np.exp(-0.5 * z * z) / max(np.sqrt(2.0 * np.pi) * std, 1e-30)
            matrix[:, state_idx] = np.clip(pdf, 1e-300, None)
        return matrix

    posterior = np.full((len(x), 2), 0.5, dtype=np.float64)
    for _ in range(max(int(n_iter), 1)):
        emission = _emission_matrix(means, variances)
        alpha = np.empty_like(emission)
        scales = np.empty(len(x), dtype=np.float64)
        alpha[0] = initial * emission[0]
        scales[0] = max(float(alpha[0].sum()), 1e-300)
        alpha[0] /= scales[0]
        for idx in range(1, len(x)):
            alpha[idx] = emission[idx] * (alpha[idx - 1] @ transition)
            scales[idx] = max(float(alpha[idx].sum()), 1e-300)
            alpha[idx] /= scales[idx]

        beta = np.ones_like(emission)
        for idx in range(len(x) - 2, -1, -1):
            beta[idx] = transition @ (emission[idx + 1] * beta[idx + 1])
            beta[idx] /= max(scales[idx + 1], 1e-300)

        posterior = alpha * beta
        posterior_sum = posterior.sum(axis=1, keepdims=True)
        posterior = np.divide(
            posterior,
            posterior_sum,
            out=np.full_like(posterior, 0.5),
            where=posterior_sum > 0,
        )

        weights = posterior.sum(axis=0)
        for state_idx in range(2):
            weight = float(weights[state_idx])
            if weight <= 1e-6:
                continue
            mean_value = float(np.sum(posterior[:, state_idx] * x) / weight)
            variance_value = float(np.sum(posterior[:, state_idx] * (x - mean_value) ** 2) / weight)
            means[state_idx] = mean_value
            variances[state_idx] = max(variance_value, 1e-6)

        order = np.argsort(means)
        means = means[order]
        variances = variances[order]
        posterior = posterior[:, order]

    emission = _emission_matrix(means, variances)
    log_initial = np.log(np.clip(initial, 1e-12, None))
    log_transition = np.log(np.clip(transition, 1e-12, None))
    log_emission = np.log(np.clip(emission, 1e-300, None))
    delta = np.empty_like(emission)
    psi = np.zeros((len(x), 2), dtype=int)
    delta[0] = log_initial + log_emission[0]
    for idx in range(1, len(x)):
        for state_idx in range(2):
            scores = delta[idx - 1] + log_transition[:, state_idx]
            psi[idx, state_idx] = int(np.argmax(scores))
            delta[idx, state_idx] = float(np.max(scores)) + log_emission[idx, state_idx]

    state_path = np.zeros(len(x), dtype=int)
    state_path[-1] = int(np.argmax(delta[-1]))
    for idx in range(len(x) - 2, -1, -1):
        state_path[idx] = psi[idx + 1, state_path[idx + 1]]

    occupancy = np.asarray(
        [
            float(np.mean(state_path == 0)),
            float(np.mean(state_path == 1)),
        ],
        dtype=np.float64,
    )
    positive = occupancy > 0
    entropy = -np.sum(np.where(positive, occupancy * np.log2(np.clip(occupancy, 1e-12, None)), 0.0))
    return {
        "state_path": state_path.astype(int),
        "high_state_prob": posterior[:, 1].astype(np.float64),
        "posterior": posterior.astype(np.float64),
        "means": means.astype(np.float64),
        "variances": variances.astype(np.float64),
        "state_entropy": float(entropy / np.log2(2.0)),
    }


def _semi_markov_2state_profile(
    signal: np.ndarray,
    *,
    means: np.ndarray | None = None,
    variances: np.ndarray | None = None,
    base_state_path: np.ndarray | None = None,
) -> dict[str, Any]:
    x = np.asarray(signal, dtype=np.float64)
    if len(x) == 0:
        empty = np.array([], dtype=np.float64)
        return {
            "state_path": np.array([], dtype=int),
            "segment_lengths": np.array([], dtype=int),
            "segment_states": np.array([], dtype=int),
            "state_entropy": float("nan"),
            "duration_surprise_mean": float("nan"),
            "short_run_count": 0,
        }

    finite = np.isfinite(x)
    fill_value = float(np.median(x[finite])) if np.any(finite) else 0.0
    x = np.where(finite, x, fill_value)
    if len(x) == 1:
        return {
            "state_path": np.array([0], dtype=int),
            "segment_lengths": np.array([1], dtype=int),
            "segment_states": np.array([0], dtype=int),
            "state_entropy": 0.0,
            "duration_surprise_mean": 0.0,
            "short_run_count": 1,
        }

    if means is None or variances is None:
        hmm_profile = _gaussian_hmm_2state_profile(x)
        means = np.asarray(hmm_profile["means"], dtype=np.float64)
        variances = np.asarray(hmm_profile["variances"], dtype=np.float64)
        if base_state_path is None:
            base_state_path = np.asarray(hmm_profile["state_path"], dtype=int)
    else:
        means = np.asarray(means, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
    if base_state_path is None or len(base_state_path) != len(x):
        base_state_path = np.asarray((x >= float(np.median(x))).astype(int), dtype=int)

    base_run_lengths: list[int] = []
    current_state = int(base_state_path[0])
    current_length = 1
    for idx in range(1, len(base_state_path)):
        next_state = int(base_state_path[idx])
        if next_state == current_state:
            current_length += 1
        else:
            base_run_lengths.append(current_length)
            current_state = next_state
            current_length = 1
    base_run_lengths.append(current_length)
    duration_mean = float(np.clip(np.mean(base_run_lengths), 2.0, float(len(x))))

    log_emission = np.empty((len(x), 2), dtype=np.float64)
    for state_idx in range(2):
        variance = max(float(variances[state_idx]), 1e-6)
        std = np.sqrt(variance)
        z = (x - float(means[state_idx])) / std
        pdf = np.exp(-0.5 * z * z) / max(np.sqrt(2.0 * np.pi) * std, 1e-30)
        log_emission[:, state_idx] = np.log(np.clip(pdf, 1e-300, None))

    max_duration = len(x)
    duration_weights = np.empty(max_duration + 1, dtype=np.float64)
    duration_weights[0] = 0.0
    for duration in range(1, max_duration + 1):
        log_weight = duration * math.log(duration_mean) - duration_mean - math.lgamma(duration + 1.0)
        duration_weights[duration] = math.exp(log_weight)
    weight_sum = float(np.sum(duration_weights[1:]))
    if not np.isfinite(weight_sum) or weight_sum <= 1e-300:
        duration_weights[1:] = 1.0 / float(max_duration)
    else:
        duration_weights[1:] /= weight_sum
    log_duration = np.log(np.clip(duration_weights, 1e-300, None))

    prefix = np.concatenate(
        [
            np.zeros((2, 1), dtype=np.float64),
            np.cumsum(log_emission.T, axis=1),
        ],
        axis=1,
    )
    dp = np.full((len(x) + 1, 2), -np.inf, dtype=np.float64)
    prev_end = np.full((len(x) + 1, 2), -1, dtype=int)
    prev_state = np.full((len(x) + 1, 2), -1, dtype=int)
    log_initial = math.log(0.5)

    for end in range(1, len(x) + 1):
        for state_idx in range(2):
            best_score = -np.inf
            best_start = -1
            best_prev_state = -1
            for duration in range(1, end + 1):
                start = end - duration
                segment_score = float(prefix[state_idx, end] - prefix[state_idx, start] + log_duration[duration])
                if start == 0:
                    score = log_initial + segment_score
                    prev_state_idx = -1
                else:
                    candidate_prev = 1 - state_idx
                    prev_score = float(dp[start, candidate_prev])
                    if not np.isfinite(prev_score):
                        continue
                    score = prev_score + segment_score
                    prev_state_idx = candidate_prev
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_prev_state = prev_state_idx
            dp[end, state_idx] = best_score
            prev_end[end, state_idx] = best_start
            prev_state[end, state_idx] = best_prev_state

    final_state = int(np.argmax(dp[len(x)]))
    if not np.isfinite(dp[len(x), final_state]):
        fallback_state = int(np.argmax(np.sum(log_emission, axis=0)))
        return {
            "state_path": np.full(len(x), fallback_state, dtype=int),
            "segment_lengths": np.array([len(x)], dtype=int),
            "segment_states": np.array([fallback_state], dtype=int),
            "state_entropy": 0.0,
            "duration_surprise_mean": float(-log_duration[len(x)]),
            "short_run_count": int(len(x) == 1),
        }

    segments: list[tuple[int, int, int]] = []
    end = len(x)
    state_idx = final_state
    while end > 0 and state_idx >= 0:
        start = int(prev_end[end, state_idx])
        if start < 0:
            break
        segments.append((start, end, state_idx))
        prev_state_idx = int(prev_state[end, state_idx])
        end = start
        state_idx = prev_state_idx
    segments.reverse()

    state_path = np.zeros(len(x), dtype=int)
    segment_lengths: list[int] = []
    segment_states: list[int] = []
    duration_surprises: list[float] = []
    for start, end, state_idx in segments:
        state_path[start:end] = int(state_idx)
        duration = int(end - start)
        segment_lengths.append(duration)
        segment_states.append(int(state_idx))
        duration_surprises.append(float(-log_duration[duration]))

    occupancy = np.asarray(
        [
            float(np.mean(state_path == 0)),
            float(np.mean(state_path == 1)),
        ],
        dtype=np.float64,
    )
    positive = occupancy > 0
    entropy = -np.sum(np.where(positive, occupancy * np.log2(np.clip(occupancy, 1e-12, None)), 0.0))
    return {
        "state_path": state_path.astype(int),
        "segment_lengths": np.asarray(segment_lengths, dtype=int),
        "segment_states": np.asarray(segment_states, dtype=int),
        "state_entropy": float(entropy / np.log2(2.0)),
        "duration_surprise_mean": float(np.mean(duration_surprises)) if duration_surprises else float("nan"),
        "short_run_count": int(sum(duration == 1 for duration in segment_lengths)),
    }


def _summarize_transition_local_sequence(
    primary_overlap_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if primary_overlap_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sequence_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    edge_universe = max(len(TRANSITION_METHOD_ORDER) ** 2, 2)

    for dataset_key, df_case in primary_overlap_df.groupby("dataset_key", dropna=False):
        ordered = _order_transition_sequence_df(df_case)
        if ordered.empty:
            continue

        methods = (
            ordered["local_selected_method"]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )
        offsets = pd.to_numeric(
            ordered.get("local_freq_offset_from_global_hz", pd.Series(index=ordered.index, dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        confidences = pd.to_numeric(
            ordered.get("local_common_axial_confidence", pd.Series(index=ordered.index, dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        sequence_positions = np.arange(1, len(ordered) + 1, dtype=int)

        run_ids: list[int] = []
        run_id = 0
        prev_method = ""
        for method in methods:
            if method != prev_method:
                run_id += 1
                prev_method = method
            run_ids.append(run_id)
        run_size_map = pd.Series(run_ids, dtype=int).value_counts().to_dict()

        sign_labels = [_offset_sign_label(value) for value in offsets]
        bocpd_signal, method_change = _build_bocpd_signal(offsets, confidences, methods)
        bocpd_profile = _bocpd_gaussian_change_profile(bocpd_signal)
        change_prob = bocpd_profile["change_prob"]
        expected_run = bocpd_profile["expected_run_length"]
        log_surprise = bocpd_profile["log_surprise"]
        hmm_profile = _gaussian_hmm_2state_profile(bocpd_signal)
        hmm_state_path = hmm_profile["state_path"]
        hmm_high_prob = hmm_profile["high_state_prob"]
        semi_markov_profile = _semi_markov_2state_profile(
            bocpd_signal,
            means=np.asarray(hmm_profile["means"], dtype=np.float64),
            variances=np.asarray(hmm_profile["variances"], dtype=np.float64),
            base_state_path=hmm_state_path,
        )
        semi_markov_state_path = semi_markov_profile["state_path"]
        sign_switch_count = 0
        edge_counter: dict[tuple[str, str], int] = {}
        offset_step_values: list[float] = []
        confidence_step_values: list[float] = []
        semi_markov_run_ids: list[int] = []
        semi_markov_run_id = 0
        prev_semi_markov_state = -1
        for state_value in semi_markov_state_path.tolist():
            state_int = int(state_value)
            if state_int != prev_semi_markov_state:
                semi_markov_run_id += 1
                prev_semi_markov_state = state_int
            semi_markov_run_ids.append(semi_markov_run_id)
        semi_markov_run_size_map = pd.Series(semi_markov_run_ids, dtype=int).value_counts().to_dict()
        for idx, (_, row) in enumerate(ordered.iterrows()):
            method = methods[idx]
            sign_label = sign_labels[idx]
            prev_method = methods[idx - 1] if idx > 0 else ""
            prev_sign = sign_labels[idx - 1] if idx > 0 else ""
            candidate_rank_raw = pd.to_numeric(row.get("candidate_rank"), errors="coerce")
            local_window_raw = pd.to_numeric(row.get("local_window_index"), errors="coerce")
            offset_step_abs = abs(offsets[idx] - offsets[idx - 1]) if idx > 0 and np.isfinite(offsets[idx - 1]) and np.isfinite(offsets[idx]) else float("nan")
            confidence_step_abs = (
                abs(confidences[idx] - confidences[idx - 1])
                if idx > 0 and np.isfinite(confidences[idx - 1]) and np.isfinite(confidences[idx])
                else float("nan")
            )
            method_changed = bool(idx > 0 and method != prev_method)
            hmm_changed = bool(idx > 0 and idx < len(hmm_state_path) and hmm_state_path[idx] != hmm_state_path[idx - 1])
            semi_markov_changed = bool(
                idx > 0
                and idx < len(semi_markov_state_path)
                and semi_markov_state_path[idx] != semi_markov_state_path[idx - 1]
            )
            sign_changed = bool(
                idx > 0
                and sign_label
                and prev_sign
                and sign_label != prev_sign
            )
            if sign_changed:
                sign_switch_count += 1
            if idx > 0 and prev_method and method:
                edge_key = (prev_method, method)
                edge_counter[edge_key] = edge_counter.get(edge_key, 0) + 1
                edge_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "discharge_type": str(ordered["discharge_type"].iloc[0]),
                        "variant": str(ordered["variant"].iloc[0]),
                        "from_method": prev_method,
                        "to_method": method,
                        "edge_count": 1,
                        "is_method_switch": int(method_changed),
                        "offset_step_abs_hz": float(offset_step_abs),
                        "confidence_step_abs": float(confidence_step_abs),
                    }
                )
                if np.isfinite(offset_step_abs):
                    offset_step_values.append(float(offset_step_abs))
                if np.isfinite(confidence_step_abs):
                    confidence_step_values.append(float(confidence_step_abs))
            sequence_rows.append(
                {
                    "dataset_key": dataset_key,
                    "discharge_type": str(ordered["discharge_type"].iloc[0]),
                    "variant": str(ordered["variant"].iloc[0]),
                    "sequence_position": int(sequence_positions[idx]),
                    "candidate_rank": int(candidate_rank_raw) if pd.notna(candidate_rank_raw) else 0,
                    "local_window_index": int(local_window_raw) if pd.notna(local_window_raw) else 0,
                    "local_selected_method": method,
                    "local_freq_offset_from_global_hz": float(offsets[idx]) if np.isfinite(offsets[idx]) else float("nan"),
                    "local_common_axial_confidence": float(confidences[idx]) if np.isfinite(confidences[idx]) else float("nan"),
                    "local_freq_offset_sign": sign_label,
                    "method_changed": int(method_changed),
                    "bocpd_method_change_flag": float(method_change[idx]) if idx < len(method_change) else 0.0,
                    "bocpd_signal": float(bocpd_signal[idx]) if idx < len(bocpd_signal) else float("nan"),
                    "bocpd_change_prob": float(change_prob[idx]) if idx < len(change_prob) else float("nan"),
                    "bocpd_expected_run_length": float(expected_run[idx]) if idx < len(expected_run) else float("nan"),
                    "bocpd_log_surprise": float(log_surprise[idx]) if idx < len(log_surprise) else float("nan"),
                    "hmm_state": int(hmm_state_path[idx]) if idx < len(hmm_state_path) else 0,
                    "hmm_high_state_prob": float(hmm_high_prob[idx]) if idx < len(hmm_high_prob) else float("nan"),
                    "hmm_state_changed": int(hmm_changed),
                    "semi_markov_state": int(semi_markov_state_path[idx]) if idx < len(semi_markov_state_path) else 0,
                    "semi_markov_state_changed": int(semi_markov_changed),
                    "semi_markov_run_length": int(semi_markov_run_size_map.get(semi_markov_run_ids[idx], 1))
                    if idx < len(semi_markov_run_ids)
                    else 1,
                    "offset_sign_changed": int(sign_changed),
                    "regime_run_id": int(run_ids[idx]),
                    "regime_run_length": int(run_size_map.get(run_ids[idx], 1)),
                    "offset_step_abs_hz": float(offset_step_abs),
                    "confidence_step_abs": float(confidence_step_abs),
                }
            )

        edge_entropy = float("nan")
        total_edges = int(sum(edge_counter.values()))
        if total_edges > 0:
            probs = np.asarray([float(count) / float(total_edges) for count in edge_counter.values()], dtype=np.float64)
            positive = probs > 0
            entropy = -np.sum(np.where(positive, probs * np.log2(np.clip(probs, 1e-12, None)), 0.0))
            edge_entropy = float(entropy / np.log2(edge_universe))

        run_lengths = list(run_size_map.values())
        run_count = int(len(run_lengths))
        mean_run_length = float(np.mean(run_lengths)) if run_lengths else float("nan")
        max_run_length = int(max(run_lengths)) if run_lengths else 0
        persistence_ratio = float(1.0 - (sum(methods[idx] != methods[idx - 1] for idx in range(1, len(methods))) / max(len(methods) - 1, 1))) if len(methods) >= 2 else 1.0
        offset_sign_switch_rate = float(sign_switch_count / max(len(methods) - 1, 1)) if len(methods) >= 2 else 0.0
        change_threshold = 0.25
        hmm_run_lengths: list[int] = []
        if len(hmm_state_path):
            current_hmm_state = int(hmm_state_path[0])
            current_hmm_run = 1
            for idx in range(1, len(hmm_state_path)):
                if int(hmm_state_path[idx]) == current_hmm_state:
                    current_hmm_run += 1
                else:
                    hmm_run_lengths.append(current_hmm_run)
                    current_hmm_state = int(hmm_state_path[idx])
                    current_hmm_run = 1
            hmm_run_lengths.append(current_hmm_run)
        hmm_switch_count = int(np.sum(hmm_state_path[1:] != hmm_state_path[:-1])) if len(hmm_state_path) >= 2 else 0
        hmm_switch_rate = float(hmm_switch_count / max(len(hmm_state_path) - 1, 1)) if len(hmm_state_path) >= 2 else 0.0
        hmm_high_state_share = float(np.mean(hmm_state_path == 1)) if len(hmm_state_path) else float("nan")
        semi_markov_segment_lengths = semi_markov_profile["segment_lengths"].astype(int).tolist()
        semi_markov_switch_count = (
            int(np.sum(semi_markov_state_path[1:] != semi_markov_state_path[:-1]))
            if len(semi_markov_state_path) >= 2
            else 0
        )
        semi_markov_switch_rate = (
            float(semi_markov_switch_count / max(len(semi_markov_state_path) - 1, 1))
            if len(semi_markov_state_path) >= 2
            else 0.0
        )
        semi_markov_high_state_share = (
            float(np.mean(semi_markov_state_path == 1))
            if len(semi_markov_state_path)
            else float("nan")
        )
        case_rows.append(
            {
                "dataset_key": dataset_key,
                "local_regime_run_count": run_count,
                "local_regime_mean_run_length": mean_run_length,
                "local_regime_max_run_length": max_run_length,
                "local_regime_persistence_ratio": persistence_ratio,
                "local_regime_transition_entropy": edge_entropy,
                "local_offset_sign_switch_count": int(sign_switch_count),
                "local_offset_sign_switch_rate": offset_sign_switch_rate,
                "local_abs_offset_step_mean_hz": float(np.mean(offset_step_values)) if offset_step_values else float("nan"),
                "local_abs_offset_step_max_hz": float(np.max(offset_step_values)) if offset_step_values else float("nan"),
                "local_abs_confidence_step_mean": float(np.mean(confidence_step_values)) if confidence_step_values else float("nan"),
                "local_abs_confidence_step_max": float(np.max(confidence_step_values)) if confidence_step_values else float("nan"),
                "bocpd_max_change_prob": float(np.max(change_prob)) if len(change_prob) else float("nan"),
                "bocpd_mean_change_prob": float(np.mean(change_prob)) if len(change_prob) else float("nan"),
                "bocpd_run_length_mean": float(np.mean(expected_run)) if len(expected_run) else float("nan"),
                "bocpd_run_length_min": float(np.min(expected_run)) if len(expected_run) else float("nan"),
                "bocpd_change_count": int(np.sum(change_prob >= change_threshold)) if len(change_prob) else 0,
                "bocpd_surprise_score": float(np.max(log_surprise)) if len(log_surprise) else float("nan"),
                "hmm_high_state_share": hmm_high_state_share,
                "hmm_high_state_prob_mean": float(np.mean(hmm_high_prob)) if len(hmm_high_prob) else float("nan"),
                "hmm_state_switch_count": hmm_switch_count,
                "hmm_state_switch_rate": hmm_switch_rate,
                "hmm_state_mean_run_length": float(np.mean(hmm_run_lengths)) if hmm_run_lengths else float("nan"),
                "hmm_state_persistence_ratio": float(1.0 - hmm_switch_rate) if len(hmm_state_path) >= 2 else 1.0,
                "hmm_state_entropy": float(hmm_profile["state_entropy"]),
                "semi_markov_high_state_share": semi_markov_high_state_share,
                "semi_markov_state_switch_count": semi_markov_switch_count,
                "semi_markov_state_switch_rate": semi_markov_switch_rate,
                "semi_markov_state_mean_run_length": float(np.mean(semi_markov_segment_lengths))
                if semi_markov_segment_lengths
                else float("nan"),
                "semi_markov_state_persistence_ratio": float(1.0 - semi_markov_switch_rate)
                if len(semi_markov_state_path) >= 2
                else 1.0,
                "semi_markov_state_entropy": float(semi_markov_profile["state_entropy"]),
                "semi_markov_segment_count": int(len(semi_markov_segment_lengths)),
                "semi_markov_short_run_count": int(semi_markov_profile["short_run_count"]),
                "semi_markov_duration_surprise_mean": float(semi_markov_profile["duration_surprise_mean"]),
            }
        )

    sequence_df = pd.DataFrame(sequence_rows)
    case_df = pd.DataFrame(case_rows).sort_values("dataset_key").reset_index(drop=True) if case_rows else pd.DataFrame()
    edge_df = pd.DataFrame(edge_rows)
    if not edge_df.empty:
        edge_df = (
            edge_df.groupby(
                ["dataset_key", "discharge_type", "variant", "from_method", "to_method"],
                dropna=False,
            )
            .agg(
                transition_count=("edge_count", "sum"),
                switch_count=("is_method_switch", "sum"),
                mean_offset_step_abs_hz=("offset_step_abs_hz", "mean"),
                max_offset_step_abs_hz=("offset_step_abs_hz", "max"),
                mean_confidence_step_abs=("confidence_step_abs", "mean"),
                max_confidence_step_abs=("confidence_step_abs", "max"),
            )
            .reset_index()
        )
    return sequence_df, case_df, edge_df


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
    known_methods = TRANSITION_METHOD_ORDER
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
        transition_entropy, dominant_share = _normalized_transition_entropy(
            method_counts,
            known_methods=known_methods,
        )
        switch_count, switch_rate = _transition_switch_metrics(df_case)
        local_offset_series = pd.to_numeric(df_case["local_freq_offset_from_global_hz"], errors="coerce")
        local_conf_series = pd.to_numeric(df_case["local_common_axial_confidence"], errors="coerce")
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
            "local_freq_offset_std_hz": float(local_offset_series.std(ddof=0)),
            "mean_local_common_axial_confidence": float(local_conf_series.mean()),
            "local_common_axial_confidence_std": float(local_conf_series.std(ddof=0)),
            "transition_method_entropy": transition_entropy,
            "transition_dominant_method_share": dominant_share,
            "local_method_switch_count": int(switch_count),
            "local_method_switch_rate": float(switch_rate),
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
        "h_test": "#3a86ff",
        "pdm": "#ff9f1c",
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


def _plot_transition_mix_stability(case_summary_df: pd.DataFrame, *, out_png: Path) -> Path | None:
    if case_summary_df.empty:
        return None
    required = {
        "dataset_key",
        "transition_method_entropy",
        "local_method_switch_rate",
        "local_freq_offset_std_hz",
        "local_common_axial_confidence_std",
        "discharge_type",
    }
    if not required.issubset(case_summary_df.columns):
        return None

    ordered = case_summary_df.sort_values("dataset_key").reset_index(drop=True)
    color_map = {
        "internal": "#2d6a8a",
        "superficial": "#c26d2d",
        "multiple": "#6d597a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    x = np.arange(len(ordered))
    width = 0.36
    entropy_vals = pd.to_numeric(ordered["transition_method_entropy"], errors="coerce").fillna(0.0).to_numpy()
    switch_vals = pd.to_numeric(ordered["local_method_switch_rate"], errors="coerce").fillna(0.0).to_numpy()
    axes[0].bar(x - width / 2.0, entropy_vals, width=width, color="#6d597a", label="Method entropy")
    axes[0].bar(x + width / 2.0, switch_vals, width=width, color="#2d6a8a", label="Switch rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ordered["dataset_key"].tolist())
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Transition-local mix entropy and switching")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(loc="upper center", ncol=2)

    for _, row in ordered.iterrows():
        x_val = _safe_float(row.get("local_freq_offset_std_hz"))
        y_val = _safe_float(row.get("local_common_axial_confidence_std"))
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        color = color_map.get(str(row.get("discharge_type", "")), "#555555")
        axes[1].scatter(
            x_val,
            y_val,
            s=90,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        axes[1].annotate(
            str(row.get("dataset_key", "")),
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Local freq offset std (Hz)")
    axes[1].set_ylabel("Local confidence std")
    axes[1].set_title("Transition-local dispersion")
    axes[1].grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_transition_regime_sequence(
    sequence_df: pd.DataFrame,
    case_summary_df: pd.DataFrame,
    *,
    out_png: Path,
) -> Path | None:
    if sequence_df.empty or case_summary_df.empty:
        return None
    required_sequence = {
        "dataset_key",
        "sequence_position",
        "local_selected_method",
        "local_common_axial_confidence",
        "local_freq_offset_sign",
    }
    required_case = {
        "dataset_key",
        "local_regime_persistence_ratio",
        "local_regime_transition_entropy",
    }
    if not required_sequence.issubset(sequence_df.columns) or not required_case.issubset(case_summary_df.columns):
        return None

    dataset_order = sorted(sequence_df["dataset_key"].astype(str).unique().tolist())
    palette = {
        "coherence": "#2d6a8a",
        "harmonic_power": "#c26d2d",
        "epoch_folding": "#6d597a",
        "h_test": "#3a86ff",
        "pdm": "#ff9f1c",
        "gregory_loredo": "#5d8f52",
        "phase_distance_correlation": "#9b2226",
    }
    marker_map = {
        "positive": "^",
        "negative": "v",
        "zero": "o",
        "": "o",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 1.1 * len(dataset_order) + 3.0),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    ax_seq, ax_case = axes
    y_positions = {dataset_key: idx for idx, dataset_key in enumerate(dataset_order)}

    for dataset_key in dataset_order:
        df_case = _order_transition_sequence_df(sequence_df[sequence_df["dataset_key"] == dataset_key])
        if df_case.empty:
            continue
        y = y_positions[dataset_key]
        x_vals = pd.to_numeric(df_case["sequence_position"], errors="coerce").to_numpy(dtype=np.float64)
        ax_seq.plot(x_vals, np.full_like(x_vals, y, dtype=np.float64), color="#d8d2c6", linewidth=1.0, zorder=1)
        for _, row in df_case.iterrows():
            method = str(row.get("local_selected_method", ""))
            color = palette.get(method, "#777777")
            marker = marker_map.get(str(row.get("local_freq_offset_sign", "")), "o")
            conf = _safe_float(row.get("local_common_axial_confidence"))
            size = 70.0 + 110.0 * np.clip(conf if np.isfinite(conf) else 0.0, 0.0, 1.0)
            ax_seq.scatter(
                _safe_float(row.get("sequence_position")),
                y,
                s=size,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
        ax_seq.annotate(dataset_key, (0.75, y), ha="right", va="center", fontsize=9, color="#16324f")

    ax_seq.set_yticks(list(y_positions.values()))
    ax_seq.set_yticklabels(dataset_order)
    ax_seq.set_xlabel("Local transition order")
    ax_seq.set_title("Local regime chain by dataset")
    ax_seq.grid(True, axis="x", linestyle="--", alpha=0.2)
    ax_seq.set_ylim(-0.75, len(dataset_order) - 0.25)
    ax_seq.invert_yaxis()

    case_ordered = case_summary_df.sort_values("dataset_key").reset_index(drop=True)
    persistence = pd.to_numeric(case_ordered["local_regime_persistence_ratio"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    entropy = pd.to_numeric(case_ordered["local_regime_transition_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y_case = np.arange(len(case_ordered))
    width = 0.36
    ax_case.barh(y_case - width / 2.0, persistence, height=width, color="#2d6a8a", label="Persistence")
    ax_case.barh(y_case + width / 2.0, entropy, height=width, color="#6d597a", label="Seq. entropy")
    ax_case.set_yticks(y_case)
    ax_case.set_yticklabels(case_ordered["dataset_key"].astype(str).tolist())
    ax_case.set_xlim(0.0, 1.02)
    ax_case.set_title("Persistence vs sequence entropy")
    ax_case.grid(True, axis="x", linestyle="--", alpha=0.2)
    ax_case.legend(loc="lower right")
    ax_case.invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette.get(method, "#777777"), label=method, markersize=8)
        for method in TRANSITION_METHOD_ORDER
        if method in sequence_df["local_selected_method"].astype(str).unique().tolist()
    ]
    if handles:
        ax_seq.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=max(1, min(3, len(handles))))

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _build_transition_local_wavelet_signal(
    df_events: pd.DataFrame,
    *,
    n_bins: int = 64,
) -> np.ndarray:
    if df_events.empty or n_bins < 8:
        return np.zeros(max(n_bins, 8), dtype=np.float64)

    work = df_events.copy()
    toa = pd.to_numeric(work.get("toa_s"), errors="coerce")
    peak = pd.to_numeric(work.get("peak_v"), errors="coerce").fillna(1.0)
    valid = pd.DataFrame({"toa_s": toa, "peak_v": peak}).dropna()
    if valid.empty:
        return np.zeros(n_bins, dtype=np.float64)

    times = valid["toa_s"].to_numpy(dtype=np.float64)
    weights = np.abs(valid["peak_v"].to_numpy(dtype=np.float64))
    signal = np.zeros(n_bins, dtype=np.float64)
    if len(times) == 1 or float(times.max() - times.min()) <= 1e-12:
        signal[0] = float(np.sum(weights))
        return signal

    scaled = (times - float(times.min())) / float(times.max() - times.min())
    bins = np.clip(np.floor(scaled * (n_bins - 1)).astype(int), 0, n_bins - 1)
    np.add.at(signal, bins, weights)
    signal = signal - float(np.mean(signal))
    scale = float(np.std(signal, ddof=0))
    if scale > 1e-12 and np.isfinite(scale):
        signal = signal / scale
    return signal


def _wavelet_energy_profile(
    signal: np.ndarray,
    *,
    wavelet: str = "db4",
    max_level: int = 3,
) -> dict[str, Any]:
    data = np.asarray(signal, dtype=np.float64)
    if data.size < 8 or not np.isfinite(data).any() or np.allclose(data, 0.0):
        return {
            "tfa_wavelet_entropy": float("nan"),
            "tfa_wavelet_dominant_band": "",
            "tfa_wavelet_dominant_band_share": float("nan"),
            "tfa_wavelet_detail_entropy": float("nan"),
            "tfa_wavelet_detail_dominant_band": "",
            "tfa_wavelet_detail_dominant_band_share": float("nan"),
        }

    level = min(int(max_level), int(pywt.dwt_max_level(len(data), wavelet)))
    if level <= 0:
        level = 1
    coeffs = pywt.wavedec(data, wavelet, level=level)
    band_labels = [f"A{level}"] + [f"D{idx}" for idx in range(level, 0, -1)]
    energies = np.asarray([float(np.sum(np.square(np.asarray(c, dtype=np.float64)))) for c in coeffs], dtype=np.float64)
    total_energy = float(np.sum(energies))
    if total_energy <= 0 or not np.isfinite(total_energy):
        return {
            "tfa_wavelet_entropy": float("nan"),
            "tfa_wavelet_dominant_band": "",
            "tfa_wavelet_dominant_band_share": float("nan"),
            "tfa_wavelet_detail_entropy": float("nan"),
            "tfa_wavelet_detail_dominant_band": "",
            "tfa_wavelet_detail_dominant_band_share": float("nan"),
        }

    shares = energies / total_energy
    positive = shares > 0
    entropy = -np.sum(np.where(positive, shares * np.log2(np.clip(shares, 1e-12, None)), 0.0))
    entropy_denom = np.log2(max(len(shares), 2))
    dominant_idx = int(np.argmax(shares))
    detail_energies = energies[1:] if len(energies) > 1 else energies[:0]
    detail_labels = band_labels[1:] if len(band_labels) > 1 else band_labels[:0]
    detail_entropy = float("nan")
    detail_dominant_band = ""
    detail_dominant_share = float("nan")
    detail_total = float(np.sum(detail_energies))
    if detail_total > 0 and len(detail_energies) > 0:
        detail_shares = detail_energies / detail_total
        detail_positive = detail_shares > 0
        detail_entropy_raw = -np.sum(
            np.where(detail_positive, detail_shares * np.log2(np.clip(detail_shares, 1e-12, None)), 0.0)
        )
        detail_denom = np.log2(max(len(detail_shares), 2))
        detail_entropy = float(detail_entropy_raw / detail_denom) if detail_denom > 0 else float("nan")
        detail_idx = int(np.argmax(detail_shares))
        detail_dominant_band = detail_labels[detail_idx]
        detail_dominant_share = float(detail_shares[detail_idx])
    return {
        "tfa_wavelet_entropy": float(entropy / entropy_denom) if entropy_denom > 0 else float("nan"),
        "tfa_wavelet_dominant_band": band_labels[dominant_idx],
        "tfa_wavelet_dominant_band_share": float(shares[dominant_idx]),
        "tfa_wavelet_detail_entropy": detail_entropy,
        "tfa_wavelet_detail_dominant_band": detail_dominant_band,
        "tfa_wavelet_detail_dominant_band_share": detail_dominant_share,
    }


def _summarize_transition_local_wavelet(
    primary_overlap_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    n_bins: int = 64,
    wavelet: str = "db4",
    max_level: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if primary_overlap_df.empty or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    window_rows: list[dict[str, Any]] = []
    band_order: dict[str, int] = {}
    for dataset_key, df_case in primary_overlap_df.groupby("dataset_key", dropna=False):
        case_meta = summary_df[summary_df["dataset_key"] == dataset_key]
        if case_meta.empty:
            continue
        study_output_dir = Path(str(case_meta["study_output_dir"].iloc[0]))
        material_output_dir = study_output_dir.parent / "material"
        blind_trace_path = material_output_dir / "blind_prpd_local_trace.csv"
        delta_path = material_output_dir / "delta_t_series_master.csv"
        if not blind_trace_path.exists() or not delta_path.exists():
            continue

        blind_trace_df = pd.read_csv(blind_trace_path)
        delta_df = pd.read_csv(delta_path, usecols=["toa_s", "peak_v"])
        if blind_trace_df.empty or delta_df.empty:
            continue

        for _, overlap_row in df_case.iterrows():
            local_idx = pd.to_numeric(overlap_row.get("local_window_index"), errors="coerce")
            if pd.isna(local_idx):
                continue
            trace_match = blind_trace_df[blind_trace_df["local_window_index"] == int(local_idx)]
            if trace_match.empty:
                continue
            trace_row = trace_match.iloc[0]
            start_idx = int(pd.to_numeric(trace_row.get("event_start_idx"), errors="coerce") or 0)
            end_idx = int(pd.to_numeric(trace_row.get("event_end_idx"), errors="coerce") or start_idx)
            df_events = delta_df.iloc[start_idx : end_idx + 1].copy()
            if df_events.empty:
                continue
            binned_signal = _build_transition_local_wavelet_signal(df_events, n_bins=n_bins)
            profile = _wavelet_energy_profile(
                binned_signal,
                wavelet=wavelet,
                max_level=max_level,
            )
            band_label = str(profile.get("tfa_wavelet_dominant_band", ""))
            if band_label and band_label not in band_order:
                band_order[band_label] = len(band_order)
            window_rows.append(
                {
                    "dataset_key": dataset_key,
                    "local_window_index": int(local_idx),
                    "candidate_rank": int(pd.to_numeric(overlap_row.get("candidate_rank"), errors="coerce") or 0),
                    "local_selected_method": str(overlap_row.get("local_selected_method", "")),
                    **profile,
                }
            )

    window_df = pd.DataFrame(window_rows)
    if window_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    case_rows: list[dict[str, Any]] = []
    for dataset_key, df_case in window_df.groupby("dataset_key", dropna=False):
        dominant_counts = (
            df_case["tfa_wavelet_dominant_band"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )
        detail_band_counts = (
            df_case["tfa_wavelet_detail_dominant_band"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .to_dict()
        )
        dominant_labels = (
            df_case.sort_values(["local_window_index", "candidate_rank"], kind="mergesort")["tfa_wavelet_dominant_band"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .tolist()
        )
        detail_labels = (
            df_case.sort_values(["local_window_index", "candidate_rank"], kind="mergesort")["tfa_wavelet_detail_dominant_band"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .tolist()
        )
        switch_count = 0
        switch_rate = 0.0
        if len(dominant_labels) >= 2:
            switch_count = int(sum(curr != prev for prev, curr in zip(dominant_labels, dominant_labels[1:])))
            switch_rate = float(switch_count / max(len(dominant_labels) - 1, 1))
        detail_switch_count = 0
        detail_switch_rate = 0.0
        if len(detail_labels) >= 2:
            detail_switch_count = int(sum(curr != prev for prev, curr in zip(detail_labels, detail_labels[1:])))
            detail_switch_rate = float(detail_switch_count / max(len(detail_labels) - 1, 1))
        total = float(sum(dominant_counts.values()))
        dominant_entropy = float("nan")
        if total > 0:
            probs = np.asarray([float(count) / total for count in dominant_counts.values()], dtype=np.float64)
            positive = probs > 0
            entropy = -np.sum(np.where(positive, probs * np.log2(np.clip(probs, 1e-12, None)), 0.0))
            denom = np.log2(max(len(band_order), 2))
            dominant_entropy = float(entropy / denom) if denom > 0 else float("nan")
        detail_band_entropy = float("nan")
        detail_total = float(sum(detail_band_counts.values()))
        if detail_total > 0:
            detail_probs = np.asarray([float(count) / detail_total for count in detail_band_counts.values()], dtype=np.float64)
            detail_positive = detail_probs > 0
            detail_entropy_raw = -np.sum(
                np.where(detail_positive, detail_probs * np.log2(np.clip(detail_probs, 1e-12, None)), 0.0)
            )
            detail_denom = np.log2(max(len(detail_band_counts), 2))
            detail_band_entropy = float(detail_entropy_raw / detail_denom) if detail_denom > 0 else float("nan")
        case_rows.append(
            {
                "dataset_key": dataset_key,
                "tfa_wavelet_entropy_mean": float(pd.to_numeric(df_case["tfa_wavelet_entropy"], errors="coerce").mean()),
                "tfa_wavelet_entropy_max": float(pd.to_numeric(df_case["tfa_wavelet_entropy"], errors="coerce").max()),
                "tfa_wavelet_dominant_band_unique_count": int(len(dominant_counts)),
                "tfa_wavelet_dominant_band_entropy": dominant_entropy,
                "tfa_wavelet_dominant_band_switch_count": int(switch_count),
                "tfa_wavelet_dominant_band_switch_rate": float(switch_rate),
                "tfa_wavelet_dominant_band_top": max(dominant_counts, key=dominant_counts.get) if dominant_counts else "",
                "tfa_wavelet_detail_entropy_mean": float(pd.to_numeric(df_case["tfa_wavelet_detail_entropy"], errors="coerce").mean()),
                "tfa_wavelet_detail_entropy_max": float(pd.to_numeric(df_case["tfa_wavelet_detail_entropy"], errors="coerce").max()),
                "tfa_wavelet_detail_dominant_band_unique_count": int(len(detail_band_counts)),
                "tfa_wavelet_detail_dominant_band_entropy": detail_band_entropy,
                "tfa_wavelet_detail_dominant_band_switch_count": int(detail_switch_count),
                "tfa_wavelet_detail_dominant_band_switch_rate": float(detail_switch_rate),
                "tfa_wavelet_detail_dominant_band_top": max(detail_band_counts, key=detail_band_counts.get) if detail_band_counts else "",
            }
        )

    return window_df, pd.DataFrame(case_rows).sort_values("dataset_key").reset_index(drop=True)


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
    primary_overlap_df = _select_primary_transition_matches(overlap_master_df)
    transition_sequence_window_df, transition_sequence_case_df, transition_sequence_edge_df = _summarize_transition_local_sequence(
        primary_overlap_df,
        summary_df,
    )
    transition_tfa_window_df, transition_tfa_case_df = _summarize_transition_local_wavelet(
        primary_overlap_df,
        summary_df,
    )
    if not case_transition_summary_df.empty and not transition_sequence_case_df.empty:
        case_transition_summary_df = case_transition_summary_df.merge(
            transition_sequence_case_df,
            on="dataset_key",
            how="left",
        )
    if not case_transition_summary_df.empty and not transition_tfa_case_df.empty:
        case_transition_summary_df = case_transition_summary_df.merge(
            transition_tfa_case_df,
            on="dataset_key",
            how="left",
        )
    case_transition_csv_path = output_root / "transition_overlap_case_summary.csv"
    method_transition_csv_path = output_root / "transition_overlap_method_summary.csv"
    transition_sequence_window_csv_path = output_root / "transition_local_sequence_windows.csv"
    transition_sequence_case_csv_path = output_root / "transition_local_sequence_case_summary.csv"
    transition_sequence_edge_csv_path = output_root / "transition_local_sequence_edges.csv"
    transition_tfa_window_csv_path = output_root / "transition_local_wavelet_windows.csv"
    transition_tfa_case_csv_path = output_root / "transition_local_wavelet_case_summary.csv"
    if not case_transition_summary_df.empty:
        case_transition_summary_df.to_csv(case_transition_csv_path, index=False, encoding="utf-8-sig")
    if not method_transition_summary_df.empty:
        method_transition_summary_df.to_csv(method_transition_csv_path, index=False, encoding="utf-8-sig")
    if not transition_sequence_window_df.empty:
        transition_sequence_window_df.to_csv(transition_sequence_window_csv_path, index=False, encoding="utf-8-sig")
    if not transition_sequence_case_df.empty:
        transition_sequence_case_df.to_csv(transition_sequence_case_csv_path, index=False, encoding="utf-8-sig")
    if not transition_sequence_edge_df.empty:
        transition_sequence_edge_df.to_csv(transition_sequence_edge_csv_path, index=False, encoding="utf-8-sig")
    if not transition_tfa_window_df.empty:
        transition_tfa_window_df.to_csv(transition_tfa_window_csv_path, index=False, encoding="utf-8-sig")
    if not transition_tfa_case_df.empty:
        transition_tfa_case_df.to_csv(transition_tfa_case_csv_path, index=False, encoding="utf-8-sig")

    transition_method_mix_png = _plot_transition_method_mix(
        case_transition_summary_df,
        out_png=output_root / "transition_method_mix.png",
    )
    transition_offset_score_png = _plot_transition_offset_vs_scores(
        case_transition_summary_df,
        out_png=output_root / "transition_offset_vs_scores.png",
    )
    transition_mix_stability_png = _plot_transition_mix_stability(
        case_transition_summary_df,
        out_png=output_root / "transition_mix_stability.png",
    )
    transition_regime_sequence_png = _plot_transition_regime_sequence(
        transition_sequence_window_df,
        case_transition_summary_df,
        out_png=output_root / "transition_regime_sequence.png",
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
        "transition_local_sequence_windows_csv": str(transition_sequence_window_csv_path) if not transition_sequence_window_df.empty else "",
        "transition_local_sequence_case_summary_csv": str(transition_sequence_case_csv_path) if not transition_sequence_case_df.empty else "",
        "transition_local_sequence_edges_csv": str(transition_sequence_edge_csv_path) if not transition_sequence_edge_df.empty else "",
        "transition_local_wavelet_windows_csv": str(transition_tfa_window_csv_path) if not transition_tfa_window_df.empty else "",
        "transition_local_wavelet_case_summary_csv": str(transition_tfa_case_csv_path) if not transition_tfa_case_df.empty else "",
        "transition_method_mix_png": str(transition_method_mix_png) if transition_method_mix_png is not None else "",
        "transition_offset_vs_scores_png": str(transition_offset_score_png) if transition_offset_score_png is not None else "",
        "transition_mix_stability_png": str(transition_mix_stability_png) if transition_mix_stability_png is not None else "",
        "transition_regime_sequence_png": str(transition_regime_sequence_png) if transition_regime_sequence_png is not None else "",
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
