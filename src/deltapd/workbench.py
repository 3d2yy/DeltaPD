from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from flask import abort, send_file

from deltapd.semaphore import build_semaphore_df, ingestion_audit_confidence
from deltapd.workbench_jobs import (
    DEFAULT_PD_CHANNELS,
    create_sensitivity_job,
    discover_pd_cases,
    latest_job_status_text,
    list_workbench_jobs,
    run_pd_selection,
    run_vna_selection,
)


THEME = {
    "paper": "#f3ede2",
    "card": "#fbf8f1",
    "ink": "#10263f",
    "muted": "#5b6977",
    "line": "#d2c4af",
    "accent": "#b15c2b",
    "accent_soft": "#e2af87",
    "sea": "#1f6b72",
    "berry": "#7a4d59",
    "rose": "#c88462",
}

CHANNEL_METADATA = {
    "CH3": {
        "role": "Thesis",
        "summary": "Canonical thesis channel and primary narrative.",
        "detail": "Use CH3 for the final thesis figures, semaphore calibration and conclusions.",
        "tone": "thesis",
    },
    "CH2": {
        "role": "Gemela support",
        "summary": "Auxiliary support channel for the gemela folders.",
        "detail": "CH2 is useful as algorithm validation when it comes from the gemela antenna path, but it should not replace CH3 as the thesis axis.",
        "tone": "auxiliary",
    },
    "CH4": {
        "role": "Experimental",
        "summary": "Algorithm sandbox outside the thesis narrative.",
        "detail": "Use CH4 to stress-test methods and robustness, not as the main thesis channel.",
        "tone": "experimental",
    },
}

CASE_IMAGE_PRIORITY = [
    "01_raw_with_detections.png",
    "02a_delta_t_series_lineal.png",
    "05_rolling_delta_t_stats.png",
    "06_ewma_cusum_robusto.png",
    "08_blind_prpd_50hz.png",
    "09_classification_trend.png",
]

STUDY_IMAGE_PRIORITY = [
    "blind_prpd_transition_map.png",
]

COMPARATIVE_IMAGE_PRIORITY = [
    "comparative_window_counts.png",
    "comparative_block_ablation.png",
    "comparative_semicycle_ablation.png",
    "comparative_transition_case_heatmap.png",
    "comparative_transition_case_scatter.png",
    "comparative_type3_boxplots.png",
    "comparative_variant2_boxplots.png",
    "comparative_dataset6_heatmap.png",
]

VNA_IMAGE_PRIORITY = [
    "vna_overview.png",
    "vna_comparative_overlay.png",
]

STATE_ALARM_IMAGE_PRIORITY = [
    "transition_method_mix.png",
    "transition_regime_sequence.png",
    "transition_mix_stability.png",
    "transition_offset_vs_scores.png",
]

SENSITIVITY_IMAGE_PRIORITY = [
    "semaphore_sensitivity_heatmap.png",
    "semaphore_sensitivity_stability.png",
]

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _blank_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": THEME["muted"]},
    )
    fig.update_layout(
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["card"],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _figure_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["card"],
        font={"family": "Georgia, Times New Roman, serif", "color": THEME["ink"]},
        title_font={"size": 18},
        legend={"bgcolor": "rgba(255,255,255,0.7)"},
        margin={"l": 50, "r": 20, "t": 60, "b": 45},
    )
    fig.update_xaxes(gridcolor="rgba(22,50,79,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(22,50,79,0.08)", zeroline=False)
    return fig


def _safe_rel_url(path: Path, repo_root: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    return "/artifacts/" + "/".join(rel.parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _normalize_channel_name(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in DEFAULT_PD_CHANNELS else ""


def _discover_channel_root_map(
    repo_root: Path,
    *,
    prefix: str,
    override_root: Path | None = None,
) -> dict[str, str]:
    channel_map: dict[str, str] = {}
    if override_root is not None:
        name = override_root.name.lower()
        for channel in DEFAULT_PD_CHANNELS:
            if name.endswith(channel.lower()):
                channel_map[channel] = str(override_root)
                break
    for channel in DEFAULT_PD_CHANNELS:
        candidate = repo_root / "outputs" / f"{prefix}_{channel.lower()}"
        if candidate.exists() and channel not in channel_map:
            channel_map[channel] = str(candidate)
    return channel_map


def _active_channel_value(channel_maps: dict[str, dict[str, str]] | None, requested: str | None) -> str:
    available: list[str] = []
    maps = channel_maps or {}
    for key in ["state_alarm_roots", "comparative_roots"]:
        for channel in maps.get(key, {}):
            if channel not in available:
                available.append(channel)
    for channel in maps.get("sensitivity_roots", {}):
        if channel not in available:
            available.append(channel)
    normalized = _normalize_channel_name(requested)
    if normalized and normalized in available:
        return normalized
    if "CH3" in available:
        return "CH3"
    return available[0] if available else "CH3"


def _ordered_existing_files(root: Path, names: list[str]) -> list[Path]:
    ordered = []
    seen: set[Path] = set()
    for name in names:
        candidate = root / name
        if candidate.exists():
            ordered.append(candidate)
            seen.add(candidate.resolve())
    for candidate in sorted(root.glob("*.png")):
        resolved = candidate.resolve()
        if resolved not in seen:
            ordered.append(candidate)
    return ordered


def _channel_meta(channel: str | None) -> dict[str, str]:
    normalized = _normalize_channel_name(channel)
    meta = CHANNEL_METADATA.get(normalized, {})
    return {
        "channel": normalized or "CH3",
        "role": str(meta.get("role", "Reference")),
        "summary": str(meta.get("summary", "Loaded workbench channel.")),
        "detail": str(meta.get("detail", "This view reads artifacts already generated by the pipeline.")),
        "tone": str(meta.get("tone", "thesis")),
    }


def _channel_dropdown_options(channels: list[str]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for channel in channels:
        meta = _channel_meta(channel)
        options.append({"label": f"{channel} · {meta['role']}", "value": channel})
    return options


def _channel_scope_banner(channel: str | None) -> html.Div:
    meta = _channel_meta(channel)
    return html.Div(
        [
            html.Div(
                [
                    html.Span(meta["channel"], className="channel-code"),
                    html.Span(meta["role"], className=f"channel-badge channel-badge-{meta['tone']}"),
                ],
                className="channel-banner-head",
            ),
            html.Div(meta["summary"], className="channel-summary"),
            html.Div(meta["detail"], className="channel-note"),
        ],
        className="channel-banner",
    )


def _thesis_scope_strip() -> html.Div:
    cards = []
    for channel in ["CH3", "CH2", "CH4"]:
        meta = _channel_meta(channel)
        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(channel, className="channel-code"),
                            html.Span(meta["role"], className=f"channel-badge channel-badge-{meta['tone']}"),
                        ],
                        className="channel-banner-head",
                    ),
                    html.Div(meta["summary"], className="channel-summary"),
                ],
                className="scope-card",
            )
        )
    return html.Div(cards, className="scope-strip")


def _discover_state_alarm_batch(batch_root: Path, repo_root: Path) -> dict[str, Any]:
    manifest_path = batch_root / "state_alarm_batch_manifest.json"
    summary_csv_path = batch_root / "state_alarm_batch_summary.csv"
    summary_md_path = batch_root / "state_alarm_batch_summary.md"
    if not manifest_path.exists() or not summary_csv_path.exists():
        return {"available": False, "case_keys": [], "cases": {}, "summary_rows": []}

    manifest = _read_json(manifest_path)
    summary_df = pd.read_csv(summary_csv_path)
    transition_case_path = batch_root / "transition_overlap_case_summary.csv"
    if transition_case_path.exists():
        transition_df = pd.read_csv(transition_case_path)
        if "dataset_key" in transition_df.columns:
            transition_df = transition_df.drop(columns=[col for col in ["discharge_type", "variant"] if col in transition_df.columns])
            summary_df = summary_df.merge(transition_df, on="dataset_key", how="left")
    summary_df = summary_df.fillna("")
    summary_rows = summary_df.to_dict("records")
    root_images = [
        {
            "name": path.name,
            "title": path.stem.replace("_", " "),
            "url": _safe_rel_url(path, repo_root),
        }
        for path in _ordered_existing_files(batch_root, STATE_ALARM_IMAGE_PRIORITY)
    ]
    artifacts = []
    for path in sorted(batch_root.glob("*")):
        if path.suffix.lower() in {".png", ".csv", ".json", ".md", ".pdf"}:
            artifacts.append({"name": path.name, "url": _safe_rel_url(path, repo_root)})
    cases: dict[str, Any] = {}

    for case in manifest.get("cases", []):
        dataset_key = str(case["dataset_key"])
        material_dir = Path(case["material_output_dir"])
        study_dir = Path(case["study_output_dir"])
        material_manifest = _read_json(material_dir / "run_manifest.json")
        material_images = [
            {
                "name": path.name,
                "title": path.stem.replace("_", " "),
                "url": _safe_rel_url(path, repo_root),
            }
            for path in _ordered_existing_files(material_dir, CASE_IMAGE_PRIORITY)
        ]
        study_images = [
            {
                "name": path.name,
                "title": f"Study: {path.stem.replace('_', ' ')}",
                "url": _safe_rel_url(path, repo_root),
            }
            for path in _ordered_existing_files(study_dir, STUDY_IMAGE_PRIORITY)
        ]
        artifact_paths = []
        for path in sorted(study_dir.glob("*")):
            if path.suffix.lower() not in {".pdf", ".csv", ".json", ".md", ".png"}:
                continue
            artifact_paths.append(
                {
                    "name": path.name,
                    "url": _safe_rel_url(path, repo_root),
                }
            )
        case_row = next((row for row in summary_rows if row["dataset_key"] == dataset_key), {})
        ingestion_audit = dict(material_manifest.get("ingestion_audit", {}))
        ingestion_confidence, ingestion_flags = ingestion_audit_confidence(ingestion_audit)
        if case_row:
            case_row["ingestion_confidence"] = float(ingestion_confidence)
            case_row["ingestion_flags"] = ", ".join(ingestion_flags)
            case_row["ingestion_loader_mode"] = str(ingestion_audit.get("loader_mode", ""))
            case_row["ingestion_fs_source"] = str(ingestion_audit.get("fs_source", ""))
            case_row["ingestion_signal_column"] = str(ingestion_audit.get("signal_column_label", ""))
        cases[dataset_key] = {
            "meta": case_row,
            "folder": case.get("folder", ""),
            "study_report": _read_text(study_dir / "study_report.md"),
            "material_images": material_images,
            "case_images": material_images + study_images,
            "artifacts": artifact_paths,
            "pdf_url": _safe_rel_url(Path(case["pdf_path"]), repo_root) if case.get("pdf_path") else "",
            "blind_prpd": material_manifest.get("blind_prpd", {}),
            "ingestion_audit": ingestion_audit,
        }

    return {
        "available": True,
        "manifest": manifest,
        "summary_rows": summary_rows,
        "transition_summary_rows": (
            pd.read_csv(transition_case_path).fillna("").to_dict("records") if transition_case_path.exists() else []
        ),
        "summary_markdown": _read_text(summary_md_path),
        "images": root_images,
        "artifacts": artifacts,
        "case_keys": [case["dataset_key"] for case in manifest.get("cases", [])],
        "cases": cases,
    }


def _discover_comparative_study(comparative_root: Path, repo_root: Path) -> dict[str, Any]:
    summary_md_path = comparative_root / "comparative_summary.md"
    if not summary_md_path.exists():
        return {"available": False, "images": []}

    pdf_candidates = sorted(comparative_root.glob("*.pdf"))
    pdf_path = pdf_candidates[0] if pdf_candidates else comparative_root / "comparative_descriptor_report.pdf"
    blind_metrics_path = comparative_root / "blind_prpd_metrics.csv"
    images = [
        {
            "name": path.name,
            "title": path.stem.replace("_", " "),
            "url": _safe_rel_url(path, repo_root),
        }
        for path in _ordered_existing_files(comparative_root, COMPARATIVE_IMAGE_PRIORITY)
    ]
    return {
        "available": True,
        "summary_markdown": _read_text(summary_md_path),
        "recommendations": _read_json(comparative_root / "study_recommendations.json"),
        "blind_metrics_rows": (
            pd.read_csv(blind_metrics_path).fillna("").to_dict("records")
            if blind_metrics_path.exists()
            else []
        ),
        "pdf_url": _safe_rel_url(pdf_path, repo_root) if pdf_path.exists() else "",
        "images": images,
    }


def _discover_semaphore_sensitivity(sensitivity_root: Path | None, repo_root: Path) -> dict[str, Any]:
    if sensitivity_root is None:
        return {"available": False, "images": [], "scenario_rows": [], "case_rows": []}
    sensitivity_root = Path(sensitivity_root)
    manifest_path = sensitivity_root / "semaphore_sensitivity_manifest.json"
    summary_md_path = sensitivity_root / "semaphore_sensitivity_summary.md"
    if not manifest_path.exists() or not summary_md_path.exists():
        return {"available": False, "images": [], "scenario_rows": [], "case_rows": []}

    manifest = _read_json(manifest_path)
    scenario_csv_path = sensitivity_root / "semaphore_sensitivity_scenario_summary.csv"
    case_csv_path = sensitivity_root / "semaphore_sensitivity_case_summary.csv"
    repeatability_path = sensitivity_root / "semaphore_sensitivity_repeatability.json"
    images = [
        {
            "name": path.name,
            "title": path.stem.replace("_", " "),
            "url": _safe_rel_url(path, repo_root),
        }
        for path in _ordered_existing_files(sensitivity_root, SENSITIVITY_IMAGE_PRIORITY)
    ]
    artifacts = []
    for path in sorted(sensitivity_root.glob("*")):
        if path.suffix.lower() in {".png", ".csv", ".json", ".md"}:
            artifacts.append({"name": path.name, "url": _safe_rel_url(path, repo_root)})
    return {
        "available": True,
        "manifest": manifest,
        "summary_markdown": _read_text(summary_md_path),
        "repeatability": _read_json(repeatability_path),
        "scenario_rows": pd.read_csv(scenario_csv_path).fillna("").to_dict("records") if scenario_csv_path.exists() else [],
        "case_rows": pd.read_csv(case_csv_path).fillna("").to_dict("records") if case_csv_path.exists() else [],
        "images": images,
        "artifacts": artifacts,
    }


def _discover_vna_outputs(vna_root: Path | None, repo_root: Path) -> dict[str, Any]:
    if vna_root is None:
        return {"available": False, "images": [], "summary_rows": []}
    vna_root = Path(vna_root)
    manifest_path = vna_root / "vna_manifest.json"
    summary_csv_path = vna_root / "vna_summary.csv"
    summary_md_path = vna_root / "vna_summary.md"
    if not manifest_path.exists() or not summary_csv_path.exists():
        return {"available": False, "images": [], "summary_rows": []}

    manifest = _read_json(manifest_path)
    summary_df = pd.read_csv(summary_csv_path).fillna("")
    images = [
        {
            "name": path.name,
            "title": path.stem.replace("_", " "),
            "url": _safe_rel_url(path, repo_root),
        }
        for path in _ordered_existing_files(vna_root, VNA_IMAGE_PRIORITY)
    ]
    seen_urls = {item["url"] for item in images}
    for item in manifest.get("items", []):
        item_dir = Path(item.get("output_dir", ""))
        if not item_dir.exists():
            continue
        for path in _ordered_existing_files(item_dir, ["vna_overview.png"]):
            url = _safe_rel_url(path, repo_root)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            images.append(
                {
                    "name": path.name,
                    "title": f"{path.stem.replace('_', ' ')} - {Path(item.get('source_file', '')).name}",
                    "url": url,
                }
            )
    artifacts = []
    for path in sorted(vna_root.glob("*")):
        if path.suffix.lower() in {".png", ".csv", ".json", ".md", ".pdf"}:
            artifacts.append({"name": path.name, "url": _safe_rel_url(path, repo_root)})
    for item in manifest.get("items", []):
        item_dir = Path(item.get("output_dir", ""))
        if not item_dir.exists():
            continue
        for path in sorted(item_dir.glob("*")):
            if path.suffix.lower() in {".png", ".csv", ".json", ".md", ".pdf"}:
                artifacts.append({"name": f"{item_dir.name}/{path.name}", "url": _safe_rel_url(path, repo_root)})
    return {
        "available": True,
        "manifest": manifest,
        "summary_rows": summary_df.to_dict("records"),
        "summary_markdown": _read_text(summary_md_path),
        "pdf_url": (
            _safe_rel_url(Path(manifest["pdf_path"]), repo_root)
            if manifest.get("pdf_path") and Path(manifest["pdf_path"]).exists()
            else ""
        ),
        "images": images,
        "artifacts": artifacts,
    }


def load_workbench_data(
    *,
    repo_root: Path | None = None,
    state_alarm_root: Path | None = None,
    comparative_root: Path | None = None,
    state_alarm_roots: dict[str, str] | None = None,
    comparative_roots: dict[str, str] | None = None,
    sensitivity_root: Path | None = None,
    sensitivity_roots: dict[str, str] | None = None,
    active_channel: str | None = None,
    vna_root: Path | None = None,
    pd_base_dir: Path | None = None,
    extra_pd_input: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    pd_base_dir = pd_base_dir or Path("E:/Carpeta definitiva de Tesis/programas")
    effective_state_alarm_roots = state_alarm_roots or _discover_channel_root_map(
        repo_root,
        prefix="state_alarm",
        override_root=state_alarm_root,
    )
    effective_comparative_roots = comparative_roots or _discover_channel_root_map(
        repo_root,
        prefix="comparative",
        override_root=comparative_root,
    )
    effective_sensitivity_roots = sensitivity_roots or _discover_channel_root_map(
        repo_root,
        prefix="semaphore_sensitivity",
        override_root=sensitivity_root,
    )
    channel_maps = {
        "state_alarm_roots": effective_state_alarm_roots,
        "comparative_roots": effective_comparative_roots,
        "sensitivity_roots": effective_sensitivity_roots,
    }
    current_channel = _active_channel_value(channel_maps, active_channel)
    state_alarm_root_active = effective_state_alarm_roots.get(current_channel)
    comparative_root_active = effective_comparative_roots.get(current_channel)
    sensitivity_root_active = effective_sensitivity_roots.get(current_channel)
    state_alarm = (
        _discover_state_alarm_batch(Path(state_alarm_root_active), repo_root)
        if state_alarm_root_active
        else {"available": False, "case_keys": [], "cases": {}, "summary_rows": []}
    )
    comparative = (
        _discover_comparative_study(Path(comparative_root_active), repo_root)
        if comparative_root_active
        else {"available": False, "images": []}
    )
    sensitivity = _discover_semaphore_sensitivity(Path(sensitivity_root_active), repo_root) if sensitivity_root_active else {
        "available": False,
        "images": [],
        "scenario_rows": [],
        "case_rows": [],
    }
    vna = _discover_vna_outputs(vna_root, repo_root)
    jobs = list_workbench_jobs(repo_root)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "pd_base_dir": str(pd_base_dir),
        "active_channel": current_channel,
        "available_channels": sorted(
            {
                channel
                for channel in list(effective_state_alarm_roots) + list(effective_comparative_roots) + list(effective_sensitivity_roots)
            }
            or DEFAULT_PD_CHANNELS
        ),
        "state_alarm_roots": effective_state_alarm_roots,
        "comparative_roots": effective_comparative_roots,
        "sensitivity_roots": effective_sensitivity_roots,
        "pd_cases": discover_pd_cases(pd_base_dir, raw_input=extra_pd_input),
        "jobs": jobs,
        "state_alarm": state_alarm,
        "comparative": comparative,
        "sensitivity": sensitivity,
        "vna": vna,
    }


def _score_figure(summary_rows: list[dict[str, Any]], score_column: str, title: str, color_column: str) -> go.Figure:
    if not summary_rows:
        return _blank_figure("No state/alarm batch results were found.")
    df = pd.DataFrame(summary_rows)
    if score_column not in df.columns:
        return _blank_figure("Score column missing.")
    fig = px.bar(
        df,
        x="dataset_key",
        y=score_column,
        color=color_column if color_column in df.columns else None,
        text=score_column,
        color_discrete_sequence=[THEME["sea"], THEME["accent"], THEME["berry"], THEME["rose"]],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, max(1.0, float(pd.to_numeric(df[score_column], errors="coerce").max()) + 0.08)])
    return _figure_layout(fig, title)


def _feature_count_figure(feature_counts: dict[str, int], title: str, color: str) -> go.Figure:
    if not feature_counts:
        return _blank_figure("No recurrent descriptor counts yet.")
    df = pd.DataFrame({"feature": list(feature_counts.keys()), "count": list(feature_counts.values())})
    fig = px.bar(df, x="count", y="feature", orientation="h", text="count")
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="total ascending")
    return _figure_layout(fig, title)


def _semaphore_figure(summary_rows: list[dict[str, Any]], channel_label: str) -> go.Figure:
    df = build_semaphore_df(summary_rows)
    if df.empty:
        return _blank_figure("No transition metrics available for the exploratory semaphore.")
    palette = {
        "green": "#5d8f52",
        "yellow": "#c26d2d",
        "red": "#9b2226",
        "gray": "#a0a7b0",
    }
    fig = px.bar(
        df,
        x="dataset_key",
        y="semaphore_risk_score",
        color="semaphore_band",
        text="semaphore_band",
        color_discrete_map=palette,
        hover_data=["semaphore_top_drivers", "semaphore_confidence_score", "ingestion_confidence", "ingestion_fs_source"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0.0, 1.02], title="Exploratory risk score")
    return _figure_layout(fig, f"Exploratory semaphore by dataset ({channel_label})")


def _semaphore_cards(summary_rows: list[dict[str, Any]]) -> html.Div:
    df = build_semaphore_df(summary_rows)
    if df.empty:
        return html.Div("No transition metrics available for the exploratory semaphore.", className="empty-note")
    counts = df["semaphore_band"].value_counts().to_dict()
    mean_risk = pd.to_numeric(df["semaphore_risk_score"], errors="coerce").mean()
    mean_confidence = pd.to_numeric(df["semaphore_confidence_score"], errors="coerce").mean()
    mean_ingestion = pd.to_numeric(df.get("ingestion_confidence"), errors="coerce").mean()
    return html.Div(
        [
            _metric_card("Gray", str(int(counts.get("gray", 0))), "insufficient or doubtful ingestion"),
            _metric_card("Red", str(int(counts.get("red", 0))), "higher exploratory local-mixture risk"),
            _metric_card("Yellow", str(int(counts.get("yellow", 0))), "intermediate"),
            _metric_card("Green", str(int(counts.get("green", 0))), "more stable local regime"),
            _metric_card("Mean risk", f"{mean_risk:.3f}" if pd.notna(mean_risk) else "n/a"),
            _metric_card("Mean confidence", f"{mean_confidence:.3f}" if pd.notna(mean_confidence) else "n/a"),
            _metric_card("Ingestion conf.", f"{mean_ingestion:.3f}" if pd.notna(mean_ingestion) else "n/a"),
        ],
        className="metric-grid",
    )


def _metric_card(title: str, value: str, subtitle: str = "") -> html.Div:
    children = [html.Div(title, className="metric-title"), html.Div(value, className="metric-value")]
    if subtitle:
        children.append(html.Div(subtitle, className="metric-subtitle"))
    return html.Div(children, className="metric-card")


def _artifact_link(name: str, url: str) -> html.A:
    return html.A(name, href=url, target="_blank", className="artifact-link")


def _image_gallery(images: list[dict[str, str]]) -> html.Div:
    if not images:
        return html.Div("No images found for this case.", className="empty-note")
    cards = []
    for image in images:
        cards.append(
            html.Div(
                [
                    html.Div(image["title"], className="image-title"),
                    html.Img(src=image["url"], className="image-preview"),
                    _artifact_link("Open image", image["url"]),
                ],
                className="image-card",
            )
        )
    return html.Div(cards, className="image-grid")


def _case_detail_layout(case: dict[str, Any]) -> html.Div:
    if not case:
        return html.Div("Select a case to review.", className="empty-note")

    meta = case.get("meta", {})
    blind = dict(case.get("blind_prpd", {}))
    blind_selected_method = str(
        meta.get("blind_selected_method", "") or blind.get("selected_method", blind.get("method", ""))
    )
    blind_freq = _safe_float(meta.get("blind_freq_hz", blind.get("calibrated_freq_hz")))
    blind_coherence = _safe_float(meta.get("blind_coherence", blind.get("coherence")))
    blind_sharpness = _safe_float(meta.get("blind_sharpness", blind.get("sharpness")))
    blind_common_confidence = _safe_float(
        meta.get("blind_common_axial_confidence", blind.get("common_axial_confidence"))
    )
    blind_peak_offset = _safe_float(
        meta.get("blind_common_axial_peak_offset_hz", blind.get("common_axial_peak_offset_hz"))
    )
    blind_bootstrap_std = _safe_float(
        meta.get("blind_bootstrap_freq_std_hz", blind.get("bootstrap_freq_std_hz"))
    )
    blind_bootstrap_agreement = _safe_float(
        meta.get("blind_bootstrap_method_agreement", blind.get("bootstrap_method_agreement"))
    )
    blind_local_std = _safe_float(
        meta.get("blind_local_freq_std_hz", blind.get("local_freq_std_hz"))
    )
    blind_local_agreement = _safe_float(
        meta.get("blind_local_method_agreement", blind.get("local_method_agreement"))
    )
    metrics = html.Div(
        [
            _metric_card("Dataset", str(meta.get("dataset_key", "")), str(meta.get("folder", ""))),
            _metric_card("Type", str(meta.get("discharge_type", "")), str(meta.get("variant", ""))),
            _metric_card("Duration", f"{_safe_float(meta.get('duration_s')):.3f} s"),
            _metric_card("Events", str(meta.get("total_events", ""))),
            _metric_card("Blind Method", blind_selected_method or "n/a", str(blind.get("requested_method", ""))),
            _metric_card("Blind Freq", f"{blind_freq:.6f} Hz" if pd.notna(blind_freq) else "n/a"),
            _metric_card("Coherence", f"{blind_coherence:.4f}" if pd.notna(blind_coherence) else "n/a"),
            _metric_card("Blind Conf.", f"{blind_common_confidence:.4f}" if pd.notna(blind_common_confidence) else "n/a"),
            _metric_card("Peak Offset", f"{blind_peak_offset:.5f} Hz" if pd.notna(blind_peak_offset) else "n/a"),
            _metric_card("Boot Std", f"{blind_bootstrap_std:.5f} Hz" if pd.notna(blind_bootstrap_std) else "n/a"),
            _metric_card("Boot Agree.", f"{blind_bootstrap_agreement:.2f}" if pd.notna(blind_bootstrap_agreement) else "n/a"),
            _metric_card("Local Std", f"{blind_local_std:.5f} Hz" if pd.notna(blind_local_std) else "n/a"),
            _metric_card("Local Agree.", f"{blind_local_agreement:.2f}" if pd.notna(blind_local_agreement) else "n/a"),
            _metric_card("Sharpness", f"{blind_sharpness:.4f}" if pd.notna(blind_sharpness) else "n/a", "within-method only"),
            _metric_card("State", f"{_safe_float(meta.get('state_primary_score')):.3f}", str(meta.get("state_features", ""))),
            _metric_card("Alarm", f"{_safe_float(meta.get('alarm_primary_score')):.3f}", str(meta.get("alarm_features", ""))),
        ],
        className="metric-grid",
    )
    semaphore_df = build_semaphore_df([meta])
    semaphore_note = html.Div()
    if not semaphore_df.empty:
        semaphore_row = semaphore_df.iloc[0]
        semaphore_note = html.Div(
            [
                html.Div(
                    f"Exploratory semaphore: {str(semaphore_row.get('semaphore_band', '')).upper()}",
                    className="panel-title",
                    style={"color": str(semaphore_row.get("semaphore_color", THEME["ink"]))},
                ),
                html.Div(
                    f"risk={_safe_float(semaphore_row.get('semaphore_risk_score')):.3f}, "
                    f"confidence={_safe_float(semaphore_row.get('semaphore_confidence_score')):.3f}, "
                    f"evidence={_safe_float(semaphore_row.get('semaphore_transition_evidence')):.3f}",
                    className="lead-note",
                ),
                html.Div(
                    f"Drivers: {str(semaphore_row.get('semaphore_top_drivers', '')) or 'n/a'}",
                    className="control-note",
                ),
            ],
            className="panel",
        )

    transition = ""
    if meta.get("transition_feature", ""):
        transition = (
            f"Top transition: {_safe_float(meta.get('transition_start_s')):.3f} - "
            f"{_safe_float(meta.get('transition_end_s')):.3f} s "
            f"({meta.get('transition_feature', '')})"
        )

    artifact_links = [
        _artifact_link(item["name"], item["url"])
        for item in case.get("artifacts", [])
    ]
    if case.get("pdf_url"):
        artifact_links.insert(0, _artifact_link("Open PDF report", case["pdf_url"]))

    blind_lines = []
    if blind:
        blind_lines = [
            f"- requested_method: {blind.get('requested_method', blind.get('method', ''))}",
            f"- selected_method: {blind.get('selected_method', blind.get('method', ''))}",
            f"- calibrated_freq_hz: {_safe_float(blind.get('calibrated_freq_hz')):.6f}",
            f"- coherence: {_safe_float(blind.get('coherence')):.6f}",
            f"- axial_entropy_score: {_safe_float(blind.get('axial_entropy_score')):.6f}",
            f"- sharpness: {_safe_float(blind.get('sharpness')):.6f}",
            f"- half_height_width_hz: {_safe_float(blind.get('half_height_width_hz')):.6f}",
            f"- common_axial_confidence: {_safe_float(blind.get('common_axial_confidence')):.6f}",
            f"- common_axial_peak_offset_hz: {_safe_float(blind.get('common_axial_peak_offset_hz')):.6f}",
            f"- bootstrap_iterations: {int(blind.get('bootstrap_iterations', 0) or 0)}",
            f"- bootstrap_freq_std_hz: {_safe_float(blind.get('bootstrap_freq_std_hz')):.6f}",
            f"- bootstrap_ci_width_hz: {_safe_float(blind.get('bootstrap_ci_width_hz')):.6f}",
            f"- bootstrap_method_agreement: {_safe_float(blind.get('bootstrap_method_agreement')):.6f}",
            f"- local_window_count: {int(blind.get('local_window_count', 0) or 0)}",
            f"- local_freq_std_hz: {_safe_float(blind.get('local_freq_std_hz')):.6f}",
            f"- local_freq_span_hz: {_safe_float(blind.get('local_freq_span_hz')):.6f}",
            f"- local_method_agreement: {_safe_float(blind.get('local_method_agreement')):.6f}",
            f"- candidate_spread_hz: {_safe_float(blind.get('candidate_spread_hz')):.6f}",
            f"- winner_margin: {_safe_float(blind.get('winner_margin')):.6f}",
        ]

    ingestion_audit = dict(case.get("ingestion_audit", {}))
    ingestion_lines = []
    if ingestion_audit:
        ordered_keys = [
            "file_type",
            "loader_mode",
            "delimiter",
            "fs_hz",
            "fs_source",
            "sample_count",
            "final_sample_count",
            "numeric_row_count",
            "column_count",
            "time_column_label",
            "signal_column_label",
            "has_absolute_times",
            "used_segment_offsets",
            "metadata_keys_count",
            "preserve_amplitude",
            "nan_filtered_count",
        ]
        for key in ordered_keys:
            if key not in ingestion_audit:
                continue
            value = ingestion_audit.get(key)
            if isinstance(value, float):
                ingestion_lines.append(f"- {key}: {value:.6g}")
            else:
                ingestion_lines.append(f"- {key}: {value}")

    return html.Div(
        [
            metrics,
            semaphore_note,
            html.Div(transition, className="lead-note"),
            html.Div(artifact_links, className="artifact-list"),
            html.Div(
                [
                    html.Div("Ingestion audit", className="panel-title"),
                    dcc.Markdown(
                        "\n".join(ingestion_lines) if ingestion_lines else "_No ingestion audit found._",
                        className="markdown-panel",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Blind PRPD diagnostics", className="panel-title"),
                    dcc.Markdown("\n".join(blind_lines) if blind_lines else "_No blind PRPD diagnostics found._", className="markdown-panel"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Study narrative", className="panel-title"),
                    dcc.Markdown(case.get("study_report", "_No study report found._"), className="markdown-panel"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Case figures", className="panel-title"),
                    _image_gallery(case.get("case_images", case.get("material_images", []))),
                ],
                className="panel",
            ),
        ]
    )


def _comparative_layout(comparative: dict[str, Any]) -> html.Div:
    if not comparative.get("available"):
        return html.Div("Comparative study outputs not found.", className="empty-note")

    recommendations = comparative.get("recommendations", {})
    blind_metrics_rows = comparative.get("blind_metrics_rows", [])
    cards = []
    for task_name in ["type3", "dataset6", "variant2"]:
        task = recommendations.get(task_name, {})
        rec = task.get("recommendation", {})
        label = rec.get("primary_metric", "")
        score = rec.get("primary_score")
        cards.append(
            _metric_card(
                task_name,
                f"{_safe_float(score):.3f}" if score is not None else "n/a",
                ", ".join(rec.get("features", [])) if rec.get("features") else label,
            )
        )

    links = []
    if comparative.get("pdf_url"):
        links.append(_artifact_link("Open comparative PDF", comparative["pdf_url"]))

    blind_metrics_panel: Any = html.Div("No blind PRPD comparative metrics found.", className="empty-note")
    if blind_metrics_rows:
        blind_metrics_panel = dash_table.DataTable(
            data=blind_metrics_rows,
            columns=[
                {"name": "dataset_key", "id": "dataset_key"},
                {"name": "blind_prpd_method", "id": "blind_prpd_method"},
                {"name": "blind_prpd_selected_method", "id": "blind_prpd_selected_method"},
                {"name": "blind_freq_hz", "id": "blind_freq_hz"},
                {"name": "blind_prpd_coherence", "id": "blind_prpd_coherence"},
                {"name": "blind_prpd_common_axial_confidence", "id": "blind_prpd_common_axial_confidence"},
                {"name": "blind_prpd_common_axial_peak_offset_hz", "id": "blind_prpd_common_axial_peak_offset_hz"},
                {"name": "blind_prpd_bootstrap_freq_std_hz", "id": "blind_prpd_bootstrap_freq_std_hz"},
                {"name": "blind_prpd_bootstrap_method_agreement", "id": "blind_prpd_bootstrap_method_agreement"},
                {"name": "blind_prpd_local_freq_std_hz", "id": "blind_prpd_local_freq_std_hz"},
                {"name": "blind_prpd_local_method_agreement", "id": "blind_prpd_local_method_agreement"},
            ],
            page_size=min(8, max(len(blind_metrics_rows), 1)),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontFamily": "Georgia, serif", "fontSize": 13},
            style_header={"backgroundColor": "#efe4d0", "fontWeight": "bold"},
        )

    return html.Div(
        [
            html.Div(cards, className="metric-grid"),
            html.Div(links, className="artifact-list"),
            html.Div(
                [
                    html.Div("Blind PRPD calibration", className="panel-title"),
                    blind_metrics_panel,
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Comparative narrative", className="panel-title"),
                    dcc.Markdown(comparative.get("summary_markdown", ""), className="markdown-panel"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Comparative figures", className="panel-title"),
                    _image_gallery(comparative.get("images", [])),
                ],
                className="panel",
            ),
        ]
    )


def _sensitivity_layout(sensitivity: dict[str, Any]) -> html.Div:
    if not sensitivity.get("available"):
        return html.Div(
            "No CH3 semaphore sensitivity battery has been generated for the current workbench session.",
            className="empty-note",
        )

    scenario_rows = sensitivity.get("scenario_rows", [])
    case_rows = sensitivity.get("case_rows", [])
    repeatability = sensitivity.get("repeatability", {})
    cards = html.Div(
        [
            _metric_card("Scenarios", str(len(scenario_rows)), "one-factor CH3 sensitivity sweep"),
            _metric_card("Cases", str(len(case_rows)), "datasets included in the battery"),
            _metric_card(
                "Repeatability",
                "stable" if bool(repeatability.get("all_bands_match", False)) else "watch",
                f"max risk delta {float(repeatability.get('max_abs_risk_delta', float('nan'))):.4f}",
            ),
        ],
        className="metric-grid",
    )
    artifacts = [_artifact_link(item["name"], item["url"]) for item in sensitivity.get("artifacts", [])]
    return html.Div(
        [
            cards,
            html.Div(
                [
                    html.Div("Sensitivity narrative", className="panel-title"),
                    dcc.Markdown(sensitivity.get("summary_markdown", ""), className="markdown-panel"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Sensitivity figures", className="panel-title"),
                    _image_gallery(sensitivity.get("images", [])),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Scenario summary", className="panel-title"),
                    dash_table.DataTable(
                        data=scenario_rows,
                        columns=[{"name": key, "id": key} for key in list(scenario_rows[0].keys())] if scenario_rows else [],
                        page_size=12,
                        style_table={"overflowX": "auto"},
                        style_header={"backgroundColor": THEME["ink"], "color": "#ffffff", "fontWeight": "bold"},
                        style_cell={
                            "backgroundColor": THEME["card"],
                            "color": THEME["ink"],
                            "padding": "10px",
                            "border": f"1px solid {THEME['line']}",
                            "fontFamily": "Georgia, Times New Roman, serif",
                            "fontSize": "13px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Case stability summary", className="panel-title"),
                    dash_table.DataTable(
                        data=case_rows,
                        columns=[{"name": key, "id": key} for key in list(case_rows[0].keys())] if case_rows else [],
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_header={"backgroundColor": THEME["ink"], "color": "#ffffff", "fontWeight": "bold"},
                        style_cell={
                            "backgroundColor": THEME["card"],
                            "color": THEME["ink"],
                            "padding": "10px",
                            "border": f"1px solid {THEME['line']}",
                            "fontFamily": "Georgia, Times New Roman, serif",
                            "fontSize": "13px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ],
                className="panel",
            ),
            html.Div(artifacts, className="artifact-list") if artifacts else html.Div(),
        ]
    )


def _overview_layout(data: dict[str, Any]) -> html.Div:
    state_alarm = data["state_alarm"]
    sensitivity = data.get("sensitivity", {})
    summary_rows = state_alarm.get("summary_rows", [])
    manifest = state_alarm.get("manifest", {})
    channel_label = str(data.get("active_channel", "CH3"))
    state_counts = manifest.get("state_feature_counts", {})
    alarm_counts = manifest.get("alarm_feature_counts", {})
    summary_df = pd.DataFrame(summary_rows)
    semaphore_df = build_semaphore_df(summary_rows)
    mean_state = pd.to_numeric(summary_df.get("state_primary_score"), errors="coerce").mean() if not summary_df.empty else 0.0
    mean_alarm = pd.to_numeric(summary_df.get("alarm_primary_score"), errors="coerce").mean() if not summary_df.empty else 0.0

    channel_meta = _channel_meta(channel_label)
    cards = html.Div(
        [
            _metric_card("Channel", channel_label, channel_meta["role"]),
            _metric_card("Batch cases", str(len(summary_rows)), "loaded in this channel"),
            _metric_card("Mean state score", f"{mean_state:.3f}" if pd.notna(mean_state) else "n/a"),
            _metric_card("Mean alarm score", f"{mean_alarm:.3f}" if pd.notna(mean_alarm) else "n/a"),
            _metric_card("Updated", data.get("generated_at", "")),
        ],
        className="metric-grid",
    )
    artifact_links = [_artifact_link(item["name"], item["url"]) for item in state_alarm.get("artifacts", [])]

    columns = [
        {"name": "Dataset", "id": "dataset_key"},
        {"name": "Type", "id": "discharge_type"},
        {"name": "Variant", "id": "variant"},
        {"name": "State score", "id": "state_primary_score"},
        {"name": "Alarm score", "id": "alarm_primary_score"},
        {"name": "State features", "id": "state_features"},
        {"name": "Alarm features", "id": "alarm_features"},
    ]

    return html.Div(
        [
            _channel_scope_banner(channel_label),
            cards,
            html.Div(
                [
                    dcc.Graph(
                        figure=_score_figure(summary_rows, "state_primary_score", "Per-test state score", "discharge_type"),
                        className="graph-card",
                    ),
                    dcc.Graph(
                        figure=_score_figure(summary_rows, "alarm_primary_score", "Per-test alarm score", "variant"),
                        className="graph-card",
                    ),
                ],
                className="graph-grid",
            ),
            html.Div(
                [
                    dcc.Graph(
                        figure=_feature_count_figure(state_counts, "Recurrent state descriptors", THEME["sea"]),
                        className="graph-card",
                    ),
                    dcc.Graph(
                        figure=_feature_count_figure(alarm_counts, "Recurrent alarm descriptors", THEME["accent"]),
                        className="graph-card",
                    ),
                ],
                className="graph-grid",
            ),
            html.Div(artifact_links, className="artifact-list") if artifact_links else html.Div(),
            html.Div(
                [
                    html.Div("Exploratory semaphore", className="panel-title"),
                    html.Div(
                        "Relative channel-level risk proxy built from offset, local confidence stability, mixture entropy and sequential persistence. The CH3-calibrated rule now downweights entropy/switching when only a few transition windows are available.",
                        className="control-note",
                    ),
                    _semaphore_cards(summary_rows),
                    dcc.Graph(
                        figure=_semaphore_figure(summary_rows, channel_label),
                        className="graph-card",
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("CH3 sensitivity battery", className="panel-title"),
                    html.Div(
                        "One-factor CH3 sweep over k sigma, wavelet preprocessing, blind PRPD method and local window geometry. Use it to calibrate the semaphore, not to replace the thesis batch.",
                        className="control-note",
                    ),
                    (
                        html.Div(
                            [
                                html.Div(
                                    f"Scenarios: {len(sensitivity.get('scenario_rows', []))} | "
                                    f"Repeatability stable: {bool(sensitivity.get('repeatability', {}).get('all_bands_match', False))}",
                                    className="lead-note",
                                ),
                                _image_gallery(sensitivity.get("images", [])),
                            ]
                        )
                        if sensitivity.get("available")
                        else html.Div("No CH3 sensitivity battery loaded yet.", className="empty-note")
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Master summary", className="panel-title"),
                    dash_table.DataTable(
                        data=semaphore_df.to_dict("records") if not semaphore_df.empty else summary_rows,
                        columns=(
                            columns
                            + [
                                {"name": "Semaphore", "id": "semaphore_band"},
                                {"name": "Risk score", "id": "semaphore_risk_score"},
                                {"name": "Confidence", "id": "semaphore_confidence_score"},
                                {"name": "Evidence", "id": "semaphore_transition_evidence"},
                                {"name": "Drivers", "id": "semaphore_top_drivers"},
                            ]
                        ),
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": THEME["ink"],
                            "color": "#ffffff",
                            "fontWeight": "bold",
                            "border": f"1px solid {THEME['ink']}",
                        },
                        style_cell={
                            "backgroundColor": THEME["card"],
                            "color": THEME["ink"],
                            "padding": "10px",
                            "border": f"1px solid {THEME['line']}",
                            "fontFamily": "Georgia, Times New Roman, serif",
                            "fontSize": "13px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Batch figures", className="panel-title"),
                    _image_gallery(state_alarm.get("images", [])),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Batch narrative", className="panel-title"),
                    dcc.Markdown(state_alarm.get("summary_markdown", ""), className="markdown-panel"),
                ],
                className="panel",
            ),
        ]
    )


def _recent_jobs_panel(job_rows: list[dict[str, Any]]) -> html.Div:
    if not job_rows:
        return html.Div("No job has been launched from the interface yet.", className="lead-note")
    columns = [
        {"name": "job_id", "id": "job_id"},
        {"name": "status", "id": "status"},
        {"name": "progress", "id": "progress_label"},
        {"name": "message", "id": "message"},
        {"name": "stable_output_root", "id": "stable_output_root"},
    ]
    rows = []
    for row in job_rows[:8]:
        rows.append(
            {
                "job_id": str(row.get("job_id", "")),
                "status": str(row.get("status", "")),
                "progress_label": str(row.get("progress_label", "")),
                "message": str(row.get("message", "")),
                "stable_output_root": str(row.get("stable_output_root", "")),
            }
        )
    return dash_table.DataTable(
        data=rows,
        columns=columns,
        page_size=min(len(rows), 8),
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": THEME["ink"],
            "color": "#ffffff",
            "fontWeight": "bold",
            "border": f"1px solid {THEME['ink']}",
        },
        style_cell={
            "backgroundColor": THEME["card"],
            "color": THEME["ink"],
            "padding": "10px",
            "border": f"1px solid {THEME['line']}",
            "fontFamily": "Georgia, Times New Roman, serif",
            "fontSize": "12px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        },
    )


def _inputs_layout(data: dict[str, Any], job_status: str) -> html.Div:
    pd_options = []
    for case in data.get("pd_cases", []):
        channels = ", ".join(case.get("available_channels", []))
        label = f"{case['dataset_key']} - {case['folder']}"
        if channels:
            label = f"{label} [{channels}]"
        pd_options.append({"label": label, "value": case["dataset_key"]})
    channel_options = _channel_dropdown_options(list(DEFAULT_PD_CHANNELS))
    job_rows = list(data.get("jobs", []))

    return html.Div(
        [
            _thesis_scope_strip(),
            html.Div(
                [
                    html.Div("PD runner", className="panel-title"),
                    html.Div(
                        "Choose one channel, then select known datasets and optionally add custom PD CSV or folder paths. CH3 is the thesis default; CH2 is auxiliary for gemela support; CH4 stays experimental. Comparative runs only when labels are reliable; otherwise the workbench falls back to a state/alarm batch.",
                        className="control-note",
                    ),
                    html.Div("Channel selection", className="control-label"),
                    dcc.Dropdown(
                        id="pd-channel-dropdown",
                        options=channel_options,
                        value="CH3",
                        clearable=False,
                        placeholder="Select one channel",
                        className="case-dropdown",
                    ),
                    dcc.Dropdown(
                        id="pd-selection-dropdown",
                        options=pd_options,
                        value=[],
                        multi=True,
                        placeholder="Select one or more PD tests",
                        className="case-dropdown",
                    ),
                    html.Div("Detection threshold (k sigma)", className="control-label"),
                    dcc.Input(id="pd-k-sigma", type="number", value=5.0, step=0.1, className="numeric-input"),
                    dcc.Checklist(
                        id="pd-wavelet-toggle",
                        options=[{"label": "Use wavelet preprocessing", "value": "wavelet"}],
                        value=["wavelet"],
                        className="inline-checklist",
                    ),
                    html.Div("Additional PD file or folder paths", className="control-label"),
                    dcc.Textarea(
                        id="pd-extra-paths-textarea",
                        className="path-textarea",
                        placeholder="E:\\datasets\\new_case\\CH2.csv\nE:\\datasets\\folder_with_pd_cases",
                    ),
                    html.Button("Run PD study", id="run-pd-button", className="refresh-button"),
                    html.Button("Run CH3 sensitivity", id="run-sensitivity-button", className="refresh-button"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("VNA runner", className="panel-title"),
                    html.Div(
                        "Paste one file path per line. If you provide a folder, the workbench will search for VNA-like files inside it.",
                        className="control-note",
                    ),
                    dcc.Textarea(
                        id="vna-paths-textarea",
                        className="path-textarea",
                        placeholder="E:\\measurements\\antena_a.s1p\nE:\\measurements\\antena_b.s1p",
                    ),
                    html.Button("Run VNA analysis", id="run-vna-button", className="refresh-button"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("Latest job", className="panel-title"),
                    html.Div(job_status or "No job has been launched from the interface yet.", className="lead-note"),
                    html.Div("Recent jobs", className="control-label"),
                    _recent_jobs_panel(job_rows),
                ],
                className="panel",
            ),
        ]
    )


def _vna_layout(vna: dict[str, Any]) -> html.Div:
    if not vna.get("available"):
        return html.Div("No VNA outputs available yet. Use the Inputs tab to launch an analysis.", className="empty-note")

    summary_rows = vna.get("summary_rows", [])
    cards = []
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        cards = [
            _metric_card("Files", str(len(summary_rows))),
            _metric_card("Mode", str(vna.get("manifest", {}).get("mode", "")), ", ".join(vna.get("manifest", {}).get("modes_detected", []))),
            _metric_card("Freq min", f"{pd.to_numeric(df.get('freq_min_hz'), errors='coerce').min() / 1e9:.3f} GHz" if "freq_min_hz" in df else "n/a"),
            _metric_card("Freq max", f"{pd.to_numeric(df.get('freq_max_hz'), errors='coerce').max() / 1e9:.3f} GHz" if "freq_max_hz" in df else "n/a"),
        ]

    columns = [{"name": column, "id": column} for column in (summary_rows[0].keys() if summary_rows else [])]
    artifact_links = [_artifact_link(item["name"], item["url"]) for item in vna.get("artifacts", [])]
    if vna.get("pdf_url"):
        artifact_links.insert(0, _artifact_link("Open VNA PDF", vna["pdf_url"]))

    return html.Div(
        [
            html.Div(cards, className="metric-grid") if cards else html.Div(),
            html.Div(artifact_links, className="artifact-list"),
            html.Div(
                [
                    html.Div("VNA summary", className="panel-title"),
                    dash_table.DataTable(
                        data=summary_rows,
                        columns=columns,
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": THEME["ink"],
                            "color": "#ffffff",
                            "fontWeight": "bold",
                            "border": f"1px solid {THEME['ink']}",
                        },
                        style_cell={
                            "backgroundColor": THEME["card"],
                            "color": THEME["ink"],
                            "padding": "10px",
                            "border": f"1px solid {THEME['line']}",
                            "fontFamily": "Georgia, Times New Roman, serif",
                            "fontSize": "13px",
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                    ),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("VNA narrative", className="panel-title"),
                    dcc.Markdown(vna.get("summary_markdown", ""), className="markdown-panel"),
                ],
                className="panel",
            ),
            html.Div(
                [
                    html.Div("VNA figures", className="panel-title"),
                    _image_gallery(vna.get("images", [])),
                ],
                className="panel",
            ),
        ]
    )


def create_workbench_app(
    *,
    repo_root: Path | None = None,
    state_alarm_root: Path | None = None,
    comparative_root: Path | None = None,
    sensitivity_root: Path | None = None,
    vna_root: Path | None = None,
    pd_base_dir: Path | None = None,
) -> Dash:
    repo_root = repo_root or _repo_root()
    pd_base_dir = pd_base_dir or Path("E:/Carpeta definitiva de Tesis/programas")
    assets_folder = Path(__file__).resolve().parent / "assets"
    app = Dash(
        __name__,
        title="DeltaPD Workbench",
        suppress_callback_exceptions=True,
        assets_folder=str(assets_folder),
    )

    @app.server.route("/artifacts/<path:artifact_path>")
    def serve_artifact(artifact_path: str):
        candidate = (repo_root / artifact_path).resolve()
        allowed_roots = [(repo_root / "outputs").resolve(), (repo_root / "docs").resolve()]
        if not candidate.exists() or not any(str(candidate).startswith(str(root)) for root in allowed_roots):
            return abort(404)
        return send_file(candidate)

    initial_data = load_workbench_data(
        repo_root=repo_root,
        state_alarm_root=state_alarm_root,
        comparative_root=comparative_root,
        sensitivity_root=sensitivity_root,
        vna_root=vna_root,
        pd_base_dir=pd_base_dir,
    )
    initial_active_paths = {
        "state_alarm_roots": initial_data.get("state_alarm_roots", {}),
        "comparative_roots": initial_data.get("comparative_roots", {}),
        "sensitivity_roots": initial_data.get("sensitivity_roots", {}),
        "vna_root": str(vna_root) if vna_root is not None else "",
        "current_channel": initial_data.get("active_channel", "CH3"),
        "extra_pd_input": "",
    }
    case_options = [
        {"label": key, "value": key}
        for key in initial_data["state_alarm"].get("case_keys", [])
    ]
    default_case = case_options[0]["value"] if case_options else None
    channel_options = _channel_dropdown_options(list(initial_data.get("available_channels", DEFAULT_PD_CHANNELS)))
    initial_job_status = latest_job_status_text(list(initial_data.get("jobs", [])))

    app.layout = html.Div(
        [
            dcc.Store(id="workbench-store", data=initial_data),
            dcc.Store(id="active-paths-store", data=initial_active_paths),
            dcc.Store(id="job-store", data=initial_data.get("jobs", [])),
            dcc.Store(id="job-status-store", data=initial_job_status),
            dcc.Interval(id="job-poll-interval", interval=4000, n_intervals=0),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("DeltaPD Thesis Workbench", className="hero-title"),
                            html.Div(
                                "CH3-first thesis cockpit for state, alarm and comparative descriptor studies with explicit separation between thesis, gemela support and experimental channels.",
                                className="hero-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button("Refresh data", id="refresh-button", className="refresh-button"),
                            html.Div(id="refresh-status", className="status-pill"),
                        ],
                        className="hero-actions",
                    ),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("View channel", className="control-label"),
                            dcc.Dropdown(
                                id="analysis-channel-dropdown",
                                options=channel_options,
                                value=initial_data.get("active_channel", "CH3"),
                                clearable=False,
                                className="case-dropdown",
                            ),
                            html.Div("Case selector", className="control-label"),
                            dcc.Dropdown(
                                id="case-dropdown",
                                options=case_options,
                                value=default_case,
                                clearable=False,
                                className="case-dropdown",
                            ),
                            html.Div(
                                "The dashboard reads manifests and artifacts already generated by the pipeline. CH3 remains the canonical thesis view.",
                                className="control-note",
                            ),
                            html.Div(
                                [
                                    html.Div("Scope", className="control-label"),
                                    html.Div("CH3 thesis · CH2 gemela support · CH4 experimental", className="lead-note"),
                                ],
                                className="scope-summary",
                            ),
                            html.A(
                                "Open workbench spec",
                                href=_safe_rel_url(repo_root / "docs" / "visual_workbench_spec.md", repo_root),
                                target="_blank",
                                className="artifact-link",
                            ),
                        ],
                        className="sidebar-card",
                    ),
                    html.Div(
                        [
                            dcc.Tabs(
                                id="main-tabs",
                                value="inputs",
                                className="tabs-shell",
                                children=[
                                    dcc.Tab(label="Inputs", value="inputs", className="tab-item", selected_className="tab-item-selected"),
                                    dcc.Tab(label="Overview", value="overview", className="tab-item", selected_className="tab-item-selected"),
                                    dcc.Tab(label="Case Review", value="case", className="tab-item", selected_className="tab-item-selected"),
                                    dcc.Tab(label="Sensitivity", value="sensitivity", className="tab-item", selected_className="tab-item-selected"),
                                    dcc.Tab(label="Comparative", value="comparative", className="tab-item", selected_className="tab-item-selected"),
                                    dcc.Tab(label="VNA", value="vna", className="tab-item", selected_className="tab-item-selected"),
                                ],
                            ),
                            html.Div(id="tab-content", className="tab-content"),
                        ],
                        className="main-column",
                    ),
                ],
                className="layout-shell",
            ),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("workbench-store", "data"),
        Output("refresh-status", "children"),
        Output("analysis-channel-dropdown", "options"),
        Output("analysis-channel-dropdown", "value"),
        Output("case-dropdown", "options"),
        Output("case-dropdown", "value"),
        Output("job-store", "data"),
        Input("refresh-button", "n_clicks"),
        Input("active-paths-store", "data"),
        State("case-dropdown", "value"),
        prevent_initial_call=False,
    )
    def refresh_data(_: int | None, active_paths: dict[str, Any], current_case: str | None):
        data = load_workbench_data(
            repo_root=repo_root,
            state_alarm_roots=dict(active_paths.get("state_alarm_roots", {})),
            comparative_roots=dict(active_paths.get("comparative_roots", {})),
            sensitivity_roots=dict(active_paths.get("sensitivity_roots", {})),
            active_channel=str(active_paths.get("current_channel", "")),
            vna_root=Path(active_paths.get("vna_root")) if active_paths.get("vna_root") else None,
            pd_base_dir=pd_base_dir,
            extra_pd_input=str(active_paths.get("extra_pd_input", "") or ""),
        )
        channel_options = _channel_dropdown_options(list(data.get("available_channels", DEFAULT_PD_CHANNELS)))
        options = [{"label": key, "value": key} for key in data["state_alarm"].get("case_keys", [])]
        valid_values = {item["value"] for item in options}
        selected = current_case if current_case in valid_values else (options[0]["value"] if options else None)
        status = f"Loaded {data.get('generated_at', '')} ({data.get('active_channel', '')})"
        return data, status, channel_options, data.get("active_channel", "CH3"), options, selected, data.get("jobs", [])

    @app.callback(
        Output("active-paths-store", "data"),
        Output("job-status-store", "data"),
        Input("run-pd-button", "n_clicks"),
        Input("run-sensitivity-button", "n_clicks"),
        Input("run-vna-button", "n_clicks"),
        State("pd-channel-dropdown", "value"),
        State("pd-selection-dropdown", "value"),
        State("pd-k-sigma", "value"),
        State("pd-wavelet-toggle", "value"),
        State("pd-extra-paths-textarea", "value"),
        State("vna-paths-textarea", "value"),
        State("active-paths-store", "data"),
        prevent_initial_call=True,
    )
    def run_input_jobs(
        run_pd_clicks: int | None,
        run_sensitivity_clicks: int | None,
        run_vna_clicks: int | None,
        pd_channel: str | None,
        pd_selection: list[str] | None,
        pd_k_sigma: float | None,
        pd_wavelet_toggle: list[str] | None,
        pd_extra_paths_text: str | None,
        vna_paths_text: str | None,
        active_paths: dict[str, Any],
    ):
        trigger_id = ctx.triggered_id
        active_paths = dict(active_paths or {})
        try:
            if trigger_id == "run-pd-button":
                outputs = run_pd_selection(
                    repo_root=repo_root,
                    base_dir=pd_base_dir,
                    dataset_keys=list(pd_selection or []),
                    channel=pd_channel,
                    raw_input=pd_extra_paths_text or "",
                    k_sigma=float(pd_k_sigma or 5.0),
                    wavelet_denoise=bool(pd_wavelet_toggle and "wavelet" in pd_wavelet_toggle),
                )
                active_paths["state_alarm_roots"] = {
                    key: str(value) for key, value in outputs.get("state_alarm_roots", {}).items()
                }
                active_paths["comparative_roots"] = {
                    key: str(value) for key, value in outputs.get("comparative_roots", {}).items()
                }
                selected_channels = list(outputs.get("state_alarm_roots", {}).keys())
                if selected_channels:
                    active_paths["current_channel"] = selected_channels[0]
                active_paths["extra_pd_input"] = str(pd_extra_paths_text or "")
                return active_paths, outputs["message"]

            if trigger_id == "run-sensitivity-button":
                job = create_sensitivity_job(
                    repo_root=repo_root,
                    base_dir=pd_base_dir,
                    dataset_keys=list(pd_selection or []),
                    channel=pd_channel,
                    raw_input=pd_extra_paths_text or "",
                )
                active_paths["current_channel"] = "CH3"
                active_paths["extra_pd_input"] = str(pd_extra_paths_text or "")
                return active_paths, f"Queued CH3 sensitivity job {job.get('job_id', '')}."

            if trigger_id == "run-vna-button":
                outputs = run_vna_selection(
                    repo_root=repo_root,
                    raw_input=vna_paths_text or "",
                )
                active_paths["vna_root"] = str(outputs["vna_root"])
                return active_paths, outputs["message"]
        except Exception as exc:
            return active_paths, f"Job failed: {exc}"

        return active_paths, "No job was triggered."

    @app.callback(
        Output("active-paths-store", "data", allow_duplicate=True),
        Input("analysis-channel-dropdown", "value"),
        State("active-paths-store", "data"),
        prevent_initial_call=True,
    )
    def update_active_channel(channel_value: str | None, active_paths: dict[str, Any]):
        work = dict(active_paths or {})
        normalized = _normalize_channel_name(channel_value)
        if normalized:
            work["current_channel"] = normalized
        return work

    @app.callback(
        Output("job-store", "data", allow_duplicate=True),
        Output("job-status-store", "data", allow_duplicate=True),
        Output("active-paths-store", "data", allow_duplicate=True),
        Input("job-poll-interval", "n_intervals"),
        State("active-paths-store", "data"),
        prevent_initial_call=True,
    )
    def poll_jobs(_: int, active_paths: dict[str, Any]):
        job_rows = list_workbench_jobs(repo_root)
        status_text = latest_job_status_text(job_rows)
        active_paths = dict(active_paths or {})
        updated_paths = dict(active_paths)
        latest_sensitivity = next(
            (
                row
                for row in job_rows
                if str(row.get("mode", "")) == "semaphore_sensitivity" and str(row.get("channel", "")) == "CH3"
            ),
            None,
        )
        if latest_sensitivity is None:
            return job_rows, status_text, no_update
        stable_root = str(latest_sensitivity.get("stable_output_root", "")).strip()
        current_root = str(updated_paths.get("sensitivity_roots", {}).get("CH3", "")).strip()
        if str(latest_sensitivity.get("status", "")) == "succeeded" and stable_root and stable_root != current_root:
            sensitivity_roots = dict(updated_paths.get("sensitivity_roots", {}))
            sensitivity_roots["CH3"] = stable_root
            updated_paths["sensitivity_roots"] = sensitivity_roots
            updated_paths["current_channel"] = "CH3"
            return job_rows, status_text, updated_paths
        return job_rows, status_text, no_update

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        Input("case-dropdown", "value"),
        Input("workbench-store", "data"),
        Input("job-status-store", "data"),
    )
    def render_tab(tab_name: str, selected_case: str | None, data: dict[str, Any], job_status: str):
        if tab_name == "inputs":
            return _inputs_layout(data, job_status)
        if tab_name == "overview":
            return _overview_layout(data)
        if tab_name == "case":
            case = data["state_alarm"].get("cases", {}).get(selected_case or "", {})
            return _case_detail_layout(case)
        if tab_name == "sensitivity":
            return _sensitivity_layout(data.get("sensitivity", {}))
        if tab_name == "vna":
            return _vna_layout(data.get("vna", {}))
        return _comparative_layout(data["comparative"])

    return app


def serve_workbench(
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    repo_root: Path | None = None,
    state_alarm_root: Path | None = None,
    comparative_root: Path | None = None,
    sensitivity_root: Path | None = None,
    vna_root: Path | None = None,
    pd_base_dir: Path | None = None,
) -> None:
    app = create_workbench_app(
        repo_root=repo_root,
        state_alarm_root=state_alarm_root,
        comparative_root=comparative_root,
        sensitivity_root=sensitivity_root,
        vna_root=vna_root,
        pd_base_dir=pd_base_dir,
    )
    app.run(host=host, port=port, debug=debug)
