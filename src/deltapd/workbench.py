from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from flask import abort, send_file

from deltapd.workbench_jobs import discover_pd_cases, run_pd_selection, run_vna_selection


THEME = {
    "paper": "#f6f2e8",
    "card": "#fffdf8",
    "ink": "#16324f",
    "muted": "#51606f",
    "line": "#c8c0af",
    "accent": "#9b5d2e",
    "accent_soft": "#d8a06f",
    "sea": "#2d728f",
    "berry": "#6d597a",
    "rose": "#c97b63",
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
    "transition_offset_vs_scores.png",
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


def _discover_state_alarm_batch(batch_root: Path, repo_root: Path) -> dict[str, Any]:
    manifest_path = batch_root / "state_alarm_batch_manifest.json"
    summary_csv_path = batch_root / "state_alarm_batch_summary.csv"
    summary_md_path = batch_root / "state_alarm_batch_summary.md"
    if not manifest_path.exists() or not summary_csv_path.exists():
        return {"available": False, "case_keys": [], "cases": {}, "summary_rows": []}

    manifest = _read_json(manifest_path)
    summary_df = pd.read_csv(summary_csv_path)
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
        cases[dataset_key] = {
            "meta": case_row,
            "folder": case.get("folder", ""),
            "study_report": _read_text(study_dir / "study_report.md"),
            "material_images": material_images,
            "case_images": material_images + study_images,
            "artifacts": artifact_paths,
            "pdf_url": _safe_rel_url(Path(case["pdf_path"]), repo_root) if case.get("pdf_path") else "",
            "blind_prpd": material_manifest.get("blind_prpd", {}),
        }

    return {
        "available": True,
        "manifest": manifest,
        "summary_rows": summary_rows,
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

    pdf_path = comparative_root / "comparative_ch3_descriptor_report.pdf"
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
    vna_root: Path | None = None,
    pd_base_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root or _repo_root()
    state_alarm_root = state_alarm_root or (repo_root / "outputs" / "state_alarm_ch3")
    comparative_root = comparative_root or (repo_root / "outputs" / "comparative_ch3")
    pd_base_dir = pd_base_dir or Path("E:/Carpeta definitiva de Tesis/programas")
    state_alarm = _discover_state_alarm_batch(state_alarm_root, repo_root)
    comparative = _discover_comparative_study(comparative_root, repo_root)
    vna = _discover_vna_outputs(vna_root, repo_root)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "pd_base_dir": str(pd_base_dir),
        "pd_cases": discover_pd_cases(pd_base_dir),
        "state_alarm": state_alarm,
        "comparative": comparative,
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

    return html.Div(
        [
            metrics,
            html.Div(transition, className="lead-note"),
            html.Div(artifact_links, className="artifact-list"),
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


def _overview_layout(data: dict[str, Any]) -> html.Div:
    state_alarm = data["state_alarm"]
    summary_rows = state_alarm.get("summary_rows", [])
    manifest = state_alarm.get("manifest", {})
    state_counts = manifest.get("state_feature_counts", {})
    alarm_counts = manifest.get("alarm_feature_counts", {})
    summary_df = pd.DataFrame(summary_rows)
    mean_state = pd.to_numeric(summary_df.get("state_primary_score"), errors="coerce").mean() if not summary_df.empty else 0.0
    mean_alarm = pd.to_numeric(summary_df.get("alarm_primary_score"), errors="coerce").mean() if not summary_df.empty else 0.0

    cards = html.Div(
        [
            _metric_card("Batch cases", str(len(summary_rows)), "P1/P2/P3/G1/G2/G3"),
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
                    html.Div("Master summary", className="panel-title"),
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


def _inputs_layout(data: dict[str, Any], job_status: str) -> html.Div:
    pd_options = []
    for case in data.get("pd_cases", []):
        label = f"{case['dataset_key']} - {case['folder']}"
        pd_options.append({"label": label, "value": case["dataset_key"]})

    return html.Div(
        [
            html.Div(
                [
                    html.Div("PD runner", className="panel-title"),
                    html.Div(
                        "Choose one test for state/alarm or several tests for comparative PD automatically.",
                        className="control-note",
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
                    html.Button("Run PD study", id="run-pd-button", className="refresh-button"),
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
    vna_root: Path | None = None,
    pd_base_dir: Path | None = None,
) -> Dash:
    repo_root = repo_root or _repo_root()
    state_alarm_root = state_alarm_root or (repo_root / "outputs" / "state_alarm_ch3")
    comparative_root = comparative_root or (repo_root / "outputs" / "comparative_ch3")
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
        vna_root=vna_root,
        pd_base_dir=pd_base_dir,
    )
    initial_active_paths = {
        "state_alarm_root": str(state_alarm_root),
        "comparative_root": str(comparative_root),
        "vna_root": str(vna_root) if vna_root is not None else "",
    }
    case_options = [
        {"label": key, "value": key}
        for key in initial_data["state_alarm"].get("case_keys", [])
    ]
    default_case = case_options[0]["value"] if case_options else None

    app.layout = html.Div(
        [
            dcc.Store(id="workbench-store", data=initial_data),
            dcc.Store(id="active-paths-store", data=initial_active_paths),
            dcc.Store(id="job-status-store", data="No job has been launched from the interface yet."),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("DeltaPD Thesis Workbench", className="hero-title"),
                            html.Div(
                                "State, alarm and comparative descriptor studies from reproducible pipeline outputs.",
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
                            html.Div("Case selector", className="control-label"),
                            dcc.Dropdown(
                                id="case-dropdown",
                                options=case_options,
                                value=default_case,
                                clearable=False,
                                className="case-dropdown",
                            ),
                            html.Div(
                                "The dashboard reads manifests and artifacts already generated by the pipeline.",
                                className="control-note",
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
        Output("case-dropdown", "options"),
        Output("case-dropdown", "value"),
        Input("refresh-button", "n_clicks"),
        Input("active-paths-store", "data"),
        State("case-dropdown", "value"),
        prevent_initial_call=False,
    )
    def refresh_data(_: int | None, active_paths: dict[str, Any], current_case: str | None):
        data = load_workbench_data(
            repo_root=repo_root,
            state_alarm_root=Path(active_paths.get("state_alarm_root")) if active_paths.get("state_alarm_root") else None,
            comparative_root=Path(active_paths.get("comparative_root")) if active_paths.get("comparative_root") else None,
            vna_root=Path(active_paths.get("vna_root")) if active_paths.get("vna_root") else None,
            pd_base_dir=pd_base_dir,
        )
        options = [{"label": key, "value": key} for key in data["state_alarm"].get("case_keys", [])]
        valid_values = {item["value"] for item in options}
        selected = current_case if current_case in valid_values else (options[0]["value"] if options else None)
        status = f"Loaded {data.get('generated_at', '')}"
        return data, status, options, selected

    @app.callback(
        Output("active-paths-store", "data"),
        Output("job-status-store", "data"),
        Input("run-pd-button", "n_clicks"),
        Input("run-vna-button", "n_clicks"),
        State("pd-selection-dropdown", "value"),
        State("pd-k-sigma", "value"),
        State("pd-wavelet-toggle", "value"),
        State("vna-paths-textarea", "value"),
        State("active-paths-store", "data"),
        prevent_initial_call=True,
    )
    def run_input_jobs(
        run_pd_clicks: int | None,
        run_vna_clicks: int | None,
        pd_selection: list[str] | None,
        pd_k_sigma: float | None,
        pd_wavelet_toggle: list[str] | None,
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
                    k_sigma=float(pd_k_sigma or 5.0),
                    wavelet_denoise=bool(pd_wavelet_toggle and "wavelet" in pd_wavelet_toggle),
                )
                if outputs.get("state_alarm_root") is not None:
                    active_paths["state_alarm_root"] = str(outputs["state_alarm_root"])
                if outputs.get("comparative_root") is not None:
                    active_paths["comparative_root"] = str(outputs["comparative_root"])
                return active_paths, outputs["message"]

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
    vna_root: Path | None = None,
    pd_base_dir: Path | None = None,
) -> None:
    app = create_workbench_app(
        repo_root=repo_root,
        state_alarm_root=state_alarm_root,
        comparative_root=comparative_root,
        vna_root=vna_root,
        pd_base_dir=pd_base_dir,
    )
    app.run(host=host, port=port, debug=debug)
