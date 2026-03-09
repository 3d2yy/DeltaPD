from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Title2",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=14,
        )
    )
    return styles


def _simple_table(rows: list[list[str]], widths: list[float], header_color: str) -> Table:
    table = Table(rows, colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _add_existing_image(story: list[Any], image_path: Path, title: str, styles: Any) -> None:
    if not image_path.exists():
        return
    story.append(PageBreak())
    story.append(Paragraph(title, styles["Section"]))
    img = Image(str(image_path))
    img._restrictSize(7.0 * inch, 9.0 * inch)
    story.append(img)


def _append_markdown_like_text(
    story: list[Any],
    markdown_path: Path,
    *,
    styles: Any,
    skip_h1: bool = True,
) -> None:
    if not markdown_path.exists():
        return
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.05 * inch))
            continue
        if stripped.startswith("# "):
            if skip_h1:
                continue
            story.append(Paragraph(escape(stripped[2:]), styles["Section"]))
            continue
        if stripped.startswith("## "):
            story.append(Paragraph(escape(stripped[3:]), styles["Section"]))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(escape(stripped[4:]), styles["BodySmall"]))
            continue
        if stripped.startswith("- "):
            story.append(Paragraph(escape(f"* {stripped[2:]}"), styles["BodySmall"]))
            continue
        if stripped.startswith("![]("):
            continue
        story.append(Paragraph(escape(stripped), styles["BodySmall"]))


def build_mat_series_pdf(
    output_dir: str | Path,
    *,
    pdf_filename: str = "serie_1_2_conclusions.pdf",
) -> Path:
    output_dir = Path(output_dir)
    pdf_path = output_dir / pdf_filename
    behavior = pd.read_csv(output_dir / "descriptor_behavior.csv").head(8)
    candidates = pd.read_csv(output_dir / "change_candidates.csv").head(6)
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    styles = _styles()
    story: list[Any] = []

    story.append(Paragraph("MAT Series Study - conclusions", styles["Title2"]))
    story.append(
        Paragraph(
            f"Source: {manifest.get('source_file', 'unknown')}<br/>"
            f"Matrix: {manifest['matrix_shape'][0]} captures x {manifest['matrix_shape'][1]} samples",
            styles["BodySmall"],
        )
    )
    primary_block = manifest.get("primary_block") or {}
    if primary_block:
        story.append(
            Paragraph(
                f"Primary activity block: rows {primary_block['row_start']}-{primary_block['row_end']} "
                f"({primary_block['length_rows']} captures).",
                styles["BodySmall"],
            )
        )

    story.append(Paragraph("Top descriptor behavior", styles["Section"]))
    behavior_rows = [["Descriptor", "activity_corr", "change_corr", "block_shift_z"]]
    for _, row in behavior.iterrows():
        behavior_rows.append(
            [
                str(row["feature"]),
                f"{float(row['activity_corr']):.3f}",
                f"{float(row['change_corr']):.3f}",
                f"{float(row['block_shift_z']):.1f}",
            ]
        )
    story.append(
        _simple_table(
            behavior_rows,
            [2.2 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch],
            "#d9e2f3",
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Top transition candidates", styles["Section"]))
    candidate_rows = [["Rank", "Row", "Score", "Dominant feature"]]
    for _, row in candidates.iterrows():
        candidate_rows.append(
            [
                str(int(row["candidate_rank"])),
                str(int(row["row_idx"])),
                f"{float(row['change_score']):.2f}",
                str(row["dominant_feature"]),
            ]
        )
    story.append(
        _simple_table(
            candidate_rows,
            [0.7 * inch, 0.8 * inch, 1.0 * inch, 2.5 * inch],
            "#fce4d6",
        )
    )

    for image_name, title in [
        ("series_matrix_overview.png", "Series matrix overview"),
        ("descriptor_trends.png", "Descriptor trends"),
        ("descriptor_heatmap.png", "Standardized descriptor heatmap"),
        ("representative_waveforms.png", "Representative waveforms"),
    ]:
        _add_existing_image(story, output_dir / image_name, title, styles)

    SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    ).build(story)
    return pdf_path


def build_descriptor_study_pdf(
    descriptor_output_dir: str | Path,
    *,
    material_output_dir: str | Path | None = None,
    title: str = "Descriptor Study Report",
    pdf_filename: str = "descriptor_study_report.pdf",
    narrative_markdown_path: str | Path | None = None,
    extra_images: list[tuple[str | Path, str]] | None = None,
) -> Path:
    descriptor_output_dir = Path(descriptor_output_dir)
    material_output_dir = Path(material_output_dir) if material_output_dir is not None else None
    pdf_path = descriptor_output_dir / pdf_filename

    recommendations = json.loads(
        (descriptor_output_dir / "study_recommendations.json").read_text(encoding="utf-8")
    )
    change_candidates = None
    change_candidates_path = descriptor_output_dir / "change_candidates.csv"
    if change_candidates_path.exists():
        change_candidates = pd.read_csv(change_candidates_path).head(8)

    styles = _styles()
    story: list[Any] = []
    story.append(Paragraph(title, styles["Title2"]))

    if material_output_dir is not None and (material_output_dir / "run_manifest.json").exists():
        manifest = json.loads((material_output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        source_file = manifest.get("source_file", "unknown")
        total_events = manifest.get("total_events", "unknown")
        df_delta = pd.read_csv(material_output_dir / "delta_t_series_master.csv", usecols=["toa_s"])
        duration_s = float(df_delta["toa_s"].max()) if not df_delta.empty else float("nan")
        blind_prpd = dict(manifest.get("blind_prpd", {}))
        story.append(
            Paragraph(
                f"Source: {source_file}<br/>"
                f"Detected events: {total_events}<br/>"
                f"Observed duration: {duration_s:.6f} s",
                styles["BodySmall"],
            )
        )
        if blind_prpd:
            story.append(Paragraph("Blind PRPD calibration", styles["Section"]))
            blind_rows = [
                ["Requested", "Selected", "Freq (Hz)", "Coherence"],
                [
                    str(blind_prpd.get("requested_method", blind_prpd.get("method", ""))),
                    str(blind_prpd.get("selected_method", blind_prpd.get("method", ""))),
                    f"{float(blind_prpd.get('calibrated_freq_hz', float('nan'))):.6f}",
                    f"{float(blind_prpd.get('coherence', float('nan'))):.6f}",
                ],
            ]
            story.append(
                _simple_table(
                    blind_rows,
                    [1.25 * inch, 1.25 * inch, 1.15 * inch, 1.0 * inch],
                    "#e8efe6",
                )
            )
            story.append(Spacer(1, 0.12 * inch))
            uncertainty_rows = [
                ["Common conf.", "Peak offset (Hz)", "Boot std (Hz)", "Boot agree."],
                [
                    f"{float(blind_prpd.get('common_axial_confidence', float('nan'))):.6f}",
                    f"{float(blind_prpd.get('common_axial_peak_offset_hz', float('nan'))):.6f}",
                    f"{float(blind_prpd.get('bootstrap_freq_std_hz', float('nan'))):.6f}",
                    f"{float(blind_prpd.get('bootstrap_method_agreement', float('nan'))):.6f}",
                ],
            ]
            story.append(
                _simple_table(
                    uncertainty_rows,
                    [1.25 * inch, 1.35 * inch, 1.45 * inch, 1.1 * inch],
                    "#f2eadc",
                )
            )
            story.append(Spacer(1, 0.12 * inch))
            local_rows = [
                ["Local windows", "Local std (Hz)", "Local span (Hz)", "Local agree."],
                [
                    str(int(blind_prpd.get("local_window_count", 0) or 0)),
                    f"{float(blind_prpd.get('local_freq_std_hz', float('nan'))):.6f}",
                    f"{float(blind_prpd.get('local_freq_span_hz', float('nan'))):.6f}",
                    f"{float(blind_prpd.get('local_method_agreement', float('nan'))):.6f}",
                ],
            ]
            story.append(
                _simple_table(
                    local_rows,
                    [1.05 * inch, 1.35 * inch, 1.35 * inch, 1.1 * inch],
                    "#e6eef7",
                )
            )
            story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Recommended subsets", styles["Section"]))
    subset_rows = [["Task", "Features", "Primary metric", "Score", "Balanced acc."]]
    for task_name, payload in recommendations.items():
        rec = payload.get("recommendation", {})
        subset_rows.append(
            [
                task_name,
                ", ".join(rec.get("features", [])),
                str(rec.get("primary_metric", "")),
                f"{float(rec.get('primary_score', float('nan'))):.4f}",
                f"{float(rec.get('balanced_accuracy', float('nan'))):.4f}",
            ]
        )
    story.append(
        _simple_table(
            subset_rows,
            [0.9 * inch, 3.2 * inch, 1.2 * inch, 0.8 * inch, 0.9 * inch],
            "#d9e2f3",
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    if change_candidates is not None and not change_candidates.empty:
        story.append(Paragraph("Top transition windows", styles["Section"]))
        transition_rows = [["Rank", "t_start_s", "t_end_s", "Score", "Dominant feature"]]
        for _, row in change_candidates.iterrows():
            transition_rows.append(
                [
                    str(int(row["candidate_rank"])),
                    f"{float(row['toa_start_s']):.6f}",
                    f"{float(row['toa_end_s']):.6f}",
                    f"{float(row['change_score']):.4f}",
                    str(row["dominant_feature"]),
                ]
            )
        story.append(
            _simple_table(
                transition_rows,
                [0.6 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch, 2.3 * inch],
                "#fce4d6",
            )
        )

    study_report_path = (
        Path(narrative_markdown_path)
        if narrative_markdown_path is not None
        else descriptor_output_dir / "study_report.md"
    )
    if study_report_path.exists():
        story.append(PageBreak())
        story.append(Paragraph("Narrative study report", styles["Section"]))
        _append_markdown_like_text(story, study_report_path, styles=styles)

    if extra_images:
        for image_path, title_text in extra_images:
            _add_existing_image(story, Path(image_path), title_text, styles)

    if material_output_dir is not None:
        for image_name, title_text in [
            ("01_raw_with_detections.png", "Raw waveform with detections"),
            ("02a_delta_t_series_lineal.png", "Delta t series"),
            ("05_rolling_delta_t_stats.png", "Rolling delta t statistics"),
            ("06_ewma_cusum_robusto.png", "EWMA and CUSUM"),
            ("08_blind_prpd_50hz.png", "Blind PRPD"),
            ("09_classification_trend.png", "Classification trend"),
        ]:
            _add_existing_image(story, material_output_dir / image_name, title_text, styles)

    SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    ).build(story)
    return pdf_path


def build_vna_pdf(
    output_dir: str | Path,
    *,
    title: str = "VNA Analysis Report",
    pdf_filename: str = "vna_report.pdf",
) -> Path:
    output_dir = Path(output_dir)
    pdf_path = output_dir / pdf_filename

    manifest_path = output_dir / "vna_manifest.json"
    summary_csv_path = output_dir / "vna_summary.csv"
    summary_md_path = output_dir / "vna_summary.md"
    if not manifest_path.exists() or not summary_csv_path.exists():
        raise FileNotFoundError("VNA report inputs are incomplete.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_df = pd.read_csv(summary_csv_path).fillna("")
    styles = _styles()
    story: list[Any] = []

    story.append(Paragraph(title, styles["Title2"]))
    story.append(
        Paragraph(
            f"Mode: {manifest.get('mode', 'unknown')}<br/>"
            f"Files analyzed: {len(manifest.get('items', []))}<br/>"
            f"Detected interpretations: {', '.join(manifest.get('modes_detected', [])) or 'unknown'}",
            styles["BodySmall"],
        )
    )

    if not summary_df.empty:
        story.append(Paragraph("Summary table", styles["Section"]))
        if set(summary_df.get("mode", pd.Series(dtype=str)).astype(str).tolist()) == {"s11"}:
            rows = [[
                "File",
                "Mode",
                "f_min (GHz)",
                "f_max (GHz)",
                "min S11 (dB)",
                "f@min (GHz)",
            ]]
            for _, row in summary_df.iterrows():
                rows.append(
                    [
                        Path(str(row.get("source_file", ""))).name,
                        str(row.get("mode", "")),
                        f"{float(row.get('freq_min_hz', 0.0)) / 1e9:.3f}",
                        f"{float(row.get('freq_max_hz', 0.0)) / 1e9:.3f}",
                        f"{float(row.get('min_s11_db', float('nan'))):.2f}",
                        f"{float(row.get('freq_at_min_s11_hz', float('nan'))) / 1e9:.3f}",
                    ]
                )
            story.append(
                _simple_table(
                    rows,
                    [2.3 * inch, 0.7 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
                    "#d9e2f3",
                )
            )
        else:
            show_columns = [column for column in ["source_file", "mode", "freq_min_hz", "freq_max_hz"] if column in summary_df.columns]
            rows = [[column for column in show_columns]]
            for _, row in summary_df.iterrows():
                rows.append(
                    [
                        Path(str(row.get(column, ""))).name if column == "source_file" else str(row.get(column, ""))
                        for column in show_columns
                    ]
                )
            widths = [2.5 * inch] + [1.2 * inch] * max(len(show_columns) - 1, 0)
            story.append(_simple_table(rows, widths, "#d9e2f3"))
        story.append(Spacer(1, 0.15 * inch))

    if summary_md_path.exists():
        story.append(Paragraph("Narrative summary", styles["Section"]))
        _append_markdown_like_text(story, summary_md_path, styles=styles)

    overlay_image = Path(str(manifest.get("overlay_image", ""))) if manifest.get("overlay_image") else None
    if overlay_image is not None and overlay_image.exists():
        _add_existing_image(story, overlay_image, "Comparative overlay", styles)

    for item in manifest.get("items", []):
        item_dir = Path(str(item.get("output_dir", "")))
        image_path = item_dir / "vna_overview.png"
        if image_path.exists():
            title_text = f"VNA overview - {Path(str(item.get('source_file', ''))).name}"
            _add_existing_image(story, image_path, title_text, styles)

    SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    ).build(story)
    return pdf_path
