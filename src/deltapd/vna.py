from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deltapd.campaign.pdf_reports import build_vna_pdf


VNA_EXTENSIONS = {".s1p", ".s2p", ".csv", ".txt", ".dat"}


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "item"


def _read_numeric_csv(file_path: Path) -> pd.DataFrame:
    for sep in [None, ",", ";", "\t"]:
        try:
            df = pd.read_csv(file_path, sep=sep, engine="python", comment="#")
        except Exception:
            continue
        if not df.empty and df.shape[1] >= 2:
            return df
    raise ValueError(f"Could not parse CSV-like VNA file: {file_path}")


def _touchstone_unit_scale(unit: str) -> float:
    unit = unit.upper()
    return {
        "HZ": 1.0,
        "KHZ": 1e3,
        "MHZ": 1e6,
        "GHZ": 1e9,
    }.get(unit, 1.0)


def _parse_touchstone(file_path: Path) -> dict[str, Any]:
    option_tokens = ["GHZ", "S", "MA", "R", "50"]
    data_rows: list[list[float]] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                option_tokens = line[1:].strip().upper().split()
                continue
            parts = line.split()
            try:
                data_rows.append([float(part) for part in parts])
            except ValueError:
                continue

    if not data_rows:
        raise ValueError(f"Touchstone file has no numeric rows: {file_path}")

    data = np.array(data_rows, dtype=np.float64)
    nports = 1 if file_path.suffix.lower() == ".s1p" else 2
    if nports == 1 and data.shape[1] < 3:
        raise ValueError("S1P file does not contain enough columns.")
    if nports == 2 and data.shape[1] < 9:
        raise ValueError("S2P file does not contain enough columns for S11.")

    unit = option_tokens[0] if option_tokens else "HZ"
    fmt = option_tokens[2] if len(option_tokens) >= 3 else "MA"
    freq_hz = data[:, 0] * _touchstone_unit_scale(unit)
    pair = data[:, 1:3]

    if fmt == "RI":
        gamma = pair[:, 0] + 1j * pair[:, 1]
    elif fmt == "DB":
        mag = np.power(10.0, pair[:, 0] / 20.0)
        gamma = mag * np.exp(1j * np.deg2rad(pair[:, 1]))
    else:
        mag = pair[:, 0]
        gamma = mag * np.exp(1j * np.deg2rad(pair[:, 1]))

    s11_db = 20.0 * np.log10(np.maximum(np.abs(gamma), 1e-12))
    return {
        "mode": "s11",
        "freq_hz": freq_hz,
        "s11_db": s11_db,
        "gamma_complex": gamma,
        "source_format": file_path.suffix.lower(),
    }


def _find_frequency_column(df: pd.DataFrame) -> tuple[str, float]:
    for column in df.columns:
        name = str(column).lower()
        if "freq" not in name and "frequency" not in name:
            continue
        if "ghz" in name:
            return column, 1e9
        if "mhz" in name:
            return column, 1e6
        if "khz" in name:
            return column, 1e3
        return column, 1.0

    numeric_cols = [column for column in df.columns if pd.to_numeric(df[column], errors="coerce").notna().mean() > 0.9]
    if numeric_cols:
        return numeric_cols[0], 1.0
    raise ValueError("No frequency-like column found.")


def _find_matching_column(columns: list[str], tokens: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    for column in columns:
        name = column.lower()
        if all(token in name for token in tokens) and not any(token in name for token in exclude):
            return column
    return None


def _parse_vna_csv(file_path: Path) -> dict[str, Any]:
    df = _read_numeric_csv(file_path)
    original_columns = [str(column) for column in df.columns]
    lower_columns = [column.lower() for column in original_columns]
    freq_column, scale = _find_frequency_column(df)
    freq_hz = pd.to_numeric(df[freq_column], errors="coerce").to_numpy(dtype=np.float64) * scale

    s11_db_col = _find_matching_column(lower_columns, ["s11", "db"])
    return_loss_col = _find_matching_column(lower_columns, ["return"], exclude=["lossless"])
    real_col = _find_matching_column(lower_columns, ["s11", "real"])
    imag_col = _find_matching_column(lower_columns, ["s11", "imag"])
    mag_col = _find_matching_column(lower_columns, ["s11", "mag"])
    ang_col = _find_matching_column(lower_columns, ["s11", "ang"])

    column_lookup = {column.lower(): column for column in original_columns}

    if s11_db_col:
        values = pd.to_numeric(df[column_lookup[s11_db_col]], errors="coerce").to_numpy(dtype=np.float64)
        s11_db = values
        gamma_mag = np.power(10.0, s11_db / 20.0)
        gamma = gamma_mag.astype(np.complex128)
        return {
            "mode": "s11",
            "freq_hz": freq_hz,
            "s11_db": s11_db,
            "gamma_complex": gamma,
            "source_format": "csv_s11_db",
        }

    if return_loss_col:
        values = pd.to_numeric(df[column_lookup[return_loss_col]], errors="coerce").to_numpy(dtype=np.float64)
        gamma_mag = np.power(10.0, -values / 20.0)
        s11_db = 20.0 * np.log10(np.maximum(gamma_mag, 1e-12))
        gamma = gamma_mag.astype(np.complex128)
        return {
            "mode": "s11",
            "freq_hz": freq_hz,
            "s11_db": s11_db,
            "gamma_complex": gamma,
            "source_format": "csv_return_loss",
        }

    if real_col and imag_col:
        real = pd.to_numeric(df[column_lookup[real_col]], errors="coerce").to_numpy(dtype=np.float64)
        imag = pd.to_numeric(df[column_lookup[imag_col]], errors="coerce").to_numpy(dtype=np.float64)
        gamma = real + 1j * imag
        s11_db = 20.0 * np.log10(np.maximum(np.abs(gamma), 1e-12))
        return {
            "mode": "s11",
            "freq_hz": freq_hz,
            "s11_db": s11_db,
            "gamma_complex": gamma,
            "source_format": "csv_real_imag",
        }

    if mag_col and ang_col:
        mag = pd.to_numeric(df[column_lookup[mag_col]], errors="coerce").to_numpy(dtype=np.float64)
        ang = pd.to_numeric(df[column_lookup[ang_col]], errors="coerce").to_numpy(dtype=np.float64)
        gamma = mag * np.exp(1j * np.deg2rad(ang))
        s11_db = 20.0 * np.log10(np.maximum(np.abs(gamma), 1e-12))
        return {
            "mode": "s11",
            "freq_hz": freq_hz,
            "s11_db": s11_db,
            "gamma_complex": gamma,
            "source_format": "csv_mag_angle",
        }

    numeric_candidates = []
    for column in original_columns:
        if column == freq_column:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().mean() > 0.9:
            numeric_candidates.append(column)

    if not numeric_candidates:
        raise ValueError(f"No VNA response column found in {file_path}")

    response_column = numeric_candidates[0]
    response = pd.to_numeric(df[response_column], errors="coerce").to_numpy(dtype=np.float64)
    return {
        "mode": "noise",
        "freq_hz": freq_hz,
        "response": response,
        "response_label": response_column,
        "source_format": "csv_generic",
    }


def analyze_vna_file(file_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    file_path = Path(file_path).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_path.suffix.lower() in {".s1p", ".s2p"}:
        parsed = _parse_touchstone(file_path)
    else:
        parsed = _parse_vna_csv(file_path)

    freq_hz = np.asarray(parsed["freq_hz"], dtype=np.float64)
    valid_mask = np.isfinite(freq_hz)
    for key in ["s11_db", "response"]:
        if key in parsed:
            values = np.asarray(parsed[key], dtype=np.float64)
            valid_mask &= np.isfinite(values)
    if "gamma_complex" in parsed:
        gamma = np.asarray(parsed["gamma_complex"], dtype=np.complex128)
        valid_mask &= np.isfinite(gamma.real) & np.isfinite(gamma.imag)
        parsed["gamma_complex"] = gamma[valid_mask]

    freq_hz = freq_hz[valid_mask]
    if freq_hz.size == 0:
        raise ValueError(f"No valid VNA points in {file_path}")

    result: dict[str, Any] = {
        "source_file": str(file_path),
        "mode": parsed["mode"],
        "source_format": parsed.get("source_format", ""),
        "n_points": int(freq_hz.size),
        "freq_min_hz": float(np.min(freq_hz)),
        "freq_max_hz": float(np.max(freq_hz)),
    }

    if parsed["mode"] == "s11":
        s11_db = np.asarray(parsed["s11_db"], dtype=np.float64)[valid_mask]
        gamma = np.asarray(parsed["gamma_complex"], dtype=np.complex128)
        gamma_mag = np.clip(np.abs(gamma), 0.0, 0.999999)
        vswr = (1.0 + gamma_mag) / np.maximum(1.0 - gamma_mag, 1e-12)
        min_idx = int(np.argmin(s11_db))
        min_s11_db = float(s11_db[min_idx])
        min_freq_hz = float(freq_hz[min_idx])
        below = np.flatnonzero(s11_db <= -10.0)
        bandwidth_hz = float(freq_hz[below[-1]] - freq_hz[below[0]]) if below.size >= 2 else float("nan")

        data_df = pd.DataFrame(
            {
                "freq_hz": freq_hz,
                "s11_db": s11_db,
                "gamma_mag": gamma_mag,
                "vswr": vswr,
            }
        )
        data_df.to_csv(output_dir / "vna_data.csv", index=False, encoding="utf-8-sig")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
        ax1.plot(freq_hz / 1e9, s11_db, color="#16324f", lw=1.7)
        ax1.axhline(-10.0, color="#9b5d2e", linestyle="--", lw=1.0)
        ax1.scatter([min_freq_hz / 1e9], [min_s11_db], color="#c97b63", zorder=3)
        ax1.set_ylabel("S11 (dB)")
        ax1.set_title(f"S11 and VSWR - {file_path.name}")
        ax1.grid(True, linestyle="--", alpha=0.3)

        ax2.plot(freq_hz / 1e9, vswr, color="#2d728f", lw=1.7)
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("VSWR")
        ax2.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "vna_overview.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        result.update(
            {
                "min_s11_db": min_s11_db,
                "freq_at_min_s11_hz": min_freq_hz,
                "bandwidth_below_-10db_hz": bandwidth_hz,
                "min_vswr": float(np.min(vswr)),
                "max_vswr": float(np.max(vswr)),
            }
        )
    else:
        response = np.asarray(parsed["response"], dtype=np.float64)[valid_mask]
        label = str(parsed.get("response_label", "response"))
        data_df = pd.DataFrame({"freq_hz": freq_hz, label: response})
        data_df.to_csv(output_dir / "vna_data.csv", index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.plot(freq_hz / 1e9, response, color="#6d597a", lw=1.7)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(label)
        ax.set_title(f"Frequency response - {file_path.name}")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "vna_overview.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        result.update(
            {
                "response_label": label,
                "response_mean": float(np.mean(response)),
                "response_max": float(np.max(response)),
                "response_min": float(np.min(response)),
            }
        )

    (output_dir / "vna_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def analyze_vna_selection(file_paths: list[str | Path], output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_files = [Path(path).expanduser().resolve() for path in file_paths]
    per_file_results: list[dict[str, Any]] = []
    per_file_data: list[pd.DataFrame] = []
    modes: list[str] = []

    for path in resolved_files:
        item_dir = output_root / _slugify(path.stem)
        result = analyze_vna_file(path, item_dir)
        result["output_dir"] = str(item_dir)
        per_file_results.append(result)
        modes.append(result["mode"])

        df = pd.read_csv(item_dir / "vna_data.csv")
        df["label"] = path.stem
        per_file_data.append(df)

    summary_df = pd.DataFrame(per_file_results)
    summary_df.to_csv(output_root / "vna_summary.csv", index=False, encoding="utf-8-sig")

    combined_df = pd.concat(per_file_data, ignore_index=True) if per_file_data else pd.DataFrame()
    overlay_image = ""
    markdown_lines = [
        "# VNA analysis summary",
        "",
        f"- Files analyzed: {len(resolved_files)}",
        f"- Modes detected: {', '.join(sorted(set(modes)))}",
        "",
    ]

    if len(resolved_files) >= 2 and not combined_df.empty:
        if set(modes) == {"s11"}:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.8), sharex=True)
            for label, df_item in combined_df.groupby("label"):
                ax1.plot(df_item["freq_hz"] / 1e9, df_item["s11_db"], lw=1.5, label=label)
                ax2.plot(df_item["freq_hz"] / 1e9, df_item["vswr"], lw=1.5, label=label)
            ax1.axhline(-10.0, color="#9b5d2e", linestyle="--", lw=1.0)
            ax1.set_ylabel("S11 (dB)")
            ax1.set_title("Comparative VNA overlay")
            ax1.grid(True, linestyle="--", alpha=0.3)
            ax1.legend()
            ax2.set_ylabel("VSWR")
            ax2.set_xlabel("Frequency (GHz)")
            ax2.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            overlay_image = "vna_comparative_overlay.png"
            fig.savefig(output_root / overlay_image, dpi=220, bbox_inches="tight")
            plt.close(fig)
        elif len(set(modes)) == 1 and modes[0] == "noise":
            response_col = [col for col in combined_df.columns if col not in {"freq_hz", "label"}][0]
            fig, ax = plt.subplots(figsize=(11, 5.8))
            for label, df_item in combined_df.groupby("label"):
                ax.plot(df_item["freq_hz"] / 1e9, df_item[response_col], lw=1.5, label=label)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(response_col)
            ax.set_title("Comparative frequency response")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            plt.tight_layout()
            overlay_image = "vna_comparative_overlay.png"
            fig.savefig(output_root / overlay_image, dpi=220, bbox_inches="tight")
            plt.close(fig)

    if set(modes) == {"s11"}:
        ranked = summary_df.sort_values("min_s11_db")
        markdown_lines.extend(
            [
                "## S11 highlights",
                "",
                f"- Best minimum S11: {ranked.iloc[0]['source_file']} at {ranked.iloc[0]['min_s11_db']:.2f} dB.",
                "",
            ]
        )
    elif len(set(modes)) == 1 and modes[0] == "noise":
        markdown_lines.extend(["## Noise highlights", "", "- Generic frequency-response mode detected.", ""])
    else:
        markdown_lines.extend(["## Mixed-mode note", "", "- The selected files do not share the same VNA interpretation.", ""])

    summary_md_path = output_root / "vna_summary.md"
    summary_md_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    manifest = {
        "mode": "comparative" if len(resolved_files) >= 2 else "single",
        "modes_detected": sorted(set(modes)),
        "summary_csv": str(output_root / "vna_summary.csv"),
        "summary_md": str(summary_md_path),
        "overlay_image": str(output_root / overlay_image) if overlay_image else "",
        "items": per_file_results,
    }
    pdf_title = "Comparative VNA Report" if len(resolved_files) >= 2 else "VNA Report"
    pdf_path = output_root / "vna_report.pdf"
    manifest["pdf_path"] = str(pdf_path)
    manifest_path = output_root / "vna_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_vna_pdf(output_root, title=pdf_title, pdf_filename=pdf_path.name)

    return {
        "summary_df": summary_df,
        "summary_csv": output_root / "vna_summary.csv",
        "summary_md": summary_md_path,
        "manifest_path": manifest_path,
        "pdf_path": pdf_path,
        "output_root": output_root,
    }
