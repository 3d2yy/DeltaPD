from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from deltapd.campaign.comparative_thesis_study import run_comparative_thesis_study
from deltapd.campaign.state_alarm_batch import DEFAULT_DATASETS, DEFAULT_MATERIAL_DEFAULTS, DEFAULT_STUDY_DEFAULTS, run_state_alarm_batch
from deltapd.loader import load_empirical_signal
from deltapd.semaphore import build_semaphore_df, ingestion_audit_confidence
from deltapd.vna import VNA_EXTENSIONS, analyze_vna_selection


DEFAULT_PD_CHANNELS = ["CH2", "CH3", "CH4"]
SUPPORTED_PD_EXTENSIONS = {".csv", ".mat", ".h5", ".hdf5"}
JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
WORKBENCH_JOB_MODE_SENSITIVITY = "semaphore_sensitivity"
THESIS_SENSITIVITY_SCENARIOS = [
    {
        "scenario_key": "baseline",
        "label": "Baseline",
        "notes": "Reference CH3 thesis configuration.",
        "material_defaults": {},
        "study_defaults": {},
    },
    {
        "scenario_key": "baseline_repeat",
        "label": "Baseline repeat",
        "notes": "Repeat baseline to check deterministic behavior.",
        "material_defaults": {},
        "study_defaults": {},
    },
    {
        "scenario_key": "k_sigma_4_5",
        "label": "k sigma 4.5",
        "notes": "Lower detection threshold.",
        "material_defaults": {"detection": {"k_sigma": 4.5}},
        "study_defaults": {},
    },
    {
        "scenario_key": "k_sigma_5_5",
        "label": "k sigma 5.5",
        "notes": "Higher detection threshold.",
        "material_defaults": {"detection": {"k_sigma": 5.5}},
        "study_defaults": {},
    },
    {
        "scenario_key": "wavelet_off",
        "label": "Wavelet off",
        "notes": "Disable wavelet preprocessing.",
        "material_defaults": {"preprocess": {"wavelet_denoise": False}},
        "study_defaults": {},
    },
    {
        "scenario_key": "method_coherence",
        "label": "Method coherence",
        "notes": "Force blind PRPD coherence.",
        "material_defaults": {"analysis": {"blind_prpd": {"calibration_method": "coherence"}}},
        "study_defaults": {},
    },
    {
        "scenario_key": "method_harmonic_power",
        "label": "Method harmonic",
        "notes": "Force blind PRPD harmonic power.",
        "material_defaults": {"analysis": {"blind_prpd": {"calibration_method": "harmonic_power"}}},
        "study_defaults": {},
    },
    {
        "scenario_key": "method_epoch_folding",
        "label": "Method epoch",
        "notes": "Force blind PRPD epoch folding.",
        "material_defaults": {"analysis": {"blind_prpd": {"calibration_method": "epoch_folding"}}},
        "study_defaults": {},
    },
    {
        "scenario_key": "window_48_step_12",
        "label": "Window 48/12",
        "notes": "Smaller local descriptor windows.",
        "material_defaults": {},
        "study_defaults": {"windowing": {"window_events": 48, "step_events": 12}},
    },
    {
        "scenario_key": "window_80_step_20",
        "label": "Window 80/20",
        "notes": "Larger local descriptor windows.",
        "material_defaults": {},
        "study_defaults": {"windowing": {"window_events": 80, "step_events": 20}},
    },
    {
        "scenario_key": "step_8",
        "label": "Step 8",
        "notes": "Finer local stride with baseline window length.",
        "material_defaults": {},
        "study_defaults": {"windowing": {"window_events": 64, "step_events": 8}},
    },
    {
        "scenario_key": "step_24",
        "label": "Step 24",
        "notes": "Coarser local stride with baseline window length.",
        "material_defaults": {},
        "study_defaults": {"windowing": {"window_events": 64, "step_events": 24}},
    },
]


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _runtime_dir(repo_root: Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = repo_root / "outputs" / "workbench_runtime" / f"{name}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sync_output_tree(source_root: Path, destination_root: Path) -> Path:
    if source_root.resolve() == destination_root.resolve():
        destination_root.mkdir(parents=True, exist_ok=True)
        return destination_root
    destination_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
    return destination_root


def _job_root(repo_root: Path) -> Path:
    root = repo_root / "outputs" / "workbench_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_dir(repo_root: Path, job_id: str) -> Path:
    path = _job_root(repo_root) / str(job_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_manifest_path(job_dir: Path) -> Path:
    return job_dir / "job_manifest.json"


def _job_progress_label(manifest: dict[str, Any]) -> str:
    current = int(manifest.get("progress_current", 0) or 0)
    total = int(manifest.get("progress_total", 0) or 0)
    if total <= 0:
        return ""
    return f"{current}/{total}"


def _summary_path_from_job(manifest: dict[str, Any]) -> Path | None:
    stable_output_root = str(manifest.get("stable_output_root", "")).strip()
    output_root = str(manifest.get("output_root", "")).strip()
    for raw_root in (stable_output_root, output_root):
        if not raw_root:
            continue
        candidate = Path(raw_root) / "semaphore_sensitivity_summary.md"
        if candidate.exists():
            return candidate
    return None


def read_workbench_job_manifest(job_manifest_path: str | Path) -> dict[str, Any]:
    path = Path(job_manifest_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_workbench_job_manifest(job_manifest_path: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(job_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def update_workbench_job_manifest(job_manifest_path: str | Path, **updates: Any) -> dict[str, Any]:
    manifest = read_workbench_job_manifest(job_manifest_path)
    manifest.update(updates)
    write_workbench_job_manifest(job_manifest_path, manifest)
    return manifest


def list_workbench_jobs(repo_root: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    jobs_root = _job_root(repo_root)
    rows: list[dict[str, Any]] = []
    for manifest_path in jobs_root.glob("*/job_manifest.json"):
        manifest = read_workbench_job_manifest(manifest_path)
        if not manifest:
            continue
        manifest["job_manifest_path"] = str(manifest_path)
        manifest["job_dir"] = str(manifest_path.parent)
        manifest["progress_label"] = _job_progress_label(manifest)
        summary_path = _summary_path_from_job(manifest)
        manifest["summary_path"] = str(summary_path) if summary_path is not None else ""
        rows.append(manifest)
    rows.sort(
        key=lambda row: (
            str(row.get("created_at", "")),
            str(row.get("job_id", "")),
        ),
        reverse=True,
    )
    return rows[: max(int(limit), 1)]


def latest_job_status_text(job_rows: list[dict[str, Any]]) -> str:
    if not job_rows:
        return "No job has been launched from the interface yet."
    row = job_rows[0]
    mode = str(row.get("mode", "")).replace("_", " ").strip() or "job"
    status = str(row.get("status", "")).strip() or "unknown"
    progress = str(row.get("progress_label", "")).strip()
    message = str(row.get("message", "")).strip()
    parts = [f"{mode} [{status}]"]
    if progress:
        parts.append(progress)
    if message:
        parts.append(message)
    return " | ".join(parts)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _available_channel_paths(folder: Path) -> dict[str, str]:
    channel_paths: dict[str, str] = {}
    for channel in DEFAULT_PD_CHANNELS:
        candidate = folder / f"{channel}.csv"
        if candidate.exists():
            channel_paths[channel] = str(candidate.resolve())
    return channel_paths


def _is_valid_empirical_trace(file_path: Path) -> bool:
    try:
        signal, fs = load_empirical_signal(str(file_path), preserve_amplitude=True)
    except Exception:
        return False
    return bool(len(signal) >= 16 and fs > 0)


def _infer_discharge_type(label: str) -> str:
    text = str(label).strip().lower()
    if "intern" in text:
        return "internal"
    if "superfic" in text:
        return "superficial"
    if "múltiple" in text or "multiple" in text:
        return "multiple"
    return "unknown"


def _infer_variant(label: str) -> str:
    text = str(label).strip().lower()
    if "gemela" in text or "gemelas" in text:
        return "gemela"
    if text:
        return "custom"
    return "unknown"


def _discover_default_pd_cases(base_dir: str | Path) -> list[dict[str, Any]]:
    base_dir = Path(base_dir).expanduser()
    cases = []
    for case in DEFAULT_DATASETS:
        folder = base_dir / case["folder"]
        channel_paths = _available_channel_paths(folder)
        if not channel_paths:
            continue
        cases.append(
            {
                **case,
                "case_id": f"default:{case['dataset_key']}",
                "path": str(folder.resolve()),
                "channel_paths": channel_paths,
                "available_channels": sorted(channel_paths),
                "is_custom": False,
            }
        )
    return cases


def expand_pd_inputs(raw_text: str) -> list[Path]:
    paths: list[Path] = []
    for line in raw_text.splitlines():
        text = line.strip().strip('"')
        if not text:
            continue
        candidate = Path(text).expanduser()
        if candidate.is_dir():
            for ext in sorted(SUPPORTED_PD_EXTENSIONS):
                paths.extend(sorted(candidate.rglob(f"*{ext}")))
            continue
        if candidate.exists() and candidate.suffix.lower() in SUPPORTED_PD_EXTENSIONS:
            paths.append(candidate)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _discover_custom_pd_cases(raw_text: str) -> list[dict[str, Any]]:
    file_paths = [path for path in expand_pd_inputs(raw_text) if _is_valid_empirical_trace(path)]
    if not file_paths:
        return []

    cases: list[dict[str, Any]] = []
    custom_idx = 1
    grouped_channel_files: dict[Path, dict[str, str]] = {}

    for file_path in file_paths:
        stem = file_path.stem.upper()
        if stem in DEFAULT_PD_CHANNELS:
            parent = file_path.parent.resolve()
            grouped_channel_files.setdefault(parent, {})
            grouped_channel_files[parent][stem] = str(file_path.resolve())
            continue

        folder_label = file_path.stem or file_path.parent.name
        cases.append(
            {
                "dataset_key": f"C{custom_idx}",
                "folder": folder_label,
                "discharge_type": _infer_discharge_type(folder_label),
                "variant": _infer_variant(folder_label),
                "case_id": f"custom:{custom_idx}",
                "path": str(file_path.resolve()),
                "file_path": str(file_path.resolve()),
                "channel_paths": {},
                "available_channels": [],
                "is_custom": True,
            }
        )
        custom_idx += 1

    for parent, channel_paths in sorted(grouped_channel_files.items()):
        folder_label = parent.name
        cases.append(
            {
                "dataset_key": f"C{custom_idx}",
                "folder": folder_label,
                "discharge_type": _infer_discharge_type(folder_label),
                "variant": _infer_variant(folder_label),
                "case_id": f"custom:{custom_idx}",
                "path": str(parent),
                "channel_paths": channel_paths,
                "available_channels": sorted(channel_paths),
                "is_custom": True,
            }
        )
        custom_idx += 1
    return cases


def discover_pd_cases(base_dir: str | Path, raw_input: str | None = None) -> list[dict[str, Any]]:
    default_cases = _discover_default_pd_cases(base_dir)
    custom_cases = _discover_custom_pd_cases(raw_input or "")
    return default_cases + custom_cases


def _selected_default_cases(base_dir: str | Path, dataset_keys: list[str]) -> list[dict[str, Any]]:
    lookup = {case["dataset_key"]: case for case in _discover_default_pd_cases(base_dir)}
    return [lookup[key] for key in dataset_keys if key in lookup]


def _selected_cases(base_dir: str | Path, dataset_keys: list[str], raw_input: str | None) -> list[dict[str, Any]]:
    selected = _selected_default_cases(base_dir, dataset_keys)
    custom_cases = _discover_custom_pd_cases(raw_input or "")
    deduped: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for case in selected + custom_cases:
        path = str(case.get("path", ""))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        deduped.append(case)
    return deduped


def _normalize_channel(channel: str | None) -> str:
    value = str(channel or "").strip().upper()
    return value if value in DEFAULT_PD_CHANNELS else "CH3"


def _case_payload_for_channel(case: dict[str, Any], channel: str) -> dict[str, Any] | None:
    channel_paths = dict(case.get("channel_paths", {}))
    file_path = channel_paths.get(channel)
    if not file_path:
        file_path = case.get("file_path")
    if not file_path:
        return None
    return {
        "dataset_key": case["dataset_key"],
        "folder": case.get("folder", ""),
        "discharge_type": case.get("discharge_type", "unknown"),
        "variant": case.get("variant", "unknown"),
        "file_path": str(file_path),
    }


def _comparative_is_supported(cases: list[dict[str, Any]]) -> bool:
    if len(cases) < 2:
        return False
    if any(bool(case.get("is_custom")) for case in cases):
        return False
    for case in cases:
        if str(case.get("discharge_type", "")).strip().lower() in {"", "unknown"}:
            return False
        if str(case.get("variant", "")).strip().lower() in {"", "unknown"}:
            return False
    return True


def _comparative_tasks(cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected_keys = [case["dataset_key"] for case in cases]
    discharge_types = sorted({str(case["discharge_type"]) for case in cases if str(case.get("discharge_type", "")).strip()})
    variants = sorted({str(case["variant"]) for case in cases if str(case.get("variant", "")).strip()})
    tasks: dict[str, Any] = {
        f"dataset{len(selected_keys)}": {
            "type": "multiclass" if len(selected_keys) > 2 else "binary",
            "label_column": "dataset_key",
        }
    }
    if len(discharge_types) >= 2:
        tasks[f"type{len(discharge_types)}"] = {
            "type": "multiclass" if len(discharge_types) > 2 else "binary",
            "label_column": "discharge_type",
            **({"positive_values": [discharge_types[-1]]} if len(discharge_types) == 2 else {}),
        }
    if len(variants) >= 2:
        tasks["variant2"] = {
            "type": "binary",
            "label_column": "acquisition_variant",
            "positive_values": ["gemela"],
        }
    return tasks


def _state_alarm_config_dict(
    *,
    base_dir: Path,
    output_root: Path,
    channel: str,
    channel_cases: list[dict[str, Any]],
    material_defaults: dict[str, Any],
    study_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "study_name": f"Workbench_State_Alarm_Selection_{channel}",
        "base_dir": str(base_dir),
        "channel": channel,
        "output_root": str(output_root),
        "datasets": channel_cases,
        "material_defaults": material_defaults,
    }
    if study_defaults:
        config["study_defaults"] = study_defaults
    return config


def _load_state_alarm_summary_rows(output_root: Path) -> list[dict[str, Any]]:
    manifest_path = output_root / "state_alarm_batch_manifest.json"
    summary_csv_path = output_root / "state_alarm_batch_summary.csv"
    if not manifest_path.exists() or not summary_csv_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_df = pd.read_csv(summary_csv_path)
    transition_case_path = output_root / "transition_overlap_case_summary.csv"
    if transition_case_path.exists():
        transition_df = pd.read_csv(transition_case_path)
        if "dataset_key" in transition_df.columns:
            transition_df = transition_df.drop(columns=[col for col in ["discharge_type", "variant"] if col in transition_df.columns])
            summary_df = summary_df.merge(transition_df, on="dataset_key", how="left")
    summary_df = summary_df.fillna("")
    summary_rows = summary_df.to_dict("records")
    for case in manifest.get("cases", []):
        dataset_key = str(case.get("dataset_key", ""))
        material_dir = Path(str(case.get("material_output_dir", "")))
        material_manifest_path = material_dir / "run_manifest.json"
        if not material_manifest_path.exists():
            continue
        material_manifest = json.loads(material_manifest_path.read_text(encoding="utf-8"))
        ingestion_audit = dict(material_manifest.get("ingestion_audit", {}))
        ingestion_confidence, ingestion_flags = ingestion_audit_confidence(ingestion_audit)
        case_row = next((row for row in summary_rows if str(row.get("dataset_key", "")) == dataset_key), None)
        if case_row is None:
            continue
        case_row["ingestion_confidence"] = float(ingestion_confidence)
        case_row["ingestion_flags"] = ", ".join(ingestion_flags)
        case_row["ingestion_loader_mode"] = str(ingestion_audit.get("loader_mode", ""))
        case_row["ingestion_fs_source"] = str(ingestion_audit.get("fs_source", ""))
    return summary_rows


def _write_sensitivity_heatmap(master_df: pd.DataFrame, out_png: Path) -> Path | None:
    if master_df.empty:
        return None
    pivot = (
        master_df.pivot_table(index="dataset_key", columns="scenario_key", values="semaphore_risk_score", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.9), max(3.6, pivot.shape[0] * 0.7)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1]), labels=list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]), labels=list(pivot.index))
    ax.set_title("CH3 semaphore sensitivity risk heatmap")
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iat[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="#10263f", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Risk score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _write_sensitivity_stability(case_df: pd.DataFrame, out_png: Path) -> Path | None:
    if case_df.empty:
        return None
    plot_df = case_df.sort_values("band_stability", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(plot_df) * 0.6)))
    ax.barh(plot_df["dataset_key"], plot_df["band_stability"], color="#1f6b72", alpha=0.88)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Band stability share")
    ax.set_title("CH3 semaphore stability by dataset")
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            min(float(row["band_stability"]) + 0.02, 0.98),
            idx,
            f"{row['modal_band']} | gray={int(row['gray_count'])}",
            va="center",
            fontsize=8,
            color="#10263f",
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _write_sensitivity_markdown(
    output_root: Path,
    *,
    cases: list[dict[str, Any]],
    scenario_df: pd.DataFrame,
    case_df: pd.DataFrame,
    repeatability: dict[str, Any],
) -> Path:
    lines = [
        "# CH3 Semaphore Sensitivity",
        "",
        "This sweep calibrates the exploratory semaphore only on CH3. It is intended as a thesis robustness check, not as a new main result.",
        "",
        f"- cases: {', '.join(str(case.get('dataset_key', '')) for case in cases)}",
        f"- scenarios: {int(len(scenario_df))}",
        f"- repeatability max abs risk delta: {float(repeatability.get('max_abs_risk_delta', float('nan'))):.4f}",
        f"- repeatability all bands stable: {bool(repeatability.get('all_bands_match', False))}",
        "",
        "## Dataset stability",
        "",
    ]
    if case_df.empty:
        lines.append("_No case-level sensitivity rows were produced._")
    else:
        for _, row in case_df.sort_values(["band_stability", "dataset_key"], ascending=[False, True]).iterrows():
            lines.append(
                "- "
                + f"{row['dataset_key']}: modal_band={row['modal_band']}, "
                + f"band_stability={float(row['band_stability']):.3f}, "
                + f"risk_std={float(row['risk_std']):.4f}, "
                + f"gray_count={int(row['gray_count'])}, "
                + f"baseline={row['baseline_band']}"
            )
    lines.extend(["", "## Scenario summary", ""])
    if scenario_df.empty:
        lines.append("_No scenario summary available._")
    else:
        for _, row in scenario_df.sort_values("scenario_order").iterrows():
            lines.append(
                "- "
                + f"{row['scenario_label']}: mean_risk={float(row['mean_risk']):.4f}, "
                + f"mean_confidence={float(row['mean_confidence']):.4f}, "
                + f"gray_cases={int(row['gray_count'])}, "
                + f"mean_abs_delta_vs_baseline={float(row['mean_abs_risk_delta_vs_baseline']):.4f}"
            )
    path = output_root / "semaphore_sensitivity_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_sensitivity_job(
    *,
    repo_root: Path,
    base_dir: str | Path,
    dataset_keys: list[str],
    channel: str | None = None,
    raw_input: str | None = None,
) -> dict[str, Any]:
    selected_channel = _normalize_channel(channel)
    if selected_channel != "CH3":
        raise ValueError("Semaphore sensitivity is calibrated only for CH3 in thesis mode.")

    base_dir = Path(base_dir).expanduser()
    selected_cases = _selected_cases(base_dir, dataset_keys, raw_input)
    if not selected_cases:
        raise ValueError("No valid PD datasets or custom CSV paths were selected.")

    channel_cases = []
    for case in selected_cases:
        payload = _case_payload_for_channel(case, selected_channel)
        if payload is not None:
            channel_cases.append(payload)
    if not channel_cases:
        raise ValueError("No selected cases contained CH3.")

    job_id = f"sensitivity_ch3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = _job_dir(repo_root, job_id)
    job_manifest_path = _job_manifest_path(job_dir)
    manifest = {
        "job_id": job_id,
        "mode": WORKBENCH_JOB_MODE_SENSITIVITY,
        "channel": selected_channel,
        "status": JOB_STATUS_PENDING,
        "created_at": _timestamp_now(),
        "started_at": "",
        "finished_at": "",
        "params": {
            "repo_root": str(repo_root),
            "base_dir": str(base_dir),
            "dataset_keys": [str(key) for key in dataset_keys],
            "raw_input": str(raw_input or ""),
            "channel": selected_channel,
        },
        "cases": channel_cases,
        "progress_current": 0,
        "progress_total": int(len(THESIS_SENSITIVITY_SCENARIOS)),
        "message": "Queued CH3 semaphore sensitivity battery.",
        "output_root": "",
        "stable_output_root": "",
        "error": "",
    }
    write_workbench_job_manifest(job_manifest_path, manifest)
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "deltapd.workbench_worker",
            "--job-manifest",
            str(job_manifest_path),
        ],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    manifest["job_manifest_path"] = str(job_manifest_path)
    manifest["job_dir"] = str(job_dir)
    manifest["progress_label"] = _job_progress_label(manifest)
    return manifest


def run_pd_sensitivity(
    *,
    repo_root: Path,
    base_dir: str | Path,
    dataset_keys: list[str],
    channel: str | None = None,
    raw_input: str | None = None,
    output_root: Path | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    base_dir = Path(base_dir).expanduser()
    selected_channel = _normalize_channel(channel)
    if selected_channel != "CH3":
        raise ValueError("Semaphore sensitivity is calibrated only for CH3 in thesis mode.")

    selected_cases = _selected_cases(base_dir, dataset_keys, raw_input)
    if not selected_cases:
        raise ValueError("No valid PD datasets or custom CSV paths were selected.")
    channel_cases = []
    for case in selected_cases:
        payload = _case_payload_for_channel(case, selected_channel)
        if payload is not None:
            channel_cases.append(payload)
    if not channel_cases:
        raise ValueError("No selected cases contained CH3.")

    output_root = output_root or _runtime_dir(repo_root, "semaphore_sensitivity_ch3")
    scenario_records: list[dict[str, Any]] = []
    scenario_summary_rows: list[dict[str, Any]] = []

    baseline_lookup: dict[str, dict[str, Any]] = {}
    progress_total = len(THESIS_SENSITIVITY_SCENARIOS)
    if callable(progress_callback):
        progress_callback(
            0,
            progress_total,
            f"Running CH3 semaphore sensitivity battery for {len(channel_cases)} cases.",
            output_root=output_root,
            stable_output_root=None,
        )

    for scenario_order, scenario in enumerate(THESIS_SENSITIVITY_SCENARIOS, start=1):
        scenario_root = output_root / scenario["scenario_key"]
        scenario_root.mkdir(parents=True, exist_ok=True)
        material_defaults = _merge_dict(DEFAULT_MATERIAL_DEFAULTS, scenario.get("material_defaults", {}))
        study_defaults = _merge_dict(DEFAULT_STUDY_DEFAULTS, scenario.get("study_defaults", {}))
        state_config = _state_alarm_config_dict(
            base_dir=base_dir,
            output_root=scenario_root,
            channel=selected_channel,
            channel_cases=channel_cases,
            material_defaults=material_defaults,
            study_defaults=study_defaults,
        )
        config_path = scenario_root / f"{scenario['scenario_key']}.yaml"
        config_path.write_text(yaml.safe_dump(state_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
        run_state_alarm_batch(config_path)

        summary_rows = _load_state_alarm_summary_rows(scenario_root)
        semaphore_df = build_semaphore_df(summary_rows)
        if semaphore_df.empty:
            continue
        semaphore_df = semaphore_df.copy()
        semaphore_df["scenario_key"] = scenario["scenario_key"]
        semaphore_df["scenario_label"] = scenario["label"]
        semaphore_df["scenario_notes"] = scenario["notes"]
        semaphore_df["scenario_order"] = scenario_order
        semaphore_df["k_sigma"] = float(material_defaults.get("detection", {}).get("k_sigma", 5.0))
        semaphore_df["wavelet_denoise"] = bool(material_defaults.get("preprocess", {}).get("wavelet_denoise", True))
        semaphore_df["blind_method"] = str(
            material_defaults.get("analysis", {}).get("blind_prpd", {}).get("calibration_method", "auto")
        )
        semaphore_df["window_events"] = int(study_defaults.get("windowing", {}).get("window_events", 64))
        semaphore_df["step_events"] = int(study_defaults.get("windowing", {}).get("step_events", 16))
        semaphore_df["scenario_output_root"] = str(scenario_root)
        scenario_records.extend(semaphore_df.to_dict("records"))

        if scenario["scenario_key"] == "baseline":
            baseline_lookup = {
                str(row["dataset_key"]): {
                    "risk": float(row["semaphore_risk_score"]),
                    "band": str(row["semaphore_band"]),
                    "confidence": float(row["semaphore_confidence_score"]),
                }
                for row in semaphore_df.to_dict("records")
            }

        abs_deltas = []
        for row in semaphore_df.to_dict("records"):
            baseline_row = baseline_lookup.get(str(row["dataset_key"]))
            if baseline_row is not None:
                abs_deltas.append(abs(float(row["semaphore_risk_score"]) - baseline_row["risk"]))
        band_counts = pd.Series(semaphore_df["semaphore_band"]).value_counts().to_dict()
        scenario_summary_rows.append(
            {
                "scenario_key": scenario["scenario_key"],
                "scenario_label": scenario["label"],
                "scenario_notes": scenario["notes"],
                "scenario_order": scenario_order,
                "k_sigma": float(material_defaults.get("detection", {}).get("k_sigma", 5.0)),
                "wavelet_denoise": bool(material_defaults.get("preprocess", {}).get("wavelet_denoise", True)),
                "blind_method": str(material_defaults.get("analysis", {}).get("blind_prpd", {}).get("calibration_method", "auto")),
                "window_events": int(study_defaults.get("windowing", {}).get("window_events", 64)),
                "step_events": int(study_defaults.get("windowing", {}).get("step_events", 16)),
                "mean_risk": float(pd.to_numeric(semaphore_df["semaphore_risk_score"], errors="coerce").mean()),
                "mean_confidence": float(pd.to_numeric(semaphore_df["semaphore_confidence_score"], errors="coerce").mean()),
                "gray_count": int(band_counts.get("gray", 0)),
                "red_count": int(band_counts.get("red", 0)),
                "yellow_count": int(band_counts.get("yellow", 0)),
                "green_count": int(band_counts.get("green", 0)),
                "mean_abs_risk_delta_vs_baseline": float(pd.Series(abs_deltas, dtype=float).mean()) if abs_deltas else 0.0,
            }
        )
        if callable(progress_callback):
            progress_callback(
                scenario_order,
                progress_total,
                f"Completed scenario {scenario_order}/{progress_total}: {scenario['label']}.",
                output_root=output_root,
                stable_output_root=None,
            )

    master_df = pd.DataFrame(scenario_records)
    scenario_df = pd.DataFrame(scenario_summary_rows)
    if not scenario_df.empty and "scenario_order" in scenario_df.columns:
        scenario_df = scenario_df.sort_values("scenario_order").reset_index(drop=True)
    case_rows: list[dict[str, Any]] = []
    if not master_df.empty:
        for dataset_key, df_case in master_df.groupby("dataset_key"):
            band_mode = pd.Series(df_case["semaphore_band"]).mode()
            modal_band = str(band_mode.iloc[0]) if not band_mode.empty else ""
            baseline_row = baseline_lookup.get(str(dataset_key), {})
            case_rows.append(
                {
                    "dataset_key": str(dataset_key),
                    "scenario_count": int(len(df_case)),
                    "modal_band": modal_band,
                    "band_stability": float((df_case["semaphore_band"] == modal_band).mean()),
                    "gray_count": int((df_case["semaphore_band"] == "gray").sum()),
                    "risk_mean": float(pd.to_numeric(df_case["semaphore_risk_score"], errors="coerce").mean()),
                    "risk_std": float(pd.to_numeric(df_case["semaphore_risk_score"], errors="coerce").std(ddof=0)),
                    "confidence_mean": float(pd.to_numeric(df_case["semaphore_confidence_score"], errors="coerce").mean()),
                    "baseline_band": str(baseline_row.get("band", "")),
                    "baseline_risk": float(baseline_row.get("risk", float("nan"))),
                    "baseline_confidence": float(baseline_row.get("confidence", float("nan"))),
                }
            )
    case_df = pd.DataFrame(case_rows).sort_values("dataset_key").reset_index(drop=True) if case_rows else pd.DataFrame()

    repeatability = {"all_bands_match": False, "max_abs_risk_delta": float("nan")}
    if not master_df.empty:
        baseline_df = master_df[master_df["scenario_key"] == "baseline"][["dataset_key", "semaphore_band", "semaphore_risk_score"]]
        repeat_df = master_df[master_df["scenario_key"] == "baseline_repeat"][["dataset_key", "semaphore_band", "semaphore_risk_score"]]
        if not baseline_df.empty and not repeat_df.empty:
            merged = baseline_df.merge(
                repeat_df,
                on="dataset_key",
                suffixes=("_baseline", "_repeat"),
            )
            if not merged.empty:
                repeatability = {
                    "all_bands_match": bool((merged["semaphore_band_baseline"] == merged["semaphore_band_repeat"]).all()),
                    "max_abs_risk_delta": float(
                        (pd.to_numeric(merged["semaphore_risk_score_baseline"], errors="coerce")
                        - pd.to_numeric(merged["semaphore_risk_score_repeat"], errors="coerce")).abs().max()
                    ),
                }

    master_csv = output_root / "semaphore_sensitivity_master.csv"
    case_csv = output_root / "semaphore_sensitivity_case_summary.csv"
    scenario_csv = output_root / "semaphore_sensitivity_scenario_summary.csv"
    repeat_json = output_root / "semaphore_sensitivity_repeatability.json"
    if not master_df.empty:
        master_df.to_csv(master_csv, index=False, encoding="utf-8-sig")
    if not case_df.empty:
        case_df.to_csv(case_csv, index=False, encoding="utf-8-sig")
    if not scenario_df.empty:
        scenario_df.to_csv(scenario_csv, index=False, encoding="utf-8-sig")
    repeat_json.write_text(json.dumps(repeatability, indent=2), encoding="utf-8")

    heatmap_png = _write_sensitivity_heatmap(master_df, output_root / "semaphore_sensitivity_heatmap.png")
    stability_png = _write_sensitivity_stability(case_df, output_root / "semaphore_sensitivity_stability.png")
    summary_md = _write_sensitivity_markdown(
        output_root,
        cases=channel_cases,
        scenario_df=scenario_df,
        case_df=case_df,
        repeatability=repeatability,
    )

    manifest = {
        "channel": selected_channel,
        "output_root": str(output_root),
        "cases": channel_cases,
        "scenario_count": int(len(scenario_df)),
        "master_csv": str(master_csv) if master_csv.exists() else "",
        "case_summary_csv": str(case_csv) if case_csv.exists() else "",
        "scenario_summary_csv": str(scenario_csv) if scenario_csv.exists() else "",
        "repeatability_json": str(repeat_json),
        "summary_md": str(summary_md),
        "heatmap_png": str(heatmap_png) if heatmap_png is not None else "",
        "stability_png": str(stability_png) if stability_png is not None else "",
    }
    manifest_path = output_root / "semaphore_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stable_root = _sync_output_tree(output_root, repo_root / "outputs" / "semaphore_sensitivity_ch3")
    if callable(progress_callback):
        progress_callback(
            progress_total,
            progress_total,
            f"Generated CH3 semaphore sensitivity battery with {len(scenario_df)} scenarios.",
            output_root=output_root,
            stable_output_root=stable_root,
        )
    return {
        "mode": "semaphore_sensitivity",
        "channel": selected_channel,
        "output_root": output_root,
        "stable_output_root": stable_root,
        "manifest_path": manifest_path,
        "message": f"Generated CH3 semaphore sensitivity battery with {len(scenario_df)} scenarios.",
    }


def run_pd_selection(
    *,
    repo_root: Path,
    base_dir: str | Path,
    dataset_keys: list[str],
    channel: str | None = None,
    raw_input: str | None = None,
    k_sigma: float = 5.0,
    wavelet_denoise: bool = True,
) -> dict[str, Any]:
    base_dir = Path(base_dir).expanduser()
    selected_cases = _selected_cases(base_dir, dataset_keys, raw_input)
    if not selected_cases:
        raise ValueError("No valid PD datasets or custom CSV paths were selected.")

    selected_channel = _normalize_channel(channel)
    channel_cases = []
    for case in selected_cases:
        payload = _case_payload_for_channel(case, selected_channel)
        if payload is not None:
            channel_cases.append(payload)
    if not channel_cases:
        raise ValueError(f"No selected cases contained {selected_channel}.")

    state_output_root = _runtime_dir(repo_root, f"state_alarm_selection_{selected_channel.lower()}")
    state_config = {
        "study_name": f"Workbench_State_Alarm_Selection_{selected_channel}",
        "base_dir": str(base_dir),
        "channel": selected_channel,
        "output_root": str(state_output_root),
        "datasets": channel_cases,
        "material_defaults": {
            "preprocess": {"wavelet_denoise": bool(wavelet_denoise)},
            "detection": {"k_sigma": float(k_sigma)},
        },
    }
    state_config_path = state_output_root / f"workbench_state_alarm_{selected_channel.lower()}.yaml"
    state_config_path.write_text(
        yaml.safe_dump(state_config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    state_outputs = run_state_alarm_batch(state_config_path)

    comparative_roots: dict[str, Path] = {}
    channel_source_cases = [case for case in selected_cases if selected_channel in case.get("available_channels", [])]
    if _comparative_is_supported(channel_source_cases):
        selected_keys = [case["dataset_key"] for case in channel_source_cases]
        comparative_output_root = _runtime_dir(repo_root, f"comparative_selection_{selected_channel.lower()}")
        comparative_config = {
            "study_name": f"Workbench_Comparative_PD_Selection_{selected_channel}",
            "input": {
                "points_csv": "outputs/thesis_master/thesis_master_prpd_points.csv",
                "channel": selected_channel,
                "dataset_keys": selected_keys,
            },
            "tasks": _comparative_tasks(channel_source_cases),
            "state_alarm_root": str(state_outputs["output_root"]),
            "output_dir": str(comparative_output_root),
            "report": {
                "export_pdf": True,
                "pdf_filename": f"comparative_selection_{selected_channel.lower()}_report.pdf",
            },
        }
        comparative_config_path = comparative_output_root / f"workbench_comparative_{selected_channel.lower()}.yaml"
        comparative_config_path.write_text(
            yaml.safe_dump(comparative_config, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        comparative_outputs = run_comparative_thesis_study(comparative_config_path)
        comparative_roots[selected_channel] = comparative_outputs["output_dir"]
        message = f"Generated {selected_channel}: state/alarm + comparative."
    else:
        message = f"Generated {selected_channel}: state/alarm."

    return {
        "mode": "single_channel",
        "state_alarm_roots": {selected_channel: state_outputs["output_root"]},
        "comparative_roots": comparative_roots,
        "channel": selected_channel,
        "message": message,
    }


def expand_vna_inputs(raw_text: str) -> list[Path]:
    paths: list[Path] = []
    for line in raw_text.splitlines():
        text = line.strip().strip('"')
        if not text:
            continue
        candidate = Path(text).expanduser()
        if candidate.is_dir():
            for ext in sorted(VNA_EXTENSIONS):
                paths.extend(sorted(candidate.rglob(f"*{ext}")))
            continue
        if candidate.exists():
            paths.append(candidate)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def run_vna_selection(
    *,
    repo_root: Path,
    raw_input: str,
) -> dict[str, Any]:
    file_paths = expand_vna_inputs(raw_input)
    if not file_paths:
        raise ValueError("No valid VNA file or folder paths were provided.")
    output_root = _runtime_dir(repo_root, "vna_selection")
    outputs = analyze_vna_selection(file_paths, output_root)
    return {
        "mode": "vna",
        "vna_root": outputs["output_root"],
        "message": f"VNA analysis generated for {len(file_paths)} file(s).",
    }
