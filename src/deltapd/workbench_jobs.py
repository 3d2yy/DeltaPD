from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from deltapd.campaign.comparative_thesis_study import run_comparative_thesis_study
from deltapd.campaign.state_alarm_batch import DEFAULT_DATASETS, run_state_alarm_batch
from deltapd.vna import VNA_EXTENSIONS, analyze_vna_selection


def discover_pd_cases(base_dir: str | Path) -> list[dict[str, Any]]:
    base_dir = Path(base_dir).expanduser()
    cases = []
    for case in DEFAULT_DATASETS:
        folder = base_dir / case["folder"]
        ch3_path = folder / "CH3.csv"
        if ch3_path.exists():
            cases.append(
                {
                    **case,
                    "path": str(folder),
                    "channel_path": str(ch3_path),
                }
            )
    return cases


def _runtime_dir(repo_root: Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = repo_root / "outputs" / "workbench_runtime" / f"{name}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_pd_selection(
    *,
    repo_root: Path,
    base_dir: str | Path,
    dataset_keys: list[str],
    k_sigma: float = 5.0,
    wavelet_denoise: bool = True,
) -> dict[str, Any]:
    base_dir = Path(base_dir).expanduser()
    available = {case["dataset_key"]: case for case in DEFAULT_DATASETS}
    selected_cases = [available[key] for key in dataset_keys if key in available]
    if not selected_cases:
        raise ValueError("No valid PD datasets were selected.")

    if len(selected_cases) == 1:
        output_root = _runtime_dir(repo_root, "state_alarm_selection")
        config = {
            "study_name": "Workbench_State_Alarm_Selection",
            "base_dir": str(base_dir),
            "output_root": str(output_root),
            "datasets": selected_cases,
            "material_defaults": {
                "preprocess": {"wavelet_denoise": bool(wavelet_denoise)},
                "detection": {"k_sigma": float(k_sigma)},
            },
        }
        config_path = output_root / "workbench_state_alarm.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
        outputs = run_state_alarm_batch(config_path)
        return {
            "mode": "state_alarm",
            "config_path": config_path,
            "state_alarm_root": outputs["output_root"],
            "comparative_root": None,
            "message": f"State/alarm study generated for {selected_cases[0]['dataset_key']}.",
        }

    output_root = _runtime_dir(repo_root, "comparative_selection")
    selected_keys = [case["dataset_key"] for case in selected_cases]
    discharge_types = sorted({case["discharge_type"] for case in selected_cases})
    variants = sorted({case["variant"] for case in selected_cases})

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

    config = {
        "study_name": "Workbench_Comparative_PD_Selection",
        "input": {
            "points_csv": "outputs/thesis_master/thesis_master_prpd_points.csv",
            "channel": "CH3",
            "dataset_keys": selected_keys,
        },
        "tasks": tasks,
        "output_dir": str(output_root),
        "report": {"export_pdf": True, "pdf_filename": "comparative_selection_report.pdf"},
    }
    config_path = output_root / "workbench_comparative.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    outputs = run_comparative_thesis_study(config_path)
    return {
        "mode": "comparative",
        "config_path": config_path,
        "state_alarm_root": None,
        "comparative_root": outputs["output_dir"],
        "message": f"Comparative study generated for {', '.join(selected_keys)}.",
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
