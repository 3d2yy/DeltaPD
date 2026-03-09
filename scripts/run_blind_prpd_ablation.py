from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from deltapd.campaign.comparative_thesis_study import run_comparative_thesis_study
from deltapd.campaign.config import load_config
from deltapd.campaign.state_alarm_batch import (
    DEFAULT_DATASETS,
    DEFAULT_MATERIAL_DEFAULTS,
    DEFAULT_STUDY_DEFAULTS,
    run_state_alarm_batch,
)
from export_thesis_master_table import export_master_tables


METHODS = ("coherence", "harmonic_power", "epoch_folding", "gregory_loredo", "auto")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _single_channel_thesis_config(config_path: Path, output_path: Path, channel: str) -> Path:
    cfg = load_config(config_path)
    datasets = copy.deepcopy(cfg.get("datasets", {}))
    for payload in datasets.values():
        channel_map = dict(payload.get("channel_map", {}))
        payload["channel_map"] = {channel: channel_map[channel]} if channel in channel_map else {}
    filtered_cfg = dict(cfg)
    filtered_cfg["datasets"] = datasets
    output_path.write_text(yaml.safe_dump(filtered_cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return output_path


def _recommendation_score(recommendations: dict[str, Any], task_name: str) -> tuple[float, str]:
    rec = recommendations.get(task_name, {}).get("recommendation", {})
    score = float(rec.get("primary_score", float("nan")))
    features = ", ".join(rec.get("features", []))
    return score, features


def _build_report(
    *,
    output_root: Path,
    channel: str,
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
) -> Path:
    type_rank = summary_df.sort_values("type3_primary_score", ascending=False)
    state_rank = summary_df.sort_values("mean_state_score", ascending=False)
    alarm_rank = summary_df.sort_values("mean_alarm_score", ascending=False)
    p3_rows = case_df[case_df["dataset_key"] == "P3"].sort_values("phase_entropy_global", ascending=False)
    g3_rows = case_df[case_df["dataset_key"] == "G3"].sort_values("phase_entropy_global", ascending=False)

    lines = [
        "# Blind PRPD method ablation",
        "",
        "## What this run compares",
        "",
        f"- Fixed-method blind PRPD calibration on `{channel}` for `coherence`, `harmonic_power`, `epoch_folding`, `gregory_loredo`, and `auto`.",
        "- Same downstream structure for all methods: `state/alarm` within each test and `type` comparative across `P1/P2/P3/G1/G2/G3`.",
        "",
        "## Type ranking",
        "",
        "| Method | type3 primary score | type3 features |",
        "| --- | ---: | --- |",
    ]
    for _, row in type_rank.iterrows():
        lines.append(
            f"| {row['method']} | {row['type3_primary_score']:.4f} | {row['type3_features']} |"
        )

    lines.extend(
        [
            "",
            "## State/alarm ranking",
            "",
            "| Method | Mean state score | Mean alarm score |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in state_rank.iterrows():
        lines.append(
            f"| {row['method']} | {row['mean_state_score']:.4f} | {row['mean_alarm_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## P3 reading",
            "",
            "| Method | Blind freq (Hz) | Phase entropy | Phase spread (deg) | Inlier ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in p3_rows.iterrows():
        lines.append(
            f"| {row['method']} | {row['blind_freq_hz']:.6f} | {row['phase_entropy_global']:.6f} | "
            f"{row['phase_spread_deg']:.6f} | {row['inlier_ratio']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## G3 reading",
            "",
            "| Method | Blind freq (Hz) | Phase entropy | Phase spread (deg) | Inlier ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in g3_rows.iterrows():
        lines.append(
            f"| {row['method']} | {row['blind_freq_hz']:.6f} | {row['phase_entropy_global']:.6f} | "
            f"{row['phase_spread_deg']:.6f} | {row['inlier_ratio']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Working interpretation",
            "",
            "- `type3` is the key cross-test discriminator for the paper.",
            "- `mean_state_score` and `mean_alarm_score` say whether the same phase method remains useful inside each acquisition.",
            "- `P3` and `G3` are the decisive stress cases because multiple-source patterns are where the simple circular concentration baseline is weakest.",
            "",
        ]
    )
    path = output_root / "blind_prpd_ablation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_ablation(
    *,
    base_dir: Path,
    thesis_config: Path,
    output_root: Path,
    methods: list[str],
    channel: str,
    n_harmonics: int = 4,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    thesis_cfg_single_channel = _single_channel_thesis_config(
        thesis_config,
        generated_dir / f"config_thesis_{channel.lower()}_only.yaml",
        channel,
    )
    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for method in methods:
        method_root = output_root / method
        method_root.mkdir(parents=True, exist_ok=True)

        material_defaults = _deep_merge(
            DEFAULT_MATERIAL_DEFAULTS,
            {
                "analysis": {
                    "export_sensitivity_report": False,
                    "blind_prpd": {
                        "calibration_method": method,
                        "n_harmonics": n_harmonics,
                    },
                },
                "plots": {
                    "show_raw_with_detections": False,
                    "show_delta_t_series": False,
                    "show_delta_t_hist": False,
                    "show_rate_series": False,
                    "show_rolling_stats": False,
                    "show_ewma_cusum": False,
                    "show_advanced_stats": False,
                    "show_blind_prpd": False,
                },
            },
        )
        study_defaults = _deep_merge(
            DEFAULT_STUDY_DEFAULTS,
            {
                "report": {
                    "export_pdf": False,
                }
            },
        )
        state_alarm_cfg = {
            "study_name": f"BlindPRPD_Ablation_{method}_{channel}",
            "base_dir": str(base_dir),
            "channel": channel,
            "output_root": str(method_root / f"state_alarm_{channel.lower()}"),
            "datasets": DEFAULT_DATASETS,
            "material_defaults": material_defaults,
            "study_defaults": study_defaults,
        }
        state_alarm_cfg_path = generated_dir / f"state_alarm_{method}.yaml"
        state_alarm_cfg_path.write_text(
            yaml.safe_dump(state_alarm_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        state_outputs = run_state_alarm_batch(state_alarm_cfg_path)

        thesis_output_dir = method_root / "thesis_master"
        thesis_output_dir.mkdir(parents=True, exist_ok=True)
        thesis_metrics_df, _, _ = export_master_tables(
            config_path=thesis_cfg_single_channel,
            base_dir=base_dir,
            out_dir=thesis_output_dir,
            threshold_sigma=5.0,
            min_separation_s=20e-9,
            blind_prpd_method=method,
            blind_prpd_harmonics=n_harmonics,
        )

        comparative_cfg = {
            "study_name": f"BlindPRPD_Comparative_{method}_{channel}",
            "input": {
                "points_csv": str(thesis_output_dir / "thesis_master_prpd_points.csv"),
                "channel": channel,
                "dataset_keys": ["P1", "P2", "P3", "G1", "G2", "G3"],
            },
            "output_dir": str(method_root / f"comparative_{channel.lower()}"),
            "report": {
                "export_pdf": False,
            },
        }
        comparative_cfg_path = generated_dir / f"comparative_{method}.yaml"
        comparative_cfg_path.write_text(
            yaml.safe_dump(comparative_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        comparative_outputs = run_comparative_thesis_study(comparative_cfg_path)

        state_summary_df = pd.read_csv(state_outputs["summary_csv"])
        recommendations = comparative_outputs["recommendations"]
        type3_score, type3_features = _recommendation_score(recommendations, "type3")
        dataset6_score, dataset6_features = _recommendation_score(recommendations, "dataset6")
        variant2_score, variant2_features = _recommendation_score(recommendations, "variant2")
        summary_rows.append(
            {
                "method": method,
                "mean_state_score": float(pd.to_numeric(state_summary_df["state_primary_score"], errors="coerce").mean()),
                "mean_alarm_score": float(pd.to_numeric(state_summary_df["alarm_primary_score"], errors="coerce").mean()),
                "type3_primary_score": type3_score,
                "type3_features": type3_features,
                "dataset6_primary_score": dataset6_score,
                "dataset6_features": dataset6_features,
                "variant2_primary_score": variant2_score,
                "variant2_features": variant2_features,
                "state_alarm_root": str(state_outputs["output_root"]),
                "comparative_root": str(comparative_outputs["output_dir"]),
                "thesis_master_root": str(thesis_output_dir),
            }
        )

        thesis_metrics_df = thesis_metrics_df.copy()
        thesis_metrics_df["method"] = method
        case_rows.extend(thesis_metrics_df.to_dict("records"))

    summary_df = pd.DataFrame(summary_rows).sort_values("type3_primary_score", ascending=False).reset_index(drop=True)
    case_df = pd.DataFrame(case_rows).sort_values(["dataset_key", "method"]).reset_index(drop=True)
    summary_csv = output_root / "blind_prpd_ablation_summary.csv"
    case_csv = output_root / "blind_prpd_ablation_case_metrics.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    case_df.to_csv(case_csv, index=False, encoding="utf-8-sig")
    report_md = _build_report(output_root=output_root, channel=channel, summary_df=summary_df, case_df=case_df)

    manifest = {
        "base_dir": str(base_dir),
        "thesis_config": str(thesis_config),
        "channel": channel,
        "methods": methods,
        "n_harmonics": n_harmonics,
        "summary_csv": str(summary_csv),
        "case_csv": str(case_csv),
        "report_md": str(report_md),
    }
    manifest_path = output_root / "blind_prpd_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "case_csv": case_csv,
        "report_md": report_md,
        "manifest_path": manifest_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run downstream blind PRPD method ablation on thesis datasets for one channel.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory containing thesis datasets.",
    )
    parser.add_argument(
        "--thesis-config",
        type=Path,
        default=Path("campaign/config_thesis.yaml"),
        help="Thesis dataset config used to build single-channel event tables.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "outputs" / "blind_prpd_ablation_ch3",
        help="Output folder for the ablation study.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="CH3",
        help="Channel to evaluate, for example CH2, CH3 or CH4.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Blind PRPD methods to compare.",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=4,
        help="Number of harmonics used by harmonic-power calibration.",
    )
    args = parser.parse_args()

    outputs = run_ablation(
        base_dir=args.base_dir,
        thesis_config=args.thesis_config,
        output_root=args.output_root,
        methods=list(args.methods),
        channel=args.channel,
        n_harmonics=args.n_harmonics,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
