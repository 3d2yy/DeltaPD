import json
import tempfile
from pathlib import Path

import pandas as pd

from deltapd.workbench import create_workbench_app, load_workbench_data


def test_load_workbench_data_discovers_state_alarm_and_comparative_outputs():
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        outputs_root = repo_root / "outputs"
        docs_root = repo_root / "docs"
        docs_root.mkdir(parents=True)
        (docs_root / "visual_workbench_spec.md").write_text("# spec\n", encoding="utf-8")

        state_root = outputs_root / "state_alarm_ch3"
        state_root.mkdir(parents=True)
        (state_root / "state_alarm_batch_manifest.json").write_text(
            json.dumps(
                {
                    "state_feature_counts": {"phase_entropy": 2},
                    "alarm_feature_counts": {"phase_width_pos_deg": 1},
                    "cases": [
                        {
                            "dataset_key": "P1",
                            "folder": "Prueba 1 - Internas",
                            "material_output_dir": str(state_root / "P1" / "material"),
                            "study_output_dir": str(state_root / "P1" / "study"),
                            "pdf_path": str(state_root / "P1" / "study" / "report.pdf"),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "dataset_key": "P1",
                    "folder": "Prueba 1 - Internas",
                    "discharge_type": "internal",
                    "variant": "benchmark",
                    "duration_s": 1.23,
                    "total_events": 20,
                    "state_features": "phase_entropy",
                    "state_primary_score": 0.8,
                    "alarm_features": "phase_width_pos_deg",
                    "alarm_primary_score": 0.9,
                }
            ]
        ).to_csv(state_root / "state_alarm_batch_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "dataset_key": "P1",
                    "max_abs_local_freq_offset_hz": 0.12,
                    "local_regime_transition_entropy": 0.34,
                    "local_regime_mean_run_length": 1.5,
                }
            ]
        ).to_csv(state_root / "transition_overlap_case_summary.csv", index=False)
        (state_root / "state_alarm_batch_summary.md").write_text("# batch\n", encoding="utf-8")
        p1_material = state_root / "P1" / "material"
        p1_study = state_root / "P1" / "study"
        p1_material.mkdir(parents=True)
        p1_study.mkdir(parents=True)
        (p1_material / "run_manifest.json").write_text(
            json.dumps(
                {
                    "blind_prpd": {"selected_method": "coherence"},
                    "ingestion_audit": {
                        "file_type": "csv",
                        "loader_mode": "generic_csv",
                        "delimiter": ";",
                        "fs_hz": 1.0e9,
                        "fs_source": "time_axis",
                        "signal_column_label": "CH3",
                    },
                }
            ),
            encoding="utf-8",
        )
        (p1_material / "08_blind_prpd_50hz.png").write_bytes(b"png")
        (p1_study / "study_report.md").write_text("# report\n", encoding="utf-8")
        (p1_study / "report.pdf").write_bytes(b"pdf")

        comparative_root = outputs_root / "comparative_ch3"
        comparative_root.mkdir(parents=True)
        (comparative_root / "comparative_summary.md").write_text("# comparative\n", encoding="utf-8")
        (comparative_root / "study_recommendations.json").write_text(
            json.dumps({"type3": {"recommendation": {"features": ["p90_dt_s"], "primary_score": 0.99}}}),
            encoding="utf-8",
        )
        (comparative_root / "comparative_ch3_descriptor_report.pdf").write_bytes(b"pdf")
        (comparative_root / "comparative_window_counts.png").write_bytes(b"png")

        sensitivity_root = outputs_root / "semaphore_sensitivity_ch3"
        sensitivity_root.mkdir(parents=True)
        (sensitivity_root / "semaphore_sensitivity_manifest.json").write_text(
            json.dumps({"channel": "CH3", "scenario_count": 2}),
            encoding="utf-8",
        )
        (sensitivity_root / "semaphore_sensitivity_summary.md").write_text("# sensitivity\n", encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "scenario_key": "baseline",
                    "scenario_label": "Baseline",
                    "scenario_order": 1,
                    "mean_risk": 0.44,
                    "mean_confidence": 0.88,
                    "gray_count": 0,
                }
            ]
        ).to_csv(sensitivity_root / "semaphore_sensitivity_scenario_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "dataset_key": "P1",
                    "modal_band": "yellow",
                    "band_stability": 0.9,
                    "gray_count": 0,
                }
            ]
        ).to_csv(sensitivity_root / "semaphore_sensitivity_case_summary.csv", index=False)
        (sensitivity_root / "semaphore_sensitivity_repeatability.json").write_text(
            json.dumps({"all_bands_match": True, "max_abs_risk_delta": 0.0}),
            encoding="utf-8",
        )
        (sensitivity_root / "semaphore_sensitivity_heatmap.png").write_bytes(b"png")

        jobs_root = outputs_root / "workbench_jobs" / "job_001"
        jobs_root.mkdir(parents=True)
        (jobs_root / "job_manifest.json").write_text(
            json.dumps(
                {
                    "job_id": "job_001",
                    "mode": "semaphore_sensitivity",
                    "status": "succeeded",
                    "created_at": "2026-03-09T10:30:00",
                    "progress_current": 12,
                    "progress_total": 12,
                    "message": "Finished.",
                    "output_root": str(sensitivity_root),
                    "stable_output_root": str(sensitivity_root),
                }
            ),
            encoding="utf-8",
        )

        vna_root = outputs_root / "workbench_vna"
        vna_root.mkdir(parents=True)
        (vna_root / "vna_manifest.json").write_text(
            json.dumps({"mode": "single", "modes_detected": ["s11"], "pdf_path": str(vna_root / "vna_report.pdf")}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "source_file": "antena.s1p",
                    "mode": "s11",
                    "freq_min_hz": 1e9,
                    "freq_max_hz": 2e9,
                    "min_s11_db": -18.0,
                }
            ]
        ).to_csv(vna_root / "vna_summary.csv", index=False)
        (vna_root / "vna_summary.md").write_text("# vna\n", encoding="utf-8")
        (vna_root / "vna_overview.png").write_bytes(b"png")
        (vna_root / "vna_report.pdf").write_bytes(b"pdf")

        data = load_workbench_data(
            repo_root=repo_root,
            state_alarm_root=state_root,
            comparative_root=comparative_root,
            sensitivity_root=sensitivity_root,
            vna_root=vna_root,
            pd_base_dir=repo_root,
        )

        assert data["active_channel"] == "CH3"
        assert "CH3" in data["available_channels"]
        assert data["state_alarm"]["available"] is True
        assert data["state_alarm"]["case_keys"] == ["P1"]
        assert data["state_alarm"]["summary_rows"][0]["max_abs_local_freq_offset_hz"] == 0.12
        assert data["state_alarm"]["cases"]["P1"]["pdf_url"].endswith("/outputs/state_alarm_ch3/P1/study/report.pdf")
        assert data["state_alarm"]["cases"]["P1"]["ingestion_audit"]["loader_mode"] == "generic_csv"
        assert data["comparative"]["available"] is True
        assert len(data["comparative"]["images"]) == 1
        assert data["sensitivity"]["available"] is True
        assert data["sensitivity"]["repeatability"]["all_bands_match"] is True
        assert len(data["sensitivity"]["images"]) == 1
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "job_001"
        assert data["vna"]["available"] is True
        assert len(data["vna"]["images"]) == 1
        assert data["vna"]["pdf_url"].endswith("/outputs/workbench_vna/vna_report.pdf")


def test_create_workbench_app_builds_layout_from_loaded_outputs():
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        (repo_root / "outputs" / "state_alarm_ch3").mkdir(parents=True)
        (repo_root / "outputs" / "comparative_ch3").mkdir(parents=True)
        (repo_root / "docs").mkdir(parents=True)
        (repo_root / "docs" / "visual_workbench_spec.md").write_text("# spec\n", encoding="utf-8")

        app = create_workbench_app(repo_root=repo_root, pd_base_dir=repo_root)
        assert app.layout is not None
