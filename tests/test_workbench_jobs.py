import tempfile
from pathlib import Path

import pandas as pd

from deltapd.workbench_jobs import create_sensitivity_job, discover_pd_cases, list_workbench_jobs, run_pd_selection, run_pd_sensitivity
from deltapd.workbench_worker import run_job


def _waveform_csv_text() -> str:
    rows = ["time,signal"]
    for idx in range(20):
        rows.append(f"{idx * 1e-9:.12g},{(idx % 5) * 0.1:.3f}")
    return "\n".join(rows)


def test_discover_pd_cases_reports_available_channels_and_custom_inputs():
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td)
        default_folder = base_dir / "Prueba 1 - Internas"
        default_folder.mkdir(parents=True)
        (default_folder / "CH2.csv").write_text("x", encoding="utf-8")
        (default_folder / "CH3.csv").write_text("x", encoding="utf-8")

        custom_folder = base_dir / "custom_case"
        custom_folder.mkdir(parents=True)
        (custom_folder / "CH4.csv").write_text(_waveform_csv_text(), encoding="utf-8")

        cases = discover_pd_cases(base_dir, raw_input=str(custom_folder))

        p1_case = next(case for case in cases if case["dataset_key"] == "P1")
        custom_case = next(case for case in cases if case["dataset_key"] == "C1")
        assert p1_case["available_channels"] == ["CH2", "CH3"]
        assert p1_case["is_custom"] is False
        assert custom_case["available_channels"] == ["CH4"]
        assert custom_case["is_custom"] is True


def test_run_pd_selection_runs_single_channel_state_and_comparative(monkeypatch):
    state_calls = []
    comparative_calls = []

    def fake_state_alarm(config_path):
        config_path = Path(config_path)
        output_root = config_path.parent
        state_calls.append(config_path)
        return {"output_root": output_root}

    def fake_comparative(config_path):
        config_path = Path(config_path)
        output_root = config_path.parent
        comparative_calls.append(config_path)
        return {"output_dir": output_root}

    monkeypatch.setattr("deltapd.workbench_jobs.run_state_alarm_batch", fake_state_alarm)
    monkeypatch.setattr("deltapd.workbench_jobs.run_comparative_thesis_study", fake_comparative)

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        for folder_name in ["Prueba 1 - Internas", "Prueba 2 - Superficiales"]:
            folder = base_dir / folder_name
            folder.mkdir(parents=True)
            for channel in ["CH2", "CH3"]:
                (folder / f"{channel}.csv").write_text("x", encoding="utf-8")

        outputs = run_pd_selection(
            repo_root=repo_root,
            base_dir=base_dir,
            dataset_keys=["P1", "P2"],
            channel="CH2",
        )

        assert sorted(outputs["state_alarm_roots"]) == ["CH2"]
        assert sorted(outputs["comparative_roots"]) == ["CH2"]
        assert outputs["channel"] == "CH2"
        assert len(state_calls) == 1
        assert len(comparative_calls) == 1


def test_run_pd_selection_falls_back_to_state_alarm_for_custom_paths(monkeypatch):
    state_calls = []
    comparative_calls = []

    def fake_state_alarm(config_path):
        config_path = Path(config_path)
        output_root = config_path.parent
        state_calls.append(config_path)
        return {"output_root": output_root}

    def fake_comparative(config_path):
        comparative_calls.append(Path(config_path))
        return {"output_dir": Path(config_path).parent}

    monkeypatch.setattr("deltapd.workbench_jobs.run_state_alarm_batch", fake_state_alarm)
    monkeypatch.setattr("deltapd.workbench_jobs.run_comparative_thesis_study", fake_comparative)

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        base_dir.mkdir(parents=True)
        custom_folder = repo_root / "incoming_case"
        custom_folder.mkdir(parents=True)
        (custom_folder / "CH4.csv").write_text(_waveform_csv_text(), encoding="utf-8")

        outputs = run_pd_selection(
            repo_root=repo_root,
            base_dir=base_dir,
            dataset_keys=[],
            channel="CH4",
            raw_input=str(custom_folder),
        )

        assert sorted(outputs["state_alarm_roots"]) == ["CH4"]
        assert outputs["comparative_roots"] == {}
        assert len(state_calls) == 1
        assert len(comparative_calls) == 0


def test_run_pd_selection_accepts_custom_csv_with_arbitrary_filename(monkeypatch):
    state_calls = []

    def fake_state_alarm(config_path):
        config_path = Path(config_path)
        state_calls.append(config_path)
        return {"output_root": config_path.parent}

    def fake_comparative(config_path):
        return {"output_dir": Path(config_path).parent}

    monkeypatch.setattr("deltapd.workbench_jobs.run_state_alarm_batch", fake_state_alarm)
    monkeypatch.setattr("deltapd.workbench_jobs.run_comparative_thesis_study", fake_comparative)

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        base_dir.mkdir(parents=True)
        custom_csv = repo_root / "incoming_case" / "capture_alpha.csv"
        custom_csv.parent.mkdir(parents=True)
        custom_csv.write_text(
            "time;signal\n"
            "0,0;0,00\n"
            "1,0E-09;0,10\n"
            "2,0E-09;0,40\n"
            "3,0E-09;-0,20\n"
            "4,0E-09;0,00\n"
            "5,0E-09;0,20\n"
            "6,0E-09;0,10\n"
            "7,0E-09;0,00\n"
            "8,0E-09;0,30\n"
            "9,0E-09;0,10\n"
            "10,0E-09;0,00\n"
            "11,0E-09;0,20\n"
            "12,0E-09;0,10\n"
            "13,0E-09;0,00\n"
            "14,0E-09;0,20\n"
            "15,0E-09;0,10\n",
            encoding="utf-8",
        )

        outputs = run_pd_selection(
            repo_root=repo_root,
            base_dir=base_dir,
            dataset_keys=[],
            channel="CH3",
            raw_input=str(custom_csv),
        )

        assert sorted(outputs["state_alarm_roots"]) == ["CH3"]
        assert outputs["comparative_roots"] == {}
        assert len(state_calls) == 1

        config_text = state_calls[0].read_text(encoding="utf-8")
        assert "capture_alpha.csv" in config_text
        assert "channel: CH3" in config_text


def _write_fake_state_alarm_outputs(output_root: Path, *, channel: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_keys = ["P1", "P2"]
    manifest_cases = []
    summary_rows = []
    transition_rows = []
    for idx, dataset_key in enumerate(dataset_keys, start=1):
        material_dir = output_root / dataset_key / "material"
        study_dir = output_root / dataset_key / "study"
        material_dir.mkdir(parents=True, exist_ok=True)
        study_dir.mkdir(parents=True, exist_ok=True)
        (material_dir / "run_manifest.json").write_text(
            '{"ingestion_audit":{"loader_mode":"generic_csv","fs_source":"time_axis","signal_column_label":"CH3","sample_count":512,"numeric_row_count":512}}',
            encoding="utf-8",
        )
        manifest_cases.append(
            {
                "dataset_key": dataset_key,
                "folder": f"Case {dataset_key}",
                "material_output_dir": str(material_dir),
                "study_output_dir": str(study_dir),
                "pdf_path": str(study_dir / "report.pdf"),
            }
        )
        summary_rows.append(
            {
                "dataset_key": dataset_key,
                "folder": f"Case {dataset_key}",
                "discharge_type": "internal" if dataset_key == "P1" else "superficial",
                "variant": "benchmark",
                "state_primary_score": 0.70 + 0.05 * idx,
                "alarm_primary_score": 0.60 + 0.05 * idx,
                "blind_local_method_agreement": 0.75,
            }
        )
        transition_rows.append(
            {
                "dataset_key": dataset_key,
                "max_abs_local_freq_offset_hz": 0.02 * idx,
                "local_freq_offset_std_hz": 0.01 * idx,
                "local_common_axial_confidence_std": 0.10 * idx,
                "transition_method_entropy": 0.20 * idx,
                "local_regime_transition_entropy": 0.15 * idx,
                "local_method_switch_rate": 0.10 * idx,
                "mean_local_common_axial_confidence": 0.70 - 0.05 * idx,
                "local_regime_mean_run_length": 2.0 + idx,
            }
        )
    (output_root / "state_alarm_batch_manifest.json").write_text(
        __import__("json").dumps({"channel": channel, "cases": manifest_cases}),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(output_root / "state_alarm_batch_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output_root / "transition_overlap_case_summary.csv", index=False)


def test_run_pd_sensitivity_generates_ch3_sweep_outputs(monkeypatch):
    def fake_state_alarm(config_path):
        config_path = Path(config_path)
        config_text = config_path.read_text(encoding="utf-8")
        assert "channel: CH3" in config_text
        assert "local_window_size_events: 256" in config_text
        assert "local_window_step_events: 128" in config_text
        _write_fake_state_alarm_outputs(config_path.parent, channel="CH3")
        return {"output_root": config_path.parent}

    monkeypatch.setattr("deltapd.workbench_jobs.run_state_alarm_batch", fake_state_alarm)

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        for folder_name in ["Prueba 1 - Internas", "Prueba 2 - Superficiales"]:
            folder = base_dir / folder_name
            folder.mkdir(parents=True)
            (folder / "CH3.csv").write_text(_waveform_csv_text(), encoding="utf-8")

        outputs = run_pd_sensitivity(
            repo_root=repo_root,
            base_dir=base_dir,
            dataset_keys=["P1", "P2"],
            channel="CH3",
        )

        assert outputs["channel"] == "CH3"
        assert outputs["output_root"].exists()
        assert outputs["stable_output_root"].exists()
        assert (outputs["output_root"] / "semaphore_sensitivity_manifest.json").exists()
        assert (outputs["output_root"] / "semaphore_sensitivity_case_summary.csv").exists()
        assert (outputs["output_root"] / "semaphore_sensitivity_scenario_summary.csv").exists()
        assert (outputs["stable_output_root"] / "semaphore_sensitivity_case_summary.csv").exists()


def test_run_pd_sensitivity_accepts_stable_output_root_without_self_copy(monkeypatch):
    def fake_state_alarm(config_path):
        config_path = Path(config_path)
        _write_fake_state_alarm_outputs(config_path.parent, channel="CH3")
        return {"output_root": config_path.parent}

    monkeypatch.setattr("deltapd.workbench_jobs.run_state_alarm_batch", fake_state_alarm)

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        for folder_name in ["Prueba 1 - Internas", "Prueba 2 - Superficiales"]:
            folder = base_dir / folder_name
            folder.mkdir(parents=True)
            (folder / "CH3.csv").write_text(_waveform_csv_text(), encoding="utf-8")

        stable_root = repo_root / "outputs" / "semaphore_sensitivity_ch3"
        outputs = run_pd_sensitivity(
            repo_root=repo_root,
            base_dir=base_dir,
            dataset_keys=["P1", "P2"],
            channel="CH3",
            output_root=stable_root,
        )

        assert outputs["output_root"] == stable_root
        assert outputs["stable_output_root"] == stable_root
        assert (stable_root / "semaphore_sensitivity_manifest.json").exists()
        assert (stable_root / "semaphore_sensitivity_summary.md").exists()


def test_run_pd_sensitivity_rejects_non_ch3_channel():
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        base_dir.mkdir(parents=True)
        try:
            run_pd_sensitivity(
                repo_root=repo_root,
                base_dir=base_dir,
                dataset_keys=[],
                channel="CH2",
            )
        except ValueError as exc:
            assert "only for CH3" in str(exc)
        else:
            raise AssertionError("Expected CH2 sensitivity request to fail.")


def test_create_sensitivity_job_writes_manifest_and_launches_worker(monkeypatch):
    popen_calls = []

    class DummyProcess:
        def __init__(self):
            self.pid = 1234

    def fake_popen(args, **kwargs):
        popen_calls.append((args, kwargs))
        return DummyProcess()

    monkeypatch.setattr("deltapd.workbench_jobs.subprocess.Popen", fake_popen)

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        base_dir = repo_root / "pd"
        folder = base_dir / "Prueba 1 - Internas"
        folder.mkdir(parents=True)
        (folder / "CH3.csv").write_text(_waveform_csv_text(), encoding="utf-8")

        manifest = create_sensitivity_job(
            repo_root=repo_root,
            base_dir=base_dir,
            dataset_keys=["P1"],
            channel="CH3",
        )

        manifest_path = Path(manifest["job_manifest_path"])
        assert manifest_path.exists()
        stored = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        assert stored["status"] == "pending"
        assert stored["mode"] == "semaphore_sensitivity"
        assert stored["params"]["repo_root"] == str(repo_root)
        assert stored["params"]["channel"] == "CH3"
        assert len(stored["cases"]) == 1
        assert len(popen_calls) == 1
        assert "deltapd.workbench_worker" in popen_calls[0][0]


def test_list_workbench_jobs_returns_latest_first():
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        jobs_root = repo_root / "outputs" / "workbench_jobs"
        older = jobs_root / "job_old"
        newer = jobs_root / "job_new"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        (older / "job_manifest.json").write_text(
            '{"job_id":"job_old","mode":"semaphore_sensitivity","status":"succeeded","created_at":"2026-03-09T09:00:00","progress_current":12,"progress_total":12,"message":"done","output_root":"","stable_output_root":""}',
            encoding="utf-8",
        )
        (newer / "job_manifest.json").write_text(
            '{"job_id":"job_new","mode":"semaphore_sensitivity","status":"running","created_at":"2026-03-09T10:00:00","progress_current":4,"progress_total":12,"message":"working","output_root":"","stable_output_root":""}',
            encoding="utf-8",
        )

        rows = list_workbench_jobs(repo_root)

        assert [row["job_id"] for row in rows] == ["job_new", "job_old"]
        assert rows[0]["progress_label"] == "4/12"


def test_workbench_worker_updates_manifest_to_succeeded(monkeypatch):
    def fake_run_pd_sensitivity(**kwargs):
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        stable_root = output_root.parent / "stable"
        stable_root.mkdir(parents=True, exist_ok=True)
        return {
            "output_root": output_root,
            "stable_output_root": stable_root,
            "message": "Generated CH3 semaphore sensitivity battery with 12 scenarios.",
        }

    monkeypatch.setattr("deltapd.workbench_worker.run_pd_sensitivity", fake_run_pd_sensitivity)

    with tempfile.TemporaryDirectory() as td:
        job_manifest_path = Path(td) / "job_manifest.json"
        job_manifest_path.write_text(
            __import__("json").dumps(
                {
                    "job_id": "job_001",
                    "mode": "semaphore_sensitivity",
                    "status": "pending",
                    "params": {
                        "repo_root": td,
                        "base_dir": str(Path(td) / "pd"),
                        "dataset_keys": ["P1"],
                        "channel": "CH3",
                        "raw_input": "",
                    },
                    "progress_current": 0,
                    "progress_total": 12,
                    "message": "Queued CH3 semaphore sensitivity battery.",
                    "output_root": "",
                    "stable_output_root": "",
                    "error": "",
                }
            ),
            encoding="utf-8",
        )

        run_job(job_manifest_path)

        stored = __import__("json").loads(job_manifest_path.read_text(encoding="utf-8"))
        assert stored["status"] == "succeeded"
        assert stored["output_root"].endswith("output")
        assert stored["stable_output_root"].endswith("stable")
