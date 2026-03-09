import json
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from deltapd.campaign.state_alarm_batch import run_state_alarm_batch


def test_run_state_alarm_batch_writes_summary_and_manifest(monkeypatch):
    def fake_run_descriptor_study(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_key = output_dir.parent.name
        material_cfg_path = Path(cfg["material_config"])
        with open(material_cfg_path, "r", encoding="utf-8") as f:
            material_cfg = yaml.safe_load(f)

        material_output_dir = Path(material_cfg["output_dir"])
        material_output_dir.mkdir(parents=True, exist_ok=True)
        (material_output_dir / "run_manifest.json").write_text(
            json.dumps({"total_events": 123}),
            encoding="utf-8",
        )
        pd.DataFrame({"toa_s": [0.1, 0.5, 0.9]}).to_csv(
            material_output_dir / "delta_t_series_master.csv",
            index=False,
        )

        pdf_path = output_dir / "fake_report.pdf"
        pdf_path.write_text("pdf", encoding="utf-8")
        overlap_rows = {
            "P1": [
                {
                    "candidate_rank": 1,
                    "local_window_index": 0,
                    "local_selected_method": "coherence",
                    "local_freq_offset_from_global_hz": 0.012,
                    "local_common_axial_confidence": 0.81,
                },
                {
                    "candidate_rank": 2,
                    "local_window_index": 0,
                    "local_selected_method": "coherence",
                    "local_freq_offset_from_global_hz": 0.012,
                    "local_common_axial_confidence": 0.81,
                },
                {
                    "candidate_rank": 3,
                    "local_window_index": 1,
                    "local_selected_method": "harmonic_power",
                    "local_freq_offset_from_global_hz": -0.021,
                    "local_common_axial_confidence": 0.62,
                },
                {
                    "candidate_rank": 4,
                    "local_window_index": 2,
                    "local_selected_method": "harmonic_power",
                    "local_freq_offset_from_global_hz": -0.033,
                    "local_common_axial_confidence": 0.55,
                },
            ],
            "P2": [
                {
                    "candidate_rank": 1,
                    "local_window_index": 0,
                    "local_selected_method": "coherence",
                    "local_freq_offset_from_global_hz": 0.008,
                    "local_common_axial_confidence": 0.74,
                },
                {
                    "candidate_rank": 2,
                    "local_window_index": 0,
                    "local_selected_method": "coherence",
                    "local_freq_offset_from_global_hz": 0.008,
                    "local_common_axial_confidence": 0.74,
                },
                {
                    "candidate_rank": 3,
                    "local_window_index": 1,
                    "local_selected_method": "epoch_folding",
                    "local_freq_offset_from_global_hz": -0.019,
                    "local_common_axial_confidence": 0.66,
                },
            ],
        }
        pd.DataFrame(overlap_rows[dataset_key]).to_csv(
            output_dir / "blind_prpd_transition_overlap.csv",
            index=False,
        )

        return {
            "output_dir": output_dir,
            "pdf_path": pdf_path,
            "recommendations": {
                "state": {
                    "recommendation": {
                        "features": ["phase_entropy", "cv_dt"],
                        "strategy": "forward",
                        "primary_metric": "macro_f1",
                        "primary_score": 0.81,
                        "balanced_accuracy": 0.79,
                    }
                },
                "alarm": {
                    "recommendation": {
                        "features": ["phase_width_pos_deg", "fano_factor"],
                        "strategy": "exhaustive",
                        "primary_metric": "auroc",
                        "primary_score": 0.88,
                        "balanced_accuracy": 0.77,
                    }
                },
            },
            "change_candidates": pd.DataFrame(
                [
                    {
                        "toa_start_s": 0.2,
                        "toa_end_s": 0.4,
                        "dominant_feature": "phase_entropy",
                    }
                ]
            ),
        }

    monkeypatch.setattr("deltapd.campaign.state_alarm_batch.run_descriptor_study", fake_run_descriptor_study)

    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "batch.yaml"
        config = {
            "base_dir": "E:/dummy",
            "output_root": str(Path(td) / "outputs"),
            "datasets": [
                {"dataset_key": "P1", "folder": "Prueba 1 - Internas", "discharge_type": "internal", "variant": "benchmark"},
                {"dataset_key": "P2", "folder": "Prueba 2 - Superficiales", "discharge_type": "superficial", "variant": "benchmark"},
            ],
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        outputs = run_state_alarm_batch(config_path)

        summary_df = outputs["summary_df"]
        assert list(summary_df["dataset_key"]) == ["P1", "P2"]
        assert summary_df.loc[0, "state_features"] == "phase_entropy, cv_dt"
        assert summary_df.loc[0, "alarm_features"] == "phase_width_pos_deg, fano_factor"
        assert Path(outputs["summary_csv"]).exists()
        assert Path(outputs["summary_md"]).exists()
        assert (Path(td) / "outputs" / "transition_overlap_master.csv").exists()
        assert (Path(td) / "outputs" / "transition_overlap_case_summary.csv").exists()
        assert (Path(td) / "outputs" / "transition_overlap_method_summary.csv").exists()
        assert (Path(td) / "outputs" / "transition_method_mix.png").exists()
        assert (Path(td) / "outputs" / "transition_offset_vs_scores.png").exists()
        transition_case_df = pd.read_csv(Path(td) / "outputs" / "transition_overlap_case_summary.csv")
        p1_row = transition_case_df.loc[transition_case_df["dataset_key"] == "P1"].iloc[0]
        p2_row = transition_case_df.loc[transition_case_df["dataset_key"] == "P2"].iloc[0]
        assert p1_row["n_ranked_transition_candidates"] == 4
        assert p1_row["n_transition_windows"] == 3
        assert p1_row["n_duplicate_candidate_matches"] == 1
        assert p1_row["transition_count_coherence"] == 1
        assert p1_row["transition_count_harmonic_power"] == 2
        assert p1_row["candidate_match_count_coherence"] == 2
        assert p2_row["n_ranked_transition_candidates"] == 3
        assert p2_row["n_transition_windows"] == 2
        assert p2_row["dominant_local_method"] == "tie: coherence, epoch_folding"
        assert p2_row["transition_count_coherence"] == 1
        assert p2_row["transition_count_epoch_folding"] == 1
        summary_md = Path(outputs["summary_md"]).read_text(encoding="utf-8")
        assert "Transition overlap summary" in summary_md
        assert "distinct matched blind-PRPD local windows" in summary_md
        manifest = json.loads(Path(outputs["manifest_path"]).read_text(encoding="utf-8"))
        assert len(manifest["cases"]) == 2
        assert manifest["state_feature_counts"]["cv_dt"] == 2
        assert manifest["transition_overlap_case_summary_csv"]
