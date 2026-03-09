import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from deltapd.campaign.state_alarm_batch import (
    _bocpd_gaussian_change_profile,
    _build_bocpd_signal,
    _gaussian_hmm_2state_profile,
    _semi_markov_2state_profile,
    run_state_alarm_batch,
)


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
        pd.DataFrame(
            {
                "toa_s": [0.1, 0.15, 0.2, 0.5, 0.55, 0.6, 0.9, 0.95, 1.0],
                "peak_v": [0.2, 0.4, 0.3, 0.5, 0.35, 0.45, 0.25, 0.3, 0.28],
            }
        ).to_csv(
            material_output_dir / "delta_t_series_master.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"local_window_index": 0, "event_start_idx": 0, "event_end_idx": 2},
                {"local_window_index": 1, "event_start_idx": 3, "event_end_idx": 5},
                {"local_window_index": 2, "event_start_idx": 6, "event_end_idx": 8},
            ]
        ).to_csv(
            material_output_dir / "blind_prpd_local_trace.csv",
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
        assert (Path(td) / "outputs" / "transition_local_sequence_windows.csv").exists()
        assert (Path(td) / "outputs" / "transition_local_sequence_case_summary.csv").exists()
        assert (Path(td) / "outputs" / "transition_local_sequence_edges.csv").exists()
        assert (Path(td) / "outputs" / "transition_local_wavelet_windows.csv").exists()
        assert (Path(td) / "outputs" / "transition_local_wavelet_case_summary.csv").exists()
        assert (Path(td) / "outputs" / "transition_method_mix.png").exists()
        assert (Path(td) / "outputs" / "transition_regime_sequence.png").exists()
        assert (Path(td) / "outputs" / "transition_mix_stability.png").exists()
        assert (Path(td) / "outputs" / "transition_offset_vs_scores.png").exists()
        transition_case_df = pd.read_csv(Path(td) / "outputs" / "transition_overlap_case_summary.csv")
        transition_sequence_df = pd.read_csv(Path(td) / "outputs" / "transition_local_sequence_windows.csv")
        p1_row = transition_case_df.loc[transition_case_df["dataset_key"] == "P1"].iloc[0]
        p2_row = transition_case_df.loc[transition_case_df["dataset_key"] == "P2"].iloc[0]
        assert p1_row["n_ranked_transition_candidates"] == 4
        assert p1_row["n_transition_windows"] == 3
        assert p1_row["n_duplicate_candidate_matches"] == 1
        assert p1_row["transition_count_coherence"] == 1
        assert p1_row["transition_count_harmonic_power"] == 2
        assert p1_row["candidate_match_count_coherence"] == 2
        assert p1_row["local_method_switch_count"] == 1
        assert p1_row["local_method_switch_rate"] == 0.5
        assert p1_row["transition_dominant_method_share"] == 2 / 3
        assert p1_row["local_freq_offset_std_hz"] > 0
        assert p1_row["local_common_axial_confidence_std"] > 0
        assert p1_row["tfa_wavelet_entropy_mean"] >= 0
        assert p1_row["tfa_wavelet_dominant_band_unique_count"] >= 1
        assert p1_row["tfa_wavelet_detail_entropy_mean"] >= 0
        assert p1_row["tfa_wavelet_detail_dominant_band_unique_count"] >= 1
        assert p1_row["local_regime_run_count"] == 2
        assert p1_row["local_regime_mean_run_length"] == 1.5
        assert p1_row["local_regime_persistence_ratio"] == 0.5
        assert p1_row["local_regime_transition_entropy"] > 0
        assert p1_row["local_offset_sign_switch_count"] == 1
        assert p1_row["local_offset_sign_switch_rate"] == 0.5
        assert 0.0 <= p1_row["bocpd_max_change_prob"] <= 1.0
        assert p1_row["bocpd_run_length_mean"] >= 0.0
        assert p1_row["bocpd_change_count"] >= 0
        assert p1_row["bocpd_surprise_score"] >= 0.0
        assert 0.0 <= p1_row["hmm_high_state_share"] <= 1.0
        assert 0.0 <= p1_row["hmm_high_state_prob_mean"] <= 1.0
        assert p1_row["hmm_state_switch_count"] >= 0
        assert p1_row["hmm_state_mean_run_length"] >= 1.0
        assert 0.0 <= p1_row["semi_markov_high_state_share"] <= 1.0
        assert p1_row["semi_markov_state_switch_count"] >= 0
        assert p1_row["semi_markov_state_mean_run_length"] >= 1.0
        assert p1_row["semi_markov_segment_count"] >= 1
        assert p2_row["n_ranked_transition_candidates"] == 3
        assert p2_row["n_transition_windows"] == 2
        assert p2_row["dominant_local_method"] == "tie: coherence, epoch_folding"
        assert p2_row["transition_count_coherence"] == 1
        assert p2_row["transition_count_epoch_folding"] == 1
        assert p2_row["local_method_switch_rate"] == 1.0
        assert p2_row["local_regime_run_count"] == 2
        assert p2_row["local_regime_mean_run_length"] == 1.0
        assert p2_row["local_offset_sign_switch_rate"] == 1.0
        assert set(transition_sequence_df["hmm_state"].astype(int).tolist()).issubset({0, 1})
        assert transition_sequence_df["hmm_high_state_prob"].between(0.0, 1.0).all()
        assert set(transition_sequence_df["semi_markov_state"].astype(int).tolist()).issubset({0, 1})
        assert (transition_sequence_df["semi_markov_run_length"] >= 1).all()
        assert transition_sequence_df["regime_run_length"].tolist() == [1, 2, 2, 1, 1]
        summary_md = Path(outputs["summary_md"]).read_text(encoding="utf-8")
        assert "Transition overlap summary" in summary_md
        assert "Method entropy" in summary_md
        assert "Highest local method switch rate" in summary_md
        assert "Highest local wavelet entropy mean" in summary_md
        assert "Highest local wavelet detail entropy mean" in summary_md
        assert "Highest local regime-transition entropy" in summary_md
        assert "Longest local mean regime run length" in summary_md
        assert "Highest local offset-sign switch rate" in summary_md
        assert "Highest BOCPD max change probability" in summary_md
        assert "Shortest BOCPD mean expected run length" in summary_md
        assert "Highest BOCPD surprise score" in summary_md
        assert "Highest HMM high-state share" in summary_md
        assert "Highest HMM switch rate" in summary_md
        assert "Shortest HMM mean run length" in summary_md
        assert "Highest semi-Markov high-state share" in summary_md
        assert "Highest semi-Markov switch rate" in summary_md
        assert "Shortest semi-Markov mean run length" in summary_md
        assert "distinct matched blind-PRPD local windows" in summary_md
        manifest = json.loads(Path(outputs["manifest_path"]).read_text(encoding="utf-8"))
        assert len(manifest["cases"]) == 2
        assert manifest["state_feature_counts"]["cv_dt"] == 2
        assert manifest["transition_overlap_case_summary_csv"]
        assert manifest["transition_mix_stability_png"]
        assert manifest["transition_regime_sequence_png"]
        assert manifest["transition_local_sequence_windows_csv"]
        assert manifest["transition_local_sequence_case_summary_csv"]
        assert manifest["transition_local_sequence_edges_csv"]
        assert manifest["transition_local_wavelet_windows_csv"]
        assert manifest["transition_local_wavelet_case_summary_csv"]


def test_bocpd_profile_detects_planted_regime_shift():
    offsets = np.asarray([0.011, 0.010, 0.012, 0.078, 0.081, 0.079], dtype=np.float64)
    confidences = np.asarray([0.82, 0.80, 0.81, 0.54, 0.52, 0.53], dtype=np.float64)
    methods = ["coherence", "coherence", "coherence", "harmonic_power", "harmonic_power", "harmonic_power"]

    signal, _ = _build_bocpd_signal(offsets, confidences, methods)
    profile = _bocpd_gaussian_change_profile(signal)

    assert len(profile["change_prob"]) == len(offsets)
    assert float(profile["change_prob"][3]) > float(profile["change_prob"][1])
    assert float(profile["log_surprise"][3]) > float(profile["log_surprise"][1])


def test_hmm_profile_detects_planted_regime_shift():
    signal = np.asarray([-0.9, -0.8, -0.7, 0.8, 0.9, 1.0], dtype=np.float64)

    profile = _gaussian_hmm_2state_profile(signal)

    assert len(profile["state_path"]) == len(signal)
    assert profile["state_path"][0] == 0
    assert profile["state_path"][-1] == 1
    assert float(profile["high_state_prob"][4]) > float(profile["high_state_prob"][1])
    assert 0.0 <= float(profile["state_entropy"]) <= 1.0


def test_semi_markov_profile_detects_planted_regime_shift_without_fragmentation():
    signal = np.asarray([-0.9, -0.8, -0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    hmm_profile = _gaussian_hmm_2state_profile(signal)

    profile = _semi_markov_2state_profile(
        signal,
        means=np.asarray(hmm_profile["means"], dtype=np.float64),
        variances=np.asarray(hmm_profile["variances"], dtype=np.float64),
        base_state_path=np.asarray(hmm_profile["state_path"], dtype=int),
    )

    assert len(profile["state_path"]) == len(signal)
    assert profile["state_path"][0] == 0
    assert profile["state_path"][-1] == 1
    assert profile["short_run_count"] == 0
    assert profile["segment_lengths"].tolist() == [3, 3]
    assert 0.0 <= float(profile["state_entropy"]) <= 1.0
