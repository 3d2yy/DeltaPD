import tempfile
from pathlib import Path
import json

import pandas as pd

from deltapd.campaign.comparative_thesis_study import (
    _augment_transition_case_metrics,
    _build_comparative_artifacts,
    _loo_nearest_centroid_case_level,
    _partition_comparative_feature_blocks,
    build_comparative_event_table,
)


def test_build_comparative_event_table_assigns_labels_and_delta_t():
    rows = []
    for dataset_key, base_toa in [("P1", 0.0), ("P2", 0.0), ("G1", 0.0)]:
        for idx, toa in enumerate([0.1, 0.3, 0.6, 1.0], start=1):
            rows.append(
                {
                    "group_family": "benchmark" if dataset_key.startswith("P") else "gemelas",
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_key,
                    "channel": "CH3",
                    "antenna_label": "ant",
                    "toa_s": base_toa + toa,
                    "peak_v": 0.1 * idx,
                    "prpd_phase_deg": 10.0 * idx,
                    "signed_peak_v": 0.1 * idx,
                }
            )

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "points.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        df = build_comparative_event_table(
            csv_path,
            channel="CH3",
            dataset_keys=["P1", "P2", "G1"],
        )

        assert set(df["dataset_key"].unique()) == {"P1", "P2", "G1"}
        assert set(df["discharge_type"].unique()) == {"internal", "superficial"}
        assert set(df["acquisition_variant"].unique()) == {"benchmark", "gemela"}
        assert df["delta_t_s"].notna().sum() == 9


def test_build_comparative_artifacts_writes_markdown_and_plots():
    with tempfile.TemporaryDirectory() as td:
        output_dir = Path(td)

        recommendations = {
            "dataset6": {
                "class_counts": {"P1": 2, "P2": 2, "G1": 2},
                "recommendation": {
                    "strategy": "forward",
                    "features": ["cv2_dt", "mean_peak_v", "phase_kuramoto_r"],
                    "primary_score": 0.88,
                    "balanced_accuracy": 0.89,
                },
            },
            "type3": {
                "class_counts": {"internal": 2, "superficial": 2},
                "recommendation": {
                    "strategy": "exhaustive",
                    "features": ["p90_dt_s", "local_variation", "phase_kuramoto_r"],
                    "primary_score": 0.99,
                    "balanced_accuracy": 1.0,
                },
            },
            "variant2": {
                "class_counts": {"benchmark": 3, "gemela": 3},
                "recommendation": {
                    "strategy": "exhaustive",
                    "features": ["mean_peak_v", "weibull_beta", "p90_dt_s"],
                    "primary_score": 0.93,
                    "balanced_accuracy": 0.83,
                },
            },
        }
        (output_dir / "study_recommendations.json").write_text(
            json.dumps(recommendations),
            encoding="utf-8",
        )

        pd.DataFrame(
            [
                {"dataset_key": "P1", "toa_s": 0.1, "discharge_type": "internal", "acquisition_variant": "benchmark"},
                {"dataset_key": "P2", "toa_s": 0.2, "discharge_type": "superficial", "acquisition_variant": "benchmark"},
                {"dataset_key": "G1", "toa_s": 0.3, "discharge_type": "internal", "acquisition_variant": "gemela"},
                {"dataset_key": "G2", "toa_s": 0.4, "discharge_type": "superficial", "acquisition_variant": "gemela"},
            ]
        ).to_csv(output_dir / "comparative_event_table.csv", index=False)

        pd.DataFrame(
            [
                {
                    "dataset_key": "P1",
                    "discharge_type": "internal",
                    "acquisition_variant": "benchmark",
                    "cv2_dt": 1.0,
                    "mean_peak_v": 0.12,
                    "phase_kuramoto_r": 0.20,
                    "p90_dt_s": 0.30,
                    "local_variation": 0.50,
                    "weibull_beta": 0.60,
                },
                {
                    "dataset_key": "P2",
                    "discharge_type": "superficial",
                    "acquisition_variant": "benchmark",
                    "cv2_dt": 1.5,
                    "mean_peak_v": 0.18,
                    "phase_kuramoto_r": 0.45,
                    "p90_dt_s": 0.42,
                    "local_variation": 0.80,
                    "weibull_beta": 0.90,
                },
                {
                    "dataset_key": "G1",
                    "discharge_type": "internal",
                    "acquisition_variant": "gemela",
                    "cv2_dt": 1.1,
                    "mean_peak_v": 0.11,
                    "phase_kuramoto_r": 0.25,
                    "p90_dt_s": 0.28,
                    "local_variation": 0.52,
                    "weibull_beta": 0.72,
                },
                {
                    "dataset_key": "G2",
                    "discharge_type": "superficial",
                    "acquisition_variant": "gemela",
                    "cv2_dt": 1.7,
                    "mean_peak_v": 0.20,
                    "phase_kuramoto_r": 0.49,
                    "p90_dt_s": 0.48,
                    "local_variation": 0.92,
                    "weibull_beta": 1.02,
                },
            ]
        ).to_csv(output_dir / "descriptor_windows.csv", index=False)

        transition_case_df = pd.DataFrame(
            [
                {
                    "dataset_key": "P1",
                    "discharge_type": "internal",
                    "variant": "benchmark",
                    "n_transition_windows": 3,
                    "n_ranked_transition_candidates": 4,
                    "n_unique_local_methods": 2,
                    "dominant_local_method": "coherence",
                    "max_abs_local_freq_offset_hz": 0.12,
                    "mean_abs_local_freq_offset_hz": 0.08,
                    "local_freq_offset_std_hz": 0.03,
                    "mean_local_common_axial_confidence": 0.61,
                    "local_common_axial_confidence_std": 0.09,
                    "transition_method_entropy": 0.42,
                    "transition_dominant_method_share": 0.67,
                    "local_method_switch_count": 1,
                    "local_method_switch_rate": 0.5,
                    "local_regime_run_count": 2,
                    "local_regime_mean_run_length": 1.5,
                    "local_regime_max_run_length": 2,
                    "local_regime_persistence_ratio": 0.5,
                    "local_regime_transition_entropy": 0.0,
                    "local_offset_sign_switch_count": 1,
                    "local_offset_sign_switch_rate": 0.5,
                    "tfa_wavelet_entropy_mean": 0.51,
                    "tfa_wavelet_entropy_max": 0.67,
                    "tfa_wavelet_dominant_band_unique_count": 2,
                    "tfa_wavelet_dominant_band_entropy": 0.44,
                    "tfa_wavelet_dominant_band_switch_count": 1,
                    "tfa_wavelet_dominant_band_switch_rate": 0.5,
                    "tfa_wavelet_detail_entropy_mean": 0.58,
                    "tfa_wavelet_detail_entropy_max": 0.74,
                    "tfa_wavelet_detail_dominant_band_unique_count": 2,
                    "tfa_wavelet_detail_dominant_band_entropy": 0.50,
                    "tfa_wavelet_detail_dominant_band_switch_count": 1,
                    "tfa_wavelet_detail_dominant_band_switch_rate": 0.5,
                    "state_primary_score": 0.82,
                    "alarm_primary_score": 0.87,
                    "transition_count_coherence": 2,
                    "transition_count_harmonic_power": 1,
                    "transition_count_epoch_folding": 0,
                },
                {
                    "dataset_key": "P2",
                    "discharge_type": "superficial",
                    "variant": "benchmark",
                    "n_transition_windows": 3,
                    "n_ranked_transition_candidates": 3,
                    "n_unique_local_methods": 1,
                    "dominant_local_method": "epoch_folding",
                    "max_abs_local_freq_offset_hz": 0.02,
                    "mean_abs_local_freq_offset_hz": 0.01,
                    "local_freq_offset_std_hz": 0.01,
                    "mean_local_common_axial_confidence": 0.30,
                    "local_common_axial_confidence_std": 0.04,
                    "transition_method_entropy": 0.00,
                    "transition_dominant_method_share": 1.00,
                    "local_method_switch_count": 0,
                    "local_method_switch_rate": 0.0,
                    "local_regime_run_count": 1,
                    "local_regime_mean_run_length": 3.0,
                    "local_regime_max_run_length": 3,
                    "local_regime_persistence_ratio": 1.0,
                    "local_regime_transition_entropy": 0.0,
                    "local_offset_sign_switch_count": 0,
                    "local_offset_sign_switch_rate": 0.0,
                    "tfa_wavelet_entropy_mean": 0.19,
                    "tfa_wavelet_entropy_max": 0.25,
                    "tfa_wavelet_dominant_band_unique_count": 1,
                    "tfa_wavelet_dominant_band_entropy": 0.0,
                    "tfa_wavelet_dominant_band_switch_count": 0,
                    "tfa_wavelet_dominant_band_switch_rate": 0.0,
                    "tfa_wavelet_detail_entropy_mean": 0.21,
                    "tfa_wavelet_detail_entropy_max": 0.28,
                    "tfa_wavelet_detail_dominant_band_unique_count": 1,
                    "tfa_wavelet_detail_dominant_band_entropy": 0.0,
                    "tfa_wavelet_detail_dominant_band_switch_count": 0,
                    "tfa_wavelet_detail_dominant_band_switch_rate": 0.0,
                    "state_primary_score": 0.90,
                    "alarm_primary_score": 0.95,
                    "transition_count_coherence": 0,
                    "transition_count_harmonic_power": 0,
                    "transition_count_epoch_folding": 3,
                },
                {
                    "dataset_key": "G1",
                    "discharge_type": "internal",
                    "variant": "gemela",
                    "n_transition_windows": 4,
                    "n_ranked_transition_candidates": 6,
                    "n_unique_local_methods": 3,
                    "dominant_local_method": "harmonic_power",
                    "max_abs_local_freq_offset_hz": 0.07,
                    "mean_abs_local_freq_offset_hz": 0.03,
                    "local_freq_offset_std_hz": 0.02,
                    "mean_local_common_axial_confidence": 0.56,
                    "local_common_axial_confidence_std": 0.07,
                    "transition_method_entropy": 0.68,
                    "transition_dominant_method_share": 0.50,
                    "local_method_switch_count": 2,
                    "local_method_switch_rate": 0.67,
                    "local_regime_run_count": 3,
                    "local_regime_mean_run_length": 1.33,
                    "local_regime_max_run_length": 2,
                    "local_regime_persistence_ratio": 0.33,
                    "local_regime_transition_entropy": 0.58,
                    "local_offset_sign_switch_count": 1,
                    "local_offset_sign_switch_rate": 0.33,
                    "tfa_wavelet_entropy_mean": 0.63,
                    "tfa_wavelet_entropy_max": 0.72,
                    "tfa_wavelet_dominant_band_unique_count": 3,
                    "tfa_wavelet_dominant_band_entropy": 0.62,
                    "tfa_wavelet_dominant_band_switch_count": 2,
                    "tfa_wavelet_dominant_band_switch_rate": 0.67,
                    "tfa_wavelet_detail_entropy_mean": 0.71,
                    "tfa_wavelet_detail_entropy_max": 0.81,
                    "tfa_wavelet_detail_dominant_band_unique_count": 3,
                    "tfa_wavelet_detail_dominant_band_entropy": 0.70,
                    "tfa_wavelet_detail_dominant_band_switch_count": 2,
                    "tfa_wavelet_detail_dominant_band_switch_rate": 0.67,
                    "state_primary_score": 0.88,
                    "alarm_primary_score": 0.93,
                    "transition_count_coherence": 1,
                    "transition_count_harmonic_power": 2,
                    "transition_count_epoch_folding": 1,
                },
                {
                    "dataset_key": "G2",
                    "discharge_type": "superficial",
                    "variant": "gemela",
                    "n_transition_windows": 3,
                    "n_ranked_transition_candidates": 5,
                    "n_unique_local_methods": 1,
                    "dominant_local_method": "epoch_folding",
                    "max_abs_local_freq_offset_hz": 0.03,
                    "mean_abs_local_freq_offset_hz": 0.02,
                    "local_freq_offset_std_hz": 0.01,
                    "mean_local_common_axial_confidence": 0.34,
                    "local_common_axial_confidence_std": 0.05,
                    "transition_method_entropy": 0.00,
                    "transition_dominant_method_share": 1.00,
                    "local_method_switch_count": 0,
                    "local_method_switch_rate": 0.0,
                    "local_regime_run_count": 1,
                    "local_regime_mean_run_length": 3.0,
                    "local_regime_max_run_length": 3,
                    "local_regime_persistence_ratio": 1.0,
                    "local_regime_transition_entropy": 0.0,
                    "local_offset_sign_switch_count": 0,
                    "local_offset_sign_switch_rate": 0.0,
                    "tfa_wavelet_entropy_mean": 0.22,
                    "tfa_wavelet_entropy_max": 0.30,
                    "tfa_wavelet_dominant_band_unique_count": 1,
                    "tfa_wavelet_dominant_band_entropy": 0.0,
                    "tfa_wavelet_dominant_band_switch_count": 0,
                    "tfa_wavelet_dominant_band_switch_rate": 0.0,
                    "tfa_wavelet_detail_entropy_mean": 0.24,
                    "tfa_wavelet_detail_entropy_max": 0.31,
                    "tfa_wavelet_detail_dominant_band_unique_count": 1,
                    "tfa_wavelet_detail_dominant_band_entropy": 0.0,
                    "tfa_wavelet_detail_dominant_band_switch_count": 0,
                    "tfa_wavelet_detail_dominant_band_switch_rate": 0.0,
                    "state_primary_score": 0.79,
                    "alarm_primary_score": 0.89,
                    "transition_count_coherence": 0,
                    "transition_count_harmonic_power": 0,
                    "transition_count_epoch_folding": 3,
                },
            ]
        )
        transition_case_df["hmm_high_state_share"] = [0.62, 0.18, 0.74, 0.16]
        transition_case_df["hmm_high_state_prob_mean"] = [0.69, 0.24, 0.77, 0.21]
        transition_case_df["hmm_state_switch_count"] = [1, 0, 2, 0]
        transition_case_df["hmm_state_switch_rate"] = [0.50, 0.00, 0.67, 0.00]
        transition_case_df["hmm_state_mean_run_length"] = [1.5, 3.0, 1.33, 3.0]
        transition_case_df["hmm_state_persistence_ratio"] = [0.50, 1.00, 0.33, 1.00]
        transition_case_df["hmm_state_entropy"] = [0.92, 0.31, 0.88, 0.28]
        transition_case_df["semi_markov_high_state_share"] = [0.55, 0.10, 0.66, 0.12]
        transition_case_df["semi_markov_state_switch_count"] = [1, 0, 1, 0]
        transition_case_df["semi_markov_state_switch_rate"] = [0.50, 0.00, 0.33, 0.00]
        transition_case_df["semi_markov_state_mean_run_length"] = [1.5, 3.0, 2.0, 3.0]
        transition_case_df["semi_markov_state_persistence_ratio"] = [0.50, 1.00, 0.67, 1.00]
        transition_case_df["semi_markov_state_entropy"] = [0.92, 0.00, 0.92, 0.00]
        transition_case_df["semi_markov_segment_count"] = [2, 1, 2, 1]
        transition_case_df["semi_markov_short_run_count"] = [0, 0, 0, 0]
        transition_case_df["semi_markov_duration_surprise_mean"] = [1.2, 0.9, 1.3, 0.8]
        block_ablation_df = pd.DataFrame(
            [
                {
                    "task_name": "dataset6",
                    "block_name": "temporal",
                    "block_label": "Temporal only",
                    "features": ["cv2_dt", "local_variation"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.71,
                    "balanced_accuracy": 0.72,
                    "strategy": "forward",
                },
                {
                    "task_name": "dataset6",
                    "block_name": "phase_prpd",
                    "block_label": "Phase/PRPD only",
                    "features": ["phase_kuramoto_r"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.62,
                    "balanced_accuracy": 0.63,
                    "strategy": "univariate",
                },
                {
                    "task_name": "dataset6",
                    "block_name": "fused",
                    "block_label": "Temporal + Phase",
                    "features": ["cv2_dt", "phase_kuramoto_r"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.83,
                    "balanced_accuracy": 0.84,
                    "strategy": "forward",
                },
                {
                    "task_name": "type3",
                    "block_name": "temporal",
                    "block_label": "Temporal only",
                    "features": ["p90_dt_s", "local_variation"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.88,
                    "balanced_accuracy": 0.89,
                    "strategy": "forward",
                },
                {
                    "task_name": "type3",
                    "block_name": "phase_prpd",
                    "block_label": "Phase/PRPD only",
                    "features": ["phase_kuramoto_r"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.74,
                    "balanced_accuracy": 0.75,
                    "strategy": "univariate",
                },
                {
                    "task_name": "type3",
                    "block_name": "fused",
                    "block_label": "Temporal + Phase",
                    "features": ["p90_dt_s", "phase_kuramoto_r"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.95,
                    "balanced_accuracy": 0.96,
                    "strategy": "forward",
                },
                {
                    "task_name": "variant2",
                    "block_name": "temporal",
                    "block_label": "Temporal only",
                    "features": ["weibull_beta", "p90_dt_s"],
                    "primary_metric": "auroc",
                    "primary_score": 0.77,
                    "balanced_accuracy": 0.71,
                    "strategy": "exhaustive",
                },
                {
                    "task_name": "variant2",
                    "block_name": "phase_prpd",
                    "block_label": "Phase/PRPD only",
                    "features": ["phase_kuramoto_r"],
                    "primary_metric": "auroc",
                    "primary_score": 0.64,
                    "balanced_accuracy": 0.60,
                    "strategy": "univariate",
                },
                {
                    "task_name": "variant2",
                    "block_name": "fused",
                    "block_label": "Temporal + Phase",
                    "features": ["weibull_beta", "phase_kuramoto_r"],
                    "primary_metric": "auroc",
                    "primary_score": 0.81,
                    "balanced_accuracy": 0.76,
                    "strategy": "forward",
                },
            ]
        )
        semicycle_ablation_df = pd.DataFrame(
            [
                {
                    "task_name": "dataset6",
                    "block_name": "negative",
                    "block_label": "Negative semicycle",
                    "features": ["phase_neg_q75_deg", "phase_neg_count"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.74,
                    "balanced_accuracy": 0.75,
                    "strategy": "forward",
                },
                {
                    "task_name": "dataset6",
                    "block_name": "positive",
                    "block_label": "Positive semicycle",
                    "features": ["phase_pos_mean_deg"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.62,
                    "balanced_accuracy": 0.63,
                    "strategy": "univariate",
                },
                {
                    "task_name": "dataset6",
                    "block_name": "both",
                    "block_label": "Both semicycles",
                    "features": ["phase_neg_q75_deg", "phase_pos_mean_deg"],
                    "primary_metric": "macro_f1",
                    "primary_score": 0.80,
                    "balanced_accuracy": 0.81,
                    "strategy": "forward",
                },
            ]
        )
        outputs = _build_comparative_artifacts(
            output_dir=output_dir,
            channel="CH3",
            dataset_keys=["P1", "P2", "G1", "G2"],
            block_ablation_df=block_ablation_df,
            block_auxiliary_features=["mean_peak_v"],
            semicycle_ablation_df=semicycle_ablation_df,
            transition_case_df=transition_case_df,
            transition_eval={
                "type3": {
                    "features": ["max_abs_local_freq_offset_hz", "mean_local_common_axial_confidence"],
                    "macro_f1": 0.75,
                    "balanced_accuracy": 0.75,
                }
            },
        )

        report_text = Path(outputs["markdown_path"]).read_text(encoding="utf-8")
        assert "type3" in report_text
        assert "p90_dt_s, local_variation, phase_kuramoto_r" in report_text
        assert "Strict block ablation" in report_text
        assert "Semicycle PRPD baseline" in report_text
        assert "Negative semicycle" in report_text
        assert "Both semicycles" in report_text
        assert "Reserve-only auxiliary features excluded from this strict comparison" in report_text
        assert "Exploratory case-level transition metrics" in report_text
        assert "Method entropy" in report_text
        assert "Switch rate" in report_text
        assert "Seq. entropy" in report_text
        assert "Mean run" in report_text
        assert "Sign-switch rate" in report_text
        assert "Wavelet H mean" in report_text
        assert "Wavelet band diversity" in report_text
        assert "Wavelet detail H" in report_text
        assert "Wavelet detail diversity" in report_text
        assert "deduplicated by matched local blind-PRPD window" in report_text
        assert "mean run length, sequence entropy, BOCPD change evidence, HMM and semi-Markov state persistence, and local dispersion" in report_text
        assert "SMM high share" in report_text
        assert "SMM switch" in report_text
        assert "SMM mean run" in report_text
        assert "HMM high share" in report_text
        assert "HMM switch" in report_text
        assert "HMM mean run" in report_text
        assert len(outputs["extra_images"]) == 8
        for image_path, _ in outputs["extra_images"]:
            assert Path(image_path).exists()


def test_build_comparative_event_table_preserves_existing_delta_t():
    rows = [
        {
            "group_family": "benchmark",
            "dataset_key": "P1",
            "dataset_label": "P1",
            "channel": "CH3",
            "antenna_label": "ant",
            "toa_s": 0.5,
            "delta_t_s": 0.11,
            "peak_v": 1.0,
            "prpd_phase_deg": 25.0,
            "signed_peak_v": 1.0,
        },
        {
            "group_family": "benchmark",
            "dataset_key": "P1",
            "dataset_label": "P1",
            "channel": "CH3",
            "antenna_label": "ant",
            "toa_s": 0.8,
            "delta_t_s": 0.22,
            "peak_v": 1.2,
            "prpd_phase_deg": 30.0,
            "signed_peak_v": 1.2,
        },
    ]

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "points.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        df = build_comparative_event_table(
            csv_path,
            channel="CH3",
            dataset_keys=["P1"],
        )

        assert df["delta_t_s"].tolist() == [0.11, 0.22]


def test_loo_nearest_centroid_case_level_returns_multiclass_metrics():
    df = pd.DataFrame(
        [
            {"dataset_key": "P1", "discharge_type": "internal", "variant": "benchmark", "max_abs_local_freq_offset_hz": 0.11, "mean_abs_local_freq_offset_hz": 0.08, "mean_local_common_axial_confidence": 0.62, "n_unique_local_methods": 2, "transition_count_coherence": 2, "transition_count_harmonic_power": 1, "transition_count_epoch_folding": 0},
            {"dataset_key": "G1", "discharge_type": "internal", "variant": "gemela", "max_abs_local_freq_offset_hz": 0.09, "mean_abs_local_freq_offset_hz": 0.05, "mean_local_common_axial_confidence": 0.58, "n_unique_local_methods": 3, "transition_count_coherence": 1, "transition_count_harmonic_power": 2, "transition_count_epoch_folding": 0},
            {"dataset_key": "P2", "discharge_type": "superficial", "variant": "benchmark", "max_abs_local_freq_offset_hz": 0.02, "mean_abs_local_freq_offset_hz": 0.01, "mean_local_common_axial_confidence": 0.31, "n_unique_local_methods": 1, "transition_count_coherence": 0, "transition_count_harmonic_power": 0, "transition_count_epoch_folding": 3},
            {"dataset_key": "G2", "discharge_type": "superficial", "variant": "gemela", "max_abs_local_freq_offset_hz": 0.03, "mean_abs_local_freq_offset_hz": 0.02, "mean_local_common_axial_confidence": 0.34, "n_unique_local_methods": 1, "transition_count_coherence": 0, "transition_count_harmonic_power": 0, "transition_count_epoch_folding": 3},
            {"dataset_key": "P3", "discharge_type": "multiple", "variant": "benchmark", "max_abs_local_freq_offset_hz": 0.13, "mean_abs_local_freq_offset_hz": 0.06, "mean_local_common_axial_confidence": 0.41, "n_unique_local_methods": 3, "transition_count_coherence": 1, "transition_count_harmonic_power": 4, "transition_count_epoch_folding": 1},
            {"dataset_key": "G3", "discharge_type": "multiple", "variant": "gemela", "max_abs_local_freq_offset_hz": 0.10, "mean_abs_local_freq_offset_hz": 0.04, "mean_local_common_axial_confidence": 0.47, "n_unique_local_methods": 3, "transition_count_coherence": 1, "transition_count_harmonic_power": 3, "transition_count_epoch_folding": 2},
        ]
    )

    result = _loo_nearest_centroid_case_level(
        df,
        features=[
            "max_abs_local_freq_offset_hz",
            "mean_abs_local_freq_offset_hz",
            "mean_local_common_axial_confidence",
            "n_unique_local_methods",
            "transition_count_coherence",
            "transition_count_harmonic_power",
            "transition_count_epoch_folding",
        ],
        label_column="discharge_type",
    )

    assert result["n_cases"] == 6
    assert "macro_f1" in result
    assert "balanced_accuracy" in result


def test_augment_transition_case_metrics_normalizes_method_mix():
    df = pd.DataFrame(
        [
            {
                "dataset_key": "P1",
                "n_transition_windows": 3,
                "transition_count_coherence": 2,
                "transition_count_harmonic_power": 1,
                "transition_count_epoch_folding": 0,
            },
            {
                "dataset_key": "P2",
                "n_transition_windows": 4,
                "transition_count_coherence": 1,
                "transition_count_harmonic_power": 1,
                "transition_count_epoch_folding": 2,
            },
        ]
    )

    out = _augment_transition_case_metrics(df)

    assert out.loc[0, "transition_window_total"] == 3
    assert out.loc[0, "transition_share_coherence"] == 2 / 3
    assert out.loc[0, "transition_share_harmonic_power"] == 1 / 3
    assert out.loc[0, "transition_share_epoch_folding"] == 0.0
    assert out.loc[0, "transition_dominant_method_share"] == 2 / 3
    assert 0.0 <= out.loc[0, "transition_method_entropy"] <= 1.0


def test_partition_comparative_feature_blocks_separates_temporal_phase_and_auxiliary():
    blocks, auxiliary = _partition_comparative_feature_blocks(
        [
            "median_dt_s",
            "local_variation",
            "phase_kuramoto_r",
            "phase_inlier_ratio",
            "amplitude_balance_ratio",
            "mean_peak_v",
            "n_events",
        ]
    )

    assert blocks["temporal"] == ["median_dt_s", "local_variation"]
    assert blocks["phase_prpd"] == ["phase_kuramoto_r", "phase_inlier_ratio", "amplitude_balance_ratio"]
    assert blocks["fused"] == [
        "median_dt_s",
        "local_variation",
        "phase_kuramoto_r",
        "phase_inlier_ratio",
        "amplitude_balance_ratio",
    ]
    assert auxiliary == ["mean_peak_v", "n_events"]
