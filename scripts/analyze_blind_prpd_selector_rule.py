from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MULTIPLE_METHODS = ("harmonic_power", "epoch_folding", "gregory_loredo", "auto")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rank_multiple_cases(real_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for dataset_key in ("P3", "G3"):
        subset = real_df[
            (real_df["dataset_key"] == dataset_key)
            & (real_df["method"].isin(MULTIPLE_METHODS))
        ].copy()
        if subset.empty:
            continue
        subset["rank_entropy"] = subset["axial_entropy_score"].rank(ascending=False, method="dense")
        subset["rank_confidence"] = subset["common_axial_confidence"].rank(ascending=False, method="dense")
        subset["rank_offset"] = subset["common_axial_peak_offset_hz"].rank(ascending=True, method="dense")
        outputs[dataset_key] = subset.sort_values(
            ["rank_entropy", "rank_confidence", "rank_offset", "method"]
        ).reset_index(drop=True)
    return outputs


def build_selector_rule_report(
    *,
    real_cases_csv: str | Path,
    stress_summary_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    real_cases_csv = Path(real_cases_csv)
    stress_summary_csv = Path(stress_summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_df = pd.read_csv(real_cases_csv)
    stress_df = pd.read_csv(stress_summary_csv)
    multiple_rankings = _rank_multiple_cases(real_df)

    stress_phase_winners = (
        stress_df.sort_values(["scenario", "mean_axial_phase_error_deg", "mean_abs_error_vs_reference_hz"])
        .groupby("scenario", as_index=False)
        .first()
    )
    gregory_stress = stress_df[stress_df["method"] == "gregory_loredo"].copy()

    real_summary_rows = []
    gregory_real_advantages = 0
    for dataset_key, subset in multiple_rankings.items():
        gregory_row = subset[subset["method"] == "gregory_loredo"].iloc[0]
        best_entropy = subset.sort_values(["rank_entropy", "rank_confidence", "rank_offset"]).iloc[0]
        best_conf = subset.sort_values(["rank_confidence", "rank_entropy", "rank_offset"]).iloc[0]
        if int(gregory_row["rank_entropy"]) == 1 and int(gregory_row["rank_confidence"]) == 1:
            gregory_real_advantages += 1
        real_summary_rows.append(
            {
                "dataset_key": dataset_key,
                "gregory_freq_hz": float(gregory_row["freq_hz"]),
                "gregory_entropy": float(gregory_row["axial_entropy_score"]),
                "gregory_confidence": float(gregory_row["common_axial_confidence"]),
                "gregory_peak_offset_hz": float(gregory_row["common_axial_peak_offset_hz"]),
                "best_entropy_method": str(best_entropy["method"]),
                "best_confidence_method": str(best_conf["method"]),
            }
        )

    stress_gregory_wins = int((stress_phase_winners["method"] == "gregory_loredo").sum())
    decision = (
        "No conditional Gregory-Loredo rule yet."
        if gregory_real_advantages == 0 and stress_gregory_wins == 0
        else "Gregory-Loredo deserves more targeted selector analysis."
    )

    summary_rows = pd.DataFrame(real_summary_rows)
    summary_csv = output_dir / "blind_prpd_selector_rule_summary.csv"
    summary_rows.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# Blind PRPD selector-rule analysis",
        "",
        "## Question",
        "",
        "- Should `gregory_loredo` enter a conditional selector for multiple-source cases, or stay as a paper baseline only?",
        "",
        "## Multiple-source real cases",
        "",
        "| Dataset | Gregory freq (Hz) | Gregory entropy | Gregory common conf. | Gregory peak offset (Hz) | Best entropy method | Best confidence method |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in real_summary_rows:
        lines.append(
            f"| {row['dataset_key']} | {row['gregory_freq_hz']:.6f} | {row['gregory_entropy']:.6f} | "
            f"{row['gregory_confidence']:.6f} | {row['gregory_peak_offset_hz']:.6f} | "
            f"{row['best_entropy_method']} | {row['best_confidence_method']} |"
        )

    lines.extend(
        [
            "",
            "## Stress scenarios",
            "",
            "| Scenario | Phase-error winner | Gregory phase error (deg) | Gregory common conf. | Gregory peak offset (Hz) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, winner_row in stress_phase_winners.iterrows():
        scenario = str(winner_row["scenario"])
        gregory_row = gregory_stress[gregory_stress["scenario"] == scenario].iloc[0]
        lines.append(
            f"| {scenario} | {winner_row['method']} | {float(gregory_row['mean_axial_phase_error_deg']):.4f} | "
            f"{float(gregory_row['mean_common_axial_confidence']):.4f} | "
            f"{float(gregory_row['mean_common_axial_peak_offset_hz']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- {decision}",
            f"- Gregory-Loredo real-case combined dominance count: {gregory_real_advantages}.",
            f"- Gregory-Loredo stress-scenario wins by phase error: {stress_gregory_wins}.",
            "- Current reading: keep `gregory_loredo` as a strong baseline, not as an operational branch in `auto`.",
        ]
    )

    report_md = output_dir / "blind_prpd_selector_rule_report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "report_md": report_md,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze whether Gregory-Loredo deserves a conditional selector rule.")
    parser.add_argument(
        "--real-cases-csv",
        type=Path,
        default=_repo_root() / "outputs" / "blind_prpd_real_cases_gl" / "blind_prpd_real_case_comparison.csv",
        help="Real-case comparison CSV produced by compare_blind_prpd_real_cases.py.",
    )
    parser.add_argument(
        "--stress-summary-csv",
        type=Path,
        default=_repo_root() / "outputs" / "blind_prpd_stress" / "blind_prpd_stress_summary.csv",
        help="Stress benchmark summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "outputs" / "blind_prpd_selector_rule",
        help="Directory where selector-rule outputs will be written.",
    )
    args = parser.parse_args()

    outputs = build_selector_rule_report(
        real_cases_csv=args.real_cases_csv,
        stress_summary_csv=args.stress_summary_csv,
        output_dir=args.output_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
