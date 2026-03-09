"""CLI entrypoint for DeltaPD Core."""

from __future__ import annotations

import argparse
from pathlib import Path

from deltapd.pipeline import main as legacy_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deltapd",
        description="DeltaPD Core -- legacy pipeline and thesis campaign tools",
    )
    subparsers = parser.add_subparsers(dest="command")

    legacy = subparsers.add_parser(
        "run-legacy",
        help="Run the original four-phase synthetic/empirical validation pipeline.",
    )
    legacy.add_argument("-n", "--n-samples", type=int, default=4096)
    legacy.add_argument("--fs", type=float, default=1e9)
    legacy.add_argument("--mc-iterations", type=int, default=100)
    legacy.add_argument("--seed", type=int, default=42)
    legacy.add_argument("-q", "--quiet", action="store_true")

    thesis = subparsers.add_parser(
        "run-thesis",
        aliases=["run-campaign"],
        help="Run thesis campaign processing from a YAML config.",
    )
    thesis.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_thesis.yaml"),
        help="Path to thesis campaign YAML config.",
    )

    material = subparsers.add_parser(
        "run-material",
        help="Run material-state time-evolution analysis from a YAML config.",
    )
    material.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_material.yaml"),
        help="Path to material-state YAML config.",
    )

    study = subparsers.add_parser(
        "run-study",
        aliases=["run-descriptor-study"],
        help="Run descriptor screening and combination search from a YAML config.",
    )
    study.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_descriptor_study.yaml"),
        help="Path to descriptor-study YAML config.",
    )

    mat_series = subparsers.add_parser(
        "run-mat-series",
        help="Run row-wise descriptor analysis for matrix-style MAT datasets.",
    )
    mat_series.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_mat_series_1_2.yaml"),
        help="Path to matrix-MAT study YAML config.",
    )

    comparative = subparsers.add_parser(
        "run-comparative-study",
        help="Run comparative CH3 descriptor study across thesis datasets.",
    )
    comparative.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_comparative_ch3.yaml"),
        help="Path to comparative-study YAML config.",
    )

    state_alarm = subparsers.add_parser(
        "run-state-alarm-batch",
        help="Run per-dataset state/alarm studies for thesis datasets.",
    )
    state_alarm.add_argument(
        "--config",
        type=Path,
        default=Path("campaign/config_state_alarm_ch3.yaml"),
        help="Path to state/alarm batch YAML config.",
    )

    workbench = subparsers.add_parser(
        "run-workbench",
        help="Launch the local visual workbench for thesis outputs.",
    )
    workbench.add_argument("--host", default="127.0.0.1", help="Host interface for the local server.")
    workbench.add_argument("--port", type=int, default=8050, help="Port for the local server.")
    workbench.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    workbench.add_argument(
        "--pd-base-dir",
        type=Path,
        default=Path("E:/Carpeta definitiva de Tesis/programas"),
        help="Base directory used to discover thesis PD datasets.",
    )
    workbench.add_argument(
        "--state-alarm-root",
        type=Path,
        default=None,
        help="Optional state/alarm output folder to load at startup.",
    )
    workbench.add_argument(
        "--comparative-root",
        type=Path,
        default=None,
        help="Optional comparative-study output folder to load at startup.",
    )
    workbench.add_argument(
        "--vna-root",
        type=Path,
        default=None,
        help="Optional VNA output folder to load at startup.",
    )

    blind_prpd_stress = subparsers.add_parser(
        "run-blind-prpd-stress",
        help="Run drift/gap stress benchmarks for blind PRPD methods.",
    )
    blind_prpd_stress.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/blind_prpd_stress"),
        help="Directory where stress benchmark outputs will be written.",
    )
    blind_prpd_stress.add_argument("--trials", type=int, default=12, help="Number of trials per scenario.")
    blind_prpd_stress.add_argument("--seed", type=int, default=142, help="Random seed.")

    return parser


def cli() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {None, "run-legacy"}:
        legacy_main(
            n_samples=getattr(args, "n_samples", 4096),
            fs=getattr(args, "fs", 1e9),
            mc_iterations=getattr(args, "mc_iterations", 100),
            seed=getattr(args, "seed", 42),
            verbose=not getattr(args, "quiet", False),
        )
        return 0

    if args.command in {"run-thesis", "run-campaign"}:
        from deltapd.campaign.thesis_campaign import run_thesis_campaign

        run_thesis_campaign(args.config)
        return 0

    if args.command == "run-material":
        from deltapd.campaign.material_state import run_material_state

        run_material_state(args.config)
        return 0

    if args.command in {"run-study", "run-descriptor-study"}:
        from deltapd.campaign.descriptor_study import run_descriptor_study

        run_descriptor_study(args.config)
        return 0

    if args.command == "run-mat-series":
        from deltapd.campaign.mat_series_study import run_mat_series_study

        run_mat_series_study(args.config)
        return 0

    if args.command == "run-comparative-study":
        from deltapd.campaign.comparative_thesis_study import run_comparative_thesis_study

        run_comparative_thesis_study(args.config)
        return 0

    if args.command == "run-state-alarm-batch":
        from deltapd.campaign.state_alarm_batch import run_state_alarm_batch

        run_state_alarm_batch(args.config)
        return 0

    if args.command == "run-workbench":
        from deltapd.workbench import serve_workbench

        serve_workbench(
            host=args.host,
            port=args.port,
            debug=args.debug,
            state_alarm_root=args.state_alarm_root,
            comparative_root=args.comparative_root,
            vna_root=args.vna_root,
            pd_base_dir=args.pd_base_dir,
        )
        return 0

    if args.command == "run-blind-prpd-stress":
        from deltapd.blind_prpd_stress import run_blind_prpd_stress_benchmark

        outputs = run_blind_prpd_stress_benchmark(
            output_dir=args.output_dir,
            n_trials=args.trials,
            seed=args.seed,
        )
        for key, value in outputs.items():
            print(f"{key}: {value}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
