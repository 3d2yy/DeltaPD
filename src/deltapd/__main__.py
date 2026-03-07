"""CLI entrypoint for DeltaPD Core."""

from __future__ import annotations

import argparse
from deltapd.pipeline import main as legacy_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deltapd",
        description="DeltaPD Core -- UHF partial-discharge Δt tracker",
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
