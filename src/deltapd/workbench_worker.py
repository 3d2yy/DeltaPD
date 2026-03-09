from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from deltapd.workbench_jobs import (
    JOB_STATUS_FAILED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    read_workbench_job_manifest,
    run_pd_sensitivity,
    update_workbench_job_manifest,
)


def _progress_callback(
    job_manifest_path: Path,
    current: int,
    total: int,
    message: str,
    *,
    output_root: Path | None = None,
    stable_output_root: Path | None = None,
) -> None:
    updates: dict[str, Any] = {
        "progress_current": int(current),
        "progress_total": int(total),
        "message": str(message),
    }
    if output_root is not None:
        updates["output_root"] = str(output_root)
    if stable_output_root is not None:
        updates["stable_output_root"] = str(stable_output_root)
    update_workbench_job_manifest(job_manifest_path, **updates)


def run_job(job_manifest_path: Path) -> None:
    manifest = read_workbench_job_manifest(job_manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Job manifest not found: {job_manifest_path}")

    params = dict(manifest.get("params", {}))
    update_workbench_job_manifest(
        job_manifest_path,
        status=JOB_STATUS_RUNNING,
        started_at=manifest.get("started_at") or datetime.now().isoformat(timespec="seconds"),
        finished_at="",
        error="",
        message="Running CH3 semaphore sensitivity battery.",
    )
    try:
        outputs = run_pd_sensitivity(
            repo_root=Path(str(params.get("repo_root", Path.cwd()))),
            base_dir=Path(str(params.get("base_dir", ""))),
            dataset_keys=[str(key) for key in params.get("dataset_keys", [])],
            channel=str(params.get("channel", "CH3")),
            raw_input=str(params.get("raw_input", "")),
            output_root=job_manifest_path.parent / "output",
            progress_callback=lambda current, total, message, output_root=None, stable_output_root=None: _progress_callback(
                job_manifest_path,
                current,
                total,
                message,
                output_root=output_root,
                stable_output_root=stable_output_root,
            ),
        )
        update_workbench_job_manifest(
            job_manifest_path,
            status=JOB_STATUS_SUCCEEDED,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            output_root=str(outputs.get("output_root", "")),
            stable_output_root=str(outputs.get("stable_output_root", "")),
            message=str(outputs.get("message", "Sensitivity job completed.")),
            error="",
        )
    except Exception as exc:
        update_workbench_job_manifest(
            job_manifest_path,
            status=JOB_STATUS_FAILED,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            message="Sensitivity job failed.",
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeltaPD workbench background jobs.")
    parser.add_argument("--job-manifest", required=True, help="Path to job_manifest.json")
    args = parser.parse_args()
    run_job(Path(args.job_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
