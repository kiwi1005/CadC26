#!/usr/bin/env python3
"""Parallel Step7V live-layout active-soft adapter runner.

Runs each representative case in a separate Python process.  This is intentionally
process-level parallelism so slow/hung cases can be timed out independently while
using the workstation's 48 CPU cores.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from puzzleplace.experiments.step7v_live_active_soft_adapter import (
    aggregate_case_summaries,
    load_specs_from_step7s,
    write_outputs,
)


def _case_paths(output_dir: Path, case_id: int) -> tuple[Path, Path]:
    return (
        output_dir / f"step7v_live_active_soft_case{case_id}.json",
        output_dir / f"step7v_live_active_soft_case{case_id}.md",
    )


def _load_completed_case(path: Path, case_id: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["case_id"] = case_id
    data["status"] = "completed"
    data["source_artifact"] = str(path)
    data["reused_existing"] = True
    return data


def _run_one_case(
    *,
    case_id: int,
    base_dir: Path,
    step7s_summary: Path,
    output_dir: Path,
    max_candidates_per_case: int,
    objective_selection_k: int,
    baseline_cache_dir: Path | None,
    timeout_seconds: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    out, md = _case_paths(output_dir, case_id)
    if reuse_existing and out.exists():
        return _load_completed_case(out, case_id)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    # Avoid accidental nested BLAS/OpenMP oversubscription when many cases run.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cmd = [
        sys.executable,
        "scripts/step7v_run_live_active_soft_adapter.py",
        "--base-dir",
        str(base_dir),
        "--step7s-summary",
        str(step7s_summary),
        "--case-id",
        str(case_id),
        "--max-candidates-per-case",
        str(max_candidates_per_case),
        "--objective-selection-k",
        str(objective_selection_k),
        "--out",
        str(out),
        "--markdown-out",
        str(md),
    ]
    if baseline_cache_dir is not None:
        cmd.extend(["--baseline-cache-dir", str(baseline_cache_dir)])
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=base_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "case_id": case_id,
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "blocker": "case_subprocess_timeout",
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }
    runtime = time.perf_counter() - started
    if proc.returncode != 0:
        return {
            "case_id": case_id,
            "status": "failed",
            "returncode": proc.returncode,
            "runtime_seconds": runtime,
            "blocker": "case_subprocess_failed",
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    data = _load_completed_case(out, case_id)
    data["runtime_seconds_wall"] = runtime
    data["stdout_tail"] = proc.stdout[-2000:]
    data["stderr_tail"] = proc.stderr[-2000:]
    data["reused_existing"] = False
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--step7s-summary",
        type=Path,
        default=Path("artifacts/research/step7s_critical_cone_summary.json"),
    )
    parser.add_argument("--case-id", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--case-timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-candidates-per-case", type=int, default=50)
    parser.add_argument("--objective-selection-k", type=int, default=1)
    parser.add_argument("--baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7v_live_active_soft_parallel_summary.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7v_live_active_soft_parallel_summary.md"),
    )
    args = parser.parse_args()

    specs = load_specs_from_step7s(args.step7s_summary, args.case_id)
    case_ids = [int(spec["case_id"]) for spec in specs]
    started = time.perf_counter()
    case_summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(
                _run_one_case,
                case_id=case_id,
                base_dir=args.base_dir,
                step7s_summary=args.step7s_summary,
                output_dir=args.output_dir,
                max_candidates_per_case=args.max_candidates_per_case,
                objective_selection_k=args.objective_selection_k,
                baseline_cache_dir=args.baseline_cache_dir,
                timeout_seconds=args.case_timeout_seconds,
                reuse_existing=args.reuse_existing,
            )
            for case_id in case_ids
        ]
        for future in as_completed(futures):
            row = future.result()
            case_summaries.append(row)
            print(
                {
                    "case_id": row.get("case_id"),
                    "status": row.get("status"),
                    "strict_winner_count": row.get("strict_winner_count"),
                    "strict_winner_case_count": row.get("strict_winner_case_count"),
                    "runtime_seconds": row.get("runtime_seconds_wall", row.get("runtime_seconds")),
                },
                flush=True,
            )

    summary = aggregate_case_summaries(case_summaries)
    summary["workers"] = int(args.workers)
    summary["case_timeout_seconds"] = int(args.case_timeout_seconds)
    summary["baseline_cache_dir"] = (
        str(args.baseline_cache_dir) if args.baseline_cache_dir is not None else None
    )
    summary["runtime_seconds_wall"] = time.perf_counter() - started
    summary["case_process_summaries"] = sorted(
        [
            {
                key: row.get(key)
                for key in (
                    "case_id",
                    "status",
                    "reused_existing",
                    "strict_winner_count",
                    "strict_winner_case_count",
                    "runtime_seconds",
                    "runtime_seconds_wall",
                    "blocker",
                    "source_artifact",
                    "returncode",
                    "timeout_seconds",
                )
                if key in row
            }
            for row in case_summaries
        ],
        key=lambda row: int(row.get("case_id", 10**9)),
    )
    write_outputs(summary, args.out, args.markdown_out)
    print(
        {
            "out": str(args.out),
            "markdown_out": str(args.markdown_out),
            "decision": summary["decision"],
            "strict_winner_count": summary["strict_winner_count"],
            "strict_winner_case_count": summary["strict_winner_case_count"],
            "phase4_gate_open": summary["phase4_gate_open"],
            "failed_case_count": summary["failed_case_count"],
            "runtime_seconds_wall": summary["runtime_seconds_wall"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
