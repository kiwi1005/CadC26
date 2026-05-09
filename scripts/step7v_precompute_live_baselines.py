#!/usr/bin/env python3
"""Pre-compute Step7V live ContestOptimizer baselines."""

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

from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.experiments.step7v_live_active_soft_adapter import (
    _live_optimizer_positions,
    load_specs_from_step7s,
)


def _cache_path(cache_dir: Path, case_id: int) -> Path:
    return cache_dir / f"case{case_id}.json"


def _tail(text: str | bytes | None, limit: int = 2000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-limit:]


def _write_one_baseline(
    *,
    base_dir: Path,
    step7s_summary: Path,
    cache_dir: Path,
    case_id: int,
    floorset_root: Path | None,
    auto_download: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    specs = load_specs_from_step7s(step7s_summary, [case_id])
    if not specs:
        raise ValueError(f"case_id {case_id} is not present in {step7s_summary}")

    cases = load_validation_cases(
        base_dir,
        [case_id],
        floorset_root=floorset_root,
        auto_download=auto_download,
    )
    case = cases[case_id]
    positions, optimizer_report = _live_optimizer_positions(base_dir, case)
    payload = {
        "case_id": case_id,
        "positions": [[float(value) for value in box] for box in positions],
        "optimizer_report": optimizer_report,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = _cache_path(cache_dir, case_id)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "case_id": case_id,
        "status": "completed",
        "cache_path": str(out),
        "runtime_seconds": time.perf_counter() - started,
    }


def _run_one_case(
    *,
    case_id: int,
    base_dir: Path,
    step7s_summary: Path,
    cache_dir: Path,
    floorset_root: Path | None,
    auto_download: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cmd = [
        sys.executable,
        "scripts/step7v_precompute_live_baselines.py",
        "--base-dir",
        str(base_dir),
        "--step7s-summary",
        str(step7s_summary),
        "--cache-dir",
        str(cache_dir),
        "--worker-case-id",
        str(case_id),
    ]
    if floorset_root is not None:
        cmd.extend(["--floorset-root", str(floorset_root)])
    if auto_download:
        cmd.append("--auto-download")

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
            "stdout_tail": _tail(exc.stdout),
            "stderr_tail": _tail(exc.stderr),
        }

    runtime = time.perf_counter() - started
    if proc.returncode != 0:
        return {
            "case_id": case_id,
            "status": "failed",
            "returncode": proc.returncode,
            "runtime_seconds": runtime,
            "blocker": "case_subprocess_failed",
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }

    out = _cache_path(cache_dir, case_id)
    if not out.exists():
        return {
            "case_id": case_id,
            "status": "failed",
            "runtime_seconds": runtime,
            "blocker": "case_cache_file_missing",
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
    return {
        "case_id": case_id,
        "status": "completed",
        "cache_path": str(out),
        "runtime_seconds": runtime,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--step7s-summary",
        type=Path,
        default=Path("artifacts/research/step7s_critical_cone_summary.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/research/step7v_live_baseline_cache"),
    )
    parser.add_argument("--case-id", type=int, action="append", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--case-timeout-seconds", type=int, default=3600)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--worker-case-id", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_case_id is not None:
        row = _write_one_baseline(
            base_dir=args.base_dir,
            step7s_summary=args.step7s_summary,
            cache_dir=args.cache_dir,
            case_id=args.worker_case_id,
            floorset_root=args.floorset_root,
            auto_download=args.auto_download,
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        return

    specs = load_specs_from_step7s(args.step7s_summary, args.case_id)
    case_ids = [int(spec["case_id"]) for spec in specs]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [
            executor.submit(
                _run_one_case,
                case_id=case_id,
                base_dir=args.base_dir,
                step7s_summary=args.step7s_summary,
                cache_dir=args.cache_dir,
                floorset_root=args.floorset_root,
                auto_download=args.auto_download,
                timeout_seconds=args.case_timeout_seconds,
            )
            for case_id in case_ids
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                {
                    "case_id": row.get("case_id"),
                    "status": row.get("status"),
                    "cache_path": row.get("cache_path"),
                    "runtime_seconds": row.get("runtime_seconds"),
                },
                flush=True,
            )

    rows = sorted(rows, key=lambda row: int(row.get("case_id", 10**9)))
    failed_case_count = sum(int(row.get("status") != "completed") for row in rows)
    summary = {
        "case_count": len(rows),
        "completed_case_count": len(rows) - failed_case_count,
        "failed_case_count": failed_case_count,
        "workers": int(args.workers),
        "case_timeout_seconds": int(args.case_timeout_seconds),
        "cache_dir": str(args.cache_dir),
        "runtime_seconds_wall": time.perf_counter() - started,
        "cases": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if failed_case_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
