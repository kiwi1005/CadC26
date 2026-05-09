#!/usr/bin/env python3
"""Run Step7T active-soft constrained descent POC."""

from __future__ import annotations

import argparse
from pathlib import Path

from puzzleplace.experiments.step7t_active_soft_cone import (
    load_step7s_case_specs,
    run_active_soft_cone,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--step7s-summary",
        type=Path,
        default=Path("artifacts/research/step7s_critical_cone_summary.json"),
    )
    parser.add_argument(
        "--case-id",
        type=int,
        action="append",
        default=None,
        help="Run one or more representative cases; default all Step7S cases.",
    )
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--max-candidates-per-case", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7t_active_soft_summary.json"),
    )
    parser.add_argument("--md", type=Path, default=Path("docs/step7t_active_soft_cone.md"))
    args = parser.parse_args()

    specs = load_step7s_case_specs(args.step7s_summary)
    if args.case_id is not None:
        wanted = set(args.case_id)
        specs = [spec for spec in specs if int(spec["case_id"]) in wanted]
    summary = run_active_soft_cone(
        args.base_dir,
        specs,
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
        max_candidates_per_case=args.max_candidates_per_case,
    )
    write_outputs(summary, args.out, args.md)
    print({
        "out": str(args.out),
        "md": str(args.md),
        "case_count": summary["case_count"],
        "strict_winner_count": summary["strict_winner_count"],
        "phase4_gate_open": summary["phase4_gate_open"],
    })


if __name__ == "__main__":
    main()
