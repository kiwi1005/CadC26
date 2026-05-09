#!/usr/bin/env python3
"""Run Step7V live-layout active-soft adapter sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from puzzleplace.experiments.step7v_live_active_soft_adapter import (
    load_specs_from_step7s,
    run_live_active_soft_adapter,
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
    parser.add_argument("--case-id", type=int, action="append", default=None)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--max-candidates-per-case", type=int, default=50)
    parser.add_argument("--objective-selection-k", type=int, default=1)
    parser.add_argument("--baseline-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7v_live_active_soft_summary.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7v_live_active_soft_summary.md"),
    )
    args = parser.parse_args()

    specs = load_specs_from_step7s(args.step7s_summary, args.case_id)
    summary = run_live_active_soft_adapter(
        args.base_dir,
        specs,
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
        max_candidates_per_case=args.max_candidates_per_case,
        objective_selection_k=args.objective_selection_k,
        baseline_cache_dir=args.baseline_cache_dir,
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
            "next_recommendation": summary["next_recommendation"],
        }
    )


if __name__ == "__main__":
    main()
