#!/usr/bin/env python3
"""Step7R-C HPWL gradient nudge replay over Step7Q-F's AVNR rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.step7r_gradient_replay import replay_gradient_deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--step7q-rows", type=Path, required=True)
    parser.add_argument(
        "--replay-rows-out",
        type=Path,
        default=Path("artifacts/research/step7r_c_gradient_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7r_c_gradient_replay_summary.json"),
    )
    parser.add_argument(
        "--failures-out",
        type=Path,
        default=Path("artifacts/research/step7r_c_gradient_replay_failures_by_case.json"),
    )
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Replay every row in step7q-rows, not only AVNR rows.",
    )
    parser.add_argument(
        "--step-ladder",
        type=str,
        default="0.25,0.5,1.0",
        help="Comma-separated list of step factors.",
    )
    args = parser.parse_args()
    step_ladder = tuple(
        float(value.strip()) for value in args.step_ladder.split(",") if value.strip()
    )

    summary = replay_gradient_deck(
        args.base_dir,
        args.examples,
        args.step7q_rows,
        args.replay_rows_out,
        args.summary_out,
        args.failures_out,
        n_workers=args.n_workers,
        step_ladder=step_ladder,
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
        avnr_only=not args.all_rows,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "avnr_input_count": summary["avnr_input_count"],
                "variant_count": summary["variant_count"],
                "fresh_hard_feasible_nonnoop_count": summary["fresh_hard_feasible_nonnoop_count"],
                "strict_meaningful_winner_count": summary["strict_meaningful_winner_count"],
                "actual_all_vector_nonregressing_count": summary[
                    "actual_all_vector_nonregressing_count"
                ],
                "hpwl_strict_improvement_count": summary["hpwl_strict_improvement_count"],
                "phase2_gate_open": summary["phase2_gate_open"],
                "n_workers_used": summary["n_workers_used"],
                "runtime_proxy_ms": round(summary["runtime_proxy_ms"], 1),
            }
        )
    )


if __name__ == "__main__":
    main()
