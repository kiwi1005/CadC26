#!/usr/bin/env python3
"""Run Step7Q-D fresh-metric replay for parameter expansion deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.step7q_fresh_metric_replay import replay_parameter_expansion_deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--expansion-deck", type=Path, required=True)
    parser.add_argument("--parameter-summary", type=Path, required=True)
    parser.add_argument("--replay-rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, required=True)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--slot-aware", action="store_true")
    parser.add_argument("--objective-aware-slot", action="store_true")
    args = parser.parse_args()
    summary = replay_parameter_expansion_deck(
        args.base_dir,
        args.examples,
        args.expansion_deck,
        args.parameter_summary,
        args.replay_rows_out,
        args.summary_out,
        args.failures_out,
        floorset_root=args.floorset_root,
        max_candidates=args.max_candidates,
        auto_download=args.auto_download,
        slot_aware=args.slot_aware or args.objective_aware_slot,
        objective_aware_slot=args.objective_aware_slot,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "request_count": summary["request_count"],
                "fresh_metric_available_count": summary["fresh_metric_available_count"],
                "fresh_hard_feasible_nonnoop_count": summary["fresh_hard_feasible_nonnoop_count"],
                "strict_meaningful_winner_count": summary["strict_meaningful_winner_count"],
                "phase4_gate_open": summary["phase4_gate_open"],
                "allowed_next_phase": summary["allowed_next_phase"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
