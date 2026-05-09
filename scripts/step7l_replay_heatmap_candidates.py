#!/usr/bin/env python3
"""Replay Step7L heatmap candidate requests through the sidecar replay bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7l_learning_guided_replay import replay_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument(
        "--requests",
        type=Path,
        default=Path("artifacts/research/step7l_phase2_candidate_requests.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7l_phase2_summary.json"),
    )
    parser.add_argument(
        "--replay-rows-out",
        type=Path,
        default=Path("artifacts/research/step7l_phase2_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--failures-out",
        type=Path,
        default=Path("artifacts/research/step7l_phase2_failures_by_case.json"),
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--auto-download", action="store_true")
    args = parser.parse_args()
    summary = replay_requests(
        args.base_dir,
        args.requests,
        args.replay_rows_out,
        args.out,
        args.failures_out,
        floorset_root=args.floorset_root,
        max_requests=args.max_requests,
        auto_download=args.auto_download,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "generated_candidate_count": summary["generated_candidate_count"],
                "fresh_hard_feasible_nonnoop_count": summary["fresh_hard_feasible_nonnoop_count"],
                "fresh_official_like_improving_count": summary[
                    "fresh_official_like_improving_count"
                ],
                "fresh_quality_gate_pass_count": summary["fresh_quality_gate_pass_count"],
                "phase3_gnn_gate_open": summary["phase3_gnn_gate_open"],
                "phase4_offline_rl_gate_open": summary["phase4_offline_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
