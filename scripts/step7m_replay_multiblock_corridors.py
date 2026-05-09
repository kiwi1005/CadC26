#!/usr/bin/env python3
"""Replay Step7M Phase 4 deterministic multiblock corridor requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7m_multiblock_corridors import replay_multiblock_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--requests",
        type=Path,
        default=Path("artifacts/research/step7m_phase4_multiblock_requests.jsonl"),
    )
    parser.add_argument(
        "--replay-rows-out",
        type=Path,
        default=Path("artifacts/research/step7m_phase4_multiblock_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7m_phase4_multiblock_replay_summary.json"),
    )
    parser.add_argument(
        "--failures-out",
        type=Path,
        default=Path("artifacts/research/step7m_phase4_multiblock_failures_by_case.json"),
    )
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--auto-download", action="store_true")
    args = parser.parse_args()
    summary = replay_multiblock_requests(
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
                "fresh_hard_feasible_nonnoop_count": summary["fresh_hard_feasible_nonnoop_count"],
                "meaningful_official_like_improving_count": summary[
                    "meaningful_official_like_improving_count"
                ],
                "actual_metric_regression_rate": summary["actual_metric_regression_rate"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
