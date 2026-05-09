#!/usr/bin/env python3
"""Run Step7M Phase 3 ablation over replayed corridor rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7m_corridor_ablation import run_corridor_ablation


def _family_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-rows",
        type=Path,
        default=Path("artifacts/research/step7m_phase2_replay_rows.jsonl"),
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=Path("artifacts/research/step7m_phase3_ablation_rows.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7m_phase3_ablation_summary.json"),
    )
    parser.add_argument("--families", type=_family_list, default=None)
    args = parser.parse_args()
    summary = run_corridor_ablation(
        args.replay_rows,
        args.rows_out,
        args.out,
        families=args.families,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "best_family": summary["best_family"],
                "best_metric_regression_rate": summary["best_metric_regression_rate"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
