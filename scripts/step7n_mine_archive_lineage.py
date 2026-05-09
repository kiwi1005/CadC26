#!/usr/bin/env python3
"""Run Step7N-ALR Phase0 archive-lineage mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7n_archive_lineage import (
    DEFAULT_MEANINGFUL_COST_EPS,
    DEFAULT_NON_MICRO_THRESHOLDS,
    mine_archive_lineage,
)


def _thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(".omx/plans/step7n-phase0-source-manifest.txt"),
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_lineage_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_lineage_summary.json"),
    )
    parser.add_argument(
        "--source-ledger-out",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_source_ledger.json"),
    )
    parser.add_argument(
        "--by-case-out",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_lineage_by_case.json"),
    )
    parser.add_argument(
        "--taxonomy-out",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_move_taxonomy.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7n_phase0_lineage_summary.md"),
    )
    parser.add_argument("--meaningful-cost-eps", type=float, default=DEFAULT_MEANINGFUL_COST_EPS)
    parser.add_argument(
        "--non-micro-thresholds",
        type=_thresholds,
        default=list(DEFAULT_NON_MICRO_THRESHOLDS),
    )
    args = parser.parse_args()

    summary = mine_archive_lineage(
        args.source_manifest,
        args.rows_out,
        args.summary_out,
        args.source_ledger_out,
        args.by_case_out,
        args.taxonomy_out,
        args.markdown_out,
        meaningful_cost_eps=args.meaningful_cost_eps,
        non_micro_thresholds=args.non_micro_thresholds,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "strict_meaningful_non_micro_winner_count": summary[
                    "strict_meaningful_non_micro_winner_count"
                ],
                "strict_winner_case_count": summary["strict_winner_case_count"],
                "case024_share": summary["case024_share"],
                "case025_share": summary["case025_share"],
                "phase1_gate_open": summary["phase1_gate_open"],
                "gnn_rl_gate_open": summary["gnn_rl_gate_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
