#!/usr/bin/env python3
"""Generate Step7M Phase 4 deterministic multiblock corridor requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.experiments.step7m_multiblock_corridors import generate_multiblock_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-requests",
        type=Path,
        default=Path("artifacts/research/step7m_phase1_corridor_requests.jsonl"),
    )
    parser.add_argument(
        "--opportunity-atlas",
        type=Path,
        default=Path("artifacts/research/step7m_phase0_opportunity_atlas.jsonl"),
    )
    parser.add_argument("--max-pairs-per-case", type=int, default=24)
    parser.add_argument("--include-soft-budgeted", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7m_phase4_multiblock_requests.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7m_phase4_multiblock_request_summary.json"),
    )
    args = parser.parse_args()
    summary = generate_multiblock_requests(
        args.phase1_requests,
        args.opportunity_atlas,
        args.out,
        args.summary_out,
        max_pairs_per_case=args.max_pairs_per_case,
        include_soft_budgeted=args.include_soft_budgeted,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "request_count": summary["request_count"],
                "represented_case_count": summary["represented_case_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
