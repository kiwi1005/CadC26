#!/usr/bin/env python3
"""Build Step7M-OAC objective opportunity atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.alternatives.objective_corridors import (
    DEFAULT_CASES,
    build_opportunity_atlas,
    load_cases_for_step7m,
    write_opportunity_artifacts,
)


def _case_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--validation-cases", type=_case_list, default=list(DEFAULT_CASES))
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--anchor-source", default="original_anchor")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7m_phase0_opportunity_atlas.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7m_phase0_opportunity_summary.json"),
    )
    args = parser.parse_args()
    cases = load_cases_for_step7m(args.base_dir, args.validation_cases)
    rows = build_opportunity_atlas(cases, grid_size=args.grid, anchor_id=args.anchor_source)
    summary = write_opportunity_artifacts(rows, args.out, args.summary_out, grid_size=args.grid)
    print(
        json.dumps({"decision": summary["decision"], "row_count": summary["row_count"]}, indent=2)
    )


if __name__ == "__main__":
    main()
