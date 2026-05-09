#!/usr/bin/env python3
"""Generate Step7M-OAC objective-corridor request deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from puzzleplace.alternatives.objective_corridors import (
    DEFAULT_CASES,
    GateMode,
    generate_corridor_requests,
    load_cases_for_step7m,
    write_corridor_artifacts,
)


def _case_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _gate_modes(value: str) -> list[GateMode]:
    allowed = {"wire_safe", "bbox_shrink_wire_safe", "soft_repair_budgeted"}
    modes = [item.strip() for item in value.split(",") if item.strip()]
    bad = [mode for mode in modes if mode not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown gate modes: {bad}")
    return [cast(GateMode, mode) for mode in modes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--validation-cases", type=_case_list, default=list(DEFAULT_CASES))
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--candidate-cells-per-block", type=int, default=24)
    parser.add_argument("--max-blocks-per-case", type=int, default=16)
    parser.add_argument("--windows-per-block", type=int, default=8)
    parser.add_argument(
        "--families",
        type=_gate_modes,
        default=["wire_safe", "bbox_shrink_wire_safe", "soft_repair_budgeted"],
    )
    parser.add_argument("--heatmap-mode", default="tiebreak_only")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7m_phase1_corridor_requests.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7m_phase1_corridor_request_summary.json"),
    )
    args = parser.parse_args()
    del args.heatmap_mode  # heatmap is currently always tiebreak/source-only.
    cases = load_cases_for_step7m(args.base_dir, args.validation_cases)
    rows, summary = generate_corridor_requests(
        cases,
        grid_size=args.grid,
        candidate_cells_per_block=args.candidate_cells_per_block,
        max_blocks_per_case=args.max_blocks_per_case,
        windows_per_block=args.windows_per_block,
        gate_modes=args.families,
    )
    summary = write_corridor_artifacts(rows, summary, args.out, args.summary_out)
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
