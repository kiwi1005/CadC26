#!/usr/bin/env python3
"""Generate Step7L Phase 2 request-only heatmap candidate targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.alternatives.learning_guided_targets import (
    DEFAULT_VALIDATION_CASES,
    generate_learning_guided_requests,
    write_request_summary,
)


def _case_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument(
        "--validation-cases",
        type=_case_list,
        default=list(DEFAULT_VALIDATION_CASES),
        help="Comma-separated validation loader indices to inspect.",
    )
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--top-blocks-per-family", type=int, default=4)
    parser.add_argument("--windows-per-block", type=int, default=3)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7l_phase2_candidate_requests.jsonl"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/research/step7l_phase2_request_summary.json"),
    )
    args = parser.parse_args()
    report = generate_learning_guided_requests(
        args.base_dir,
        args.out,
        floorset_root=args.floorset_root,
        validation_case_ids=args.validation_cases,
        grid_size=args.grid,
        top_blocks_per_family=args.top_blocks_per_family,
        windows_per_block=args.windows_per_block,
        auto_download=args.auto_download,
    )
    write_request_summary(report, args.summary_out)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "request_count": report["request_count"],
                "request_count_by_family": report["request_count_by_family"],
                "selected_case_count": report["selected_case_count"],
                "uses_validation_target_labels": report["uses_validation_target_labels"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
