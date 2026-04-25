#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from puzzleplace.actions import ExecutionState
from puzzleplace.research.puzzle_candidate_payload import (
    DENIED_PUZZLE_INFERENCE_FIELDS,
    build_puzzle_candidate_descriptors,
    masked_anchor_xywh,
    validate_puzzle_candidate_payload,
)
from puzzleplace.train.dataset_bc import load_validation_cases

FROZEN_PATHS = [
    "contest_optimizer.py",
    "src/puzzleplace/optimizer/contest.py",
    "src/puzzleplace/repair/finalizer.py",
    "src/puzzleplace/scoring",
    "src/puzzleplace/rollout",
]


def _frozen_diff_count() -> int:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *FROZEN_PATHS],
        check=False,
        text=True,
        capture_output=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*", default=["0"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    requested_ids = [int(v) for v in args.case_ids]
    cases = load_validation_cases(case_limit=max(requested_ids) + 1)
    case = cases[requested_ids[0]]
    descriptor = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[0]
    payload = descriptor.inference_payload()
    validate_puzzle_candidate_payload(payload)

    top_level = 0
    for field in DENIED_PUZZLE_INFERENCE_FIELDS:
        bad = dict(payload)
        bad[field] = 1
        try:
            validate_puzzle_candidate_payload(bad)
        except ValueError:
            top_level += 1
    nested_ok = False
    bad = dict(payload)
    bad["normalized_features"] = {"nested": {"target_positions": [1]}}
    try:
        validate_puzzle_candidate_payload(bad)
    except ValueError:
        nested_ok = True
    metadata_ok = False
    bad = dict(payload)
    bad["action_token"].metadata["manual_delta_rows"] = [[1.0]]
    try:
        validate_puzzle_candidate_payload(bad)
    except ValueError:
        metadata_ok = True

    missing_ok = False
    bad = dict(payload)
    bad.pop("site_id")
    try:
        validate_puzzle_candidate_payload(bad)
    except ValueError:
        missing_ok = True
    extra_ok = False
    bad = dict(payload)
    bad["debug_feature"] = 1
    try:
        validate_puzzle_candidate_payload(bad)
    except ValueError:
        extra_ok = True

    anchors = masked_anchor_xywh(case)
    leak_count = 0
    for idx in range(case.block_count):
        fixed = bool(case.constraints[idx, 0].item())
        preplaced = bool(case.constraints[idx, 1].item())
        if not (fixed or preplaced) and anchors[idx].tolist() != [-1.0, -1.0, -1.0, -1.0]:
            leak_count += 1

    report = {
        "case_ids": args.case_ids,
        "all_denied_fields_rejected": top_level == len(DENIED_PUZZLE_INFERENCE_FIELDS),
        "denied_field_rejection_count": top_level,
        "denied_field_total": len(DENIED_PUZZLE_INFERENCE_FIELDS),
        "nested_denied_fields_rejected": nested_ok,
        "action_metadata_denied_fields_rejected": metadata_ok,
        "missing_required_rejection": missing_ok,
        "unknown_extra_rejection": extra_ok,
        "soft_block_masked_anchor_leak_count": leak_count,
        "runtime_frozen_diff_count": _frozen_diff_count(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    md = Path(args.output).with_suffix(".md")
    md.write_text(
        "# Step6G Payload Guardrail\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
