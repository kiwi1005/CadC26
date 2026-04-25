#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.actions import ActionExecutor, ExecutionState
from puzzleplace.research.puzzle_candidate_payload import (
    build_puzzle_candidate_descriptors,
    choose_expert_descriptor,
    empty_mask_reason_buckets,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from puzzleplace.trajectory import generate_pseudo_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*", default=["0"])
    parser.add_argument("--max-traces", type=int, default=1)
    parser.add_argument("--first-steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    total = matched = expert_masked = illegal_to_scorer = 0
    misses: list[dict[str, object]] = []
    families: set[str] = set()
    contact_modes: set[str] = set()
    anchor_modes: set[str] = set()
    mask_reason_buckets = empty_mask_reason_buckets()
    requested_ids = [int(v) for v in args.case_ids]
    cases = load_validation_cases(case_limit=max(requested_ids) + 1)
    for case_id_int in requested_ids:
        case_id = str(case_id_int)
        case = cases[case_id_int]
        for trace in generate_pseudo_traces(case, max_traces=args.max_traces):
            state = ExecutionState()
            executor = ActionExecutor(case)
            for action in trace.actions:
                if action.primitive.value == "freeze":
                    executor.apply(state, action)
                    continue
                if total >= args.first_steps:
                    break
                remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
                descriptors = build_puzzle_candidate_descriptors(
                    case,
                    state,
                    remaining_blocks=remaining,
                    mask_reason_buckets=mask_reason_buckets,
                )
                illegal_to_scorer += sum(desc.legality_status != "legal" for desc in descriptors)
                families.update(desc.candidate_family for desc in descriptors)
                contact_modes.update(desc.contact_mode for desc in descriptors)
                anchor_modes.update(desc.anchor_kind for desc in descriptors)
                expert = choose_expert_descriptor(descriptors, action)
                total += 1
                if expert is None:
                    expert_masked += 1
                    misses.append(
                        {
                            "case_id": case_id,
                            "step": total,
                            "block_index": action.block_index,
                            "miss_bucket": "no_same_block_candidate",
                        }
                    )
                else:
                    matched += 1
                executor.apply(state, action)
            if total >= args.first_steps:
                break
        if total >= args.first_steps:
            break

    miss_rate = 1.0 - matched / max(total, 1)
    report = {
        "case_ids": args.case_ids,
        "total_steps": total,
        "matched_steps": matched,
        "candidate_miss_rate": miss_rate,
        "expert_masked_rate": expert_masked / max(total, 1),
        "illegal_candidate_to_scorer_count": int(illegal_to_scorer),
        "shape_bin_family_count": sum("shape_bin" in family for family in families),
        "contact_mode_family_count": len(contact_modes),
        "boundary_or_anchor_family_count": len(anchor_modes),
        "misses": misses,
        "mask_reason_buckets": mask_reason_buckets,
        "families": sorted(families)[:50],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
