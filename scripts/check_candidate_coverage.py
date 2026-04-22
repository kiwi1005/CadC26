#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import actions_match, generate_candidate_actions
from puzzleplace.actions.executor import ExecutionState, replay_actions
from puzzleplace.train import load_validation_cases
from puzzleplace.trajectory import generate_pseudo_traces


def _reason(mode: str, action) -> str:
    if mode == "strict":
        return "masked_out"
    return "relation_not_generated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--case-ids", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--modes", nargs="*", default=["semantic", "relaxed", "strict"])
    parser.add_argument("--max-traces", type=int, default=2)
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "reports" / "sprint2_pivot_candidate_coverage.json"),
    )
    args = parser.parse_args()

    selected = set(args.case_ids)
    max_case_id = max(selected) if selected else 4
    cases = load_validation_cases(case_limit=max_case_id + 1)
    totals = {mode: 0 for mode in args.modes}
    hits = {mode: 0 for mode in args.modes}
    teacher_hint_hits = 0
    missing_examples: list[dict[str, object]] = []

    for case in cases:
        case_idx = int(str(case.case_id).split("-")[-1])
        if case_idx not in selected:
            continue
        for trace in generate_pseudo_traces(case, max_traces=args.max_traces):
            state = ExecutionState()
            for step, action in enumerate(trace.actions):
                remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
                for mode in args.modes:
                    totals[mode] += 1
                    candidates = generate_candidate_actions(
                        case,
                        state,
                        remaining_blocks=remaining,
                        mode=mode,
                    )
                    if any(
                        actions_match(candidate, action, mode=mode)
                        for candidate in candidates
                    ):
                        hits[mode] += 1
                    else:
                        missing_examples.append(
                            {
                                "case_id": case.case_id,
                                "step": step,
                                "mode": mode,
                                "primitive": str(action.primitive),
                                "reason": _reason(mode, action),
                            }
                        )
                teacher_hint = generate_candidate_actions(
                    case,
                    state,
                    remaining_blocks=remaining,
                    teacher_action=action,
                    include_teacher_hint=True,
                    mode="semantic",
                )
                teacher_hint_hits += int(
                    any(
                        actions_match(candidate, action, mode="strict")
                        for candidate in teacher_hint
                    )
                )
                replay_actions(case, [action], initial_state=state)

    payload = {
        "semantic_coverage": hits.get("semantic", 0) / max(totals.get("semantic", 1), 1),
        "relaxed_coverage": hits.get("relaxed", 0) / max(totals.get("relaxed", 1), 1),
        "strict_coverage": hits.get("strict", 0) / max(totals.get("strict", 1), 1),
        "teacher_hint_coverage": teacher_hint_hits
        / max(sum(totals.values()) // max(len(args.modes), 1), 1),
        "coverage_by_primitive": {},
        "missing_examples": missing_examples[:25],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
