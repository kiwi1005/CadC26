#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.rollout import (
    beam_rollout,
    greedy_rollout,
    relaxed_rollout,
    semantic_rollout,
)
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases, run_bc_overfit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument(
        "--rollout-mode",
        choices=["semantic", "relaxed", "strict"],
        default=os.environ.get("ROLLOUT_MODE", "strict"),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    case_limit = int(os.environ.get("ROLLOUT_CASE_LIMIT", "5"))
    max_traces = int(os.environ.get("ROLLOUT_MAX_TRACES", "2"))
    hidden_dim = int(os.environ.get("ROLLOUT_HIDDEN_DIM", "64"))
    epochs = int(os.environ.get("ROLLOUT_EPOCHS", "20"))

    cases = load_validation_cases(case_limit=case_limit)
    if args.case_ids:
        selected = {f"validation-{idx}" for idx in args.case_ids}
        cases = [case for case in cases if str(case.case_id) in selected]
    dataset = build_bc_dataset_from_cases(cases, max_traces_per_case=max_traces)
    policy, summary = run_bc_overfit(dataset, hidden_dim=hidden_dim, epochs=epochs, lr=1e-3, seed=0)

    results = []
    for case in cases:
        if args.rollout_mode == "semantic":
            semantic = semantic_rollout(case, policy)
            results.append(
                {
                    "case_id": case.case_id,
                    "semantic_completed": semantic.semantic_completed,
                    "semantic_placed_fraction": semantic.semantic_placed_fraction,
                    "stopped_reason": semantic.stopped_reason,
                    "fallback_fraction": semantic.fallback_fraction,
                    "violation_profile": asdict(semantic.violation_profile),
                }
            )
        elif args.rollout_mode == "relaxed":
            relaxed = relaxed_rollout(case, policy)
            results.append(
                {
                    "case_id": case.case_id,
                    "semantic_completed": relaxed.semantic_completed,
                    "semantic_placed_fraction": relaxed.semantic_placed_fraction,
                    "stopped_reason": relaxed.stopped_reason,
                    "fallback_fraction": relaxed.fallback_fraction,
                    "violation_profile": asdict(relaxed.violation_profile),
                }
            )
        else:
            greedy = greedy_rollout(case, policy)
            beam = beam_rollout(case, policy, beam_width=4, per_state_candidates=3)
            results.append(
                {
                    "case_id": case.case_id,
                    "greedy": {
                        "placed_count": greedy.placed_count,
                        "all_blocks_placed": greedy.all_blocks_placed,
                        "stopped_reason": greedy.stopped_reason,
                        "feasible": greedy.feasible,
                    },
                    "beam": {
                        "placed_count": beam.placed_count,
                        "all_blocks_placed": beam.all_blocks_placed,
                        "stopped_reason": beam.stopped_reason,
                        "feasible": beam.feasible,
                    },
                }
            )
    payload = {
        "rollout_mode": args.rollout_mode,
        "bc_summary": {
            "initial_loss": summary.initial_loss,
            "final_loss": summary.final_loss,
            "primitive_accuracy": summary.primitive_accuracy,
            "block_accuracy": summary.block_accuracy,
        },
        "results": results,
    }
    output_path = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / "reports" / "agent9_rollout_validation0_4.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
