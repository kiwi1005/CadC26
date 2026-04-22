#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.repair import finalize_layout
from puzzleplace.rollout import semantic_rollout
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases, run_bc_overfit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "reports" / "sprint2_pivot_repair_validate.json"),
    )
    args = parser.parse_args()

    cases = load_validation_cases(case_limit=max(args.case_ids) + 1 if args.case_ids else 5)
    selected = {f"validation-{idx}" for idx in args.case_ids}
    cases = [case for case in cases if str(case.case_id) in selected]
    dataset = build_bc_dataset_from_cases(cases, max_traces_per_case=2)
    policy, _summary = run_bc_overfit(dataset, hidden_dim=64, epochs=5, lr=1e-3, seed=0)

    results = []
    for case in cases:
        semantic = semantic_rollout(case, policy)
        repair = finalize_layout(case, semantic.proposed_positions)
        results.append(
            {
                "case_id": case.case_id,
                "semantic_completed": semantic.semantic_completed,
                "semantic_placed_fraction": semantic.semantic_placed_fraction,
                "hard_feasible_after_repair": repair.report.hard_feasible_after,
                "repair_report": asdict(repair.report),
            }
        )

    payload = {
        "results": results,
        "repair_success_rate": sum(1 for result in results if result["hard_feasible_after_repair"])
        / max(len(results), 1),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
