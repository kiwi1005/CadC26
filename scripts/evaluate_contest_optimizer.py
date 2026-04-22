#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEST_ROOT = ROOT / "external" / "FloorSet" / "iccad2026contest"

for path in (ROOT, SRC, CONTEST_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from iccad2026_evaluate import validate_submission

from scripts.download_smoke import _auto_approve_downloads, _ensure_import_paths
from puzzleplace.eval import evaluate_positions
from puzzleplace.optimizer import ContestOptimizer
from puzzleplace.train import load_validation_cases


def main() -> None:
    case_limit = int(os.environ.get("CONTEST_CASE_LIMIT", "5"))
    optimizer_path = ROOT / "contest_optimizer.py"
    _ensure_import_paths()
    _auto_approve_downloads()

    quick_validate = validate_submission(str(optimizer_path), quick=True, verbose=False)
    optimizer = ContestOptimizer()
    cases = load_validation_cases(case_limit=case_limit)
    test_results: list[dict[str, Any]] = []
    total_cost = 0.0
    feasible_count = 0
    fallback_total = 0.0
    for test_id, case in enumerate(cases):
        positions, report = optimizer.solve_with_report(
            case.block_count,
            case.area_targets,
            case.b2b_edges,
            case.p2b_edges,
            case.pins_pos,
            case.constraints,
            case.target_positions,
        )
        evaluation = evaluate_positions(case, positions)
        feasible = bool(evaluation["official"]["is_feasible"])
        feasible_count += int(feasible)
        total_cost += float(evaluation["official"]["cost"])
        fallback_total += float(report["fallback_fraction"])
        test_results.append(
            {
                "test_id": test_id,
                "case_id": case.case_id,
                "is_feasible": feasible,
                "cost": evaluation["official"]["cost"],
                "official": evaluation["official"],
                "legality": evaluation["legality"],
                "optimizer_report": report,
            }
        )
    total_score = feasible_count / max(len(test_results), 1) * 10.0 + total_cost / max(
        len(test_results),
        1,
    )
    summary = {
        "num_tests": len(test_results),
        "num_feasible": feasible_count,
        "avg_cost": total_cost / max(len(test_results), 1),
        "avg_fallback_fraction": fallback_total / max(len(test_results), 1),
    }
    payload = {
        "optimizer_path": str(optimizer_path.relative_to(ROOT)),
        "quick_validate": bool(quick_validate),
        "case_limit": case_limit,
        "submission_name": "contest_optimizer",
        "total_score": total_score,
        "summary": summary,
        "test_results": test_results,
    }
    report_dir = ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "agent12_contest_validation0_4.json").write_text(json.dumps(payload, indent=2))
    (report_dir / "agent12_contest_summary.md").write_text(
        "\n".join(
            [
                "# Agent 12 Summary",
                "",
                f"- quick validator: `{quick_validate}`",
                "- submission: `contest_optimizer`",
                f"- evaluated cases: `{case_limit}`",
                f"- total score: `{total_score:.4f}`",
                f"- feasible cases: `{summary['num_feasible']}`",
                f"- mean cost: `{summary['avg_cost']}`",
                f"- mean fallback fraction: `{summary['avg_fallback_fraction']}`",
            ]
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
