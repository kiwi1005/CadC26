#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from puzzleplace.research.pareto_selector import (
    build_pareto_candidates,
    compact_candidate,
    objective_ranges,
    pareto_front,
    representative_to_selection,
    select_representatives,
    summarize_case_fronts,
)
from puzzleplace.research.selector_replay import compare_selection_sets


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _group_by_case(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["case_id"])].append(row)
    return dict(sorted(grouped.items()))


def _selected_by_case(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return list(_load_json(file_path))


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    comparisons = report["comparisons"]
    lines = [
        "# Step6P Pareto Alternative Selector",
        "",
        "## Scope",
        "",
        "- Sidecar only; no contest runtime integration changed.",
        "- Original is included as a Pareto candidate with zero deltas/disruption.",
        "- Hard-invalid alternatives are filtered before Pareto ranking.",
        "- Applicability filter removes no-effect / role-inapplicable moves before selection.",
        "- Depth-2 combos are limited to the AGENT.md-approved templates and are marked estimated.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Closest-to-ideal comparison",
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- Pareto front output is the deliverable; `closest_to_ideal` is only a",
        "  representative, not a runtime rule.",
        "- `original` can win naturally when moves are dominated by",
        "  zero-disruption/no-regression baseline.",
        "- Low-risk boundary alternatives remain visible when non-dominated,",
        "  instead of being hidden by boundary-first lexicographic order.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-rows",
        default="artifacts/research/step6m_move_library_eval.json",
        help="Step6M move-evaluation rows used as the Step6P alternative pool.",
    )
    parser.add_argument(
        "--step6m-selected",
        default="artifacts/research/step6m_selected_alternatives.json",
    )
    parser.add_argument(
        "--step6o-selected",
        default="artifacts/research/step6o_guarded_selected_alternatives.json",
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    eval_rows = list(_load_json(args.eval_rows))
    by_case = _group_by_case(eval_rows)
    case_fronts: list[dict[str, Any]] = []
    closest_to_ideal_selections: list[dict[str, Any]] = []
    objective_range_rows: list[dict[str, Any]] = []

    for case_id, rows in by_case.items():
        candidates, filter_stats = build_pareto_candidates(rows)
        front = pareto_front(candidates)
        reps = select_representatives(front)
        compact_front = [compact_candidate(row) for row in front]
        compact_reps = {name: compact_candidate(row) for name, row in reps.items()}
        ranges = objective_ranges(candidates)
        case_fronts.append(
            {
                "case_id": case_id,
                "input_alternative_count": len(rows),
                "candidate_count": len(candidates),
                "front_size": len(front),
                "filter_stats": filter_stats,
                "pareto_front": compact_front,
                "representatives": compact_reps,
            }
        )
        objective_range_rows.append({"case_id": case_id, "objective_ranges": ranges})
        if reps:
            closest_to_ideal_selections.append(
                representative_to_selection(reps["closest_to_ideal"], "closest_to_ideal")
            )

    summary = summarize_case_fronts(case_fronts)
    step6m_selected = _selected_by_case(args.step6m_selected)
    step6o_selected = _selected_by_case(args.step6o_selected)
    comparisons: dict[str, Any] = {}
    if step6m_selected:
        comparisons["vs_step6m"] = compare_selection_sets(
            step6m_selected,
            closest_to_ideal_selections,
        )
    if step6o_selected:
        comparisons["vs_step6o"] = compare_selection_sets(
            step6o_selected,
            closest_to_ideal_selections,
        )

    summary["case15_closest_to_ideal"] = next(
        (row for row in closest_to_ideal_selections if int(row.get("case_id", -1)) == 15),
        None,
    )
    summary["representative_counts"] = dict(
        Counter(str(row["selected_move_type"]) for row in closest_to_ideal_selections)
    )
    report = {
        "input_eval_rows": args.eval_rows,
        "case_count": len(case_fronts),
        "summary": summary,
        "comparisons": comparisons,
        "notes": {
            "runtime_integration": "not_changed",
            "primary_representative": "closest_to_ideal_for_comparison_only",
            "depth2_combos": (
                "estimated from valid single-move deltas; "
                "not a full move-sequence search"
            ),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "step6p_pareto_alternatives.json").write_text(json.dumps(case_fronts, indent=2))
    (output_dir / "step6p_pareto_fronts.json").write_text(
        json.dumps(
            [
                {
                    "case_id": row["case_id"],
                    "front_size": row["front_size"],
                    "pareto_front": row["pareto_front"],
                }
                for row in case_fronts
            ],
            indent=2,
        )
    )
    (output_dir / "step6p_selected_representatives.json").write_text(
        json.dumps(
            [
                {"case_id": row["case_id"], "representatives": row["representatives"]}
                for row in case_fronts
            ],
            indent=2,
        )
    )
    (output_dir / "step6p_objective_ranges.json").write_text(
        json.dumps(objective_range_rows, indent=2)
    )
    (output_dir / "step6p_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "step6p_profile_summary.md").write_text(_markdown(report))
    print(
        json.dumps(
            {
                "output": str(output_dir / "step6p_report.json"),
                "case_count": len(case_fronts),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
