#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"


def _read(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def main() -> None:
    coverage = _read("sprint2_pivot_candidate_coverage.json")
    semantic = _read("sprint2_pivot_semantic_rollout.json")
    repair = _read("sprint2_pivot_repair_validate.json")
    contest = _read("agent12_contest_validation0_4.json")

    semantic_completed = [
        result.get("semantic_completed", False) for result in semantic["results"]
    ]
    semantic_fractions = [
        result.get("semantic_placed_fraction", 0.0) for result in semantic["results"]
    ]
    repair_success = [
        result.get("hard_feasible_after_repair", False) for result in repair["results"]
    ]
    overlap_before = [
        result["repair_report"]["total_overlap_area_before"] for result in repair["results"]
    ]
    overlap_after = [
        result["repair_report"]["total_overlap_area_after"] for result in repair["results"]
    ]

    payload = {
        "semantic_candidate_coverage": coverage["semantic_coverage"],
        "relaxed_candidate_coverage": coverage["relaxed_coverage"],
        "strict_candidate_coverage": coverage["strict_coverage"],
        "teacher_hint_coverage": coverage["teacher_hint_coverage"],
        "semantic_rollout_completion_rate": sum(semantic_completed)
        / max(len(semantic_completed), 1),
        "avg_semantic_placed_fraction": sum(semantic_fractions) / max(len(semantic_fractions), 1),
        "repair_success_rate": sum(repair_success) / max(len(repair_success), 1),
        "avg_overlap_area_before": sum(overlap_before) / max(len(overlap_before), 1),
        "avg_overlap_area_after": sum(overlap_after) / max(len(overlap_after), 1),
        "contest_quick_validate": contest["quick_validate"],
        "contest_avg_fallback_fraction": contest["summary"].get("avg_fallback_fraction", 0.0),
        "contest_num_feasible": contest["summary"].get("num_feasible", 0),
    }

    output_path = REPORTS / "sprint2_pivot_summary.json"
    output_path.write_text(json.dumps(payload, indent=2))
    (REPORTS / "sprint2_pivot_summary.md").write_text(
        "\n".join(
            [
                "# Sprint 2 Pivot Summary",
                "",
                f"- semantic candidate coverage: `{payload['semantic_candidate_coverage']:.4f}`",
                f"- relaxed candidate coverage: `{payload['relaxed_candidate_coverage']:.4f}`",
                f"- strict candidate coverage: `{payload['strict_candidate_coverage']:.4f}`",
                f"- teacher hint coverage: `{payload['teacher_hint_coverage']:.4f}`",
                (
                    f"- semantic rollout completion rate: "
                    f"`{payload['semantic_rollout_completion_rate']:.4f}`"
                ),
                f"- avg semantic placed fraction: `{payload['avg_semantic_placed_fraction']:.4f}`",
                f"- repair success rate: `{payload['repair_success_rate']:.4f}`",
                f"- avg overlap area before repair: `{payload['avg_overlap_area_before']:.4f}`",
                f"- avg overlap area after repair: `{payload['avg_overlap_area_after']:.4f}`",
                f"- contest quick validate: `{payload['contest_quick_validate']}`",
                f"- contest avg fallback fraction: `{payload['contest_avg_fallback_fraction']}`",
                f"- contest feasible cases: `{payload['contest_num_feasible']}`",
            ]
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
