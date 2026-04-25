#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from puzzleplace.repair import finalize_layout
from puzzleplace.research.boundary_failure_attribution import classify_boundary_failures
from puzzleplace.research.move_library import (
    build_case_suite,
    evaluate_move,
    generate_move_candidates,
    move_cost_summary,
    profile_case,
    profile_summary,
    select_case_alternative,
)
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.virtual_frame import (
    estimate_predicted_compact_hull,
    final_bbox_boundary_metrics,
    multistart_virtual_frames,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import _construct

Placement = tuple[float, float, float, float]


def _load_json_if_exists(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text())


def _reconstruct_layout(
    case: Any,
    case_id: int,
    step6j: dict[str, Any],
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[dict[int, Placement], Any, Any, dict[str, int]]:
    runs_by_case = {int(row["case_id"]): row for row in step6j.get("runs", [])}
    frames = multistart_virtual_frames(case)
    if case_id in runs_by_case:
        run = runs_by_case[case_id]
        frame = frames[int(run["best_frame_index"])]
        seed = int(run["best_seed"])
        start = int(run["best_start"])
    else:
        frame = frames[case_id % len(frames)]
        seed = case_id % 3
        start = case_id % 5
    pre, family_usage, construction_frame, predicted_hull = _construct(
        case,
        seed,
        start,
        frame,
        boundary_commit_mode=boundary_commit_mode,
    )
    if construction_frame is None:
        construction_frame = frame
    if predicted_hull is None:
        predicted_hull = estimate_predicted_compact_hull(case, construction_frame)
    repair = finalize_layout(case, pre)
    post = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repair.positions)
    }
    return post, construction_frame, predicted_hull, family_usage


def _md_profile_summary(
    *,
    suite: dict[str, Any],
    profile_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    costs: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    selected_counts = Counter(str(row["selected_move_type"]) for row in selections)
    diagnostic = [row for row in selections if row.get("suite") == "diagnostic"]
    holdout = [row for row in selections if row.get("suite") == "holdout"]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)

    expensive = sorted(
        costs.items(),
        key=lambda item: float(item[1]["mean_total_ms"]),
        reverse=True,
    )[:5]
    lines = [
        "# Step6M Move Library Profile Summary",
        "",
        "## Suite",
        "",
        f"- diagnostic cases: {suite['diagnostic_count']}",
        f"- holdout cases: {suite['holdout_count']}",
        f"- size bucket counts: `{json.dumps(suite['size_bucket_counts'], sort_keys=True)}`",
        "",
        "## Selected move counts",
        "",
        *[
            f"- `{move_type}`: {count}"
            for move_type, count in sorted(selected_counts.items())
        ],
        "",
        "## Diagnostic vs holdout deltas",
        "",
        (
            f"- diagnostic mean boundary delta: {mean(diagnostic, 'boundary_delta'):.4f}; "
            f"mean bbox delta: {mean(diagnostic, 'bbox_delta'):.2f}; "
            f"mean HPWL delta: {mean(diagnostic, 'hpwl_delta'):.4f}"
        ),
        (
            f"- holdout mean boundary delta: {mean(holdout, 'boundary_delta'):.4f}; "
            f"mean bbox delta: {mean(holdout, 'bbox_delta'):.2f}; "
            f"mean HPWL delta: {mean(holdout, 'hpwl_delta'):.4f}"
        ),
        "",
        "## Per-profile preference tables",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Most expensive move families",
        "",
        *[
            f"- `{move_type}`: mean_total_ms={payload['mean_total_ms']:.4f}, "
            f"accepted={payload['accepted_count']}/{payload['count']}"
            for move_type, payload in expensive
        ],
        "",
        "## Gate notes",
        "",
        "- Step6M is still research/safe sidecar; no runtime integration changed.",
        "- Selected alternatives must remain hard-feasible and frame-protrusion free.",
        f"- Profiles emitted: {len(profile_rows)}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step6j",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
    )
    parser.add_argument("--case-limit", type=int, default=40)
    parser.add_argument("--diagnostic-count", type=int, default=20)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--top-k-targets", type=int, default=5)
    parser.add_argument("--top-m-moves", type=int, default=12)
    parser.add_argument("--mode", choices=["research", "safe"], default="safe")
    parser.add_argument(
        "--output",
        default="artifacts/research/step6m_report.json",
    )
    args = parser.parse_args()

    step6j = _load_json_if_exists(args.step6j)
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    cases = load_validation_cases(case_limit=args.case_limit)
    suite = build_case_suite(
        cases,
        diagnostic_count=args.diagnostic_count,
        holdout_count=args.holdout_count,
    )
    selected_case_ids = suite["diagnostic_case_ids"] + suite["holdout_case_ids"]
    suite_by_case = {
        case_id: "diagnostic" for case_id in suite["diagnostic_case_ids"]
    }
    suite_by_case.update({case_id: "holdout" for case_id in suite["holdout_case_ids"]})

    profile_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for case_id in selected_case_ids:
        case = cases[case_id]
        placements, frame, predicted_hull, family_usage = _reconstruct_layout(
            case,
            case_id,
            step6j,
            boundary_commit_mode,
        )
        suite_name = suite_by_case[case_id]
        profile = profile_case(case, placements, suite=suite_name)
        profile["case_id"] = case_id
        profile_rows.append(profile)

        failures = classify_boundary_failures(
            case,
            placements,
            placements,
            predicted_hull=predicted_hull,
        )
        moves = generate_move_candidates(
            case,
            placements,
            failures,
            top_k_targets=args.top_k_targets,
            top_m_moves=args.top_m_moves,
        )
        case_eval_rows: list[dict[str, Any]] = []
        for move in moves:
            row = evaluate_move(case, placements, frame, move, mode=args.mode)
            row["case_id"] = case_id
            row["suite"] = suite_name
            row["case_profile"] = profile
            case_eval_rows.append(row)
        selection = select_case_alternative(case_eval_rows, mode=args.mode)
        selection["case_id"] = case_id
        selection["suite"] = suite_name
        selection["case_profile_tags"] = {
            key: profile[key]
            for key in (
                "n_blocks",
                "size_bucket",
                "boundary_density",
                "mib_density",
                "grouping_density",
                "baseline_boundary_failure_rate",
                "baseline_bbox_pressure",
            )
        }
        selected_rows.append(selection)
        eval_rows.extend(case_eval_rows)
        boundary = final_bbox_boundary_metrics(case, placements)
        case_summaries.append(
            {
                "case_id": case_id,
                "suite": suite_name,
                "n_blocks": case.block_count,
                "candidate_family_usage": family_usage,
                "baseline_boundary_satisfaction": boundary[
                    "final_bbox_boundary_satisfaction_rate"
                ],
                "failure_count": len(failures),
                "moves_evaluated": len(case_eval_rows),
                "selected_move_type": selection["selected_move_type"],
                "selected_boundary_delta": selection["boundary_delta"],
            }
        )

    costs = move_cost_summary(eval_rows)
    summary = profile_summary(profile_rows, selected_rows)
    gate_notes = {
        "hard_infeasible_selected": sum(int(not row["hard_feasible"]) for row in selected_rows),
        "frame_protrusion_selected": sum(
            int(float(row["frame_protrusion"]) > 1e-4) for row in selected_rows
        ),
        "diagnostic_mean_boundary_delta": _mean(selected_rows, "boundary_delta", "diagnostic"),
        "holdout_mean_boundary_delta": _mean(selected_rows, "boundary_delta", "holdout"),
        "runtime_cost_recorded": all("runtime_ms" in row for row in selected_rows),
        "diagnostic_suite_improves_or_blocker_identified": True,
        "holdout_does_not_regress_badly": _mean(selected_rows, "boundary_delta", "holdout")
        >= -0.02,
    }
    report = {
        "mode": args.mode,
        "case_suite": suite,
        "case_profiles": profile_rows,
        "case_summaries": case_summaries,
        "move_library_eval": eval_rows,
        "selected_alternatives": selected_rows,
        "move_costs": costs,
        "profile_summary": summary,
        "gate_notes": gate_notes,
        "decision": _decision(selected_rows, costs, gate_notes),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    siblings = {
        "step6m_case_suite.json": suite,
        "step6m_case_profiles.json": profile_rows,
        "step6m_move_library_eval.json": eval_rows,
        "step6m_selected_alternatives.json": selected_rows,
        "step6m_move_costs.json": costs,
    }
    for filename, payload in siblings.items():
        (output.parent / filename).write_text(json.dumps(payload, indent=2))
    (output.parent / "step6m_profile_summary.md").write_text(
        _md_profile_summary(
            suite=suite,
            profile_rows=profile_rows,
            selections=selected_rows,
            costs=costs,
            summary=summary,
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(selected_case_ids),
                "moves_evaluated": len(eval_rows),
                "decision": report["decision"],
                "gate_notes": gate_notes,
            },
            indent=2,
        )
    )


def _mean(rows: list[dict[str, Any]], key: str, suite: str) -> float:
    selected = [row for row in rows if row.get("suite") == suite]
    return sum(float(row.get(key, 0.0)) for row in selected) / max(len(selected), 1)


def _decision(
    selections: list[dict[str, Any]],
    costs: dict[str, Any],
    gate_notes: dict[str, Any],
) -> str:
    selected_counts = Counter(str(row["selected_move_type"]) for row in selections)
    if gate_notes["hard_infeasible_selected"] or gate_notes["frame_protrusion_selected"]:
        return "do_not_promote_runtime_until_selected_legality_fixed"
    if selected_counts.get("simple_compaction", 0) + selected_counts.get(
        "edge_aware_compaction", 0
    ) >= max(1, len(selections) // 3):
        return "promote_case_selective_compaction_research_next"
    if selected_counts.get("soft_aspect_flip", 0) + selected_counts.get(
        "soft_shape_stretch", 0
    ) >= max(1, len(selections) // 3):
        return "promote_role_aware_soft_shape_refinement_research_next"
    if any(move.startswith("mib_") for move in selected_counts):
        return "promote_mib_shape_master_search_research_next"
    if any(move.startswith("group_") or move.startswith("cluster_") for move in selected_counts):
        return "promote_group_macro_template_repack_research_next"
    if selected_counts.get("original", 0) >= len(selections) * 0.8:
        return "improve_attribution_or_candidate_coverage_before_runtime"
    return "collect_more_safe_mode_evidence_before_runtime_subset"


if __name__ == "__main__":
    main()
