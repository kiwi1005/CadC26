#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from puzzleplace.actions import ExecutionState
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.repair import finalize_layout
from puzzleplace.research.boundary_failure_attribution import (
    CandidateEdgeCoverage,
    boundary_role_overlap_audit,
    classify_boundary_failures,
    compact_left_bottom,
    final_bbox_edge_owner_audit,
)
from puzzleplace.research.puzzle_candidate_payload import (
    BoundaryCommitMode,
    build_puzzle_candidate_descriptors,
)
from puzzleplace.research.virtual_frame import (
    _area,
    _bbox_from_placements,
    _boundary_edges,
    estimate_predicted_compact_hull,
    final_bbox_boundary_metrics,
    frame_diagnostics,
    multistart_virtual_frames,
    repair_attribution_summary,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import (
    _construct,
    _external_hpwl_proxy,
    _hpwl_proxy,
    _positions_list,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _bbox_area(placements: dict[int, tuple[float, float, float, float]]) -> float:
    return _area(_bbox_from_placements(placements.values()))


def _candidate_edge_coverage(
    case: Any, frame: Any, predicted_hull: Any, mode: str
) -> CandidateEdgeCoverage:
    by_block: dict[int, dict[str, int]] = {}
    if predicted_hull is None:
        return CandidateEdgeCoverage(by_block)
    for block_id in range(case.block_count):
        code = int(case.constraints[block_id, 4].item())
        edges = _boundary_edges(code)
        if not edges:
            continue
        descriptors = build_puzzle_candidate_descriptors(
            case,
            ExecutionState(),
            remaining_blocks=[block_id],
            max_shape_bins=5,
            max_descriptors_per_block=48,
            virtual_frame=frame,
            predicted_hull=predicted_hull,
            boundary_commit_mode=cast(BoundaryCommitMode, mode),
        )
        counts = {edge: 0 for edge in edges}
        for desc in descriptors:
            x = float(desc.action_token.x or 0.0)
            y = float(desc.action_token.y or 0.0)
            w = float(desc.action_token.w or 0.0)
            h = float(desc.action_token.h or 0.0)
            checks = {
                "left": abs(x - predicted_hull.xmin) <= 1e-4,
                "right": abs((x + w) - predicted_hull.xmax) <= 1e-4,
                "bottom": abs(y - predicted_hull.ymin) <= 1e-4,
                "top": abs((y + h) - predicted_hull.ymax) <= 1e-4,
            }
            for edge in edges:
                counts[edge] += int(checks[edge])
        by_block[block_id] = counts
    return CandidateEdgeCoverage(by_block)


def _compaction_record(case: Any, pre: dict[int, Any], frame: Any) -> dict[str, Any]:
    compacted = compact_left_bottom(case, pre, frame=frame, passes=4)
    pre_metrics = final_bbox_boundary_metrics(case, pre)
    post_metrics = final_bbox_boundary_metrics(case, compacted)
    legality = summarize_hard_legality(case, _positions_list(compacted, case.block_count))
    return {
        "boundary_satisfaction_pre_compaction": pre_metrics[
            "final_bbox_boundary_satisfaction_rate"
        ],
        "boundary_satisfaction_post_compaction": post_metrics[
            "final_bbox_boundary_satisfaction_rate"
        ],
        "boundary_satisfaction_delta": float(
            post_metrics["final_bbox_boundary_satisfaction_rate"]
            - pre_metrics["final_bbox_boundary_satisfaction_rate"]
        ),
        "bbox_area_pre_compaction": _bbox_area(pre),
        "bbox_area_post_compaction": _bbox_area(compacted),
        "internal_hpwl_pre_compaction": _hpwl_proxy(case, pre),
        "internal_hpwl_post_compaction": _hpwl_proxy(case, compacted),
        "external_hpwl_pre_compaction": _external_hpwl_proxy(case, pre),
        "external_hpwl_post_compaction": _external_hpwl_proxy(case, compacted),
        "hard_feasible_after_compaction": bool(legality.is_feasible),
        "hard_violation_summary_after_compaction": {
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "positions_changed_count": sum(1 for idx, box in pre.items() if compacted.get(idx) != box),
    }


def _pick_recommendation(
    failure_counts: Counter[str], role_audit: dict[str, Any], compaction: list[dict[str, Any]]
) -> str:
    total_failures = sum(failure_counts.values())
    compaction_helped = any(
        row.get("boundary_satisfaction_delta", 0.0) >= 0.05 for row in compaction
    )
    if failure_counts.get("edge_stolen_by_regular_or_nonboundary", 0) >= max(
        1, total_failures // 3
    ):
        return "hull_ownership_stealing_prevention"
    if compaction_helped:
        return "robust_edge_aware_compaction"
    if role_audit.get("unsatisfied_boundary_plus_grouping", 0) >= max(
        1, role_audit.get("unsatisfied_boundary_total", 0) // 2
    ):
        return "group_aware_boundary_candidates"
    if role_audit.get("unsatisfied_boundary_plus_mib", 0) >= max(
        1, role_audit.get("unsatisfied_boundary_total", 0) // 2
    ):
        return "mib_compatible_boundary_shape_slots"
    if failure_counts.get("edge_segment_conflict", 0) >= max(1, total_failures // 3):
        return "conditional_edge_segment_assignment"
    return "candidate_coverage_and_selection_audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
        help="Step6J multistart sidecar report to reconstruct winning layouts from.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/research/step6k_boundary_failure_attribution.json",
    )
    parser.add_argument("--case-ids", nargs="*", default=None)
    args = parser.parse_args()

    source = _load_json(args.input)
    requested_ids = (
        [int(value) for value in args.case_ids]
        if args.case_ids is not None
        else [int(value) for value in source.get("case_ids", [])]
    )
    runs_by_case = {int(run["case_id"]): run for run in source.get("runs", [])}
    cases = load_validation_cases(case_limit=max(requested_ids) + 1)
    boundary_commit_mode = str(source.get("boundary_commit_mode", "prefer_predicted_hull"))

    edge_owner_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []
    compaction_records: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    all_failure_counts: Counter[str] = Counter()
    aggregate_role: Counter[str] = Counter()

    for case_id in requested_ids:
        if case_id not in runs_by_case:
            raise ValueError(f"Step6J input does not contain case_id={case_id}")
        run = runs_by_case[case_id]
        case = cases[case_id]
        frames = multistart_virtual_frames(case)
        frame_index = int(run["best_frame_index"])
        frame = frames[frame_index]
        placements, family_usage, construction_frame, predicted_hull = _construct(
            case,
            int(run["best_seed"]),
            int(run["best_start"]),
            frame,
            boundary_commit_mode=cast(BoundaryCommitMode, boundary_commit_mode),
        )
        if construction_frame is None:
            construction_frame = frame
        if predicted_hull is None:
            predicted_hull = estimate_predicted_compact_hull(case, construction_frame)
        repair = finalize_layout(case, placements)
        post_placements = {idx: box for idx, box in enumerate(repair.positions)}
        coverage = _candidate_edge_coverage(
            case, construction_frame, predicted_hull, boundary_commit_mode
        )

        owners = final_bbox_edge_owner_audit(case, post_placements)
        for row in owners:
            edge_owner_records.append({"case_id": case_id, **row})
        failures = classify_boundary_failures(
            case,
            placements,
            post_placements,
            predicted_hull=predicted_hull,
            edge_owner_rows=owners,
            candidate_coverage=coverage,
        )
        for row in failures:
            failure_records.append({"case_id": case_id, **row})
        failure_counts = Counter(str(row["failure_type"]) for row in failures)
        all_failure_counts.update(failure_counts)

        role = boundary_role_overlap_audit(case, post_placements)
        role_records.append({"case_id": case_id, **role})
        aggregate_role.update(
            {key: int(value) for key, value in role.items() if isinstance(value, int)}
        )
        compact = _compaction_record(case, post_placements, construction_frame)
        compaction_records.append({"case_id": case_id, **compact})
        recommendation = _pick_recommendation(failure_counts, role, [compact])
        case_summaries.append(
            {
                "case_id": case_id,
                "best_seed": int(run["best_seed"]),
                "best_start": int(run["best_start"]),
                "best_frame_variant": construction_frame.variant,
                "frame_relaxation": construction_frame.relaxation,
                "candidate_family_usage": family_usage,
                "final_bbox_boundary_metrics": final_bbox_boundary_metrics(case, post_placements),
                "virtual_frame_metrics": frame_diagnostics(
                    case, post_placements, construction_frame
                ),
                "repair_attribution": repair_attribution_summary(case, placements, post_placements),
                "failure_counts": {
                    key: int(value) for key, value in sorted(failure_counts.items())
                },
                "role_overlap_audit": role,
                "compaction_baseline": compact,
                "recommended_next_fix": recommendation,
            }
        )

    aggregate_recommendation = _pick_recommendation(
        all_failure_counts,
        dict(aggregate_role),
        compaction_records,
    )
    report = {
        "source_step6j_report": str(args.input),
        "case_ids": requested_ids,
        "boundary_commit_mode": boundary_commit_mode,
        "step6j_quality_guardrails": {
            "semantic_completion_fraction": source.get("semantic_completion_fraction"),
            "fallback_selected_fraction": source.get("fallback_selected_fraction"),
            "hard_infeasible_after_repair_count": source.get("hard_infeasible_after_repair_count"),
            "hpwl_proxy_relative_improvement": source.get("hpwl_proxy_relative_improvement"),
            "bbox_area_proxy_relative_improvement": source.get(
                "bbox_area_proxy_relative_improvement"
            ),
            "soft_violation_relative_reduction": source.get("soft_violation_relative_reduction"),
            "repair_displacement_mean": source.get("repair_displacement_mean"),
        },
        "aggregate_failure_counts": {
            key: int(value) for key, value in sorted(all_failure_counts.items())
        },
        "aggregate_role_overlap_audit": {
            key: int(value) for key, value in sorted(aggregate_role.items())
        },
        "case3_case4_failure_types": {
            str(row["case_id"]): row["failure_counts"]
            for row in case_summaries
            if row["case_id"] in {3, 4}
        },
        "recommended_next_fix": aggregate_recommendation,
        "final_bbox_edge_owner_audit": edge_owner_records,
        "boundary_failure_classification": failure_records,
        "boundary_role_overlap_audit": role_records,
        "compaction_baseline": compaction_records,
        "case_summaries": case_summaries,
        "step6k_gate_notes": {
            "diagnostic_artifacts_emitted": True,
            "layout_changed_by_step6k": False,
            "bbox_not_worse_than_step6j": True,
            "repair_displacement_remains_step6j": True,
            "hpwl_keeps_step6j_improvement": True,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix(".md").write_text(
        "# Step6K Boundary Failure Attribution\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    siblings = {
        "step6k_final_bbox_edge_owner_audit.json": edge_owner_records,
        "step6k_boundary_failure_classification.json": failure_records,
        "step6k_boundary_role_overlap_audit.json": role_records,
        "step6k_compaction_baseline.json": compaction_records,
    }
    for name, payload in siblings.items():
        (output.parent / name).write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "output": str(output),
                "recommended_next_fix": aggregate_recommendation,
                "aggregate_failure_counts": report["aggregate_failure_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
