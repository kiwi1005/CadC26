#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.repair import finalize_layout
from puzzleplace.research.boundary_failure_attribution import (
    compact_left_bottom,
    final_bbox_edge_owner_audit,
)
from puzzleplace.research.compaction_alternatives import edge_aware_compaction
from puzzleplace.research.hull_stabilization import (
    attribution_cooccurrence,
    hull_drift_metrics,
    hull_stealing_guard_audit,
    select_alternative,
)
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.shape_group_probe import shape_group_intervention_probes
from puzzleplace.research.virtual_frame import (
    _area,
    _bbox_from_placements,
    estimate_predicted_compact_hull,
    final_bbox_boundary_metrics,
    frame_diagnostics,
    multistart_virtual_frames,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import (
    _construct,
    _external_hpwl_proxy,
    _hpwl_proxy,
    _positions_list,
    _soft_boundary_violations,
)

Placement = tuple[float, float, float, float]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _bbox_area(placements: dict[int, Placement]) -> float:
    return _area(_bbox_from_placements(placements.values()))


def _layout_from_json(value: dict[str, Any] | None) -> dict[int, Placement] | None:
    if value is None:
        return None
    return {
        int(idx): tuple(float(part) for part in box)  # type: ignore[misc]
        for idx, box in value.items()
    }


def _evaluate_alternative(
    case: Any,
    placements: dict[int, Placement],
    frame: Any,
    alternative_type: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legality = summarize_hard_legality(case, _positions_list(placements, case.block_count))
    boundary = final_bbox_boundary_metrics(case, placements)
    frame_metrics = frame_diagnostics(case, placements, frame)
    return {
        "alternative_type": alternative_type,
        "bbox_area": _bbox_area(placements),
        "hpwl_proxy": _hpwl_proxy(case, placements),
        "external_hpwl_proxy": _external_hpwl_proxy(case, placements),
        "soft_boundary_violations": _soft_boundary_violations(case, placements),
        "boundary_satisfaction_rate": boundary["final_bbox_boundary_satisfaction_rate"],
        "boundary_unsatisfied_blocks": boundary["final_bbox_boundary_unsatisfied_blocks"],
        "hard_feasible": bool(legality.is_feasible),
        "hard_violation_summary": {
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "frame_num_violations": frame_metrics["num_frame_violations"],
        "frame_max_protrusion_distance": frame_metrics["max_protrusion_distance"],
        "frame_outside_area_ratio": frame_metrics["outside_frame_area_ratio"],
        "role_conflicts_resolved": int((metadata or {}).get("role_conflicts_resolved", 0)),
        "role_conflicts_created": int((metadata or {}).get("role_conflicts_created", 0)),
        "metadata": metadata or {},
    }


def _case_failure_rows(step6k: dict[str, Any], case_id: int) -> list[dict[str, Any]]:
    return [
        row
        for row in step6k.get("boundary_failure_classification", [])
        if int(row.get("case_id", -1)) == case_id
    ]


def _recommend_next(report: dict[str, Any]) -> str:
    selections = report["alternative_selection"]
    selected_types = [row["selected_alternative_type"] for row in selections]
    bad_case_gains = [
        row["boundary_gain"] for row in selections if int(row["case_id"]) in {3, 4}
    ]
    guard_rows = report["hull_stealing_guard_audit"]
    shape_summary = report["shape_group_probe"]["aggregate_summary"]
    drift_errors = sum(
        1
        for row in report["hull_drift_metrics"]
        if row["drift_summary"].get("looks_like_estimator_error")
    )
    if any(gain > 0 for gain in bad_case_gains) and any(
        kind in selected_types for kind in {"simple_compaction", "edge_aware_compaction"}
    ):
        if any(row.get("guard_applied") for row in guard_rows):
            return "case_selective_compaction_then_selective_hull_stealing_guard"
        return "case_selective_compaction"
    if any(row.get("guard_applied") for row in guard_rows):
        return "selective_hull_stealing_guard"
    if shape_summary.get("accepted_shape_probe_count", 0) > 0:
        return "role_aware_soft_shape_refinement"
    if drift_errors >= max(1, len(report["hull_drift_metrics"]) // 2):
        return "improve_predicted_hull_estimator"
    return "edge_segment_assignment_or_group_macro_probe_next"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step6j",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
    )
    parser.add_argument(
        "--step6k",
        default="artifacts/research/step6k_boundary_failure_attribution.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/research/step6l_selective_hull_stabilization.json",
    )
    parser.add_argument("--case-ids", nargs="*", default=None)
    args = parser.parse_args()

    step6j = _load_json(args.step6j)
    step6k = _load_json(args.step6k)
    requested_ids = (
        [int(value) for value in args.case_ids]
        if args.case_ids is not None
        else [int(value) for value in step6k.get("case_ids", step6j.get("case_ids", []))]
    )
    cases = load_validation_cases(case_limit=max(requested_ids) + 1)
    runs_by_case = {int(row["case_id"]): row for row in step6j.get("runs", [])}
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(
            step6j.get(
                "boundary_commit_mode",
                step6k.get("boundary_commit_mode", "prefer_predicted_hull"),
            )
        ),
    )

    hull_drift_records: list[dict[str, Any]] = []
    guard_records: list[dict[str, Any]] = []
    compaction_records: list[dict[str, Any]] = []
    shape_case_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    total_shape_probe_count = 0
    total_accepted_shape_probe_count = 0

    for case_id in requested_ids:
        if case_id not in runs_by_case:
            raise ValueError(f"Step6J input does not contain case_id={case_id}")
        case = cases[case_id]
        run = runs_by_case[case_id]
        frames = multistart_virtual_frames(case)
        frame = frames[int(run["best_frame_index"])]
        placements, family_usage, construction_frame, predicted_hull = _construct(
            case,
            int(run["best_seed"]),
            int(run["best_start"]),
            frame,
            boundary_commit_mode=boundary_commit_mode,
        )
        if construction_frame is None:
            construction_frame = frame
        if predicted_hull is None:
            predicted_hull = estimate_predicted_compact_hull(case, construction_frame)
        repair = finalize_layout(case, placements)
        post_placements = {idx: box for idx, box in enumerate(repair.positions)}
        owners = final_bbox_edge_owner_audit(case, post_placements)
        failures = _case_failure_rows(step6k, case_id)

        drift = hull_drift_metrics(case, post_placements, predicted_hull)
        hull_drift_records.append({"case_id": case_id, **drift})

        case_guard = hull_stealing_guard_audit(case, owners, failures)
        guard_records.extend({"case_id": case_id, **row} for row in case_guard)

        simple = compact_left_bottom(case, post_placements, frame=construction_frame, passes=4)
        edge_result = edge_aware_compaction(
            case,
            post_placements,
            frame=construction_frame,
            target_hull=predicted_hull,
            passes=4,
        )
        probes = shape_group_intervention_probes(
            case,
            post_placements,
            failures,
            owners,
            frame=construction_frame,
        )
        total_shape_probe_count += int(probes["summary"]["shape_probe_count"])
        total_accepted_shape_probe_count += int(probes["summary"]["accepted_shape_probe_count"])

        alternatives = [
            _evaluate_alternative(case, post_placements, construction_frame, "original"),
            _evaluate_alternative(
                case,
                simple,
                construction_frame,
                "simple_compaction",
                metadata={
                    "positions_changed_count": sum(
                        simple[idx] != post_placements[idx] for idx in simple
                    )
                },
            ),
            _evaluate_alternative(
                case,
                edge_result.placements,
                construction_frame,
                "edge_aware_compaction",
                metadata={
                    "locked_satisfied_boundary_owner_ids": (
                        edge_result.locked_satisfied_boundary_owner_ids
                    ),
                    "promoted_boundary_block_ids": edge_result.promoted_boundary_block_ids,
                    "demoted_regular_hull_owner_ids": edge_result.demoted_regular_hull_owner_ids,
                    "rejected_promotion_block_ids": edge_result.rejected_promotion_block_ids,
                    "role_conflicts_resolved": len(edge_result.demoted_regular_hull_owner_ids),
                },
            ),
        ]
        shape_layout = _layout_from_json(probes.get("best_shape_probe_layout"))
        if shape_layout is not None:
            alternatives.append(
                _evaluate_alternative(
                    case,
                    shape_layout,
                    construction_frame,
                    "soft_shape_probe_best",
                    metadata={"source": "best accepted Step6L-E soft-shape probe"},
                )
            )
        selected = select_alternative(alternatives)
        selection_records.append({"case_id": case_id, **selected})
        for alt in alternatives:
            original = alternatives[0]
            compaction_records.append(
                {
                    "case_id": case_id,
                    **alt,
                    "boundary_gain_vs_original": float(
                        alt["boundary_satisfaction_rate"]
                        - original["boundary_satisfaction_rate"]
                    ),
                    "bbox_delta_vs_original": float(alt["bbox_area"] - original["bbox_area"]),
                    "hpwl_delta_vs_original": float(alt["hpwl_proxy"] - original["hpwl_proxy"]),
                }
            )

        shape_case_records.append(
            {
                "case_id": case_id,
                "shape_probe_records": probes["shape_probe_records"],
                "mib_probe_records": probes["mib_probe_records"],
                "group_probe_records": probes["group_probe_records"],
                "summary": probes["summary"],
            }
        )
        case_summaries.append(
            {
                "case_id": case_id,
                "best_seed": int(run["best_seed"]),
                "best_start": int(run["best_start"]),
                "best_frame_variant": construction_frame.variant,
                "candidate_family_usage": family_usage,
                "original_boundary_satisfaction": alternatives[0]["boundary_satisfaction_rate"],
                "selected_alternative_type": selected["selected_alternative_type"],
                "selected_boundary_gain": selected["boundary_gain"],
                "hull_drift_summary": drift["drift_summary"],
                "guard_candidates": len(case_guard),
                "shape_probe_summary": probes["summary"],
            }
        )

    cooccurrence = attribution_cooccurrence(step6k.get("boundary_failure_classification", []))
    shape_group_report = {
        "case_records": shape_case_records,
        "aggregate_summary": {
            "shape_probe_count": total_shape_probe_count,
            "accepted_shape_probe_count": total_accepted_shape_probe_count,
            "mib_probe_count": sum(len(row["mib_probe_records"]) for row in shape_case_records),
            "group_probe_count": sum(len(row["group_probe_records"]) for row in shape_case_records),
        },
    }
    report = {
        "source_step6j_report": str(args.step6j),
        "source_step6k_report": str(args.step6k),
        "case_ids": requested_ids,
        "boundary_commit_mode": boundary_commit_mode,
        "step6j_quality_guardrails": step6k.get("step6j_quality_guardrails", {}),
        "hull_drift_metrics": hull_drift_records,
        "attribution_cooccurrence": cooccurrence,
        "hull_stealing_guard_audit": guard_records,
        "compaction_alternatives": compaction_records,
        "shape_group_probe": shape_group_report,
        "alternative_selection": selection_records,
        "case_summaries": case_summaries,
        "recommended_next_fix": "",
        "step6l_gate_notes": {
            "layout_changed_by_step6l_runtime": False,
            "diagnostic_sidecar_only": True,
            "semantic_completion_fraction": step6k.get("step6j_quality_guardrails", {}).get(
                "semantic_completion_fraction"
            ),
            "fallback_selected_fraction": step6k.get("step6j_quality_guardrails", {}).get(
                "fallback_selected_fraction"
            ),
            "hard_infeasible_after_repair_count": step6k.get("step6j_quality_guardrails", {}).get(
                "hard_infeasible_after_repair_count"
            ),
        },
    }
    report["recommended_next_fix"] = _recommend_next(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix(".md").write_text(
        "# Step6L Selective Hull Stabilization\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    siblings = {
        "step6l_hull_drift_metrics.json": hull_drift_records,
        "step6l_attribution_cooccurrence.json": cooccurrence,
        "step6l_hull_stealing_guard_audit.json": guard_records,
        "step6l_compaction_alternatives.json": compaction_records,
        "step6l_shape_group_probe.json": shape_group_report,
        "step6l_alternative_selection.json": selection_records,
    }
    for name, payload in siblings.items():
        (output.parent / name).write_text(json.dumps(payload, indent=2))

    print(
        json.dumps(
            {
                "output": str(output),
                "recommended_next_fix": report["recommended_next_fix"],
                "selected_alternatives": {
                    str(row["case_id"]): row["selected_alternative_type"]
                    for row in selection_records
                },
                "case_boundary_gains": {
                    str(row["case_id"]): row["boundary_gain"] for row in selection_records
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
