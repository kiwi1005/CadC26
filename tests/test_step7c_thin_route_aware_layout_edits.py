from __future__ import annotations

from puzzleplace.alternatives.route_aware_layout_edits import (
    apply_layout_edit,
    build_layout_edit_candidates,
    confusion_report,
    decision_for_step7c_thin,
    evaluate_layout_edit,
    evaluate_layout_edits,
    feasibility_report,
    pareto_report,
    synthetic_baseline_layout,
    synthetic_frame,
)


def _descriptor(case_id: int, family: str, locality: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "candidate_id": f"case{case_id:03d}:{family}",
        "family": family,
        "predicted_locality_class": locality,
        "predicted_repair_mode": {
            "local": "bounded_repair_pareto",
            "regional": "region_repair_or_planner",
            "macro": "macro_legalizer",
            "global": "global_route_not_local_selector",
        }[locality],
        "predicted_affected_blocks": 1,
        "predicted_affected_block_fraction": 0.025,
    }


def test_local_layout_edit_stays_local_and_feasible() -> None:
    candidate = build_layout_edit_candidates(
        [_descriptor(1, "vacancy_aware_local_insertion", "local")]
    )[0]

    row = evaluate_layout_edit(candidate)

    assert row["actual_locality_class"] == "local"
    assert row["hard_feasible_after_route_proxy"] is True
    assert row["failure_attribution"] == "none"


def test_macro_layout_edit_preserves_macro_route() -> None:
    candidate = build_layout_edit_candidates(
        [_descriptor(1, "mib_group_closure_macro", "macro")]
    )[0]

    row = evaluate_layout_edit(candidate)

    assert row["actual_locality_class"] == "macro"
    assert row["route_lane"] == "macro_report_lane"


def test_global_layout_edit_is_report_only_not_local_repair() -> None:
    candidate = build_layout_edit_candidates(
        [_descriptor(1, "legacy_step7g_global_move", "global")]
    )[0]

    row = evaluate_layout_edit(candidate)

    assert row["actual_locality_class"] == "global"
    assert row["report_only"] is True
    assert row["actual_repair_mode"] == "global_route_not_local_selector"
    assert row["failure_attribution"] in {
        "actual_edit_overlap",
        "actual_edit_frame_protrusion",
        "global_report_only_not_repaired",
    }


def test_step7c_thin_reports_stable_diverse_routes() -> None:
    descriptors = [
        _descriptor(1, "original_layout", "local"),
        _descriptor(1, "vacancy_aware_local_insertion", "local"),
        _descriptor(1, "adjacent_region_reassignment", "regional"),
        _descriptor(1, "mib_group_closure_macro", "macro"),
        _descriptor(1, "legacy_step7g_global_move", "global"),
    ]
    rows = evaluate_layout_edits(build_layout_edit_candidates(descriptors))
    feasibility = feasibility_report(rows, descriptor_candidate_count=len(descriptors))
    confusion = confusion_report(rows)
    pareto = pareto_report(rows)

    assert feasibility["invalid_local_attempt_rate"] == 0.0
    assert feasibility["actual_route_count_by_class"]["local"] >= 1
    assert feasibility["actual_route_count_by_class"]["regional"] >= 1
    assert feasibility["actual_route_count_by_class"]["macro"] >= 1
    assert feasibility["global_report_only_count"] == 1
    assert confusion["route_stability_descriptor_to_edit"] == 1.0
    assert pareto["actual_pareto_front_non_empty_count"] == 1
    assert decision_for_step7c_thin(feasibility, confusion, pareto) in {
        "promote_to_step7c_iterative_loop",
        "pivot_to_coarse_region_planner",
    }


def test_apply_layout_edit_keeps_regional_edit_inside_frame() -> None:
    frame = synthetic_frame(40)
    baseline = synthetic_baseline_layout(40)
    edited = apply_layout_edit("adjacent_region_reassignment", baseline, frame, 40)

    assert all(frame.contains_box(box) for box in edited.values())
