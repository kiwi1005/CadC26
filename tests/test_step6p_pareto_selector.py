from puzzleplace.research.pareto_selector import (
    applicability_filter_reason,
    build_pareto_candidates,
    dominates,
    pareto_front,
    select_representatives,
)


def _row(move_type, *, boundary=0.0, hpwl=0.0, bbox=0.0, roles=None, rejected=None):
    roles = roles or {
        "1": {
            "is_boundary": True,
            "is_mib": False,
            "is_grouping": False,
            "is_fixed": False,
            "is_preplaced": False,
        }
    }
    return {
        "case_id": 7,
        "move_type": move_type,
        "target_blocks": [1],
        "target_roles": roles,
        "rejected_reason": rejected or [],
        "boundary_delta": boundary,
        "hpwl_delta": hpwl,
        "bbox_delta": bbox,
        "soft_delta": 0.0,
        "grouping_delta": 0.0,
        "mib_delta": 0.0,
        "hard_feasible": True,
        "frame_protrusion": 0.0,
        "generation_time_ms": 1.0,
        "repair_time_ms": 0.0,
        "eval_time_ms": 1.0,
        "before_metrics": {"hpwl_proxy": 10.0, "bbox_area": 100.0},
        "case_profile": {"n_blocks": 10},
    }


def test_applicability_filter_is_role_aware_and_drops_no_effect():
    assert (
        applicability_filter_reason(_row("mib_master_aspect_flip"))
        == "reject_mib_move_without_mib_target"
    )
    assert (
        applicability_filter_reason(_row("mib_master_aspect_flip", roles={"1": {"is_mib": True}}))
        == "keep_mib_compatible_target"
    )
    assert (
        applicability_filter_reason(_row("simple_compaction", rejected=["no_effect"]))
        == "reject_no_effect_move"
    )


def test_pareto_front_keeps_original_when_move_is_worse_on_all_objectives():
    candidates, stats = build_pareto_candidates(
        [
            _row("simple_compaction", boundary=-0.1, hpwl=1.0, bbox=5.0),
        ]
    )
    front = pareto_front(candidates)
    assert stats["applicable_count"] == 1
    assert [row["move_type"] for row in front] == ["original"]


def test_pareto_front_keeps_tradeoff_and_representatives():
    candidates, _stats = build_pareto_candidates(
        [
            _row("boundary_edge_reassign", boundary=0.2, hpwl=1.0, bbox=0.0),
            _row("simple_compaction", boundary=0.1, hpwl=-1.0, bbox=-10.0),
        ]
    )
    front = pareto_front(candidates)
    move_types = {row["move_type"] for row in front}
    assert "boundary_edge_reassign" in move_types
    assert "simple_compaction" in move_types
    reps = select_representatives(front)
    assert set(reps) == {"min_disruption", "closest_to_ideal", "best_boundary", "best_hpwl"}


def test_dominance_requires_no_worse_all_and_better_one():
    a = {
        "objectives": {
            "boundary_violation_delta_norm": -0.1,
            "hpwl_delta_norm": 0.0,
            "bbox_delta_norm": 0.0,
            "disruption_cost_norm": 0.1,
        }
    }
    b = {
        "objectives": {
            "boundary_violation_delta_norm": 0.0,
            "hpwl_delta_norm": 0.1,
            "bbox_delta_norm": 0.0,
            "disruption_cost_norm": 0.1,
        }
    }
    assert dominates(a, b)
    assert not dominates(b, a)
