from puzzleplace.research.selector_replay import (
    compare_selection_sets,
    select_guarded_case_alternative,
    spatial_balance_worsening,
    step6o_guard_reasons,
)


def _row(move_type="simple_compaction", *, boundary=0.1, hpwl=1.0, accepted=True):
    return {
        "case_id": 1,
        "move_type": move_type,
        "target_blocks": [1],
        "accepted": accepted,
        "guard_rejected": False,
        "boundary_delta": boundary,
        "bbox_delta": -1.0,
        "hpwl_delta": hpwl,
        "soft_delta": 0.0,
        "grouping_delta": 0.0,
        "mib_delta": 0.0,
        "hard_feasible": True,
        "frame_protrusion": 0.0,
        "generation_time_ms": 1.0,
        "repair_time_ms": 0.0,
        "eval_time_ms": 1.0,
    }


def test_step6o_guards_only_simple_compaction():
    path_delta = {"left_right_balance_delta": 0.2, "top_bottom_balance_delta": 0.0}
    assert step6o_guard_reasons(_row(), path_delta) == [
        "spatial_balance_worsening_gt_0.10"
    ]
    assert step6o_guard_reasons(_row("boundary_edge_reassign"), path_delta) == []


def test_step6o_hpwl_per_boundary_gain_guard():
    reasons = step6o_guard_reasons(
        _row(boundary=0.05, hpwl=3.0),
        {"left_right_balance_delta": 0.0, "top_bottom_balance_delta": 0.0},
    )
    assert reasons == ["hpwl_regression_per_boundary_gain_gt_40"]


def test_select_guarded_case_alternative_falls_back_when_only_guarded_move():
    row = {**_row(boundary=0.2, hpwl=1.0), "guard_rejected": True}
    selected = select_guarded_case_alternative([row])
    assert selected["selected_move_type"] == "original"
    assert selected["selection_reason"] == "step6o_no_guarded_accepted_move"


def test_select_guarded_case_alternative_uses_next_safe_move():
    rejected = {**_row(boundary=0.2, hpwl=1.0), "guard_rejected": True}
    kept = _row("boundary_edge_reassign", boundary=0.1, hpwl=0.0)
    selected = select_guarded_case_alternative([rejected, kept])
    assert selected["selected_move_type"] == "boundary_edge_reassign"


def _selection(case_id, move_type, *, boundary=0.0, hpwl=0.0):
    return {
        "case_id": case_id,
        "selected_move_type": move_type,
        "boundary_delta": boundary,
        "bbox_delta": 0.0,
        "hpwl_delta": hpwl,
        "soft_delta": 0.0,
    }


def test_compare_selection_sets_counts_suspicious_and_fallbacks():
    before = [
        _selection(8, "simple_compaction", boundary=0.1, hpwl=1.0),
        _selection(9, "original"),
    ]
    after = [
        _selection(8, "original"),
        _selection(9, "original"),
    ]
    comparison = compare_selection_sets(before, after, suspicious_case_ids={8})
    assert comparison["suspicious_selected_count_before"] == 1
    assert comparison["suspicious_selected_count_after"] == 0
    assert comparison["original_fallback_count_after"] == 2


def test_spatial_balance_worsening_takes_positive_axis_max():
    assert spatial_balance_worsening(
        {"left_right_balance_delta": -0.4, "top_bottom_balance_delta": 0.12}
    ) == 0.12
