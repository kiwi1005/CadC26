from __future__ import annotations

from typing import Any

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.geometry.legality import positions_from_case_targets
from puzzleplace.ml.step7q_fresh_metric_replay import (
    affected_blocks_from_example,
    nearest_nonoverlap_slot,
    objective_aware_nonoverlap_slot,
    replay_expansion_row,
    summarize_fresh_replay,
    target_box_for_action,
)


def _case() -> FloorSetCase:
    return FloorSetCase(
        case_id="19",
        block_count=2,
        area_targets=torch.tensor([1.0, 1.0]),
        b2b_edges=torch.empty((0, 3)),
        p2b_edges=torch.tensor([[0.0, 0.0, 1.0]]),
        pins_pos=torch.tensor([[2.0, 0.5]]),
        constraints=torch.zeros((2, 5)),
        target_positions=torch.tensor([[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]]),
    )


def _eval(
    _case: FloorSetCase, positions: list[tuple[float, float, float, float]]
) -> dict[str, Any]:
    x0 = positions[0][0]
    return {
        "quality": {
            "cost": -x0,
            "HPWLgap": 0.0,
            "Areagap_bbox": 0.0,
            "Violationsrelative": 0.0,
            "feasible": True,
        }
    }


def _parent() -> dict[str, object]:
    return {
        "example_id": "ex0",
        "graph": {
            "edges": [
                {"src": "request:r0", "dst": "block:19:0", "type": "affected_block"},
                {"src": "request:r0", "dst": "block:19:1", "type": "soft_linked_block"},
            ]
        },
    }


def _request() -> dict[str, object]:
    return {
        "candidate_id": "cand0",
        "case_id": "19",
        "parent_example_id": "ex0",
        "source_subproblem_id": "sp0",
        "source_candidate_id": "src0",
        "expansion_score": 1.0,
        "operator_action": {
            "operator_family": "closure_translate_with_repair",
            "direction_bin": "hpwl_sink_toward_pins",
            "magnitude_bin": "small",
            "bbox_guard_mode": "nonexpand",
            "vector_guard_mode": "all_vector_nonregress",
        },
    }


def test_fresh_replay_realizes_finite_action_inside_replay_boundary() -> None:
    case = _case()
    baseline = positions_from_case_targets(case)
    before = _eval(case, baseline)
    row = replay_expansion_row(_request(), _parent(), case, baseline, before, _eval)

    assert affected_blocks_from_example(_parent()) == [0]
    assert row["fresh_metric_available"] is True
    assert row["hard_feasible_nonnoop"] is True
    assert row["strict_meaningful_winner"] is True
    assert row["target_box"] != baseline[0]
    assert (
        row["validation_label_policy"]
        == "labels used for replay/evaluation only, not request generation"
    )


def test_target_box_generation_does_not_require_request_coordinates() -> None:
    case = _case()
    baseline = positions_from_case_targets(case)
    target = target_box_for_action(
        case,
        baseline,
        0,
        {"direction_bin": "bbox_shrink_x", "magnitude_bin": "tiny"},
    )

    assert target is not None
    assert target[2:] == baseline[0][2:]
    assert target[0] > baseline[0][0]


def test_slot_search_resolves_overlapping_target_without_request_coordinates() -> None:
    positions = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
    target = (1.0, 0.0, 1.0, 1.0)

    slot = nearest_nonoverlap_slot(positions, 0, target)

    assert slot is not None
    assert slot != target
    assert slot[2:] == positions[0][2:]


def test_objective_aware_slot_prefers_nonregressing_feasible_slot() -> None:
    case = _case()
    positions = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
    before = _eval(case, positions)

    choice = objective_aware_nonoverlap_slot(
        case, positions, before, _eval, 0, (1.0, 0.0, 1.0, 1.0)
    )

    assert choice is not None
    assert choice["slot_candidate_count"] >= 1
    assert choice["slot_predicted_hard_feasible"] is True
    assert choice["slot_predicted_objective_vector"]["official_like_cost_delta"] <= 0.0


def test_replay_can_use_objective_aware_slot_adjustment() -> None:
    case = _case()
    baseline = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
    before = _eval(case, baseline)
    row = replay_expansion_row(
        _request(),
        _parent(),
        case,
        baseline,
        before,
        _eval,
        slot_aware=True,
        objective_aware_slot=True,
    )

    assert row["fresh_metric_available"] is True
    assert row["slot_adjustment"]["objective_aware_slot"] is True
    assert row["slot_adjustment"]["slot_candidate_count"] >= 1


def test_summary_keeps_phase4_closed_without_enough_strict_winners() -> None:
    rows = [
        {
            "case_id": str(case_id),
            "fresh_metric_available": True,
            "hard_feasible_nonnoop": True,
            "strict_meaningful_winner": False,
            "actual_all_vector_nonregressing": True,
            "soft_constraint_delta": 0.0,
            "bbox_area_delta": 0.0,
            "hpwl_delta": 0.0,
            "failure_attribution": "metric_tradeoff",
            "quality_gate_status": "metric_tradeoff_report_only",
            "replayed_signature": f"sig{idx}",
        }
        for idx, case_id in enumerate([19, 24, 25, 51, 76, 79, 91, 99] * 12)
    ]
    summary = summarize_fresh_replay(
        rows,
        {"decision": "parameter_expansion_deck_ready_for_fresh_replay"},
        request_count=96,
        expansion_deck_path=__import__("pathlib").Path("deck.jsonl"),
        replay_rows_path=__import__("pathlib").Path("rows.jsonl"),
        runtime_proxy_ms=1.0,
    )

    assert summary["risk_replay_gate_open"] is True
    assert summary["strict_meaningful_winner_count"] == 0
    assert summary["phase4_gate_open"] is False
    assert summary["decision"] == "fresh_replay_executed_strict_gate_closed"
