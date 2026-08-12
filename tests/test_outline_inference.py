from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.case import from_official  # noqa: E402
from hcfp.outline_inference import OutlineHypothesis, infer_outline_hypotheses  # noqa: E402


def _case_with_pins_and_anchors():
    return from_official(
        block_count=4,
        area_targets=[4.0, 6.0, 9.0, 16.0],
        b2b_connectivity=[],
        p2b_connectivity=[[0, 0, 1.0], [1, 1, 1.0], [2, 2, 1.0]],
        pins_pos=[[0.0, 0.0], [4.0, 1.0], [2.0, 3.0]],
        constraints=[
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        target_positions=[
            [0.0, 0.0, 2.0, 2.0],
            [5.0, 1.0, 3.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
    )


def test_inference_is_deterministic_and_auditable() -> None:
    case = _case_with_pins_and_anchors()

    first = infer_outline_hypotheses(case)
    second = infer_outline_hypotheses(case)

    assert 4 <= len(first) <= 8
    assert [hypothesis.to_dict() for hypothesis in first] == [
        hypothesis.to_dict() for hypothesis in second
    ]
    assert all(isinstance(hypothesis, OutlineHypothesis) for hypothesis in first)
    assert len({hypothesis.hypothesis_id for hypothesis in first}) == len(first)
    assert all(hypothesis.source and hypothesis.provenance for hypothesis in first)
    assert all("pin_residual" in hypothesis.scores for hypothesis in first)
    assert all("area_prior_residual" in hypothesis.scores for hypothesis in first)
    assert all("anchor_residual" in hypothesis.scores for hypothesis in first)
    assert all("fixed_shape_residual" in hypothesis.scores for hypothesis in first)
    assert all(len(hypothesis.pin_side_assignment) == case.pins.shape[0] for hypothesis in first)


def test_utilization_aspect_variants_and_hard_anchor_containment() -> None:
    case = _case_with_pins_and_anchors()
    hypotheses = infer_outline_hypotheses(case, max_hypotheses=8)
    preplaced = case.preplaced_mask

    assert all(0.95 - 1.0e-8 <= hypothesis.utilization <= 1.0 + 1.0e-8 for hypothesis in hypotheses)
    assert len(
        {
            round(
                (hypothesis.x_right - hypothesis.x_left)
                / (hypothesis.y_top - hypothesis.y_bottom),
                5,
            )
            for hypothesis in hypotheses
        }
    ) >= 2
    assert all(hypothesis.anchor_coverage == pytest.approx(1.0) for hypothesis in hypotheses)
    assert all(hypothesis.anchor_residual == pytest.approx(0.0) for hypothesis in hypotheses)
    for hypothesis in hypotheses:
        selected = case.target[preplaced]
        assert bool(
            (
                (selected[:, 0] >= hypothesis.x_left - 1.0e-8)
                & (selected[:, 1] >= hypothesis.y_bottom - 1.0e-8)
                & (selected[:, 0] + selected[:, 2] <= hypothesis.x_right + 1.0e-8)
                & (selected[:, 1] + selected[:, 3] <= hypothesis.y_top + 1.0e-8)
            ).all()
        )
    assert all(0.0 <= hypothesis.pin_side_coverage <= 1.0 for hypothesis in hypotheses)
    assert all(0.0 <= hypothesis.side_coverage <= 1.0 for hypothesis in hypotheses)


def test_requested_beam_size() -> None:
    case = from_official(
        block_count=3,
        area_targets=[1.0, 1.0, 1.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]],
        constraints=[[0, 0, 0, 0, 0]] * 3,
    )

    direct = infer_outline_hypotheses(case, max_hypotheses=4)
    assert len(direct) == 4


def test_incompatible_anchor_span_is_rejected() -> None:
    case = from_official(
        block_count=1,
        area_targets=[1.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[1, 0, 0, 0, 0]],
        target_positions=[[0.0, 0.0, 100.0, 1.0]],
    )

    assert infer_outline_hypotheses(case) == ()


def test_pin_perimeter_family_is_retained_for_coordinate_recovery() -> None:
    case = from_official(
        block_count=3,
        area_targets=[1.0, 1.0, 1.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
        constraints=[[0, 0, 0, 0, 0]] * 3,
    )

    hypotheses = infer_outline_hypotheses(case, max_hypotheses=4)
    perimeter = [hypothesis for hypothesis in hypotheses if hypothesis.source == "pin_perimeter"]

    assert len(perimeter) == 1
    assert "pin_bounds_exact" in perimeter[0].provenance
    assert perimeter[0].utilization < 0.95
    assert perimeter[0].pin_side_coverage == pytest.approx(1.0)


def test_fixed_shape_is_fit_without_treating_target_xy_as_an_anchor() -> None:
    case = from_official(
        block_count=1,
        area_targets=[4.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[1, 0, 0, 0, 0]],
        target_positions=[[100.0, 100.0, 2.0, 2.0]],
    )

    hypotheses = infer_outline_hypotheses(case, max_hypotheses=4)

    assert len(hypotheses) == 4
    assert all(hypothesis.anchor_span == (0.0, 0.0) for hypothesis in hypotheses)
    assert all(hypothesis.anchor_coverage == pytest.approx(1.0) for hypothesis in hypotheses)
    assert all(hypothesis.x_right < 10.0 for hypothesis in hypotheses)


def test_invalid_beam_size_fails_fast() -> None:
    case = from_official(
        block_count=1,
        area_targets=[1.0],
        b2b_connectivity=[],
        p2b_connectivity=[],
        pins_pos=[],
        constraints=[[0, 0, 0, 0, 0]],
    )

    with pytest.raises(ValueError, match="max_hypotheses"):
        infer_outline_hypotheses(case, max_hypotheses=3)
