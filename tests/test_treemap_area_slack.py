from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hcfp.case import from_official
from hcfp.treemap import exact_treemap_candidates
from hcfp.verify import verify_feasible


def _hypothesis(bounds=(-1.0, -1.0, 1.0, 1.5)) -> SimpleNamespace:
    return SimpleNamespace(
        bounds=bounds,
        confidence=1.0,
        hypothesis_id="area-slack-test",
    )


def _ordinary_case():
    return from_official(
        3,
        [1.0, 2.0, 3.0],
        [],
        [],
        [],
        [[0, 0, 0, 0, 0]] * 3,
    )


def test_default_and_explicit_exact_area_slack_are_identical() -> None:
    case = _ordinary_case()
    reference = torch.tensor(
        [[-0.5, -0.5, 0.3, 0.3], [-0.2, -0.2, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]]
    )

    default, default_records = exact_treemap_candidates(
        case, reference, (_hypothesis(),), count=1
    )
    explicit, explicit_records = exact_treemap_candidates(
        case, reference, (_hypothesis(),), count=1, area_slack=1.0
    )

    assert torch.equal(default, explicit)
    assert default_records[0]["area_slack"] == 1.0
    assert explicit_records[0]["area_slack"] == 1.0


def test_area_slack_scales_only_ordinary_blocks_and_stays_feasible() -> None:
    case = from_official(
        3,
        [4.0, 3.0, 3.0],
        [],
        [],
        [],
        [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[-1.0, -1.0, 2.0, 2.0], [-1.0] * 4, [-1.0] * 4],
    )
    reference = torch.tensor(
        [[-1.0, -1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )

    candidates, records = exact_treemap_candidates(
        case,
        reference,
        (_hypothesis(bounds=(-1.0, -1.0, 1.0, 1.625)),),
        count=1,
        area_slack=0.995,
    )
    candidate = candidates[0]
    actual_area = candidate[:, 2] * candidate[:, 3]
    ordinary = ~case.fixed_mask & ~case.preplaced_mask

    assert records[0]["area_slack"] == pytest.approx(0.995)
    assert torch.equal(candidate[case.preplaced_mask], case.target[case.preplaced_mask])
    assert torch.allclose(
        actual_area[case.preplaced_mask], case.area[case.preplaced_mask]
    )
    assert torch.allclose(
        actual_area[ordinary], case.area[ordinary] * 0.995, atol=2.0e-5, rtol=0.0
    )
    assert verify_feasible(case, candidate)


def test_area_slack_rejects_out_of_range_values() -> None:
    case = _ordinary_case()
    reference = torch.ones((3, 4))
    with pytest.raises(ValueError, match="area_slack"):
        exact_treemap_candidates(
            case, reference, (_hypothesis(),), count=1, area_slack=0.0
        )
    with pytest.raises(ValueError, match="area_slack"):
        exact_treemap_candidates(
            case, reference, (_hypothesis(),), count=1, area_slack=1.01
        )
