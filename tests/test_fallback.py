from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.fallback import safe_shelf  # noqa: E402
from hcfp.verify import overlap_pairs, verify  # noqa: E402


@dataclass(frozen=True)
class Case:
    area: torch.Tensor
    target: torch.Tensor
    fixed_mask: torch.Tensor
    preplaced_mask: torch.Tensor
    scale: float = 1.0


def test_safe_shelf_copies_hard_targets_and_avoids_overlaps() -> None:
    target = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 9.0, 1.0],
            [2.0, -6.0, 3.0, 2.0],
        ]
    )
    case = Case(
        area=torch.tensor([16.0, 6.0, 9.0, 6.0]),
        target=target,
        fixed_mask=torch.tensor([False, True, False, False]),
        preplaced_mask=torch.tensor([True, False, False, True]),
    )

    placed = safe_shelf(case)
    result = verify(case, placed)

    assert result.feasible
    assert torch.equal(placed[case.preplaced_mask], target[case.preplaced_mask])
    assert torch.equal(placed[1, 2:4], target[1, 2:4])
    assert torch.allclose(placed[:, 2] * placed[:, 3], case.area)
    assert overlap_pairs(placed) == ()


def test_safe_shelf_is_deterministic_and_allows_negative_preplaced() -> None:
    target = torch.tensor([[-5.0, -3.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 4.0, 1.0]])
    case = Case(
        area=torch.tensor([4.0, 1.0, 4.0]),
        target=target,
        fixed_mask=torch.tensor([False, False, True]),
        preplaced_mask=torch.tensor([True, False, False]),
        scale=10.0,
    )

    first = safe_shelf(case)
    second = safe_shelf(case)

    assert torch.equal(first, second)
    assert torch.equal(first[0], target[0])
    assert verify(case, first).feasible


def test_safe_shelf_fails_closed_when_hard_anchors_overlap() -> None:
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    case = Case(
        area=torch.tensor([4.0, 4.0]),
        target=target,
        fixed_mask=torch.tensor([False, False]),
        preplaced_mask=torch.tensor([True, True]),
    )

    with pytest.raises(ValueError, match="preplaced anchors overlap"):
        safe_shelf(case)
