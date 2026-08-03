from __future__ import annotations

import pytest
import torch

from hcfp.listwise import listmle_loss


def test_listmle_cost_ordering_prefers_low_cost_for_best_rank() -> None:
    target_rank = torch.tensor([0, 1, 2], dtype=torch.long)
    correct_cost = torch.tensor([0.0, 1.0, 2.0])
    reversed_cost = torch.tensor([2.0, 1.0, 0.0])

    assert listmle_loss(correct_cost, target_rank) < listmle_loss(reversed_cost, target_rank)


def test_listmle_is_invariant_to_row_permutation() -> None:
    predicted_cost = torch.tensor([0.25, 1.0, -0.5, 2.0])
    target_rank = torch.tensor([1, 2, 0, 3], dtype=torch.long)
    weight = torch.tensor([1.0, 0.5, 2.0, 0.25])
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)

    original = listmle_loss(predicted_cost, target_rank, weight=weight)
    permuted = listmle_loss(
        predicted_cost[permutation],
        target_rank[permutation],
        weight=weight[permutation],
    )

    assert original == pytest.approx(float(permuted))


def test_listmle_is_finite_with_extreme_costs() -> None:
    predicted_cost = torch.tensor([-1.0e4, 0.0, 1.0e4])
    target_rank = torch.tensor([0, 1, 2], dtype=torch.long)

    loss = listmle_loss(predicted_cost, target_rank)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("predicted_score", "target_rank", "match"),
    [
        (torch.tensor([[0.0, 1.0]]), torch.tensor([0, 1], dtype=torch.long), "1-D"),
        (torch.tensor([0.0]), torch.tensor([[0]], dtype=torch.long), "1-D"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0], dtype=torch.long), "equal shape"),
        (torch.tensor([]), torch.tensor([], dtype=torch.long), "nonempty"),
        (torch.tensor([float("nan")]), torch.tensor([0], dtype=torch.long), "finite"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), "torch.long"),
        (torch.tensor([0.0, 1.0]), torch.tensor([False, True]), "torch.long"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 0], dtype=torch.long), "permutation"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 2], dtype=torch.long), "permutation"),
    ],
)
def test_listmle_rejects_invalid_scores_and_ranks(
    predicted_score: torch.Tensor,
    target_rank: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        listmle_loss(predicted_score, target_rank)


@pytest.mark.parametrize(
    ("weight", "match"),
    [
        (torch.tensor([[1.0, 1.0]]), "1-D"),
        (torch.tensor([1.0]), "shape"),
        (torch.tensor([1.0, float("nan")]), "finite"),
        (torch.tensor([1.0, -0.1]), "nonnegative"),
        (torch.tensor([0.0, 0.0]), "positive total"),
        (torch.tensor([False, True]), "not bool"),
        (torch.tensor([0.0, 1.0]), "nontrivial"),
    ],
)
def test_listmle_rejects_invalid_weights(weight: torch.Tensor, match: str) -> None:
    predicted_score = torch.tensor([0.0, 1.0])
    target_rank = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match=match):
        listmle_loss(predicted_score, target_rank, weight=weight)


def test_listmle_gradients_are_finite_and_nonzero() -> None:
    predicted_cost = torch.tensor([0.5, 0.0, 1.0], requires_grad=True)
    target_rank = torch.tensor([1, 0, 2], dtype=torch.long)

    loss = listmle_loss(predicted_cost, target_rank)
    loss.backward()

    assert predicted_cost.grad is not None
    assert torch.isfinite(predicted_cost.grad).all()
    assert float(predicted_cost.grad.abs().sum()) > 0.0
