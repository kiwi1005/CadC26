from __future__ import annotations

import torch

from puzzleplace.data import ConstraintColumns, adapt_training_batch, adapt_validation_batch


def test_adapt_training_batch_trims_padding_and_preserves_labels() -> None:
    batch = (
        torch.tensor([[10.0, 20.0, -1.0]]),
        torch.tensor([[[0.0, 1.0, 2.0], [-1.0, -1.0, -1.0]]]),
        torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]]),
        torch.tensor([[[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]]),
        torch.zeros((1, 2, 3)),
        torch.tensor([[[2.0, 3.0, 0.0, 0.0], [4.0, 5.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]]),
        torch.tensor([[100.0] * 8]),
    )
    case = adapt_training_batch(batch, case_id="train-1")
    assert case.case_id == "train-1"
    assert case.block_count == 2
    assert case.area_targets.tolist() == [10.0, 20.0]
    assert case.target_positions is not None
    assert case.target_positions.shape == (2, 4)
    assert int(case.constraints[0, ConstraintColumns.FIXED].item()) == 1
    assert case.b2b_edges.shape == (1, 3)


def test_adapt_validation_batch_converts_polygons_to_boxes() -> None:
    inputs = (
        torch.tensor([[10.0, 20.0]]),
        torch.tensor([[[0.0, 1.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 2.0]]]),
    )
    polygons = torch.tensor([
        [
            [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]],
            [[5.0, 5.0], [8.0, 5.0], [8.0, 7.0], [5.0, 7.0]],
        ]
    ])
    metrics = torch.tensor([[50.0] * 8])
    case = adapt_validation_batch((inputs, (polygons, metrics)), case_id="val-1")
    assert case.block_count == 2
    assert case.target_positions is not None
    assert torch.allclose(case.target_positions[0], torch.tensor([0.0, 0.0, 2.0, 3.0]))
    assert torch.allclose(case.target_positions[1], torch.tensor([5.0, 5.0, 3.0, 2.0]))
