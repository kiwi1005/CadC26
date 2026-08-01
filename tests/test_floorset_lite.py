from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hcfp.floorset_lite import (
    fp_sol_to_xywh,
    iter_floorset_lite,
    sample_from_lite_tensors,
    target_positions_from_solution,
)


def _layout_payload() -> list[torch.Tensor]:
    area_constraints = torch.tensor(
        [
            [4.0, 0, 1, 0, 0, 1],
            [9.0, 1, 0, 0, 0, 0],
            [16.0, 0, 0, 0, 0, 0],
        ]
    ).unsqueeze(0)
    b2b = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 2.0, 1.0]]])
    p2b = torch.tensor([[[-1.0, -1.0, -1.0]]])
    pins = torch.tensor([[[-1.0, -1.0]]])
    tree = torch.zeros(1, 2, 3)
    fp_sol = torch.tensor([[[2.0, 2.0, 0.0, 0.0], [3.0, 3.0, 3.0, 0.0], [4.0, 4.0, 0.0, 4.0]]])
    metrics = torch.zeros(1, 8)
    return [area_constraints, b2b, p2b, pins, tree, fp_sol, metrics]


def test_lite_rectangles_and_targets_match_official_semantics() -> None:
    payload = _layout_payload()
    rectangles = fp_sol_to_xywh(payload[5][0], 3)
    constraints = payload[0][0, :, 1:6]
    targets = target_positions_from_solution(constraints, rectangles)
    sample = sample_from_lite_tensors("worker/layout:0", payload[0][0], payload[1][0], payload[2][0], payload[3][0], payload[5][0])

    assert torch.equal(rectangles[0], torch.tensor([0.0, 0.0, 2.0, 2.0]))
    assert torch.equal(targets[0], rectangles[0])
    assert torch.equal(targets[1], torch.tensor([-1.0, -1.0, 3.0, 3.0]))
    assert torch.equal(targets[2], torch.full((4,), -1.0))
    assert sample.case.preplaced_mask.tolist() == [True, False, False]
    assert sample.case.fixed_mask.tolist() == [False, True, False]


def test_direct_training_stream_loads_layouts_without_creating_shards(tmp_path: Path) -> None:
    layout = tmp_path / "floorset_lite/worker_0/layouts_0.pth"
    layout.parent.mkdir(parents=True)
    torch.save(_layout_payload(), layout)

    samples = list(iter_floorset_lite(tmp_path, limit=1))

    assert [sample.sample_id for sample in samples] == ["worker_0/layouts_0.pth:0"]
    assert not list(tmp_path.rglob("*.tar"))


def test_visible_validation_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        list(iter_floorset_lite(tmp_path / "LiteTensorDataTest"))
