from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_projection():
    spec = importlib.util.spec_from_file_location("hcfp_projection_test_mod", ROOT / "src/hcfp/projection.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


projection = _load_projection()


def devices():
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize(
    ("boxes", "direction"),
    [
        ([[0.0, 0.0, 4.0, 10.0], [3.0, 0.0, 4.0, 10.0]], 0),
        ([[3.0, 0.0, 4.0, 10.0], [0.0, 0.0, 4.0, 10.0]], 1),
        ([[0.0, 0.0, 10.0, 4.0], [0.0, 3.0, 10.0, 4.0]], 2),
        ([[0.0, 3.0, 10.0, 4.0], [0.0, 0.0, 10.0, 4.0]], 3),
    ],
)
def test_assigns_minimum_displacement_direction_for_each_side(boxes, direction):
    pairs, directions = projection.assign_directions(torch.tensor(boxes))
    assert pairs.tolist() == [[0, 1]]
    assert directions.item() == direction


@pytest.mark.parametrize("device", devices())
def test_project_resolves_overlap_and_preserves_dimensions(device):
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [2.0, 0.0, 4.0, 4.0]], device=device)
    result = projection.project_disjunctive(boxes, iterations=8)
    assert result.ok
    assert result.status == "ok"
    assert bool(result.ok_mask)
    assert result.failure_reasons == ("ok",)
    assert result.active_pair_count.item() == 0
    assert result.displacement.item() > 0.0
    assert result.xywh.device.type == torch.device(device).type
    assert torch.allclose(result.xywh[:, 2:4], boxes[:, 2:4])
    assert projection.overlap_matrix(result.xywh).max().item() <= 1.0e-5


@pytest.mark.parametrize("device", devices())
def test_preplaced_centers_are_anchors(device):
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0],
            [1.0, 0.0, 4.0, 4.0],
            [8.0, 0.0, 2.0, 2.0],
        ],
        device=device,
    )
    result = projection.project_disjunctive(boxes, preplaced_mask=torch.tensor([True, False, False], device=device), iterations=10)
    assert result.ok
    assert torch.equal(result.xywh[0, :2], boxes[0, :2])
    assert torch.allclose(result.xywh[:, 2:4], boxes[:, 2:4])


@pytest.mark.parametrize("device", devices())
def test_batch_projection_is_independent_and_deterministic(device):
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 4.0], [2.0, 0.0, 4.0, 4.0]],
            [[0.0, 0.0, 3.0, 3.0], [0.0, 2.0, 3.0, 3.0]],
        ],
        device=device,
    )
    first = projection.project_disjunctive(boxes, iterations=10)
    second = projection.project_disjunctive(boxes, iterations=10)
    assert first.ok
    assert torch.allclose(first.xywh, second.xywh)
    assert torch.allclose(first.xywh[..., 2:4], boxes[..., 2:4])
    assert torch.all(projection.overlap_matrix(first.xywh).amax(dim=(1, 2)) <= 1.0e-5)
    assert first.ok_mask.tolist() == [True, True]
    assert first.active_pair_count.tolist() == second.active_pair_count.tolist()
    assert torch.all(first.displacement >= 0.0)


@pytest.mark.parametrize("device", devices())
def test_overlapping_preplaced_assignment_fails_closed(device):
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [2.0, 0.0, 4.0, 4.0]], device=device)
    result = projection.project_disjunctive(boxes, preplaced_mask=torch.tensor([True, True], device=device), iterations=8)
    assert not result.ok
    assert result.status == "infeasible"
    assert result.ok_mask.item() is False
    assert result.failure_reasons == ("fixed_pair_overlap",)
    assert result.max_overlap.item() > 0.0
    assert torch.equal(result.xywh, boxes)


@pytest.mark.parametrize("device", devices())
def test_outer_rebuild_catches_newly_created_overlaps(device):
    boxes = torch.tensor(
        [
            [0.6079329252, 3.9580490589, 2.1840376854, 1.6304452419],
            [0.0058529293, 2.8178839684, 3.2314422131, 3.6374118328],
            [2.7010056973, 0.1247475445, 1.9217861891, 2.4937322140],
        ],
        device=device,
    )

    one_pass = projection.project_disjunctive(boxes, iterations=6, outer_iterations=1, beam=1)
    rebuilt = projection.project_disjunctive(boxes, iterations=6, outer_iterations=3, beam=1)

    assert not one_pass.ok
    assert one_pass.failure_reasons == ("residual_overlap",)
    assert rebuilt.ok
    assert rebuilt.failure_reasons == ("ok",)
    assert projection.overlap_matrix(rebuilt.xywh).max().item() <= 1.0e-5


def test_normalized_problem_uses_raw_coordinate_overlap_tolerance() -> None:
    class Problem:
        scale = 100.0
        normalized = True
        preplaced_mask = torch.tensor([False, False])

    boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.05, 0.0, 0.1, 0.1]])
    result = projection.project_disjunctive(boxes, problem=Problem(), iterations=8)

    assert result.ok
    assert result.max_overlap.item() <= 1.0e-8
