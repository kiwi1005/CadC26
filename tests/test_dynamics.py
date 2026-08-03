from __future__ import annotations

import pytest
import torch

from hcfp.case import from_official
from hcfp.dynamics import DynamicsConfig, initialize_population, relax, typed_forces
from hcfp.geometry import centers_from_xywh, normalize_xywh


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _case(device: str):
    return from_official(
        3,
        [4.0, 4.0, 4.0],
        [[0, 1, 3.0], [1, 2, 1.0]],
        [],
        [],
        [[0, 1, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 2]],
        [[0.0, 0.0, 2.0, 2.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 2.0, 2.0]],
        device=device,
    )


@pytest.mark.parametrize("device", _devices())
def test_parallel_dynamics_preserves_hard_geometry_and_is_finite(device: str) -> None:
    case = _case(device)
    cfg = DynamicsConfig(population=4, steps=4)
    initial = normalize_xywh(case, torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.5, 0.0, 2.0, 2.0], [1.0, 0.0, 2.0, 2.0]],
        device=device,
    ))
    result = relax(case, cfg, initial_xywh=initial)

    assert result.initial_boxes.shape == (4, 3, 4)
    assert result.boxes.shape == (4, 3, 4)
    assert torch.isfinite(result.boxes).all()
    assert torch.equal(result.boxes[:, 0], case.target[0].expand(4, -1))
    assert torch.equal(result.boxes[:, 2, 2:4], case.target[2, 2:4].expand(4, -1))
    assert result.state.energy_history.shape == (4, 4, 3)


def test_overlap_force_is_antisymmetric_for_center_tie() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=1, steps=0)
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 1.0, 1.0]])
    state = initialize_population(case, cfg, initial_xywh=boxes)
    state.center[:] = centers_from_xywh(boxes)
    channels, _ = typed_forces(case, state, cfg)

    pair_force = channels["overlap"][0, :2]
    assert torch.allclose(pair_force[0], -pair_force[1])
    assert torch.linalg.vector_norm(pair_force[0]) > 0


def test_full_supplied_population_is_retained_as_initial_candidates() -> None:
    case = _case("cpu")
    candidate = normalize_xywh(
        case,
        torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [4.0, 0.0, 2.0, 2.0]]
        ),
    )
    population = torch.stack((candidate, candidate + torch.tensor([0.0, 1.0, 0.0, 0.0])))
    population[:, case.preplaced_mask] = case.target[case.preplaced_mask]

    result = relax(
        case,
        DynamicsConfig(population=2, steps=0),
        initial_xywh=population,
    )

    assert torch.equal(result.initial_boxes, population)
    assert torch.equal(result.boxes, population)
