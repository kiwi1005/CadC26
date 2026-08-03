from __future__ import annotations

import pytest
import torch

from hcfp.case import from_official
from hcfp.dynamics import (
    DynamicsConfig,
    ForceControl,
    initialize_population,
    relax,
    typed_forces,
)
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


def test_force_controller_none_and_ones_match_default_dynamics() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=3, steps=3)
    initial = _overlapped_initial(case)

    baseline = relax(case, cfg, initial_xywh=initial)
    explicit_none = relax(case, cfg, initial_xywh=initial, force_controller=None)

    def ones_controller(_case, state, _step_fraction):
        return torch.ones((*state.center.shape[:2], 7), device=state.center.device)

    ones = relax(case, cfg, initial_xywh=initial, force_controller=ones_controller)

    assert torch.equal(explicit_none.boxes, baseline.boxes)
    assert torch.equal(explicit_none.state.energy_history, baseline.state.energy_history)
    assert torch.equal(ones.boxes, baseline.boxes)
    assert torch.equal(ones.state.energy_history, baseline.state.energy_history)


def test_zero_force_gate_changes_motion_and_preserves_hard_invariants() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=3, steps=2)
    initial = _overlapped_initial(case)
    baseline = relax(case, cfg, initial_xywh=initial)

    def zero_controller(_case, state, _step_fraction):
        return torch.zeros((*state.center.shape[:2], 7), device=state.center.device)

    gated = relax(case, cfg, initial_xywh=initial, force_controller=zero_controller)

    assert not torch.equal(gated.boxes, baseline.boxes)
    assert torch.equal(gated.boxes[:, 0], case.target[0].expand(3, -1))
    assert torch.equal(gated.boxes[:, 2, 2:4], case.target[2, 2:4].expand(3, -1))
    assert torch.isfinite(gated.boxes).all()


def test_controller_call_count_step_fraction_and_gate_diagnostics() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=2, steps=3)
    fractions: list[float] = []

    def controller(_case, state, step_fraction):
        fractions.append(step_fraction)
        return torch.ones((*state.center.shape[:2], 7), device=state.center.device)

    result = relax(case, cfg, initial_xywh=_overlapped_initial(case), force_controller=controller)

    assert fractions == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0])
    assert torch.equal(result.diagnostics["force_gate"], torch.ones((2, 3, 7)))


def test_steps_zero_does_not_call_controller_and_keeps_population_byte_identical() -> None:
    case = _case("cpu")
    candidate = _overlapped_initial(case)
    population = torch.stack((candidate, candidate + torch.tensor([0.0, 0.1, 0.0, 0.0])))
    population[:, case.preplaced_mask] = case.target[case.preplaced_mask]

    def controller(*_args):
        raise AssertionError("controller must not be called for steps=0")

    result = relax(
        case,
        DynamicsConfig(population=2, steps=0),
        initial_xywh=population,
        force_controller=controller,
    )

    assert torch.equal(result.initial_boxes, population)
    assert torch.equal(result.boxes, population)


def test_force_gate_validation_rejects_shape_nonfinite_and_negative() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=2, steps=1)
    state = initialize_population(case, cfg, _overlapped_initial(case))

    with pytest.raises(ValueError, match="shape"):
        typed_forces(case, state, cfg, force_gates=torch.ones((2, 3, 6)))
    bad = torch.ones((2, 3, 7))
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        typed_forces(case, state, cfg, force_gates=bad)
    bad = torch.ones((2, 3, 7))
    bad[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        typed_forces(case, state, cfg, force_gates=bad)


def test_collective_velocity_changes_free_geometry_and_preserves_hard_invariants() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=2, steps=1)
    initial = _overlapped_initial(case)

    def controller(_case, state, _step_fraction):
        gates = torch.zeros((*state.center.shape[:2], 7), device=state.center.device)
        velocity = torch.full_like(state.velocity, 0.01)
        return ForceControl(gates, velocity)

    controlled = relax(case, cfg, initial_xywh=initial, force_controller=controller)

    assert not torch.equal(controlled.boxes, controlled.initial_boxes)
    assert torch.equal(controlled.boxes[:, 0], case.target[0].expand(2, -1))
    assert torch.equal(controlled.boxes[:, 2, 2:4], case.target[2, 2:4].expand(2, -1))
    assert torch.equal(
        controlled.diagnostics["learned_velocity"],
        torch.full((2, 3, 3), 0.01),
    )


def test_collective_velocity_validation_rejects_shape_and_nonfinite() -> None:
    case = _case("cpu")
    cfg = DynamicsConfig(population=2, steps=1)

    def bad_shape(_case, state, _step_fraction):
        gates = torch.ones((*state.center.shape[:2], 7))
        return ForceControl(gates, torch.zeros((2, 3, 2)))

    with pytest.raises(ValueError, match="learned_velocity"):
        relax(case, cfg, initial_xywh=_overlapped_initial(case), force_controller=bad_shape)

    def nonfinite(_case, state, _step_fraction):
        gates = torch.ones((*state.center.shape[:2], 7))
        velocity = torch.zeros_like(state.velocity)
        velocity[0, 0, 0] = float("nan")
        return ForceControl(gates, velocity)

    with pytest.raises(ValueError, match="finite"):
        relax(case, cfg, initial_xywh=_overlapped_initial(case), force_controller=nonfinite)


def _overlapped_initial(case):
    return normalize_xywh(
        case,
        torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.5, 0.0, 2.0, 2.0], [1.0, 0.0, 2.0, 2.0]]
        ),
    )
