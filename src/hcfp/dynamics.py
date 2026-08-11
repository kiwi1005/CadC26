"""Analytic HCFP population dynamics implemented with dense FP32 tensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from hcfp.case import FloorplanCase
from hcfp.fallback import safe_shelf
from hcfp.geometry import (
    bbox_area_tensor,
    bbox_tensor,
    centers_from_xywh,
    exact_shape_projection,
    hpwl_tensor,
    log_aspect_from_xywh,
    overlap_area_matrix,
    xywh_from_state,
)


Tensor = torch.Tensor
FORCE_CHANNELS = (
    "net",
    "pin",
    "overlap",
    "boundary",
    "grouping",
    "compaction",
    "mib",
)


@dataclass(frozen=True)
class ForceControl:
    force_gates: Tensor
    learned_velocity: Tensor | None = None


ForceController = Callable[
    [FloorplanCase, "PopulationState", float], Tensor | ForceControl
]


@dataclass(frozen=True)
class DynamicsConfig:
    population: int = 8
    steps: int = 12
    momentum: float = 0.60
    step_size: float = 0.045
    shape_step: float = 0.020
    max_position_step: float = 0.080
    max_shape_step: float = 0.040
    net_weight: float = 0.90
    pin_weight: float = 0.90
    overlap_weight: float = 1.80
    boundary_weight: float = 0.35
    grouping_weight: float = 0.30
    compaction_weight: float = 0.18
    mib_weight: float = 0.20
    net_temperature: float = 0.20

    def __post_init__(self) -> None:
        if self.population <= 0 or self.steps < 0:
            raise ValueError("population must be positive and steps non-negative")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if self.step_size <= 0 or self.max_position_step <= 0:
            raise ValueError("position step sizes must be positive")


@dataclass
class PopulationState:
    center: Tensor
    log_aspect: Tensor
    velocity: Tensor
    alive_mask: Tensor
    energy_history: Tensor


@dataclass(frozen=True)
class DynamicsResult:
    initial_boxes: Tensor
    boxes: Tensor
    state: PopulationState
    diagnostics: dict[str, Tensor]


def _rms_normalize(force: Tensor) -> Tensor:
    dims = tuple(range(1, force.ndim))
    scale = torch.sqrt(torch.mean(force.square(), dim=dims, keepdim=True) + 1.0e-12)
    return force / scale


def initialize_population(
    case: FloorplanCase,
    config: DynamicsConfig,
    initial_xywh: Tensor | None = None,
    *,
    enforce_mib: bool = True,
) -> PopulationState:
    k, n = config.population, case.n
    device = case.area.device
    log_aspect = torch.zeros((k, n), dtype=torch.float32, device=device)
    hard = case.fixed_mask | case.preplaced_mask
    target_wh = case.target[:, 2:4].clamp_min(1.0e-30)
    hard_ratio = torch.log(target_wh[:, 0] / target_wh[:, 1]).to(device=device, dtype=torch.float32)
    log_aspect[:, hard] = hard_ratio[hard]
    dimensions = exact_shape_projection(case, log_aspect, enforce_mib=enforce_mib)

    candidate_id = torch.arange(k, dtype=torch.float32, device=device)
    layout_ratio = torch.exp(torch.linspace(-0.7, 0.7, k, device=device))
    columns = torch.ceil(torch.sqrt(torch.tensor(float(n), device=device) * layout_ratio)).to(torch.long).clamp_min(1)
    block_id = torch.arange(n, device=device).view(1, n)
    shifts = torch.arange(k, device=device).view(k, 1)
    slot = (block_id + shifts * max(1, n // max(k, 1))) % n
    row = torch.div(slot, columns.view(k, 1), rounding_mode="floor").to(torch.float32)
    column = (slot % columns.view(k, 1)).to(torch.float32)
    cell_width = dimensions[..., 0].amax(dim=1, keepdim=True) * 1.08
    cell_height = dimensions[..., 1].amax(dim=1, keepdim=True) * 1.08
    x = column * cell_width
    y = row * cell_height
    center = torch.stack((x, y), dim=-1) + 0.5 * dimensions
    bounds_center = 0.5 * (center.amin(dim=1, keepdim=True) + center.amax(dim=1, keepdim=True))
    phase = torch.stack((torch.sin(candidate_id), torch.cos(candidate_id)), dim=-1).view(k, 1, 2) * 0.03
    center = center - bounds_center + phase

    if initial_xywh is not None:
        initial = torch.as_tensor(initial_xywh, dtype=torch.float32, device=device)
        if initial.ndim == 2:
            if initial.shape != (n, 4):
                raise ValueError("initial_xywh must have shape [N,4] or [K,N,4]")
            center[0] = centers_from_xywh(initial)
            log_aspect[0] = log_aspect_from_xywh(initial)
        elif initial.shape == (k, n, 4):
            center = centers_from_xywh(initial)
            log_aspect = log_aspect_from_xywh(initial)
        else:
            raise ValueError("initial_xywh must have shape [N,4] or [K,N,4]")

    preplaced_center = centers_from_xywh(case.target).to(device=device, dtype=torch.float32)
    center[:, case.preplaced_mask] = preplaced_center[case.preplaced_mask]
    log_aspect[:, hard] = hard_ratio[hard]
    velocity = torch.zeros((k, n, 3), dtype=torch.float32, device=device)
    alive = torch.ones(k, dtype=torch.bool, device=device)
    history = torch.empty((k, 0, 3), dtype=torch.float32, device=device)
    return PopulationState(center, log_aspect, velocity, alive, history)


def _net_force(case: FloorplanCase, center: Tensor, temperature: float) -> Tensor:
    delta = center[:, None, :, :] - center[:, :, None, :]
    weights = case.b2b_weight.to(device=center.device, dtype=torch.float32)
    return (weights.view(1, case.n, case.n, 1) * torch.tanh(delta / temperature)).sum(dim=2)


def _pin_force(case: FloorplanCase, center: Tensor, temperature: float) -> Tensor:
    force = torch.zeros_like(center)
    if not case.p2b_edges.numel():
        return force
    edges = case.p2b_edges.to(device=center.device)
    pin_index = edges[:, 0].to(torch.long)
    block_index = edges[:, 1].to(torch.long)
    weight = edges[:, 2].to(torch.float32)
    pins = case.pins.to(device=center.device, dtype=torch.float32)
    edge_force = torch.tanh((pins[pin_index].unsqueeze(0) - center[:, block_index]) / temperature)
    force.index_add_(1, block_index, edge_force * weight.view(1, -1, 1))
    return force


def _overlap_force(center: Tensor, dimensions: Tensor) -> Tensor:
    delta = center[:, :, None, :] - center[:, None, :, :]
    overlap_x = 0.5 * (dimensions[:, :, None, 0] + dimensions[:, None, :, 0]) - delta[..., 0].abs()
    overlap_y = 0.5 * (dimensions[:, :, None, 1] + dimensions[:, None, :, 1]) - delta[..., 1].abs()
    n = center.shape[1]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=center.device).view(1, n, n)
    active = (overlap_x > 0.0) & (overlap_y > 0.0) & off_diagonal
    choose_x = overlap_x <= overlap_y
    index = torch.arange(n, device=center.device)
    tie_sign = torch.sign((index[:, None] - index[None, :]).to(torch.float32)).view(1, n, n)
    sign_x = torch.where(delta[..., 0] == 0.0, tie_sign, torch.sign(delta[..., 0]))
    sign_y = torch.where(delta[..., 1] == 0.0, tie_sign, torch.sign(delta[..., 1]))
    pair_x = torch.where(active & choose_x, sign_x * overlap_x.clamp_min(0.0), 0.0)
    pair_y = torch.where(active & ~choose_x, sign_y * overlap_y.clamp_min(0.0), 0.0)
    return torch.stack((pair_x, pair_y), dim=-1).sum(dim=2)


def _boundary_force(case: FloorplanCase, center: Tensor, dimensions: Tensor) -> Tensor:
    boxes = torch.cat((center - 0.5 * dimensions, dimensions), dim=-1)
    bounds = bbox_tensor(boxes)
    left_target = bounds[:, 0:1] + 0.5 * dimensions[..., 0]
    right_target = bounds[:, 2:3] - 0.5 * dimensions[..., 0]
    bottom_target = bounds[:, 1:2] + 0.5 * dimensions[..., 1]
    top_target = bounds[:, 3:4] - 0.5 * dimensions[..., 1]
    bits = case.boundary_bits.to(device=center.device)
    force_x = torch.where(bits[None, :, 0], left_target - center[..., 0], 0.0)
    force_x += torch.where(bits[None, :, 1], right_target - center[..., 0], 0.0)
    force_y = torch.where(bits[None, :, 2], top_target - center[..., 1], 0.0)
    force_y += torch.where(bits[None, :, 3], bottom_target - center[..., 1], 0.0)
    return torch.stack((force_x, force_y), dim=-1)


def _group_force(case: FloorplanCase, center: Tensor) -> Tensor:
    membership = case.group_membership.to(device=center.device, dtype=torch.float32)
    if not membership.numel():
        return torch.zeros_like(center)
    count = membership.sum(dim=1).clamp_min(1.0)
    group_center = torch.einsum("gn,knd->kgd", membership, center) / count.view(1, -1, 1)
    target = torch.einsum("gn,kgd->knd", membership, group_center)
    present = membership.sum(dim=0).clamp_max(1.0).view(1, -1, 1)
    return (target - center) * present


def _compaction_force(center: Tensor, dimensions: Tensor) -> Tensor:
    boxes = torch.cat((center - 0.5 * dimensions, dimensions), dim=-1)
    bounds = bbox_tensor(boxes)
    target = 0.5 * (bounds[:, :2] + bounds[:, 2:4])
    return target[:, None, :] - center


def _mib_shape_force(case: FloorplanCase, log_aspect: Tensor) -> Tensor:
    membership = case.mib_membership.to(device=log_aspect.device, dtype=torch.float32)
    if not membership.numel():
        return torch.zeros_like(log_aspect)
    count = membership.sum(dim=1).clamp_min(1.0)
    mean = torch.einsum("gn,kn->kg", membership, log_aspect) / count.view(1, -1)
    target = torch.einsum("gn,kg->kn", membership, mean)
    present = membership.sum(dim=0).clamp_max(1.0).view(1, -1)
    return (target - log_aspect) * present


def typed_forces(
    case: FloorplanCase,
    state: PopulationState,
    config: DynamicsConfig,
    force_gates: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    dimensions = exact_shape_projection(case, state.log_aspect)
    channels = {
        "net": _net_force(case, state.center, config.net_temperature),
        "pin": _pin_force(case, state.center, config.net_temperature),
        "overlap": _overlap_force(state.center, dimensions),
        "boundary": _boundary_force(case, state.center, dimensions),
        "grouping": _group_force(case, state.center),
        "compaction": _compaction_force(state.center, dimensions),
    }
    normalized = {name: _rms_normalize(force) for name, force in channels.items()}
    shape_force = _rms_normalize(_mib_shape_force(case, state.log_aspect))
    gates = _validate_force_gates(force_gates, state) if force_gates is not None else None
    if gates is None:
        position_force = (
            config.net_weight * normalized["net"]
            + config.pin_weight * normalized["pin"]
            + config.overlap_weight * normalized["overlap"]
            + config.boundary_weight * normalized["boundary"]
            + config.grouping_weight * normalized["grouping"]
            + config.compaction_weight * normalized["compaction"]
        )
        gated_shape = config.mib_weight * shape_force.unsqueeze(-1)
    else:
        position_force = (
            config.net_weight * normalized["net"] * gates[..., 0:1]
            + config.pin_weight * normalized["pin"] * gates[..., 1:2]
            + config.overlap_weight * normalized["overlap"] * gates[..., 2:3]
            + config.boundary_weight * normalized["boundary"] * gates[..., 3:4]
            + config.grouping_weight * normalized["grouping"] * gates[..., 4:5]
            + config.compaction_weight * normalized["compaction"] * gates[..., 5:6]
        )
        gated_shape = config.mib_weight * shape_force.unsqueeze(-1) * gates[..., 6:7]
    return channels, torch.cat((position_force, gated_shape), dim=-1)


def _validate_force_gates(force_gates: Tensor, state: PopulationState) -> Tensor:
    gates = torch.as_tensor(
        force_gates,
        dtype=torch.float32,
        device=state.center.device,
    )
    expected = (*state.center.shape[:2], len(FORCE_CHANNELS))
    if gates.shape != expected:
        raise ValueError("force_gates must have shape [K,N,7]")
    if not bool(torch.isfinite(gates).all()):
        raise ValueError("force_gates must be finite")
    if bool((gates < 0.0).any()):
        raise ValueError("force_gates must be nonnegative")
    return gates


def _diagnostics(case: FloorplanCase, state: PopulationState) -> dict[str, Tensor]:
    boxes = xywh_from_state(case, state.center, state.log_aspect)
    overlap = torch.triu(overlap_area_matrix(boxes), diagonal=1).sum(dim=(1, 2))
    hpwl = hpwl_tensor(case, state.center)
    area = bbox_area_tensor(boxes)
    return {"overlap": overlap, "hpwl": hpwl, "bbox_area": area}


def step(
    case: FloorplanCase,
    state: PopulationState,
    config: DynamicsConfig,
    *,
    force_gates: Tensor | None = None,
    learned_velocity: Tensor | None = None,
) -> tuple[PopulationState, dict[str, Tensor]]:
    gates = _validate_force_gates(force_gates, state) if force_gates is not None else None
    channels, force = typed_forces(case, state, config, force_gates=gates)
    velocity = config.momentum * state.velocity
    velocity[..., :2] += config.step_size * force[..., :2]
    velocity[..., 2] += config.shape_step * force[..., 2]
    learned = _validate_learned_velocity(learned_velocity, state)
    if learned is not None:
        velocity += learned
    velocity[..., :2] = velocity[..., :2].clamp(-config.max_position_step, config.max_position_step)
    velocity[..., 2] = velocity[..., 2].clamp(-config.max_shape_step, config.max_shape_step)

    preplaced = case.preplaced_mask.to(device=state.center.device)
    hard_shape = (case.fixed_mask | case.preplaced_mask).to(device=state.center.device)
    velocity[:, preplaced, :2] = 0.0
    velocity[:, hard_shape, 2] = 0.0
    center = state.center + velocity[..., :2]
    log_aspect = (state.log_aspect + velocity[..., 2]).clamp(-4.0, 4.0)

    target_center = centers_from_xywh(case.target).to(device=center.device, dtype=torch.float32)
    target_wh = case.target[:, 2:4].clamp_min(1.0e-30)
    target_ratio = torch.log(target_wh[:, 0] / target_wh[:, 1]).to(device=center.device, dtype=torch.float32)
    center[:, preplaced] = target_center[preplaced]
    log_aspect[:, hard_shape] = target_ratio[hard_shape]

    next_state = PopulationState(center, log_aspect, velocity, state.alive_mask, state.energy_history)
    diagnostics = _diagnostics(case, next_state)
    snapshot = torch.stack((diagnostics["overlap"], diagnostics["hpwl"], diagnostics["bbox_area"]), dim=-1)
    next_state.energy_history = torch.cat((state.energy_history, snapshot.unsqueeze(1)), dim=1)
    diagnostics.update({f"force_{name}": value for name, value in channels.items()})
    if gates is not None:
        diagnostics["force_gate"] = gates
    if learned is not None:
        diagnostics["learned_velocity"] = learned
    return next_state, diagnostics


def _validate_learned_velocity(
    learned_velocity: Tensor | None,
    state: PopulationState,
) -> Tensor | None:
    if learned_velocity is None:
        return None
    velocity = torch.as_tensor(
        learned_velocity,
        dtype=torch.float32,
        device=state.center.device,
    )
    if velocity.shape != state.velocity.shape:
        raise ValueError("learned_velocity must have shape [K,N,3]")
    if not bool(torch.isfinite(velocity).all()):
        raise ValueError("learned_velocity must be finite")
    return velocity


def relax(
    case: FloorplanCase,
    config: DynamicsConfig | None = None,
    *,
    initial_xywh: Tensor | None = None,
    force_controller: ForceController | None = None,
) -> DynamicsResult:
    cfg = config or DynamicsConfig()
    if initial_xywh is None:
        initial_xywh = safe_shelf(case).to(device=case.area.device, dtype=torch.float32)
    supplied = torch.as_tensor(
        initial_xywh,
        dtype=torch.float32,
        device=case.area.device,
    )
    state = initialize_population(case, cfg, initial_xywh)
    initial_boxes = (
        supplied.clone()
        if supplied.shape == (cfg.population, case.n, 4)
        else xywh_from_state(case, state.center, state.log_aspect)
    )
    diagnostics = _diagnostics(case, state)
    for step_index in range(cfg.steps):
        force_gates: Tensor | None = None
        learned_velocity: Tensor | None = None
        if force_controller is not None:
            control = force_controller(
                case,
                state,
                step_index / max(cfg.steps, 1),
            )
            if isinstance(control, ForceControl):
                force_gates = control.force_gates
                learned_velocity = control.learned_velocity
            else:
                force_gates = control
        state, diagnostics = step(
            case,
            state,
            cfg,
            force_gates=force_gates,
            learned_velocity=learned_velocity,
        )
        if not bool(torch.isfinite(state.center).all() and torch.isfinite(state.log_aspect).all()):
            raise FloatingPointError("collective dynamics produced non-finite geometry")
    boxes = (
        initial_boxes.clone()
        if cfg.steps == 0
        else xywh_from_state(case, state.center, state.log_aspect)
    )
    return DynamicsResult(initial_boxes=initial_boxes, boxes=boxes, state=state, diagnostics=diagnostics)
