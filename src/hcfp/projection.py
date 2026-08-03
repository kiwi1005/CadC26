"""Batched disjunctive projection v0 for HCFP rectangles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hcfp.geometry import bbox_area_tensor, centers_from_xywh, hpwl_tensor
from hcfp.projection_guidance import ProjectionGuidance


Tensor = torch.Tensor
EPS = 1.0e-6
BDP_LEFT = 0
BDP_RIGHT = 1
BDP_BELOW = 2
BDP_ABOVE = 3
TOPOLOGY_TO_BDP = (BDP_LEFT, BDP_RIGHT, BDP_ABOVE, BDP_BELOW)
_PROPOSAL_REASONS = (
    "not_component",
    "already_feasible",
    "projector_incomplete",
    "construction_regression",
    "committed",
)
_PROPOSAL_NOT_COMPONENT = 0
_PROPOSAL_ALREADY_FEASIBLE = 1
_PROPOSAL_PROJECTOR_INCOMPLETE = 2
_PROPOSAL_CONSTRUCTION_REGRESSION = 3
_PROPOSAL_COMMITTED = 4


@dataclass(frozen=True)
class ComponentBDPConfig:
    enabled: bool = False
    beam_width: int = 4
    component_limit: int = 24
    max_uncertain_pairs: int = 6
    outer_sweeps: int = 4
    reset_limit: int = 2
    preserve_feasible: bool = True
    topology_weight: float = 1.0
    contact_weight: float = 4.0
    boundary_weight: float = 4.0

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.component_limit < 2:
            raise ValueError("component_limit must be at least 2")
        if self.max_uncertain_pairs < 0:
            raise ValueError("max_uncertain_pairs must be non-negative")
        if self.outer_sweeps < 1:
            raise ValueError("outer_sweeps must be positive")
        if self.reset_limit < 0:
            raise ValueError("reset_limit must be non-negative")
        for name in ("topology_weight", "contact_weight", "boundary_weight"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.tensor(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ProjectionResult:
    xywh: Tensor
    ok: bool
    status: str
    max_overlap: Tensor
    ok_mask: Tensor
    displacement: Tensor
    active_pair_count: Tensor
    failure_reasons: tuple[str, ...]
    iterations: int
    directions: Tensor
    active_pairs: Tensor
    initial_pair_count: Tensor
    final_pair_count: Tensor
    component_rebuilds: Tensor
    new_pairs_detected: Tensor
    reset_count: Tensor
    beam_states_evaluated: Tensor
    max_component_size: Tensor
    component_proposal_available: Tensor
    component_proposal_xywh: Tensor
    component_proposal_hard_ok: Tensor
    component_proposal_structure_ok: Tensor
    component_proposal_final_pair_count: Tensor
    component_proposal_displacement: Tensor
    component_proposal_rollback_reason: tuple[str, ...]


def _get_field(source: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _as_boxes(boxes: Any) -> tuple[Tensor, bool]:
    raw = _get_field(boxes, ("xywh", "boxes", "rects", "placements"), boxes)
    tensor = torch.as_tensor(raw)
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != 4:
        raise ValueError("boxes must have shape [N,4] or [C,N,4]")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("boxes must be finite")
    if not bool((tensor[..., 2:4] > 0).all()):
        raise ValueError("rectangle dimensions must be positive")
    single = tensor.ndim == 2
    return (tensor.unsqueeze(0) if single else tensor).float(), single


def _as_mask(mask: Any, n: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.zeros(n, dtype=torch.bool, device=device)
    result = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(-1)
    if result.numel() != n:
        raise ValueError(f"preplaced mask length {result.numel()} does not match block count {n}")
    return result


def _as_guidance(
    guidance: ProjectionGuidance | None,
    candidates: int,
    n: int,
    device: torch.device,
) -> ProjectionGuidance | None:
    if guidance is None:
        return None
    if guidance.preferred_direction.shape != (candidates, n, n):
        raise ValueError("projection guidance must match candidate and block counts")
    return ProjectionGuidance(
        preferred_direction=guidance.preferred_direction.to(device=device),
        preferred_confidence=guidance.preferred_confidence.to(device=device),
        contact_direction=guidance.contact_direction.to(device=device),
        contact_confidence=guidance.contact_confidence.to(device=device),
        boundary_axis_lock=guidance.boundary_axis_lock.to(device=device),
    )


def _guidance_row(
    guidance: ProjectionGuidance | None,
    index: int,
) -> ProjectionGuidance | None:
    if guidance is None:
        return None
    return ProjectionGuidance(
        preferred_direction=guidance.preferred_direction[index : index + 1],
        preferred_confidence=guidance.preferred_confidence[index : index + 1],
        contact_direction=guidance.contact_direction[index : index + 1],
        contact_confidence=guidance.contact_confidence[index : index + 1],
        boundary_axis_lock=guidance.boundary_axis_lock[index : index + 1],
    )


def _guidance_subset(
    guidance: ProjectionGuidance,
    mask: Tensor,
) -> ProjectionGuidance:
    return ProjectionGuidance(
        preferred_direction=guidance.preferred_direction[mask],
        preferred_confidence=guidance.preferred_confidence[mask],
        contact_direction=guidance.contact_direction[mask],
        contact_confidence=guidance.contact_confidence[mask],
        boundary_axis_lock=guidance.boundary_axis_lock[mask],
    )


def _guided_candidate_mask(guidance: ProjectionGuidance) -> Tensor:
    return (
        (guidance.preferred_confidence > 0.0).any(dim=(1, 2))
        | (guidance.contact_confidence > 0.0).any(dim=(1, 2))
        | guidance.boundary_axis_lock.any(dim=(1, 2))
    )


def _contact_membership(guidance: ProjectionGuidance) -> Tensor:
    candidates, n = guidance.contact_confidence.shape[:2]
    rows: list[Tensor] = []
    for candidate in range(candidates):
        parent = list(range(n))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(first: int, second: int) -> None:
            left, right = find(first), find(second)
            if left != right:
                parent[right] = left

        active = torch.triu(
            guidance.contact_confidence[candidate] > 0.0,
            diagonal=1,
        ).detach().cpu()
        for first, second in torch.nonzero(
            active, as_tuple=False
        ).tolist():
            union(int(first), int(second))
        roots = torch.tensor(
            [find(node) for node in range(n)],
            dtype=torch.long,
            device=guidance.contact_confidence.device,
        )
        rows.append(roots[:, None] == roots[None, :])
    return torch.stack(rows)


def _activate_contact_guidance(
    guidance: ProjectionGuidance,
    boxes: Tensor,
    tolerance: float,
) -> ProjectionGuidance:
    work = boxes.to(dtype=torch.float64)
    first = work[:, :, None]
    second = work[:, None, :]
    direction = guidance.contact_direction
    gap = torch.stack(
        (
            (first[..., 0] + first[..., 2] - second[..., 0]).abs(),
            (second[..., 0] + second[..., 2] - first[..., 0]).abs(),
            (first[..., 1] + first[..., 3] - second[..., 1]).abs(),
            (second[..., 1] + second[..., 3] - first[..., 1]).abs(),
        ),
        dim=-1,
    ).gather(-1, direction.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    overlap_x = (
        torch.minimum(
            first[..., 0] + first[..., 2],
            second[..., 0] + second[..., 2],
        )
        - torch.maximum(first[..., 0], second[..., 0])
    ).clamp_min(0.0)
    overlap_y = (
        torch.minimum(
            first[..., 1] + first[..., 3],
            second[..., 1] + second[..., 3],
        )
        - torch.maximum(first[..., 1], second[..., 1])
    ).clamp_min(0.0)
    orthogonal = torch.where(direction < BDP_BELOW, overlap_y, overlap_x)
    active = (direction >= 0) & (gap <= tolerance) & (orthogonal > tolerance)
    return ProjectionGuidance(
        preferred_direction=guidance.preferred_direction,
        preferred_confidence=guidance.preferred_confidence,
        contact_direction=guidance.contact_direction,
        contact_confidence=guidance.contact_confidence
        * active.to(dtype=guidance.contact_confidence.dtype),
        boundary_axis_lock=guidance.boundary_axis_lock,
    )


def _merge_projection_results(
    guided_mask: Tensor,
    guided: ProjectionResult,
    neutral: ProjectionResult,
) -> ProjectionResult:
    if not torch.equal(guided.active_pairs, neutral.active_pairs):
        raise RuntimeError("split projection returned inconsistent pair indices")
    count = int(guided_mask.numel())

    def merge(guided_value: Tensor, neutral_value: Tensor) -> Tensor:
        result = torch.empty(
            (count, *guided_value.shape[1:]),
            dtype=guided_value.dtype,
            device=guided_value.device,
        )
        result[guided_mask] = guided_value
        result[~guided_mask] = neutral_value
        return result

    ok_mask = merge(guided.ok_mask, neutral.ok_mask)
    reasons = [""] * count
    guided_indices = guided_mask.nonzero(as_tuple=False).flatten().tolist()
    neutral_indices = (~guided_mask).nonzero(as_tuple=False).flatten().tolist()
    for index, reason in zip(guided_indices, guided.failure_reasons, strict=True):
        reasons[index] = reason
    for index, reason in zip(neutral_indices, neutral.failure_reasons, strict=True):
        reasons[index] = reason
    proposal_reasons = [""] * count
    for index, reason in zip(
        guided_indices, guided.component_proposal_rollback_reason, strict=True
    ):
        proposal_reasons[index] = reason
    for index, reason in zip(
        neutral_indices, neutral.component_proposal_rollback_reason, strict=True
    ):
        proposal_reasons[index] = reason
    status = (
        "ok"
        if bool(ok_mask.all().item())
        else ("partial" if bool(ok_mask.any().item()) else "infeasible")
    )
    return ProjectionResult(
        xywh=merge(guided.xywh, neutral.xywh),
        ok=bool(ok_mask.all().item()),
        status=status,
        max_overlap=merge(guided.max_overlap, neutral.max_overlap),
        ok_mask=ok_mask,
        displacement=merge(guided.displacement, neutral.displacement),
        active_pair_count=merge(
            guided.active_pair_count, neutral.active_pair_count
        ),
        failure_reasons=tuple(reasons),
        iterations=guided.iterations,
        directions=merge(guided.directions, neutral.directions),
        active_pairs=guided.active_pairs,
        initial_pair_count=merge(
            guided.initial_pair_count, neutral.initial_pair_count
        ),
        final_pair_count=merge(
            guided.final_pair_count, neutral.final_pair_count
        ),
        component_rebuilds=merge(
            guided.component_rebuilds, neutral.component_rebuilds
        ),
        new_pairs_detected=merge(
            guided.new_pairs_detected, neutral.new_pairs_detected
        ),
        reset_count=merge(guided.reset_count, neutral.reset_count),
        beam_states_evaluated=merge(
            guided.beam_states_evaluated, neutral.beam_states_evaluated
        ),
        max_component_size=merge(
            guided.max_component_size, neutral.max_component_size
        ),
        component_proposal_available=merge(
            guided.component_proposal_available,
            neutral.component_proposal_available,
        ),
        component_proposal_xywh=merge(
            guided.component_proposal_xywh,
            neutral.component_proposal_xywh,
        ),
        component_proposal_hard_ok=merge(
            guided.component_proposal_hard_ok,
            neutral.component_proposal_hard_ok,
        ),
        component_proposal_structure_ok=merge(
            guided.component_proposal_structure_ok,
            neutral.component_proposal_structure_ok,
        ),
        component_proposal_final_pair_count=merge(
            guided.component_proposal_final_pair_count,
            neutral.component_proposal_final_pair_count,
        ),
        component_proposal_displacement=merge(
            guided.component_proposal_displacement,
            neutral.component_proposal_displacement,
        ),
        component_proposal_rollback_reason=tuple(proposal_reasons),
    )


def _pair_index(n: int, device: torch.device) -> tuple[Tensor, Tensor]:
    return torch.triu_indices(n, n, offset=1, device=device)


def _overlap_extents(xywh: Tensor) -> tuple[Tensor, Tensor, bool]:
    boxes, single = _as_boxes(xywh)
    x0 = boxes[..., 0]
    y0 = boxes[..., 1]
    x1 = x0 + boxes[..., 2]
    y1 = y0 + boxes[..., 3]
    ox = (torch.minimum(x1[:, :, None], x1[:, None, :]) - torch.maximum(x0[:, :, None], x0[:, None, :])).clamp_min(0.0)
    oy = (torch.minimum(y1[:, :, None], y1[:, None, :]) - torch.maximum(y0[:, :, None], y0[:, None, :])).clamp_min(0.0)
    eye = torch.eye(boxes.shape[1], dtype=torch.bool, device=boxes.device).view(1, boxes.shape[1], boxes.shape[1])
    ox = torch.where(eye, torch.zeros_like(ox), ox)
    oy = torch.where(eye, torch.zeros_like(oy), oy)
    return ox, oy, single


def overlap_matrix(xywh: Tensor) -> Tensor:
    ox, oy, single = _overlap_extents(xywh)
    overlap = ox * oy
    return overlap[0] if single else overlap


def _active_overlap_matrix(xywh: Tensor, tolerance: float) -> Tensor:
    ox, oy, single = _overlap_extents(xywh)
    active = (ox > tolerance) & (oy > tolerance)
    return active[0] if single else active


def _active_overlap_matrix_exact(xywh: Tensor, tolerance: float) -> Tensor:
    boxes = torch.as_tensor(xywh, device=xywh.device).to(dtype=torch.float64)
    single = boxes.ndim == 2
    if single:
        boxes = boxes.unsqueeze(0)
    low = boxes[..., :2]
    high = low + boxes[..., 2:4]
    overlap = torch.minimum(high[:, :, None], high[:, None, :]) - torch.maximum(
        low[:, :, None], low[:, None, :]
    )
    active = (overlap > tolerance).all(dim=-1)
    eye = torch.eye(
        boxes.shape[1], dtype=torch.bool, device=boxes.device
    ).unsqueeze(0)
    active = active & ~eye
    return active[0] if single else active


def _component_clearance(
    work: Tensor,
    tolerance: float,
    requested: float,
) -> float:
    finfo = torch.finfo(work.dtype)
    edges = torch.cat((work[..., :2], work[..., :2] + work[..., 2:4]), dim=-1)
    magnitude = max(1.0, float(edges.detach().abs().amax().item()))
    return max(requested, 2.0 * tolerance, 8.0 * finfo.eps * magnitude)


def _direction_costs(
    boxes: Tensor,
    pair_i: Tensor,
    pair_j: Tensor,
    *,
    guidance: ProjectionGuidance | None = None,
    config: ComponentBDPConfig | None = None,
) -> Tensor:
    bi = boxes[:, pair_i]
    bj = boxes[:, pair_j]
    left = (bi[..., 0] + bi[..., 2] - bj[..., 0]).clamp_min(0.0)
    right = (bj[..., 0] + bj[..., 2] - bi[..., 0]).clamp_min(0.0)
    below = (bi[..., 1] + bi[..., 3] - bj[..., 1]).clamp_min(0.0)
    above = (bj[..., 1] + bj[..., 3] - bi[..., 1]).clamp_min(0.0)
    costs = torch.stack((left, right, below, above), dim=-1)
    if guidance is None:
        return costs
    if config is None:
        raise ValueError("component guidance requires a component config")

    directions = torch.arange(4, device=boxes.device).view(1, 1, 4)
    pair_scale = 0.25 * (
        bi[..., 2] + bi[..., 3] + bj[..., 2] + bj[..., 3]
    )
    preferred = guidance.preferred_direction[:, pair_i, pair_j]
    preferred_confidence = guidance.preferred_confidence[:, pair_i, pair_j]
    preferred_penalty = (
        (directions != preferred.unsqueeze(-1))
        & (preferred.unsqueeze(-1) >= 0)
    ).to(dtype=costs.dtype)
    costs = costs + (
        config.topology_weight
        * pair_scale.unsqueeze(-1)
        * preferred_confidence.unsqueeze(-1)
        * preferred_penalty
    )

    contact = guidance.contact_direction[:, pair_i, pair_j]
    contact_confidence = guidance.contact_confidence[:, pair_i, pair_j]
    contact_penalty = (
        (directions != contact.unsqueeze(-1)) & (contact.unsqueeze(-1) >= 0)
    ).to(dtype=costs.dtype)
    costs = costs + (
        config.contact_weight
        * pair_scale.unsqueeze(-1)
        * contact_confidence.unsqueeze(-1)
        * contact_penalty
    )

    lock_i = guidance.boundary_axis_lock[:, pair_i]
    lock_j = guidance.boundary_axis_lock[:, pair_j]
    locked_axis = torch.stack(
        (
            lock_i[..., 0] & lock_j[..., 0],
            lock_i[..., 0] & lock_j[..., 0],
            lock_i[..., 1] & lock_j[..., 1],
            lock_i[..., 1] & lock_j[..., 1],
        ),
        dim=-1,
    )
    return costs + (
        config.boundary_weight
        * pair_scale.unsqueeze(-1)
        * locked_axis.to(dtype=costs.dtype)
    )


def _direction_order(
    boxes: Tensor,
    pair_i: Tensor,
    pair_j: Tensor,
    beam_variant: int,
    *,
    guidance: ProjectionGuidance | None = None,
    config: ComponentBDPConfig | None = None,
) -> Tensor:
    costs = _direction_costs(
        boxes,
        pair_i,
        pair_j,
        guidance=guidance,
        config=config,
    )
    order = costs.argsort(dim=-1, stable=True)
    pick = min(max(beam_variant, 0), 3)
    return order[..., pick]


def assign_directions(
    boxes: Any,
    *,
    beam_variant: int = 0,
    active_pairs: Tensor | None = None,
    tolerance: float = 0.0,
    guidance: ProjectionGuidance | None = None,
    component_config: ComponentBDPConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Return dense active overlap pairs and L/R/T/B direction ids 0..3."""

    work, _ = _as_boxes(boxes)
    c, n = work.shape[:2]
    if active_pairs is None:
        pair_i, pair_j = _pair_index(n, work.device)
        active_matrix = (
            _active_overlap_matrix_exact(work, tolerance)
            if component_config is not None and component_config.enabled
            else _active_overlap_matrix(work, tolerance)
        )
        active = active_matrix[:, pair_i, pair_j]
    else:
        active_pairs = active_pairs.to(device=work.device, dtype=torch.long)
        pair_i, pair_j = active_pairs[:, 0], active_pairs[:, 1]
        active = torch.ones((c, active_pairs.shape[0]), dtype=torch.bool, device=work.device)
    directions = _direction_order(
        work,
        pair_i,
        pair_j,
        beam_variant,
        guidance=guidance,
        config=component_config,
    )
    pairs = torch.stack((pair_i, pair_j), dim=1)
    return pairs, torch.where(active, directions, torch.full_like(directions, -1))


def _project_fixed_iterations(
    boxes: Tensor,
    pairs: Tensor,
    directions: Tensor,
    preplaced_mask: Tensor,
    *,
    iterations: int,
    clearance: float,
    axis_lock: Tensor | None = None,
    rigid_membership: Tensor | None = None,
) -> tuple[Tensor, bool]:
    c, n = boxes.shape[:2]
    if pairs.numel() == 0:
        return boxes, torch.zeros(c, dtype=torch.bool, device=boxes.device)
    pair_i = pairs[:, 0]
    pair_j = pairs[:, 1]
    preplaced_i = preplaced_mask[pair_i].view(1, -1)
    preplaced_j = preplaced_mask[pair_j].view(1, -1)
    active = directions >= 0
    if axis_lock is None:
        axis_lock = torch.zeros((c, n, 2), dtype=torch.bool, device=boxes.device)
    if axis_lock.shape != (c, n, 2):
        raise ValueError("axis_lock must have shape [C,N,2]")
    if rigid_membership is None:
        rigid_membership = torch.eye(
            n, dtype=torch.bool, device=boxes.device
        ).unsqueeze(0).expand(c, -1, -1)
    if rigid_membership.shape != (c, n, n):
        raise ValueError("rigid_membership must have shape [C,N,N]")
    base_fixed = axis_lock | preplaced_mask.view(1, n, 1)
    grouped_fixed = (
        rigid_membership.unsqueeze(-1) & base_fixed[:, None]
    ).any(dim=2)
    axis = (directions.clamp_min(0) >= BDP_BELOW).to(dtype=torch.long)
    fixed_i = preplaced_i | grouped_fixed[:, pair_i].gather(
        -1, axis.unsqueeze(-1)
    ).squeeze(-1)
    fixed_j = preplaced_j | grouped_fixed[:, pair_j].gather(
        -1, axis.unsqueeze(-1)
    ).squeeze(-1)
    infeasible = (active & fixed_i & fixed_j).any(dim=1)
    pos = boxes[..., :2].clone()
    wh = boxes[..., 2:4]
    for _ in range(iterations):
        pi = pos[:, pair_i]
        pj = pos[:, pair_j]
        wi = wh[:, pair_i]
        wj = wh[:, pair_j]
        amount = torch.stack(
            (
                pi[..., 0] + wi[..., 0] - pj[..., 0] + clearance,
                pj[..., 0] + wj[..., 0] - pi[..., 0] + clearance,
                pi[..., 1] + wi[..., 1] - pj[..., 1] + clearance,
                pj[..., 1] + wj[..., 1] - pi[..., 1] + clearance,
            ),
            dim=-1,
        )
        d = directions.clamp_min(0)
        violation = amount.gather(-1, d.unsqueeze(-1)).squeeze(-1).clamp_min(0.0) * active
        both_free = (~fixed_i & ~fixed_j).float()
        i_free = (~fixed_i & fixed_j).float()
        j_free = (fixed_i & ~fixed_j).float()
        share_i = 0.5 * both_free + i_free
        share_j = 0.5 * both_free + j_free
        sign_i = torch.stack(
            (
                torch.where((d == 0) | (d == 1), torch.where(d == 0, -1.0, 1.0), 0.0),
                torch.where((d == 2) | (d == 3), torch.where(d == 2, -1.0, 1.0), 0.0),
            ),
            dim=-1,
        )
        delta_i = sign_i * violation.unsqueeze(-1) * share_i.unsqueeze(-1)
        delta_j = -sign_i * violation.unsqueeze(-1) * share_j.unsqueeze(-1)
        updates = torch.zeros((c, n, 2), dtype=boxes.dtype, device=boxes.device)
        updates.scatter_add_(1, pair_i.view(1, -1, 1).expand(c, -1, 2), delta_i)
        updates.scatter_add_(1, pair_j.view(1, -1, 1).expand(c, -1, 2), delta_j)
        updates = torch.bmm(
            rigid_membership.to(dtype=updates.dtype), updates
        )
        updates = torch.where(grouped_fixed, torch.zeros_like(updates), updates)
        pos = pos + updates
        pos = torch.where(preplaced_mask.view(1, n, 1), boxes[..., :2], pos)
    return torch.cat((pos, wh), dim=-1), infeasible


def _verified_status(
    boxes: Tensor,
    original: Tensor,
    preplaced_mask: Tensor,
    tolerance: float,
    fixed_pair_overlap: Tensor,
    ignore_preplaced_pairs: bool,
) -> tuple[Tensor, tuple[str, ...], Tensor]:
    overlap = overlap_matrix(boxes)
    active = _active_overlap_matrix_exact(boxes, tolerance)
    if ignore_preplaced_pairs:
        exempt = preplaced_mask[:, None] & preplaced_mask[None, :]
        overlap = torch.where(exempt.unsqueeze(0), torch.zeros_like(overlap), overlap)
        active = torch.where(exempt.unsqueeze(0), torch.zeros_like(active), active)
    max_overlap = overlap.amax(dim=(1, 2))
    dims_ok = (boxes[..., 2:4] > 0).all(dim=(1, 2))
    if bool(preplaced_mask.any()):
        preplaced_ok = torch.isclose(boxes[:, preplaced_mask, :2], original[:, preplaced_mask, :2]).all(dim=(1, 2))
    else:
        preplaced_ok = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
    residual_ok = ~active.any(dim=(1, 2))
    ok = dims_ok & preplaced_ok & residual_ok & ~fixed_pair_overlap
    reasons = []
    for i in range(boxes.shape[0]):
        if bool(ok[i]):
            reasons.append("ok")
        elif bool(fixed_pair_overlap[i]):
            reasons.append("fixed_pair_overlap")
        elif not bool(dims_ok[i] and preplaced_ok[i]):
            reasons.append("invalid_geometry")
        else:
            reasons.append("residual_overlap")
    return ok, tuple(reasons), max_overlap


def _largest_component_size(pairs: Tensor, directions: Tensor, n: int) -> Tensor:
    """Return the largest active conflict component per candidate."""

    if pairs.numel() == 0:
        return torch.zeros(directions.shape[0], dtype=torch.long, device=directions.device)
    out = []
    active_cpu = (directions >= 0).detach().cpu()
    pairs_cpu = pairs.detach().cpu().tolist()
    for row in active_cpu:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        active_nodes: set[int] = set()
        for is_active, (i, j) in zip(row.tolist(), pairs_cpu, strict=True):
            if is_active:
                union(i, j)
                active_nodes.add(i)
                active_nodes.add(j)
        counts: dict[int, int] = {}
        for node in active_nodes:
            root = find(node)
            counts[root] = counts.get(root, 0) + 1
        out.append(max(counts.values(), default=0))
    return torch.tensor(out, dtype=torch.long, device=directions.device)


def _active_components(pairs: Tensor, directions: Tensor, n: int) -> tuple[list[list[int]], int]:
    active_indices = [idx for idx, direction in enumerate(directions.tolist()) if direction >= 0]
    if not active_indices:
        return [], 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs_cpu = pairs.detach().cpu().tolist()
    active_nodes: set[int] = set()
    for idx in active_indices:
        i, j = pairs_cpu[idx]
        union(i, j)
        active_nodes.add(i)
        active_nodes.add(j)
    by_root: dict[int, list[int]] = {}
    component_nodes: dict[int, set[int]] = {}
    for idx in active_indices:
        i, j = pairs_cpu[idx]
        root = find(i)
        by_root.setdefault(root, []).append(idx)
        component_nodes.setdefault(root, set()).update((i, j))
    max_size = max((len(nodes) for nodes in component_nodes.values()), default=0)
    return [sorted(items) for _, items in sorted(by_root.items())], max_size


def _has_cycle(n: int, edges: list[tuple[int, int]]) -> bool:
    graph = [[] for _ in range(n)]
    for src, dst in edges:
        graph[src].append(dst)
    color = [0] * n

    def visit(node: int) -> bool:
        color[node] = 1
        for nxt in graph[node]:
            if color[nxt] == 1:
                return True
            if color[nxt] == 0 and visit(nxt):
                return True
        color[node] = 2
        return False

    return any(color[node] == 0 and visit(node) for node in range(n))


def _cycle_free_values(
    pairs: list[list[int]],
    directions: list[int] | tuple[int, ...],
    n: int,
) -> bool:
    h_edges: list[tuple[int, int]] = []
    v_edges: list[tuple[int, int]] = []
    for (i, j), direction in zip(pairs, directions, strict=True):
        if direction == BDP_LEFT:
            h_edges.append((i, j))
        elif direction == BDP_RIGHT:
            h_edges.append((j, i))
        elif direction == BDP_BELOW:
            v_edges.append((i, j))
        elif direction == BDP_ABOVE:
            v_edges.append((j, i))
    return not _has_cycle(n, h_edges) and not _has_cycle(n, v_edges)


def _cycle_free(pairs: Tensor, directions: Tensor, n: int) -> bool:
    return _cycle_free_values(
        pairs.detach().cpu().tolist(),
        directions.detach().cpu().tolist(),
        n,
    )


def _cycle_free_base(
    pairs: Tensor,
    directions: Tensor,
    costs: Tensor,
    n: int,
) -> Tensor:
    """Repair independent minimum-cost choices into a deterministic DAG row."""

    order = costs.argsort(dim=-1, stable=True)
    active = [
        index
        for index, direction in enumerate(directions.detach().cpu().tolist())
        if direction >= 0
    ]
    certainty = {
        index: float(costs[index, order[index, 1]] - costs[index, order[index, 0]])
        for index in active
    }
    pairs_cpu = pairs.detach().cpu().tolist()
    result = [-1] * int(directions.numel())
    for index in sorted(active, key=lambda item: (-certainty[item], item)):
        accepted = False
        for rank in range(4):
            result[index] = int(order[index, rank])
            if _cycle_free_values(pairs_cpu, result, n):
                accepted = True
                break
        if not accepted:
            raise RuntimeError("no cycle-free disjunctive direction is available")
    return torch.tensor(
        result,
        dtype=directions.dtype,
        device=directions.device,
    )


def _branch_direction_rows(
    boxes: Tensor,
    pairs: Tensor,
    directions: Tensor,
    config: ComponentBDPConfig,
    guidance: ProjectionGuidance | None = None,
) -> Tensor:
    """Enumerate independent component beams, then merge them deterministically."""

    pair_i, pair_j = pairs[:, 0], pairs[:, 1]
    costs = _direction_costs(
        boxes.unsqueeze(0),
        pair_i,
        pair_j,
        guidance=guidance,
        config=config,
    )[0].detach().cpu()
    order = costs.argsort(dim=-1, stable=True)
    components, _ = _active_components(pairs, directions, boxes.shape[0])
    pairs_cpu = pairs.detach().cpu().tolist()
    directions = _cycle_free_base(
        pairs,
        directions,
        costs,
        boxes.shape[0],
    )
    base = tuple(int(value) for value in directions.detach().cpu().tolist())
    component_beams: list[tuple[list[int], list[tuple[float, tuple[int, ...]]]]] = []
    for component in components:
        nodes = {node for idx in component for node in pairs_cpu[idx]}
        if len(nodes) > config.component_limit:
            continue
        sortable = []
        for idx in component:
            margin = float(costs[idx, order[idx, 1]] - costs[idx, order[idx, 0]])
            sortable.append((margin, idx))
        local_rows: list[tuple[float, tuple[int, ...]]] = [(0.0, base)]
        for _, pair_index in sorted(sortable)[: config.max_uncertain_pairs]:
            expanded: dict[tuple[int, ...], float] = {}
            for score, row in local_rows:
                current = row[pair_index]
                current_cost = float(costs[pair_index, max(current, 0)])
                for rank in range(4):
                    direction = int(order[pair_index, rank])
                    proposal = list(row)
                    proposal[pair_index] = direction
                    proposal_tuple = tuple(proposal)
                    if not _cycle_free_values(
                        pairs_cpu,
                        proposal_tuple,
                        boxes.shape[0],
                    ):
                        continue
                    proposal_score = score + max(
                        0.0,
                        float(costs[pair_index, direction]) - current_cost,
                    )
                    expanded[proposal_tuple] = min(
                        proposal_score,
                        expanded.get(proposal_tuple, float("inf")),
                    )
            if expanded:
                local_rows = sorted(
                    ((score, row) for row, score in expanded.items()),
                    key=lambda item: (item[0], item[1]),
                )[: config.beam_width]
        component_beams.append((component, local_rows))

    beam_rows: list[tuple[float, tuple[int, ...]]] = [(0.0, base)]
    for component, local_rows in component_beams:
        merged: dict[tuple[int, ...], float] = {}
        for combined_score, combined_row in beam_rows:
            for local_score, local_row in local_rows:
                proposal = list(combined_row)
                for pair_index in component:
                    proposal[pair_index] = local_row[pair_index]
                proposal_tuple = tuple(proposal)
                if not _cycle_free_values(
                    pairs_cpu,
                    proposal_tuple,
                    boxes.shape[0],
                ):
                    continue
                score = combined_score + local_score
                merged[proposal_tuple] = min(
                    score,
                    merged.get(proposal_tuple, float("inf")),
                )
        if merged:
            beam_rows = sorted(
                ((score, row) for row, score in merged.items()),
                key=lambda item: (item[0], item[1]),
            )[: config.beam_width]

    return torch.stack(
        [
            torch.tensor(row, dtype=directions.dtype, device=directions.device)
            for _, row in beam_rows
        ]
    )


def _pareto_dominance_count(values: Tensor) -> Tensor:
    """Count how many same-candidate branches dominate each branch."""

    source = values[:, :, None, :]
    target = values[:, None, :, :]
    dominates = (source <= target).all(dim=-1) & (source < target).any(dim=-1)
    return dominates.sum(dim=1)


def _official_boundary_missing(
    boxes: Tensor,
    problem: Any | None,
    tolerance: float,
) -> Tensor | None:
    boundary_bits = _get_field(problem, ("boundary_bits",))
    if boundary_bits is None:
        return None
    required = torch.as_tensor(
        boundary_bits,
        dtype=torch.bool,
        device=boxes.device,
    )
    if required.shape != (boxes.shape[-2], 4):
        return None
    work = boxes.to(dtype=torch.float64)
    left = work[..., 0]
    bottom = work[..., 1]
    right = left + work[..., 2]
    top = bottom + work[..., 3]
    membership = torch.stack(
        (
            left <= left.amin(dim=-1, keepdim=True) + tolerance,
            right >= right.amax(dim=-1, keepdim=True) - tolerance,
            top >= top.amax(dim=-1, keepdim=True) - tolerance,
            bottom <= bottom.amin(dim=-1, keepdim=True) + tolerance,
        ),
        dim=-1,
    )
    return (required.view(*(1,) * (boxes.ndim - 2), *required.shape) & ~membership).sum(
        dim=(-2, -1)
    )


def _quality_metrics(
    branch_boxes: Tensor,
    original: Tensor,
    problem: Any | None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    required = ("area", "b2b_weight", "p2b_edges", "pins")
    if any(_get_field(problem, (name,)) is None for name in required):
        return None
    c, beam, n = branch_boxes.shape[:3]
    flat = branch_boxes.reshape(c * beam, n, 4)
    try:
        hpwl = hpwl_tensor(problem, centers_from_xywh(flat)).reshape(c, beam)
        bbox = bbox_area_tensor(flat).reshape(c, beam)
        reference_hpwl = hpwl_tensor(
            problem,
            centers_from_xywh(original),
        ).reshape(c, 1)
        reference_bbox = bbox_area_tensor(original).reshape(c, 1)
    except (AttributeError, TypeError, ValueError):
        return None
    eps = torch.finfo(hpwl.dtype).eps
    hpwl_ratio = hpwl / reference_hpwl.clamp_min(eps)
    bbox_ratio = bbox / reference_bbox.clamp_min(eps)
    official_local_q = (hpwl_ratio - 1.0).clamp_min(0.0) + (
        bbox_ratio - 1.0
    ).clamp_min(0.0)
    quality = torch.stack((hpwl_ratio, bbox_ratio), dim=-1)
    return _pareto_dominance_count(quality), official_local_q, quality.sum(dim=-1)


def _structure_nonregression(
    branch_boxes: Tensor,
    original: Tensor,
    guidance: ProjectionGuidance | None,
    problem: Any | None,
    tolerance: float,
) -> Tensor:
    c, beam = branch_boxes.shape[:2]
    boundary_missing = _official_boundary_missing(
        branch_boxes,
        problem,
        tolerance,
    )
    if boundary_missing is None:
        boundary_ok = (
            torch.ones((c, beam), dtype=torch.bool, device=branch_boxes.device)
            if guidance is None
            else _boundary_regression(
                branch_boxes,
                original,
                guidance.boundary_axis_lock,
                tolerance,
            )
            == 0
        )
    else:
        reference_missing = _official_boundary_missing(
            original,
            problem,
            tolerance,
        )
        assert reference_missing is not None
        boundary_ok = boundary_missing <= reference_missing.reshape(c, 1)
    if guidance is None:
        contact_ok = torch.ones_like(boundary_ok)
    else:
        contact_ok = _contact_residual(
            branch_boxes,
            guidance,
            tolerance,
        ) <= _contact_residual(
            original[:, None],
            guidance,
            tolerance,
        ) + tolerance
    return boundary_ok & contact_ok


def _best_branch_order(
    branch_boxes: Tensor,
    original: Tensor,
    mask: Tensor,
    fixed_pair_overlap: Tensor,
    tolerance: float,
    ignore_preplaced_pairs: bool,
    guidance: ProjectionGuidance | None,
    problem: Any | None,
) -> Tensor:
    """Return deterministic lexicographic branch order for each candidate."""

    c, beam, n = branch_boxes.shape[:3]
    flat = branch_boxes.reshape(c * beam, n, 4)
    final_active = _active_overlap_matrix_exact(flat, tolerance)
    if ignore_preplaced_pairs:
        exempt = mask[:, None] & mask[None, :]
        final_active = torch.where(exempt.unsqueeze(0), torch.zeros_like(final_active), final_active)
    conflict_count = final_active.sum(dim=(1, 2)).reshape(c, beam)
    max_overlap = overlap_matrix(flat).amax(dim=(1, 2)).reshape(c, beam)
    displacement = torch.linalg.vector_norm(
        branch_boxes[..., :2] - original[:, None, :, :2], dim=-1
    ).sum(dim=2)
    boundary_regression = (
        torch.zeros_like(displacement, dtype=torch.long)
        if guidance is None
        else _boundary_regression(
            branch_boxes,
            original,
            guidance.boundary_axis_lock,
            tolerance,
        )
    )
    contact_regression = (
        torch.zeros_like(displacement)
        if guidance is None
        else (
            _contact_residual(branch_boxes, guidance, tolerance)
            - _contact_residual(
                original[:, None], guidance, tolerance
            )
        ).clamp_min(0.0)
    )

    order = torch.arange(beam, device=branch_boxes.device).view(1, beam).expand(c, -1)
    quality = _quality_metrics(branch_boxes, original, problem)
    official_boundary = _official_boundary_missing(
        branch_boxes,
        problem,
        tolerance,
    )
    if quality is None and official_boundary is None:
        # Keep the standalone projector's historical ordering when no case
        # objective is available.  The official-aware path is only used by the
        # HCFP runtime, which always passes a complete FloorplanCase.
        keys = (
            displacement,
            max_overlap,
            contact_regression,
            boundary_regression,
            conflict_count,
            fixed_pair_overlap.to(dtype=torch.long),
        )
    else:
        if official_boundary is None:
            boundary_missing = boundary_regression
        else:
            boundary_missing = official_boundary
        if guidance is None:
            contact = torch.zeros_like(displacement)
        else:
            contact = _contact_residual(branch_boxes, guidance, tolerance)
        construction = torch.stack(
            (boundary_missing.to(dtype=torch.float64), contact),
            dim=-1,
        )
        construction_rank = _pareto_dominance_count(construction)
        construction_regression = ~_structure_nonregression(
            branch_boxes,
            original,
            guidance,
            problem,
            tolerance,
        )
        if quality is None:
            quality_rank = torch.zeros_like(conflict_count)
            official_local_q = torch.zeros_like(displacement)
            quality_sum = torch.zeros_like(displacement)
        else:
            quality_rank, official_local_q, quality_sum = quality
        hard_infeasible = conflict_count > 0
        feasible_construction_regression = construction_regression & ~hard_infeasible
        keys = (
            displacement,
            quality_sum,
            official_local_q,
            quality_rank,
            construction_rank,
            max_overlap,
            conflict_count,
            feasible_construction_regression.to(dtype=torch.long),
            hard_infeasible.to(dtype=torch.long),
            fixed_pair_overlap.to(dtype=torch.long),
        )
    # Stable sorts are applied from least to most significant key.
    for key in keys:
        values = key.gather(1, order)
        order = order.gather(1, values.argsort(dim=1, stable=True))
    return order


def _boundary_regression(
    boxes: Tensor,
    original: Tensor,
    axis_lock: Tensor,
    tolerance: float,
) -> Tensor:
    work = boxes.to(dtype=torch.float64)
    reference = original.to(dtype=torch.float64)

    def membership(value: Tensor) -> Tensor:
        left = value[..., 0]
        bottom = value[..., 1]
        right = left + value[..., 2]
        top = bottom + value[..., 3]
        return torch.stack(
            (
                left <= left.amin(dim=-1, keepdim=True) + tolerance,
                right >= right.amax(dim=-1, keepdim=True) - tolerance,
                top >= top.amax(dim=-1, keepdim=True) - tolerance,
                bottom <= bottom.amin(dim=-1, keepdim=True) + tolerance,
            ),
            dim=-1,
        )

    protected_axis = torch.stack(
        (
            axis_lock[..., 0],
            axis_lock[..., 0],
            axis_lock[..., 1],
            axis_lock[..., 1],
        ),
        dim=-1,
    )
    protected = membership(reference) & protected_axis
    return (protected[:, None] & ~membership(work)).sum(dim=(2, 3))


def _contact_residual(
    boxes: Tensor,
    guidance: ProjectionGuidance,
    tolerance: float,
) -> Tensor:
    work = boxes.to(dtype=torch.float64)
    c, _, n = work.shape[:3]
    pair_i, pair_j = _pair_index(n, work.device)
    bi = work[:, :, pair_i]
    bj = work[:, :, pair_j]
    direction = guidance.contact_direction[:, pair_i, pair_j]
    confidence = guidance.contact_confidence[:, pair_i, pair_j].to(
        dtype=torch.float64
    )
    gap = torch.stack(
        (
            (bi[..., 0] + bi[..., 2] - bj[..., 0]).abs(),
            (bj[..., 0] + bj[..., 2] - bi[..., 0]).abs(),
            (bi[..., 1] + bi[..., 3] - bj[..., 1]).abs(),
            (bj[..., 1] + bj[..., 3] - bi[..., 1]).abs(),
        ),
        dim=-1,
    ).gather(-1, direction.clamp_min(0)[:, None, :, None].expand(c, work.shape[1], -1, 1)).squeeze(-1)
    overlap_x = (
        torch.minimum(
            bi[..., 0] + bi[..., 2], bj[..., 0] + bj[..., 2]
        )
        - torch.maximum(bi[..., 0], bj[..., 0])
    ).clamp_min(0.0)
    overlap_y = (
        torch.minimum(
            bi[..., 1] + bi[..., 3], bj[..., 1] + bj[..., 3]
        )
        - torch.maximum(bi[..., 1], bj[..., 1])
    ).clamp_min(0.0)
    horizontal = direction < BDP_BELOW
    orthogonal = torch.where(
        horizontal[:, None], overlap_y, overlap_x
    )
    active = direction >= 0
    residual = gap + (tolerance - orthogonal).clamp_min(0.0)
    return (
        residual
        * confidence[:, None]
        * active[:, None].to(dtype=torch.float64)
    ).sum(dim=2)


def _project_component_mode(
    work: Tensor,
    pairs: Tensor,
    directions: Tensor,
    mask: Tensor,
    *,
    iterations: int,
    clearance: float,
    tolerance: float,
    ignore_preplaced_pairs: bool,
    config: ComponentBDPConfig,
    guidance: ProjectionGuidance | None,
    problem: Any | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    c, n = work.shape[:2]
    candidate = work.clone()
    final_dirs = directions.clone()
    final_active_count = (directions >= 0).sum(dim=1)
    fixed_pair_overlap = torch.zeros(c, dtype=torch.bool, device=work.device)
    component_rebuilds = torch.zeros(c, dtype=torch.long, device=work.device)
    new_pairs_detected = torch.zeros(c, dtype=torch.long, device=work.device)
    reset_count = torch.zeros(c, dtype=torch.long, device=work.device)
    beam_states_evaluated = torch.zeros(c, dtype=torch.long, device=work.device)
    max_component_size = _largest_component_size(pairs, directions, n)
    guided_candidates = (
        torch.ones(c, dtype=torch.bool, device=work.device)
        if guidance is None
        else _guided_candidate_mask(guidance)
    )
    active_contact_guidance = (
        None
        if guidance is None
        else _activate_contact_guidance(guidance, work, clearance)
    )
    rigid_membership = (
        None
        if active_contact_guidance is None
        else _contact_membership(active_contact_guidance)
    )
    initial_sets = [
        {
            index
            for index, direction in enumerate(row)
            if direction >= 0
        }
        for row in directions.detach().cpu().tolist()
    ]
    seen_pairs = [set(active) for active in initial_sets]
    seen_signatures: list[set[tuple[int, ...]]] = [set() for _ in range(c)]
    for _ in range(config.outer_sweeps):
        pairs, current_dirs = assign_directions(
            candidate,
            beam_variant=0,
            tolerance=tolerance,
            guidance=guidance,
            component_config=config,
        )
        current_dirs[~guided_candidates] = -1
        if ignore_preplaced_pairs:
            exempt = mask[pairs[:, 0]] & mask[pairs[:, 1]]
            current_dirs[:, exempt] = -1
        active_count = (current_dirs >= 0).sum(dim=1)
        if not bool((active_count > 0).any().item()):
            final_active_count = active_count
            break
        branch_rows: list[Tensor] = []
        branch_counts: list[int] = []
        repeated = torch.zeros(c, dtype=torch.bool, device=work.device)
        for row_idx in range(c):
            if int(active_count[row_idx].item()) == 0:
                branch = current_dirs[row_idx].unsqueeze(0)
                branch_rows.append(branch)
                branch_counts.append(1)
                continue
            active_set = {
                idx
                for idx, direction in enumerate(
                    current_dirs[row_idx].detach().cpu().tolist()
                )
                if direction >= 0
            }
            new_pairs_detected[row_idx] += len(active_set - seen_pairs[row_idx])
            seen_pairs[row_idx].update(active_set)
            component_rebuilds[row_idx] += 1
            signature = tuple(sorted(active_set))
            repeated[row_idx] = signature in seen_signatures[row_idx]
            seen_signatures[row_idx].add(signature)
            _, component_size = _active_components(
                pairs, current_dirs[row_idx], n
            )
            max_component_size[row_idx] = max(
                int(max_component_size[row_idx]), component_size
            )
            branch = _branch_direction_rows(
                candidate[row_idx],
                pairs,
                current_dirs[row_idx],
                config,
                _guidance_row(guidance, row_idx),
            )
            branch_rows.append(branch)
            branch_counts.append(int(branch.shape[0]))

        beam_width = max(branch_counts)
        padded = []
        for branch in branch_rows:
            if branch.shape[0] < beam_width:
                branch = torch.cat(
                    (
                        branch,
                        branch[:1].expand(beam_width - branch.shape[0], -1),
                    ),
                    dim=0,
                )
            padded.append(branch)
        branch_dirs = torch.stack(padded)
        flat_dirs = branch_dirs.reshape(c * beam_width, -1)
        flat_boxes = (
            candidate[:, None]
            .expand(c, beam_width, n, 4)
            .reshape(c * beam_width, n, 4)
            .clone()
        )
        projected, fixed_now = _project_fixed_iterations(
            flat_boxes,
            pairs,
            flat_dirs,
            mask,
            iterations=iterations,
            clearance=clearance,
            axis_lock=(
                None
                if guidance is None
                else guidance.boundary_axis_lock[:, None]
                .expand(c, beam_width, n, 2)
                .reshape(c * beam_width, n, 2)
            ),
            rigid_membership=(
                None
                if rigid_membership is None
                else rigid_membership[:, None]
                .expand(c, beam_width, n, n)
                .reshape(c * beam_width, n, n)
            ),
        )
        projected = projected.reshape(c, beam_width, n, 4)
        fixed_now = fixed_now.reshape(c, beam_width)
        order = _best_branch_order(
            projected,
            work,
            mask,
            fixed_now,
            tolerance,
            ignore_preplaced_pairs,
            guidance,
            problem,
        )
        selected = order[:, 0].clone()
        branch_count_tensor = torch.tensor(
            branch_counts, dtype=torch.long, device=work.device
        )
        can_reset = (
            repeated
            & (reset_count < config.reset_limit)
            & (branch_count_tensor > 1)
        )
        if beam_width > 1:
            construction_ok = _structure_nonregression(
                projected,
                work,
                guidance,
                problem,
                tolerance,
            )
            first_construction = construction_ok.gather(
                1, order[:, :1]
            ).squeeze(1)
            second_construction = construction_ok.gather(
                1, order[:, 1:2]
            ).squeeze(1)
            can_reset &= first_construction == second_construction
            selected = torch.where(can_reset, order[:, 1], selected)
        else:
            can_reset.zero_()
        reset_count += can_reset.to(dtype=torch.long)
        row = torch.arange(c, device=work.device)
        candidate = projected[row, selected]
        final_dirs = branch_dirs[row, selected]
        final_active_count = active_count
        fixed_pair_overlap |= fixed_now[row, selected]
        beam_states_evaluated += branch_count_tensor
    return (
        candidate,
        final_dirs,
        fixed_pair_overlap,
        final_active_count,
        component_rebuilds,
        new_pairs_detected,
        reset_count,
        beam_states_evaluated,
        max_component_size,
    )


def project_disjunctive(
    boxes: Any,
    problem: Any | None = None,
    *,
    preplaced_mask: Any | None = None,
    iterations: int = 16,
    beam: int = 1,
    outer_iterations: int = 3,
    clearance: float = 0.0,
    tolerance: float | None = None,
    component_config: ComponentBDPConfig | None = None,
    guidance: ProjectionGuidance | None = None,
) -> ProjectionResult:
    """Project overlapped ``xywh`` boxes onto fixed pairwise disjunctions.

    The result is fail-closed: callers must check ``ok``/``status`` before using
    ``xywh`` as legal geometry.
    """

    work, single = _as_boxes(boxes)
    if tolerance is None:
        scale = float(_get_field(problem, ("scale",), 1.0))
        normalized = bool(_get_field(problem, ("normalized",), False))
        tolerance = 1.0e-6 / max(scale, 1.0e-30) if normalized else 1.0e-6
    legacy_clearance = max(clearance, 2.0 * tolerance)
    clearance = legacy_clearance
    component_config = component_config or ComponentBDPConfig()
    component_enabled = component_config.enabled
    mask = _as_mask(preplaced_mask if preplaced_mask is not None else _get_field(problem, ("preplaced_mask", "is_preplaced")), work.shape[1], work.device)
    guidance = _as_guidance(
        guidance,
        work.shape[0],
        work.shape[1],
        work.device,
    )
    ignore_preplaced_pairs = bool(_get_field(problem, ("raw_preplaced_validated",), False))
    if component_enabled:
        clearance = _component_clearance(work, tolerance, clearance)
    if component_enabled and guidance is not None:
        guided_candidates = _guided_candidate_mask(guidance)
        if not bool(guided_candidates.any().item()):
            component_enabled = False
            guidance = None
            clearance = legacy_clearance
        elif not bool(guided_candidates.all().item()):
            common = {
                "problem": problem,
                "preplaced_mask": mask,
                "iterations": iterations,
                "beam": beam,
                "outer_iterations": outer_iterations,
                "tolerance": tolerance,
            }
            component = project_disjunctive(
                work[guided_candidates],
                **common,
                clearance=clearance,
                component_config=component_config,
                guidance=_guidance_subset(guidance, guided_candidates),
            )
            neutral = project_disjunctive(
                work[~guided_candidates],
                **common,
                clearance=legacy_clearance,
                component_config=ComponentBDPConfig(),
            )
            return _merge_projection_results(
                guided_candidates,
                component,
                neutral,
            )
    pairs, directions = assign_directions(
        work,
        beam_variant=0,
        tolerance=tolerance,
        guidance=guidance if component_enabled else None,
        component_config=component_config if component_enabled else None,
    )
    if ignore_preplaced_pairs:
        exempt = mask[pairs[:, 0]] & mask[pairs[:, 1]]
        directions[:, exempt] = -1
    active_count = (directions >= 0).sum(dim=1)
    initial_pair_count = active_count.clone()
    max_component_size = (
        _largest_component_size(pairs, directions, work.shape[1])
        if component_enabled
        else torch.zeros_like(active_count)
    )
    best_xywh = work
    best_dirs = directions
    initial_overlap = overlap_matrix(work)
    if ignore_preplaced_pairs:
        exempt = mask[:, None] & mask[None, :]
        initial_overlap = torch.where(exempt.unsqueeze(0), torch.zeros_like(initial_overlap), initial_overlap)
    best_max = initial_overlap.amax(dim=(1, 2))
    fixed_pair = mask[pairs[:, 0]] & mask[pairs[:, 1]]
    best_fixed_pair_overlap = ((directions >= 0) & fixed_pair.unsqueeze(0)).any(dim=1)
    original_ok, _, best_max = _verified_status(
        work,
        work,
        mask,
        tolerance,
        best_fixed_pair_overlap,
        ignore_preplaced_pairs,
    )
    best_ok = original_ok if component_enabled else torch.zeros_like(original_ok)
    best_active_count = active_count
    component_rebuilds = torch.zeros(work.shape[0], dtype=torch.long, device=work.device)
    new_pairs_detected = torch.zeros(work.shape[0], dtype=torch.long, device=work.device)
    reset_count = torch.zeros(work.shape[0], dtype=torch.long, device=work.device)
    beam_states_evaluated = torch.zeros(work.shape[0], dtype=torch.long, device=work.device)
    proposal_available = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
    proposal_xywh = work.clone()
    proposal_hard_ok = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
    proposal_structure_ok = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
    proposal_final_pair_count = torch.zeros(work.shape[0], dtype=torch.long, device=work.device)
    proposal_displacement = torch.zeros(work.shape[0], dtype=work.dtype, device=work.device)
    proposal_rollback_reason = torch.full(
        (work.shape[0],),
        _PROPOSAL_NOT_COMPONENT,
        dtype=torch.long,
        device=work.device,
    )
    if component_enabled:
        (
            candidate,
            candidate_dirs,
            candidate_fixed_pair_overlap,
            candidate_active_count,
            component_rebuilds,
            new_pairs_detected,
            reset_count,
            beam_states_evaluated,
            max_component_size,
        ) = _project_component_mode(
            work,
            pairs,
            directions,
            mask,
            iterations=iterations,
            clearance=clearance,
            tolerance=tolerance,
            ignore_preplaced_pairs=ignore_preplaced_pairs,
            config=component_config,
            guidance=guidance,
            problem=problem,
        )
        candidate_hard_ok, _, candidate_max = _verified_status(
            candidate,
            work,
            mask,
            tolerance,
            candidate_fixed_pair_overlap,
            ignore_preplaced_pairs,
        )
        structure_ok = _structure_nonregression(
            candidate[:, None],
            work,
            guidance,
            problem,
            tolerance,
        )[:, 0]
        proposal_available = torch.ones(work.shape[0], dtype=torch.bool, device=work.device)
        proposal_xywh = candidate
        proposal_hard_ok = candidate_hard_ok
        proposal_structure_ok = structure_ok
        proposal_displacement = torch.linalg.vector_norm(candidate[..., :2] - work[..., :2], dim=-1).sum(dim=1)
        candidate_admissible = candidate_hard_ok & structure_ok
        candidate_active = _active_overlap_matrix_exact(candidate, tolerance)
        original_active = _active_overlap_matrix_exact(work, tolerance)
        if ignore_preplaced_pairs:
            exempt = mask[:, None] & mask[None, :]
            candidate_active = torch.where(
                exempt.unsqueeze(0), torch.zeros_like(candidate_active), candidate_active
            )
            original_active = torch.where(
                exempt.unsqueeze(0), torch.zeros_like(original_active), original_active
            )
        candidate_conflicts = candidate_active.sum(dim=(1, 2))
        proposal_final_pair_count = candidate_active[:, pairs[:, 0], pairs[:, 1]].sum(dim=1)
        original_conflicts = original_active.sum(dim=(1, 2))
        if component_config.preserve_feasible:
            # Preserve the primary candidate unless the component proposal
            # crosses the hard-feasibility boundary without structural rollback.
            better = candidate_admissible & ~original_ok
        else:
            same_feasibility = candidate_hard_ok == original_ok
            same_fixed_status = (
                candidate_fixed_pair_overlap == best_fixed_pair_overlap
            )
            better = structure_ok & (
                (candidate_hard_ok & ~original_ok)
                | (
                    same_feasibility
                    & (
                        (~candidate_fixed_pair_overlap & best_fixed_pair_overlap)
                        | (
                            same_fixed_status
                            & (candidate_conflicts < original_conflicts)
                        )
                        | (
                            same_fixed_status
                            & (candidate_conflicts == original_conflicts)
                            & (candidate_max < best_max)
                        )
                    )
                )
            )
        best_xywh = torch.where(better.view(-1, 1, 1), candidate, work)
        best_dirs = torch.where(better.view(-1, 1), candidate_dirs, directions)
        best_max = torch.where(better, candidate_max, best_max)
        best_ok = torch.where(better, candidate_hard_ok, original_ok)
        best_fixed_pair_overlap = torch.where(
            better, candidate_fixed_pair_overlap, best_fixed_pair_overlap
        )
        best_active_count = torch.where(
            better, candidate_active_count, initial_pair_count
        )
        proposal_rollback_reason = torch.where(
            better,
            torch.full_like(proposal_rollback_reason, _PROPOSAL_COMMITTED),
            torch.where(
                original_ok,
                torch.full_like(
                    proposal_rollback_reason,
                    _PROPOSAL_ALREADY_FEASIBLE,
                ),
                torch.where(
                    ~candidate_hard_ok,
                    torch.full_like(
                        proposal_rollback_reason,
                        _PROPOSAL_PROJECTOR_INCOMPLETE,
                    ),
                    torch.where(
                        ~structure_ok,
                        torch.full_like(
                            proposal_rollback_reason,
                            _PROPOSAL_CONSTRUCTION_REGRESSION,
                        ),
                        torch.full_like(
                            proposal_rollback_reason,
                            _PROPOSAL_PROJECTOR_INCOMPLETE,
                        ),
                    ),
                ),
            ),
        )
        proposal_available = (
            (proposal_rollback_reason == _PROPOSAL_PROJECTOR_INCOMPLETE)
            | (proposal_rollback_reason == _PROPOSAL_CONSTRUCTION_REGRESSION)
        )
    else:
        for variant in range(max(1, min(beam, 4))):
            candidate = work
            variant_dirs = directions
            fixed_pair_overlap = torch.zeros(
                work.shape[0], dtype=torch.bool, device=work.device
            )
            for _ in range(max(1, outer_iterations)):
                _, variant_dirs = assign_directions(
                    candidate,
                    beam_variant=variant,
                    tolerance=tolerance,
                )
                if ignore_preplaced_pairs:
                    exempt = mask[pairs[:, 0]] & mask[pairs[:, 1]]
                    variant_dirs[:, exempt] = -1
                active_count = (variant_dirs >= 0).sum(dim=1)
                if not bool((active_count > 0).any().item()):
                    break
                candidate, fixed_now = _project_fixed_iterations(
                    candidate,
                    pairs,
                    variant_dirs,
                    mask,
                    iterations=iterations,
                    clearance=clearance,
                )
                fixed_pair_overlap |= fixed_now
            ok, _, max_overlap = _verified_status(
                candidate,
                work,
                mask,
                tolerance,
                fixed_pair_overlap,
                ignore_preplaced_pairs,
            )
            same_feasibility = ok == best_ok
            same_fixed_status = fixed_pair_overlap == best_fixed_pair_overlap
            better = (
                (ok & ~best_ok)
                | (same_feasibility & ~fixed_pair_overlap & best_fixed_pair_overlap)
                | (
                    same_feasibility
                    & same_fixed_status
                    & (max_overlap < best_max)
                )
            )
            best_xywh = torch.where(
                better.view(-1, 1, 1), candidate, best_xywh
            )
            best_dirs = torch.where(
                better.view(-1, 1), variant_dirs, best_dirs
            )
            best_max = torch.where(better, max_overlap, best_max)
            best_ok = torch.where(better, ok, best_ok)
            best_fixed_pair_overlap = torch.where(
                better, fixed_pair_overlap, best_fixed_pair_overlap
            )
            best_active_count = torch.where(
                better, active_count, best_active_count
            )
    ok_mask, reasons, best_max = _verified_status(
        best_xywh,
        work,
        mask,
        tolerance,
        best_fixed_pair_overlap,
        ignore_preplaced_pairs,
    )
    final_active = _active_overlap_matrix_exact(best_xywh, tolerance)
    if ignore_preplaced_pairs:
        exempt = mask[:, None] & mask[None, :]
        final_active = torch.where(exempt.unsqueeze(0), torch.zeros_like(final_active), final_active)
    final_pair_count = final_active[:, pairs[:, 0], pairs[:, 1]].sum(dim=1)
    displacement = torch.linalg.vector_norm(best_xywh[..., :2] - work[..., :2], dim=-1).sum(dim=1)
    status = "ok" if bool(ok_mask.all().item()) else ("partial" if bool(ok_mask.any().item()) else "infeasible")
    proposal_reasons = tuple(
        _PROPOSAL_REASONS[int(code)]
        for code in proposal_rollback_reason.detach().cpu().tolist()
    )
    return ProjectionResult(
        xywh=best_xywh[0] if single else best_xywh,
        ok=bool(ok_mask.all().item()),
        status=status,
        max_overlap=best_max[0] if single else best_max,
        ok_mask=ok_mask[0] if single else ok_mask,
        displacement=displacement[0] if single else displacement,
        active_pair_count=best_active_count[0] if single else best_active_count,
        failure_reasons=(reasons[0],) if single else reasons,
        iterations=iterations,
        directions=best_dirs[0] if single else best_dirs,
        active_pairs=pairs,
        initial_pair_count=initial_pair_count[0] if single else initial_pair_count,
        final_pair_count=final_pair_count[0] if single else final_pair_count,
        component_rebuilds=component_rebuilds[0] if single else component_rebuilds,
        new_pairs_detected=new_pairs_detected[0] if single else new_pairs_detected,
        reset_count=reset_count[0] if single else reset_count,
        beam_states_evaluated=beam_states_evaluated[0] if single else beam_states_evaluated,
        max_component_size=max_component_size[0] if single else max_component_size,
        component_proposal_available=proposal_available[0] if single else proposal_available,
        component_proposal_xywh=proposal_xywh[0] if single else proposal_xywh,
        component_proposal_hard_ok=proposal_hard_ok[0] if single else proposal_hard_ok,
        component_proposal_structure_ok=proposal_structure_ok[0] if single else proposal_structure_ok,
        component_proposal_final_pair_count=proposal_final_pair_count[0] if single else proposal_final_pair_count,
        component_proposal_displacement=proposal_displacement[0] if single else proposal_displacement,
        component_proposal_rollback_reason=(proposal_reasons[0],) if single else proposal_reasons,
    )


project = project_disjunctive
