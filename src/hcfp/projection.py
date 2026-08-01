"""Batched disjunctive projection v0 for HCFP rectangles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


Tensor = torch.Tensor
EPS = 1.0e-6


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


def _pair_index(n: int, device: torch.device) -> tuple[Tensor, Tensor]:
    return torch.triu_indices(n, n, offset=1, device=device)


def overlap_matrix(xywh: Tensor) -> Tensor:
    boxes, single = _as_boxes(xywh)
    x0 = boxes[..., 0]
    y0 = boxes[..., 1]
    x1 = x0 + boxes[..., 2]
    y1 = y0 + boxes[..., 3]
    ox = (torch.minimum(x1[:, :, None], x1[:, None, :]) - torch.maximum(x0[:, :, None], x0[:, None, :])).clamp_min(0.0)
    oy = (torch.minimum(y1[:, :, None], y1[:, None, :]) - torch.maximum(y0[:, :, None], y0[:, None, :])).clamp_min(0.0)
    overlap = ox * oy
    eye = torch.eye(boxes.shape[1], dtype=torch.bool, device=boxes.device).view(1, boxes.shape[1], boxes.shape[1])
    overlap = torch.where(eye, torch.zeros_like(overlap), overlap)
    return overlap[0] if single else overlap


def _direction_order(boxes: Tensor, pair_i: Tensor, pair_j: Tensor, beam_variant: int) -> Tensor:
    bi = boxes[:, pair_i]
    bj = boxes[:, pair_j]
    left = (bi[..., 0] + bi[..., 2] - bj[..., 0]).clamp_min(0.0)
    right = (bj[..., 0] + bj[..., 2] - bi[..., 0]).clamp_min(0.0)
    top = (bi[..., 1] + bi[..., 3] - bj[..., 1]).clamp_min(0.0)
    bottom = (bj[..., 1] + bj[..., 3] - bi[..., 1]).clamp_min(0.0)
    costs = torch.stack((left, right, top, bottom), dim=-1)
    order = costs.argsort(dim=-1, stable=True)
    pick = min(max(beam_variant, 0), 3)
    return order[..., pick]


def assign_directions(
    boxes: Any,
    *,
    beam_variant: int = 0,
    active_pairs: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return dense active overlap pairs and L/R/T/B direction ids 0..3."""

    work, _ = _as_boxes(boxes)
    c, n = work.shape[:2]
    if active_pairs is None:
        pair_i, pair_j = _pair_index(n, work.device)
        active = overlap_matrix(work)[:, pair_i, pair_j] > 0.0
    else:
        active_pairs = active_pairs.to(device=work.device, dtype=torch.long)
        pair_i, pair_j = active_pairs[:, 0], active_pairs[:, 1]
        active = torch.ones((c, active_pairs.shape[0]), dtype=torch.bool, device=work.device)
    directions = _direction_order(work, pair_i, pair_j, beam_variant)
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
) -> tuple[Tensor, bool]:
    c, n = boxes.shape[:2]
    if pairs.numel() == 0:
        return boxes, torch.zeros(c, dtype=torch.bool, device=boxes.device)
    pair_i = pairs[:, 0]
    pair_j = pairs[:, 1]
    fixed_i = preplaced_mask[pair_i].view(1, -1)
    fixed_j = preplaced_mask[pair_j].view(1, -1)
    active = directions >= 0
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
        pos = pos + updates
        pos = torch.where(preplaced_mask.view(1, n, 1), boxes[..., :2], pos)
    return torch.cat((pos, wh), dim=-1), infeasible


def _verified_status(
    boxes: Tensor,
    original: Tensor,
    preplaced_mask: Tensor,
    tolerance: float,
    fixed_pair_overlap: Tensor,
) -> tuple[Tensor, tuple[str, ...], Tensor]:
    overlap = overlap_matrix(boxes)
    max_overlap = overlap.amax(dim=(1, 2))
    dims_ok = (boxes[..., 2:4] > 0).all(dim=(1, 2))
    if bool(preplaced_mask.any()):
        preplaced_ok = torch.isclose(boxes[:, preplaced_mask, :2], original[:, preplaced_mask, :2]).all(dim=(1, 2))
    else:
        preplaced_ok = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)
    residual_ok = max_overlap <= tolerance
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
    clearance = max(clearance, 2.0 * tolerance)
    mask = _as_mask(preplaced_mask if preplaced_mask is not None else _get_field(problem, ("preplaced_mask", "is_preplaced")), work.shape[1], work.device)
    pairs, directions = assign_directions(work, beam_variant=0)
    active_count = (directions >= 0).sum(dim=1)
    best_xywh = work
    best_dirs = directions
    best_max = overlap_matrix(work).amax(dim=(1, 2))
    best_ok = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
    best_fixed_pair_overlap = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
    best_active_count = active_count
    for variant in range(max(1, min(beam, 4))):
        candidate = work
        variant_dirs = directions
        fixed_pair_overlap = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
        for _ in range(max(1, outer_iterations)):
            pairs, variant_dirs = assign_directions(candidate, beam_variant=variant)
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
        ok, _, max_overlap = _verified_status(candidate, work, mask, tolerance, fixed_pair_overlap)
        better = (max_overlap < best_max) | (ok & ~best_ok) | (fixed_pair_overlap & ~best_fixed_pair_overlap)
        best_xywh = torch.where(better.view(-1, 1, 1), candidate, best_xywh)
        best_dirs = torch.where(better.view(-1, 1), variant_dirs, best_dirs)
        best_max = torch.where(better, max_overlap, best_max)
        best_ok = torch.where(better, ok, best_ok)
        best_fixed_pair_overlap = torch.where(better, fixed_pair_overlap, best_fixed_pair_overlap)
        best_active_count = torch.where(better, active_count, best_active_count)
    ok_mask, reasons, best_max = _verified_status(best_xywh, work, mask, tolerance, best_fixed_pair_overlap)
    displacement = torch.linalg.vector_norm(best_xywh[..., :2] - work[..., :2], dim=-1).sum(dim=1)
    status = "ok" if bool(ok_mask.all().item()) else ("partial" if bool(ok_mask.any().item()) else "infeasible")
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
    )


project = project_disjunctive
