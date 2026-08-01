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
        return boxes, True
    pair_i = pairs[:, 0]
    pair_j = pairs[:, 1]
    fixed_i = preplaced_mask[pair_i].view(1, -1)
    fixed_j = preplaced_mask[pair_j].view(1, -1)
    active = directions >= 0
    infeasible = bool((active & fixed_i & fixed_j).any().item())
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
    return torch.cat((pos, wh), dim=-1), not infeasible


def _verified_status(boxes: Tensor, original: Tensor, preplaced_mask: Tensor, tolerance: float) -> tuple[bool, str, Tensor]:
    overlap = overlap_matrix(boxes)
    max_overlap = overlap.amax(dim=(1, 2))
    dims_ok = bool((boxes[..., 2:4] > 0).all().item())
    preplaced_ok = bool(torch.allclose(boxes[:, preplaced_mask, :2], original[:, preplaced_mask, :2]))
    ok = dims_ok and preplaced_ok and bool((max_overlap <= tolerance).all().item())
    return ok, "ok" if ok else "infeasible", max_overlap


def project_disjunctive(
    boxes: Any,
    problem: Any | None = None,
    *,
    preplaced_mask: Any | None = None,
    iterations: int = 16,
    beam: int = 1,
    clearance: float = 0.0,
    tolerance: float = 1.0e-5,
) -> ProjectionResult:
    """Project overlapped ``xywh`` boxes onto fixed pairwise disjunctions.

    The result is fail-closed: callers must check ``ok``/``status`` before using
    ``xywh`` as legal geometry.
    """

    work, single = _as_boxes(boxes)
    mask = _as_mask(preplaced_mask if preplaced_mask is not None else _get_field(problem, ("preplaced_mask", "is_preplaced")), work.shape[1], work.device)
    pairs, directions = assign_directions(work, beam_variant=0)
    best_xywh = work
    best_dirs = directions
    best_max = overlap_matrix(work).amax(dim=(1, 2))
    best_ok = False
    best_status = "infeasible"
    for variant in range(max(1, min(beam, 4))):
        _, variant_dirs = assign_directions(work, beam_variant=variant)
        candidate, assign_ok = _project_fixed_iterations(work, pairs, variant_dirs, mask, iterations=iterations, clearance=clearance)
        ok, status, max_overlap = _verified_status(candidate, work, mask, tolerance)
        ok = ok and assign_ok
        better = max_overlap < best_max
        best_xywh = torch.where(better.view(-1, 1, 1), candidate, best_xywh)
        best_dirs = torch.where(better.view(-1, 1), variant_dirs, best_dirs)
        best_max = torch.where(better, max_overlap, best_max)
        best_ok = bool(ok or best_ok)
        if ok:
            best_status = "ok"
    if not best_ok:
        best_status = "infeasible"
    return ProjectionResult(
        xywh=best_xywh[0] if single else best_xywh,
        ok=best_ok,
        status=best_status,
        max_overlap=best_max[0] if single else best_max,
        iterations=iterations,
        directions=best_dirs[0] if single else best_dirs,
        active_pairs=pairs,
    )


project = project_disjunctive
