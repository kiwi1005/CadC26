from __future__ import annotations

from typing import Any

import torch

from .schema import BOUNDARY_CODES, ConstraintColumns, FloorSetCase


def _squeeze_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _trim_padded_rows(tensor: torch.Tensor, pad_value: int | float = -1) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1:
        return tensor[tensor != pad_value]
    mask = ~(tensor == pad_value).all(dim=-1)
    return tensor[mask]


def _trim_case_tensors(
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    b2b_edges: torch.Tensor,
    p2b_edges: torch.Tensor,
    pins_pos: torch.Tensor,
    target_positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    valid_block_mask = area_targets.ne(-1)
    area_targets = area_targets[valid_block_mask]
    constraints = constraints[valid_block_mask]
    if target_positions is not None:
        target_positions = target_positions[valid_block_mask]

    b2b_edges = _trim_padded_rows(b2b_edges)
    p2b_edges = _trim_padded_rows(p2b_edges)
    pins_pos = _trim_padded_rows(pins_pos)
    return area_targets, constraints, b2b_edges, p2b_edges, pins_pos, target_positions


def infer_case_id(raw: dict[str, Any] | None, fallback_prefix: str = "case") -> str:
    if raw:
        for key in ("case_id", "case_idx", "index", "id"):
            if key in raw and raw[key] is not None:
                return str(raw[key])
    return fallback_prefix


def _base_case(
    area_targets: torch.Tensor,
    b2b_edges: torch.Tensor,
    p2b_edges: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    *,
    case_id: str | int,
    target_positions: torch.Tensor | None,
    metrics: torch.Tensor | None,
    raw: dict[str, Any] | None,
) -> FloorSetCase:
    area_targets = _squeeze_batch_dim(area_targets).to(torch.float32)
    b2b_edges = _squeeze_batch_dim(b2b_edges)
    p2b_edges = _squeeze_batch_dim(p2b_edges)
    pins_pos = _squeeze_batch_dim(pins_pos).to(torch.float32)
    constraints = _squeeze_batch_dim(constraints).to(torch.float32)
    metrics = _squeeze_batch_dim(metrics).to(torch.float32) if metrics is not None else None
    target_positions = _squeeze_batch_dim(target_positions).to(torch.float32) if target_positions is not None else None

    (
        area_targets,
        constraints,
        b2b_edges,
        p2b_edges,
        pins_pos,
        target_positions,
    ) = _trim_case_tensors(area_targets, constraints, b2b_edges, p2b_edges, pins_pos, target_positions)

    return FloorSetCase(
        case_id=case_id,
        block_count=int(area_targets.shape[0]),
        area_targets=area_targets,
        b2b_edges=b2b_edges.to(torch.float32),
        p2b_edges=p2b_edges.to(torch.float32),
        pins_pos=pins_pos,
        constraints=constraints,
        target_positions=target_positions,
        metrics=metrics,
        raw=raw,
    )


def adapt_training_batch(
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    case_id: str | int = "train-case",
    raw: dict[str, Any] | None = None,
) -> FloorSetCase:
    (
        area_targets,
        b2b_edges,
        p2b_edges,
        pins_pos,
        constraints,
        _tree_sol,
        fp_sol,
        metrics,
    ) = batch
    target_positions = _squeeze_batch_dim(fp_sol).to(torch.float32)
    target_positions = target_positions[..., [2, 3, 0, 1]]
    return _base_case(
        area_targets,
        b2b_edges,
        p2b_edges,
        pins_pos,
        constraints,
        case_id=case_id,
        target_positions=target_positions,
        metrics=metrics,
        raw=raw,
    )


def _polygon_to_box(poly: torch.Tensor) -> torch.Tensor:
    valid = poly[~(poly == -1).all(dim=-1)]
    if valid.numel() == 0:
        return torch.full((4,), -1.0, dtype=torch.float32)
    x_min = valid[:, 0].min()
    y_min = valid[:, 1].min()
    x_max = valid[:, 0].max()
    y_max = valid[:, 1].max()
    return torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], dtype=torch.float32)


def polygons_to_boxes(polygons: torch.Tensor) -> torch.Tensor:
    polygons = _squeeze_batch_dim(polygons)
    return torch.stack([_polygon_to_box(poly) for poly in polygons], dim=0)


def adapt_validation_batch(
    batch: tuple[list[torch.Tensor], list[torch.Tensor]] | tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    *,
    case_id: str | int = "validation-case",
    raw: dict[str, Any] | None = None,
) -> FloorSetCase:
    inputs, labels = batch
    area_targets, b2b_edges, p2b_edges, pins_pos, constraints = inputs
    polygons, metrics = labels
    target_positions = polygons_to_boxes(polygons)
    return _base_case(
        area_targets,
        b2b_edges,
        p2b_edges,
        pins_pos,
        constraints,
        case_id=case_id,
        target_positions=target_positions,
        metrics=metrics,
        raw=raw,
    )
