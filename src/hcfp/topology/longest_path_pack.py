"""Deterministic compact packing over acyclic sequence-pair constraints."""

from __future__ import annotations

import heapq

import torch

from hcfp.topology.sequence_pair import decode_sequence_pair


Tensor = torch.Tensor


def longest_path_coordinates(
    lengths: Tensor,
    edges: Tensor,
    *,
    lower_bounds: Tensor | float | None = None,
) -> Tensor:
    """Solve ``x[v] >= x[u] + lengths[u]`` at the compact lower envelope."""

    size = torch.as_tensor(lengths)
    if size.ndim != 1:
        raise ValueError("lengths must have shape [N]")
    if not torch.is_floating_point(size):
        size = size.float()
    elif size.dtype not in (torch.float32, torch.float64):
        size = size.float()
    if not bool(torch.isfinite(size).all()) or bool((size <= 0.0).any()):
        raise ValueError("lengths must be finite and positive")

    edge_list = _validated_edges(edges, size.numel())
    adjacency: list[list[int]] = [[] for _ in range(size.numel())]
    indegree = [0] * size.numel()
    for source, target in edge_list:
        adjacency[source].append(target)
        indegree[target] += 1
    for targets in adjacency:
        targets.sort()

    if lower_bounds is None:
        coordinates = torch.zeros_like(size)
    else:
        bounds = torch.as_tensor(lower_bounds, dtype=size.dtype, device=size.device)
        try:
            coordinates = torch.broadcast_to(bounds, size.shape).clone()
        except RuntimeError as exc:
            raise ValueError("lower_bounds must be scalar or shape [N]") from exc
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("lower_bounds must be finite")

    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        source = heapq.heappop(ready)
        visited += 1
        candidate = coordinates[source] + size[source]
        for target in adjacency[source]:
            coordinates[target] = torch.maximum(coordinates[target], candidate)
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if visited != size.numel():
        raise ValueError("constraint graph contains a directed cycle")
    return coordinates


def longest_path_pack(
    dimensions: Tensor,
    horizontal_edges: Tensor,
    vertical_edges: Tensor,
    *,
    origin: Tensor | tuple[float, float] | None = None,
) -> Tensor:
    """Return compact ``[x,y,w,h]`` rectangles for two constraint DAGs."""

    shape = torch.as_tensor(dimensions)
    if shape.ndim != 2 or shape.shape[1] != 2:
        raise ValueError("dimensions must have shape [N,2]")
    if not torch.is_floating_point(shape):
        shape = shape.float()
    elif shape.dtype not in (torch.float32, torch.float64):
        shape = shape.float()
    if not bool(torch.isfinite(shape).all()) or bool((shape <= 0.0).any()):
        raise ValueError("dimensions must be finite and positive")

    base = torch.zeros(2, dtype=shape.dtype, device=shape.device)
    if origin is not None:
        base = torch.as_tensor(origin, dtype=shape.dtype, device=shape.device)
        if base.shape != (2,) or not bool(torch.isfinite(base).all()):
            raise ValueError("origin must contain two finite coordinates")
    x = longest_path_coordinates(shape[:, 0], horizontal_edges, lower_bounds=base[0])
    y = longest_path_coordinates(shape[:, 1], vertical_edges, lower_bounds=base[1])
    return torch.cat((torch.stack((x, y), dim=1), shape), dim=1)


def pack_sequence_pair(
    dimensions: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    origin: Tensor | tuple[float, float] | None = None,
) -> Tensor:
    """Decode a complete sequence pair and pack it by longest paths."""

    shape = torch.as_tensor(dimensions)
    if shape.ndim != 2 or shape.shape[1] != 2:
        raise ValueError("dimensions must have shape [N,2]")
    topology = decode_sequence_pair(positive, negative, n=shape.shape[0])
    if not bool(topology.active_mask.all()):
        raise ValueError("pack_sequence_pair requires permutations of every block")
    return longest_path_pack(
        shape,
        topology.horizontal_edges,
        topology.vertical_edges,
        origin=origin,
    )


def _validated_edges(edges: Tensor, n: int) -> list[tuple[int, int]]:
    tensor = torch.as_tensor(edges, dtype=torch.long, device="cpu")
    if tensor.numel() == 0:
        return []
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    if bool((tensor < 0).any()) or bool((tensor >= n).any()):
        raise ValueError("edge endpoint is outside [0,N)")
    edge_list = {(int(source), int(target)) for source, target in tensor.tolist()}
    if any(source == target for source, target in edge_list):
        raise ValueError("self edges are not allowed")
    return sorted(edge_list)
