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


def anchored_longest_path_coordinates(
    lengths: Tensor,
    edges: Tensor,
    fixed_coordinates: Tensor,
    fixed_mask: Tensor,
    *,
    origin: Tensor | float = 0.0,
    spacing: float = 0.0,
    tolerance: float = 1.0e-6,
) -> Tensor:
    """Pack one DAG axis while preserving selected coordinates exactly.

    Unanchored predecessors of a fixed node may move below ``origin``.  The
    reverse upper-bound pass makes that shift before the ordinary forward
    longest-path pass.  Incompatible anchors fail closed.
    """

    size = torch.as_tensor(lengths)
    if size.ndim != 1:
        raise ValueError("lengths must have shape [N]")
    if not torch.is_floating_point(size):
        size = size.float()
    elif size.dtype not in (torch.float32, torch.float64):
        size = size.float()
    if not bool(torch.isfinite(size).all()) or bool((size <= 0.0).any()):
        raise ValueError("lengths must be finite and positive")
    if tolerance < 0.0 or spacing < 0.0:
        raise ValueError("spacing and tolerance must be non-negative")

    fixed = torch.as_tensor(
        fixed_coordinates,
        dtype=size.dtype,
        device=size.device,
    )
    mask = torch.as_tensor(fixed_mask, dtype=torch.bool, device=size.device)
    if fixed.shape != size.shape or mask.shape != size.shape:
        raise ValueError("fixed_coordinates and fixed_mask must have shape [N]")
    if bool(mask.any()) and not bool(torch.isfinite(fixed[mask]).all()):
        raise ValueError("anchored coordinates must be finite")
    base = torch.as_tensor(origin, dtype=size.dtype, device=size.device)
    if base.numel() != 1 or not bool(torch.isfinite(base)):
        raise ValueError("origin must be one finite scalar")

    edge_list = _validated_edges(edges, size.numel())
    adjacency: list[list[int]] = [[] for _ in range(size.numel())]
    predecessors: list[list[int]] = [[] for _ in range(size.numel())]
    indegree = [0] * size.numel()
    for source, target in edge_list:
        adjacency[source].append(target)
        predecessors[target].append(source)
        indegree[target] += 1
    for nodes in (*adjacency, *predecessors):
        nodes.sort()

    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        source = heapq.heappop(ready)
        order.append(source)
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != size.numel():
        raise ValueError("constraint graph contains a directed cycle")

    upper = torch.full_like(size, torch.inf)
    upper[mask] = fixed[mask]
    for source in reversed(order):
        for target in adjacency[source]:
            if bool(torch.isfinite(upper[target])):
                edge_spacing = 0.0 if bool(mask[source] and mask[target]) else spacing
                upper[source] = torch.minimum(
                    upper[source],
                    upper[target] - size[source] - edge_spacing,
                )
        if bool(mask[source]) and float(upper[source]) < float(fixed[source]) - tolerance:
            raise ValueError("fixed coordinates contradict sequence constraints")

    coordinates = torch.empty_like(size)
    for node in order:
        floor = size.new_tensor(-torch.inf)
        if predecessors[node]:
            floor = torch.stack(
                tuple(
                    coordinates[source] + size[source]
                    + (0.0 if bool(mask[source] and mask[node]) else spacing)
                    for source in predecessors[node]
                )
            ).amax()
        if bool(mask[node]):
            coordinate = fixed[node]
            if float(floor) > float(coordinate) + tolerance:
                raise ValueError("fixed coordinates contradict sequence constraints")
        else:
            local_origin = torch.minimum(base, upper[node])
            coordinate = torch.maximum(local_origin, floor)
            if float(coordinate) > float(upper[node]) + tolerance:
                raise ValueError("fixed coordinates leave no feasible placement")
        coordinates[node] = coordinate

    for source, target in edge_list:
        edge_spacing = 0.0 if bool(mask[source] and mask[target]) else spacing
        if (
            float(coordinates[source] + size[source] + edge_spacing)
            > float(coordinates[target]) + tolerance
        ):
            raise ValueError("anchored packing failed a sequence constraint")
    if bool(mask.any()) and not torch.equal(coordinates[mask], fixed[mask]):
        raise ValueError("anchored packing did not preserve exact coordinates")
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


def pack_sequence_pair_with_anchors(
    dimensions: Tensor,
    positive: Tensor,
    negative: Tensor,
    target_xywh: Tensor,
    preplaced_mask: Tensor,
    *,
    origin: Tensor | tuple[float, float] | None = None,
    spacing: float = 0.0,
    tolerance: float = 1.0e-6,
) -> Tensor:
    """Pack a complete sequence pair with exact preplaced coordinates."""

    shape = torch.as_tensor(dimensions)
    if shape.ndim != 2 or shape.shape[1] != 2:
        raise ValueError("dimensions must have shape [N,2]")
    if not torch.is_floating_point(shape):
        shape = shape.float()
    elif shape.dtype not in (torch.float32, torch.float64):
        shape = shape.float()
    targets = torch.as_tensor(target_xywh, dtype=shape.dtype, device=shape.device)
    mask = torch.as_tensor(preplaced_mask, dtype=torch.bool, device=shape.device)
    if targets.shape != (shape.shape[0], 4) or mask.shape != (shape.shape[0],):
        raise ValueError("target_xywh and preplaced_mask must match dimensions")
    if bool(mask.any()):
        hard = targets[mask]
        if not bool(torch.isfinite(hard).all()) or bool((hard[:, 2:4] <= 0.0).any()):
            raise ValueError("preplaced targets must be finite with positive dimensions")
        shape = shape.clone()
        shape[mask] = hard[:, 2:4]

    topology = decode_sequence_pair(positive, negative, n=shape.shape[0])
    if not bool(topology.active_mask.all()):
        raise ValueError("anchored packing requires permutations of every block")
    base = torch.zeros(2, dtype=shape.dtype, device=shape.device)
    if origin is not None:
        base = torch.as_tensor(origin, dtype=shape.dtype, device=shape.device)
        if base.shape != (2,) or not bool(torch.isfinite(base).all()):
            raise ValueError("origin must contain two finite coordinates")
    x = anchored_longest_path_coordinates(
        shape[:, 0],
        topology.horizontal_edges,
        targets[:, 0],
        mask,
        origin=base[0],
        spacing=spacing,
        tolerance=tolerance,
    )
    y = anchored_longest_path_coordinates(
        shape[:, 1],
        topology.vertical_edges,
        targets[:, 1],
        mask,
        origin=base[1],
        spacing=spacing,
        tolerance=tolerance,
    )
    result = torch.cat((torch.stack((x, y), dim=1), shape), dim=1)
    if bool(mask.any()) and not torch.equal(result[mask], targets[mask]):
        raise ValueError("preplaced targets were not preserved exactly")
    return result


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
