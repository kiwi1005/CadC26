"""Fail-closed sequence-pair adaptation around exact preplaced anchors."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import torch

from hcfp.topology.relation_labels import (
    INVERSE_RELATION,
    REL_ABOVE,
    REL_BELOW,
    REL_LEFT,
    REL_RIGHT,
    relation_mask_from_rectangles,
)
from hcfp.topology.sequence_pair import decode_sequence_pair


Tensor = torch.Tensor
MAX_REPAIR_SEARCH_STATES = 4096


@dataclass(frozen=True)
class PreplacedConflict:
    first: int
    second: int
    predicted: int
    allowed: tuple[int, ...]


@dataclass(frozen=True)
class PreplacedCompatibility:
    compatible: bool
    conflicts: tuple[PreplacedConflict, ...]


@dataclass(frozen=True)
class AnchorSafeOrderVariant:
    name: str
    positive: Tensor
    negative: Tensor


def check_preplaced_compatibility(
    positive: Tensor,
    negative: Tensor,
    target_xywh: Tensor,
    preplaced_mask: Tensor,
    *,
    tolerance: float = 1.0e-7,
) -> PreplacedCompatibility:
    """Report sequence-pair relations that contradict exact preplaced geometry."""

    targets, mask = _targets_and_mask(target_xywh, preplaced_mask)
    topology = decode_sequence_pair(positive, negative, n=targets.shape[0])
    if not bool(topology.active_mask.all()):
        raise ValueError("preplaced adaptation requires permutations of every block")
    allowed = relation_mask_from_rectangles(
        targets, valid_mask=mask, tolerance=tolerance
    )
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    conflicts: list[PreplacedConflict] = []
    for offset, first in enumerate(indices):
        for second in indices[offset + 1 :]:
            predicted = int(topology.relation[first, second])
            valid = tuple(
                int(value)
                for value in torch.nonzero(allowed[first, second], as_tuple=False)
                .flatten()
                .tolist()
            )
            if predicted not in valid:
                conflicts.append(PreplacedConflict(first, second, predicted, valid))
    result = tuple(conflicts)
    return PreplacedCompatibility(not result, result)


def anchor_safe_order_variants(
    positive: Tensor,
    negative: Tensor,
    preplaced_mask: Tensor,
) -> tuple[AnchorSafeOrderVariant, ...]:
    """Group anchors before/after movables without changing within-group order."""

    plus = torch.as_tensor(positive, dtype=torch.long)
    minus = torch.as_tensor(negative, dtype=torch.long, device=plus.device)
    topology = decode_sequence_pair(plus, minus)
    if not bool(topology.active_mask.all()):
        raise ValueError("anchor-safe variants require permutations of every block")
    mask = torch.as_tensor(preplaced_mask, dtype=torch.bool, device=plus.device)
    if mask.shape != topology.active_mask.shape:
        raise ValueError("preplaced_mask must have shape [N]")
    if not bool(mask.any()) or bool(mask.all()):
        return ()

    def grouped(order: Tensor, anchors_first: bool) -> Tensor:
        anchors = order[mask[order]]
        movable = order[~mask[order]]
        parts = (anchors, movable) if anchors_first else (movable, anchors)
        return torch.cat(parts)

    original = (tuple(plus.tolist()), tuple(minus.tolist()))
    seen = {original}
    variants = []
    for positive_prefix, negative_prefix in (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ):
        variant_plus = grouped(plus, positive_prefix)
        variant_minus = grouped(minus, negative_prefix)
        key = (tuple(variant_plus.tolist()), tuple(variant_minus.tolist()))
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            AnchorSafeOrderVariant(
                name=(
                    f"positive_{'prefix' if positive_prefix else 'suffix'}_"
                    f"negative_{'prefix' if negative_prefix else 'suffix'}"
                ),
                positive=variant_plus,
                negative=variant_minus,
            )
        )
    return tuple(variants)


def adapt_preplaced_topology(
    positive: Tensor,
    negative: Tensor,
    target_xywh: Tensor,
    preplaced_mask: Tensor,
    *,
    relation_confidence: Tensor | None = None,
    repair_threshold: float = 0.5,
    tolerance: float = 1.0e-7,
) -> tuple[Tensor, Tensor]:
    """Repair only low-confidence anchor conflicts, otherwise reject.

    Repair is deterministic and greedy rather than exhaustive. Each changed
    relation must keep both sequence orders acyclic; if no safe choice exists,
    the topology is rejected.
    """

    if not 0.0 <= repair_threshold <= 1.0:
        raise ValueError("repair_threshold must be in [0,1]")
    targets, mask = _targets_and_mask(target_xywh, preplaced_mask)
    topology = decode_sequence_pair(positive, negative, n=targets.shape[0])
    if not bool(topology.active_mask.all()):
        raise ValueError("preplaced adaptation requires permutations of every block")
    report = check_preplaced_compatibility(
        positive, negative, targets, mask, tolerance=tolerance
    )
    if report.compatible:
        return torch.as_tensor(positive, dtype=torch.long).clone(), torch.as_tensor(
            negative, dtype=torch.long
        ).clone()
    if any(not conflict.allowed for conflict in report.conflicts):
        raise ValueError("overlapping preplaced targets cannot be repaired safely")
    confidence = _confidence_tensor(
        relation_confidence, targets.shape[0], targets.device
    )
    if confidence is None:
        raise ValueError(
            "relation_confidence is required to repair preplaced conflicts"
        )
    for conflict in report.conflicts:
        value = _predicted_confidence(confidence, conflict)
        if value > repair_threshold:
            raise ValueError(
                "high-confidence preplaced conflict cannot be repaired safely"
            )

    plus = torch.as_tensor(positive, dtype=torch.long, device=targets.device)
    minus = torch.as_tensor(negative, dtype=torch.long, device=targets.device)
    plus_edges: set[tuple[int, int]] = set()
    minus_edges: set[tuple[int, int]] = set()
    conflict_pairs = {(item.first, item.second): item for item in report.conflicts}
    allowed = relation_mask_from_rectangles(
        targets, valid_mask=mask, tolerance=tolerance
    )
    gaps = _relation_gaps(targets)
    pending: list[PreplacedConflict] = []
    reserved_pairs: set[tuple[int, int]] = set()
    preplaced_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    for offset, first in enumerate(preplaced_indices):
        for second in preplaced_indices[offset + 1 :]:
            predicted = int(topology.relation[first, second])
            choices = tuple(
                int(value)
                for value in torch.nonzero(
                    allowed[first, second], as_tuple=False
                ).flatten()
            )
            conflict = conflict_pairs.get((first, second))
            confidence_value = _relation_confidence_value(
                confidence,
                first,
                second,
                predicted,
            )
            if len(choices) == 1:
                _add_relation_edges(
                    plus_edges,
                    minus_edges,
                    first,
                    second,
                    choices[0],
                )
                reserved_pairs.add((first, second))
            elif conflict is not None or confidence_value <= repair_threshold:
                pending.append(
                    PreplacedConflict(first, second, predicted, choices)
                )
                reserved_pairs.add((first, second))

    for first in range(targets.shape[0]):
        for second in range(first + 1, targets.shape[0]):
            if (first, second) in reserved_pairs:
                continue
            relation = int(topology.relation[first, second])
            both_preplaced = bool(mask[first] and mask[second])
            relation_confidence_value = _relation_confidence_value(
                confidence,
                first,
                second,
                relation,
            )
            if both_preplaced or relation_confidence_value > repair_threshold:
                _add_relation_edges(plus_edges, minus_edges, first, second, relation)

    if not _acyclic(targets.shape[0], plus_edges) or not _acyclic(
        targets.shape[0], minus_edges
    ):
        raise ValueError(
            "high-confidence relations contradict exact preplaced constraints"
        )

    pending.sort(
        key=lambda item: (
            len(item.allowed),
            _predicted_confidence(confidence, item),
            item.first,
            item.second,
        )
    )
    pending_choices: list[tuple[int, ...]] = []
    for conflict in pending:
        candidates = (
            torch.nonzero(allowed[conflict.first, conflict.second], as_tuple=False)
            .flatten()
            .tolist()
        )
        candidates.sort(
            key=lambda relation: (
                int(relation != conflict.predicted),
                -float(gaps[conflict.first, conflict.second, relation]),
                int(relation),
            )
        )
        pending_choices.append(tuple(int(relation) for relation in candidates))

    stack = [(0, plus_edges, minus_edges)]
    repaired_edges: tuple[set[tuple[int, int]], set[tuple[int, int]]] | None = None
    states = 0
    while stack and states < MAX_REPAIR_SEARCH_STATES:
        index, current_plus, current_minus = stack.pop()
        states += 1
        if index == len(pending):
            repaired_edges = current_plus, current_minus
            break
        conflict = pending[index]
        for relation in reversed(pending_choices[index]):
            plus_trial = set(current_plus)
            minus_trial = set(current_minus)
            _add_relation_edges(
                plus_trial,
                minus_trial,
                conflict.first,
                conflict.second,
                int(relation),
            )
            if _acyclic(targets.shape[0], plus_trial) and _acyclic(
                targets.shape[0], minus_trial
            ):
                stack.append((index + 1, plus_trial, minus_trial))
    if repaired_edges is None:
        if stack:
            raise ValueError("low-confidence repair exceeded bounded search")
        raise ValueError("low-confidence repair would create a sequence cycle")
    plus_edges, minus_edges = repaired_edges

    preferred_plus = torch.cat((plus[mask[plus]], plus[~mask[plus]]))
    preferred_minus = torch.cat((minus[mask[minus]], minus[~mask[minus]]))
    new_plus = _stable_topological_order(
        targets.shape[0], plus_edges, preferred_plus
    )
    new_minus = _stable_topological_order(
        targets.shape[0], minus_edges, preferred_minus
    )
    repaired_report = check_preplaced_compatibility(
        new_plus, new_minus, targets, mask, tolerance=tolerance
    )
    if not repaired_report.compatible:
        raise ValueError("preplaced repair failed compatibility verification")
    return new_plus.to(device=torch.as_tensor(positive).device), new_minus.to(
        device=torch.as_tensor(negative).device
    )


def copy_preplaced_targets(
    rectangles: Tensor,
    target_xywh: Tensor,
    preplaced_mask: Tensor,
    *,
    tolerance: float = 1.0e-7,
) -> Tensor:
    """Exact-copy preplaced rows without mutating inputs; reject resulting overlap."""

    candidate = torch.as_tensor(rectangles)
    if candidate.ndim != 2 or candidate.shape[1] != 4:
        raise ValueError("rectangles must have shape [N,4]")
    if candidate.dtype not in (torch.float32, torch.float64):
        raise ValueError("rectangles must use float32 or float64 exact geometry")
    raw_targets = torch.as_tensor(target_xywh, device=candidate.device)
    if raw_targets.dtype not in (torch.float32, torch.float64):
        raise ValueError("target_xywh must use float32 or float64 exact geometry")
    dtype = torch.promote_types(candidate.dtype, raw_targets.dtype)
    candidate = candidate.to(dtype=dtype)
    targets, mask = _targets_and_mask(
        raw_targets.to(dtype=dtype),
        preplaced_mask,
    )
    if targets.shape != candidate.shape:
        raise ValueError("target_xywh must match rectangles shape")
    result = candidate.clone()
    result[mask] = targets[mask]
    allowed = relation_mask_from_rectangles(result, tolerance=tolerance).any(dim=-1)
    relevant = mask[:, None] | mask[None, :]
    relevant.fill_diagonal_(False)
    if bool((relevant & ~allowed).any()):
        raise ValueError("exact preplaced copy overlaps another block; repair rejected")
    if not torch.equal(result[mask], targets[mask]):
        raise ValueError("preplaced targets were not preserved exactly")
    return result


def _targets_and_mask(targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    boxes = torch.as_tensor(targets)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("target_xywh must have shape [N,4]")
    active = torch.as_tensor(mask, dtype=torch.bool, device=boxes.device)
    if active.shape != (boxes.shape[0],):
        raise ValueError("preplaced_mask must have shape [N]")
    if bool(active.any()):
        hard = boxes[active]
        if not bool(torch.isfinite(hard).all()) or bool((hard[:, 2:4] <= 0.0).any()):
            raise ValueError(
                "preplaced targets must be finite with positive dimensions"
            )
    safe = boxes.clone()
    safe[~active] = torch.tensor(
        (0.0, 0.0, 1.0, 1.0), dtype=boxes.dtype, device=boxes.device
    )
    return safe, active


def _confidence_tensor(
    value: Tensor | None, n: int, device: torch.device
) -> Tensor | None:
    if value is None:
        return None
    confidence = torch.as_tensor(value, dtype=torch.float32, device=device)
    if confidence.shape not in ((n, n), (n, n, 4)):
        raise ValueError("relation_confidence must have shape [N,N] or [N,N,4]")
    if not bool(torch.isfinite(confidence).all()) or bool(
        ((confidence < 0.0) | (confidence > 1.0)).any()
    ):
        raise ValueError("relation_confidence values must be finite probabilities")
    return confidence


def _predicted_confidence(confidence: Tensor, conflict: PreplacedConflict) -> float:
    return _relation_confidence_value(
        confidence,
        conflict.first,
        conflict.second,
        conflict.predicted,
    )


def _relation_confidence_value(
    confidence: Tensor,
    first: int,
    second: int,
    relation: int,
) -> float:
    if confidence.ndim == 2:
        return max(float(confidence[first, second]), float(confidence[second, first]))
    inverse = INVERSE_RELATION[relation]
    return max(
        float(confidence[first, second, relation]),
        float(confidence[second, first, inverse]),
    )


def _relation_gaps(boxes: Tensor) -> Tensor:
    left, bottom = boxes[:, 0], boxes[:, 1]
    right, top = left + boxes[:, 2], bottom + boxes[:, 3]
    return torch.stack(
        (
            left[None, :] - right[:, None],
            left[:, None] - right[None, :],
            bottom[:, None] - top[None, :],
            bottom[None, :] - top[:, None],
        ),
        dim=-1,
    )


def _add_relation_edges(
    plus_edges: set[tuple[int, int]],
    minus_edges: set[tuple[int, int]],
    first: int,
    second: int,
    relation: int,
) -> None:
    if relation == REL_LEFT:
        plus_edges.add((first, second))
        minus_edges.add((first, second))
    elif relation == REL_RIGHT:
        plus_edges.add((second, first))
        minus_edges.add((second, first))
    elif relation == REL_ABOVE:
        plus_edges.add((first, second))
        minus_edges.add((second, first))
    elif relation == REL_BELOW:
        plus_edges.add((second, first))
        minus_edges.add((first, second))
    else:
        raise ValueError("relation must be one of L/R/U/D")


def _acyclic(n: int, edges: set[tuple[int, int]]) -> bool:
    try:
        _topological_order(n, edges, range(n))
    except ValueError:
        return False
    return True


def _stable_topological_order(
    n: int, edges: set[tuple[int, int]], order: Tensor
) -> Tensor:
    result = _topological_order(n, edges, torch.as_tensor(order).tolist())
    return torch.tensor(result, dtype=torch.long, device=torch.as_tensor(order).device)


def _topological_order(
    n: int,
    edges: set[tuple[int, int]],
    preferred_order: object,
) -> list[int]:
    rank = {int(node): index for index, node in enumerate(preferred_order)}
    adjacency: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n
    for source, target in sorted(edges):
        adjacency[source].append(target)
        indegree[target] += 1
    ready = [(rank[node], node) for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    result: list[int] = []
    while ready:
        _, source = heapq.heappop(ready)
        result.append(source)
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, (rank[target], target))
    if len(result) != n:
        raise ValueError("preplaced relation constraints contain a cycle")
    return result
