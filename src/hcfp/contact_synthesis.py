"""Runtime synthesis of contacts for disconnected grouping groups.

The first API in this module is intentionally an inspector: it proposes
contact obligations without changing geometry.  ``apply_contact_obligations``
is the small challenger on top of it.  It performs only rigid translations of
movable components, and returns an intermediate candidate only when the exact
hard verifier accepts the result and the grouping violation count decreases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hcfp.case import FloorplanCase
from hcfp.constraints.contact_tree import extract_contacts
from hcfp.constraints.construction import _contact_targets
from hcfp.verify import grouping_violation, verify_feasible


Tensor = torch.Tensor


@dataclass(frozen=True)
class ContactObligation:
    """One cheapest estimated contact between two grouping components.

    ``moving_member`` and ``anchor_member`` describe the orientation used by
    the estimate.  They are suggestions only: no translated placement is
    returned or applied here.
    """

    group_index: int
    component_a: tuple[int, ...]
    component_b: tuple[int, ...]
    member_a: int
    member_b: int
    bridge_member: int
    moving_component: tuple[int, ...]
    moving_member: int
    anchor_member: int
    side: str
    delta: tuple[float, float]
    move_distance: float
    bbox_expansion: float
    net_incident: float

    @property
    def members(self) -> tuple[int, int]:
        """Return the selected member pair in component order."""

        return self.member_a, self.member_b

    @property
    def components(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the selected component pair in deterministic order."""

        return self.component_a, self.component_b

    @property
    def estimated_cost(self) -> tuple[float, float, float]:
        """Return the sortable cost terms used for candidate selection."""

        return self.move_distance, self.bbox_expansion, -self.net_incident


@dataclass(frozen=True)
class ContactSynthesis:
    """Complete candidate graph plus its selected spanning-tree obligations."""

    obligations: tuple[ContactObligation, ...]
    candidate_edges: tuple[ContactObligation, ...]
    component_groups: tuple[tuple[tuple[int, ...], ...], ...]
    bridge_members: tuple[int, ...]

    @property
    def required_contacts(self) -> tuple[ContactObligation, ...]:
        """Alias for callers that use the runtime terminology."""

        return self.obligations

    @property
    def components(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Alias exposing each group's exact-contact components."""

        return self.component_groups

    @property
    def suggested_bridge_member(self) -> int | None:
        """Return the first deterministic bridge suggestion, if any."""

        return self.bridge_members[0] if self.bridge_members else None


def synthesize_contact_obligations(
    case: FloorplanCase,
    placements: Any,
    group_membership: Any | None = None,
    *,
    tolerance: float = 0.0,
    preplaced_mask: Any | None = None,
) -> ContactSynthesis:
    """Synthesize a deterministic spanning tree for disconnected groups.

    Components are connected only by exact positive-length rectangle edge
    contact.  For every disconnected grouping row, one candidate is retained
    for every component pair; candidates estimate the cheapest orientation by
    component movement, layout-bbox expansion, and crossing net weight.  A
    minimum spanning tree then yields exactly ``component_count - 1``
    obligations.  The input placement is never modified.
    """

    boxes = _boxes(placements, int(case.n))
    groups = _groups(
        getattr(case, "group_membership", None)
        if group_membership is None
        else group_membership,
        int(case.n),
    )
    weights = _weights(getattr(case, "b2b_weight", None), int(case.n))
    preplaced = (
        _preplaced_mask(case, int(case.n))
        if preplaced_mask is None
        else _preplaced_mask({"preplaced_mask": preplaced_mask}, int(case.n))
    )
    contacts = extract_contacts(
        boxes,
        net_weight=weights,
        tolerance=tolerance,
    )

    obligations: list[ContactObligation] = []
    candidates: list[ContactObligation] = []
    component_groups: list[tuple[tuple[int, ...], ...]] = []
    bridge_members: list[int] = []

    for group_index, row in enumerate(groups):
        members = tuple(
            int(index)
            for index in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        )
        components, degree = _components_and_degree(members, contacts)
        component_groups.append(components)
        if len(components) <= 1:
            continue

        bridge_members.extend(
            min(component, key=lambda member: (degree[member], member))
            for component in components
        )
        group_candidates = [
            _best_component_candidate(
                group_index,
                first,
                second,
                boxes,
                weights,
                degree,
                preplaced,
            )
            for first_index, first in enumerate(components)
            for second in components[first_index + 1 :]
        ]
        candidates.extend(group_candidates)
        obligations.extend(_minimum_spanning_tree(components, group_candidates))

    return ContactSynthesis(
        obligations=tuple(obligations),
        candidate_edges=tuple(sorted(candidates, key=_candidate_order)),
        component_groups=tuple(component_groups),
        bridge_members=tuple(bridge_members),
    )


def apply_contact_obligations(
    case: FloorplanCase,
    placements: Any,
    *,
    synthesis: ContactSynthesis | None = None,
    max_candidates: int = 8,
    bridge_only: bool = True,
) -> tuple[Tensor, ...]:
    """Return exact-safe rigid-move challengers for disconnected groups.

    The routine is deliberately bounded and deterministic.  It repeatedly
    re-synthesizes obligations from the latest accepted candidate, then tries
    the selected MST edges followed by the remaining component edges.  A move
    is admitted only when:

    * no member of the moved set is preplaced;
    * the exact verifier reports no overlap, area, fixed-shape, or preplaced
      violation; and
    * the *global* grouping disconnected-component count strictly decreases.

    ``bridge_only`` adds a cheap singleton variation for the selected
    low-degree bridge member.  It is accepted only if it passes the same exact
    checks, so it cannot silently disconnect a previously connected component.
    Returned tensors are detached CPU ``float64`` copies and the input is
    never mutated.
    """

    if int(max_candidates) < 0:
        raise ValueError("max_candidates must be non-negative")
    if not max_candidates:
        return ()

    current = _boxes(placements, int(case.n))
    preplaced = _preplaced_mask(case, int(case.n))
    candidates: list[Tensor] = []
    seen: set[tuple[float, ...]] = set()

    for step in range(int(max_candidates)):
        before = grouping_violation(case, current)
        if before <= 0:
            break
        # Recompute after every accepted move: an obligation's delta is local
        # to the geometry from which it was synthesized.
        current_synthesis = (
            synthesis
            if step == 0 and synthesis is not None
            else synthesize_contact_obligations(
                case,
                current,
                preplaced_mask=preplaced,
            )
        )
        obligations = tuple(
            sorted(
                current_synthesis.obligations + current_synthesis.candidate_edges,
                key=_candidate_order,
            )
        )
        accepted: Tensor | None = None
        for obligation in obligations:
            move_sets = [tuple(sorted(set(obligation.moving_component)))]
            if bridge_only and obligation.bridge_member in move_sets[0]:
                bridge = (int(obligation.bridge_member),)
                if bridge != move_sets[0] and bridge not in move_sets:
                    move_sets.append(bridge)
            for moving in move_sets:
                if bool(preplaced[list(moving)].any()):
                    continue
                moved = current.clone()
                delta = moved.new_tensor(obligation.delta, dtype=torch.float64)
                moved[list(moving), :2] += delta
                if not verify_feasible(case, moved):
                    continue
                if grouping_violation(case, moved) >= before:
                    continue
                key = tuple(
                    round(float(value), 12) for value in moved.reshape(-1).tolist()
                )
                if key in seen:
                    continue
                seen.add(key)
                accepted = moved
                break
            if accepted is not None:
                break
        if accepted is None:
            break
        current = accepted
        candidates.append(current.clone())

        # A caller-provided synthesis is tied to the original geometry; after
        # the first accepted move the loop always uses a fresh synthesis.
        synthesis = None

    return tuple(candidate.detach().cpu() for candidate in candidates)


# Name used by experiment scripts that treat each geometry constructor as a
# candidate family.  Keep the alias tiny rather than introducing a wrapper
# object or a second candidate representation.
contact_synthesis_candidates = apply_contact_obligations


def _boxes(value: Any, n: int) -> Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if boxes.shape != (n, 4):
        raise ValueError(f"placements must have shape ({n}, 4)")
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:4] <= 0.0).any()):
        raise ValueError("placements must be finite with positive dimensions")
    return boxes.clone()


def _groups(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((0, n), dtype=torch.bool)
    groups = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if groups.numel() == 0:
        return torch.zeros((0, n), dtype=torch.bool)
    if groups.ndim == 1:
        groups = groups.reshape(1, -1)
    if groups.ndim != 2 or groups.shape[1] != n:
        raise ValueError(f"group_membership must have shape [G, {n}]")
    return groups


def _preplaced_mask(case: Any, n: int) -> Tensor:
    value = None
    if isinstance(case, dict):
        value = case.get("preplaced_mask", case.get("is_preplaced"))
    else:
        value = getattr(case, "preplaced_mask", getattr(case, "is_preplaced", None))
    if value is None:
        return torch.zeros(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool, device="cpu").reshape(-1)
    if mask.numel() != n:
        raise ValueError(f"preplaced_mask must have length {n}")
    return mask


def _weights(value: Any, n: int) -> Tensor:
    if value is None:
        return torch.zeros((n, n), dtype=torch.float64)
    weights = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if weights.shape != (n, n):
        raise ValueError(f"b2b_weight must have shape ({n}, {n})")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("b2b_weight must be finite and non-negative")
    return weights


def _components_and_degree(
    members: tuple[int, ...], contacts: tuple[Any, ...]
) -> tuple[tuple[tuple[int, ...], ...], dict[int, int]]:
    member_set = set(members)
    adjacency = {member: set() for member in members}
    for contact in contacts:
        if contact.first in member_set and contact.second in member_set:
            adjacency[contact.first].add(contact.second)
            adjacency[contact.second].add(contact.first)

    degree = {member: len(adjacency[member]) for member in members}
    remaining = set(members)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            first = stack.pop()
            neighbors = sorted(adjacency[first] & remaining)
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
            component.extend(neighbors)
        components.append(tuple(sorted(component)))
    return tuple(components), degree


def _best_component_candidate(
    group_index: int,
    component_a: tuple[int, ...],
    component_b: tuple[int, ...],
    boxes: Tensor,
    weights: Tensor,
    degree: dict[int, int],
    preplaced: Tensor | None = None,
) -> ContactObligation:
    choices: list[ContactObligation] = []
    all_choices: list[ContactObligation] = []
    for member_a in component_a:
        for member_b in component_b:
            orientations = (
                _estimate_orientation(
                    group_index,
                    component_a,
                    component_b,
                    member_a,
                    member_b,
                    moving_component=component_b,
                    anchor_member=member_a,
                    moving_member=member_b,
                    boxes=boxes,
                    weights=weights,
                    degree=degree,
                ),
                _estimate_orientation(
                    group_index,
                    component_a,
                    component_b,
                    member_a,
                    member_b,
                    moving_component=component_a,
                    anchor_member=member_b,
                    moving_member=member_a,
                    boxes=boxes,
                    weights=weights,
                    degree=degree,
                ),
            )
            all_choices.extend(orientations)
            if preplaced is None:
                choices.extend(orientations)
            else:
                choices.extend(
                    candidate
                    for candidate in orientations
                    if not bool(preplaced[list(candidate.moving_component)].any())
                )
    return min(choices or all_choices, key=_candidate_order)


def _estimate_orientation(
    group_index: int,
    component_a: tuple[int, ...],
    component_b: tuple[int, ...],
    member_a: int,
    member_b: int,
    *,
    moving_component: tuple[int, ...],
    anchor_member: int,
    moving_member: int,
    boxes: Tensor,
    weights: Tensor,
    degree: dict[int, int],
) -> ContactObligation:
    anchor = boxes[anchor_member]
    child = boxes[moving_member]
    target_options = _contact_targets(anchor, child)
    options: list[tuple[tuple[float, float, float, int, str], str, Tensor]] = []
    for side, target, _ in target_options:
        delta = target.to(dtype=torch.float64) - child[:2]
        moved = boxes.clone()
        moved[list(moving_component), :2] += delta
        distance = float(torch.linalg.vector_norm(delta).item()) * len(moving_component)
        expansion = max(0.0, _bbox_area(moved) - _bbox_area(boxes))
        net_incident = _cross_net_incident(
            member_a,
            member_b,
            component_a,
            component_b,
            weights,
        )
        key = (distance, expansion, -net_incident, moving_member, side)
        options.append((key, side, delta))
    _, side, delta = min(options, key=lambda item: item[0])
    net_incident = _cross_net_incident(
        member_a,
        member_b,
        component_a,
        component_b,
        weights,
    )
    moved = boxes.clone()
    moved[list(moving_component), :2] += delta
    distance = float(torch.linalg.vector_norm(delta).item()) * len(moving_component)
    expansion = max(0.0, _bbox_area(moved) - _bbox_area(boxes))
    bridge = min((member_a, member_b), key=lambda member: (degree[member], member))
    return ContactObligation(
        group_index=group_index,
        component_a=component_a,
        component_b=component_b,
        member_a=member_a,
        member_b=member_b,
        bridge_member=bridge,
        moving_component=moving_component,
        moving_member=moving_member,
        anchor_member=anchor_member,
        side=side,
        delta=(float(delta[0]), float(delta[1])),
        move_distance=distance,
        bbox_expansion=expansion,
        net_incident=net_incident,
    )


def _cross_net_incident(
    member_a: int,
    member_b: int,
    component_a: tuple[int, ...],
    component_b: tuple[int, ...],
    weights: Tensor,
) -> float:
    incident_a = sum(float(weights[member_a, member].item()) for member in component_b)
    incident_b = sum(float(weights[member_b, member].item()) for member in component_a)
    return incident_a + incident_b


def _bbox_area(boxes: Tensor) -> float:
    left = float(boxes[:, 0].amin().item())
    bottom = float(boxes[:, 1].amin().item())
    right = float((boxes[:, 0] + boxes[:, 2]).amax().item())
    top = float((boxes[:, 1] + boxes[:, 3]).amax().item())
    return max(0.0, right - left) * max(0.0, top - bottom)


def _candidate_order(candidate: ContactObligation) -> tuple[object, ...]:
    return (
        candidate.move_distance,
        candidate.bbox_expansion,
        -candidate.net_incident,
        candidate.group_index,
        candidate.component_a,
        candidate.component_b,
        candidate.member_a,
        candidate.member_b,
        candidate.moving_member,
        candidate.side,
    )


def _minimum_spanning_tree(
    components: tuple[tuple[int, ...], ...],
    candidates: list[ContactObligation],
) -> tuple[ContactObligation, ...]:
    parent = {component: component for component in components}

    def find(component: tuple[int, ...]) -> tuple[int, ...]:
        root = component
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    chosen: list[ContactObligation] = []
    for candidate in sorted(candidates, key=_candidate_order):
        first_root = find(candidate.component_a)
        second_root = find(candidate.component_b)
        if first_root == second_root:
            continue
        parent[second_root] = first_root
        chosen.append(candidate)
        if len(chosen) == len(components) - 1:
            break
    return tuple(chosen)


__all__ = [
    "ContactObligation",
    "ContactSynthesis",
    "apply_contact_obligations",
    "contact_synthesis_candidates",
    "synthesize_contact_obligations",
]
