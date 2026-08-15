"""Deterministic contact labels and contact-tree extraction from gold geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch


LEFT = "LEFT"
RIGHT = "RIGHT"
TOP = "TOP"
BOTTOM = "BOTTOM"
Side = Literal["LEFT", "RIGHT", "TOP", "BOTTOM"]


@dataclass(frozen=True)
class Contact:
    first: int
    second: int
    first_side: Side
    second_side: Side
    length: float
    center_distance: float
    net_weight: float
    score: float

    @property
    def sequence_precedence(self) -> tuple[int, int, Literal["horizontal", "vertical"]]:
        """Return the block order implied by this side contact."""

        if self.first_side == RIGHT:
            return self.first, self.second, "horizontal"
        if self.first_side == LEFT:
            return self.second, self.first, "horizontal"
        if self.first_side == TOP:
            return self.second, self.first, "vertical"
        return self.first, self.second, "vertical"


@dataclass(frozen=True)
class ContactTree:
    group_index: int
    members: tuple[int, ...]
    edges: tuple[Contact, ...]
    connected: bool


@dataclass(frozen=True)
class ContactTreeReport:
    contacts: tuple[Contact, ...]
    trees: tuple[ContactTree, ...]
    disconnected_groups: tuple[int, ...]


def extract_contacts(
    xywh: Any,
    *,
    net_weight: Any | None = None,
    tolerance: float,
    contact_length_weight: float = 1.0,
    net_weight_weight: float = 1.0,
    distance_weight: float = 1.0e-6,
) -> tuple[Contact, ...]:
    """Return side-specific contacts for every touching rectangle pair."""

    boxes = _boxes(xywh)
    _check_tolerance(tolerance)
    weights = _weights(net_weight, int(boxes.shape[0]))
    coeffs = (contact_length_weight, net_weight_weight, distance_weight)
    if not all(math.isfinite(float(value)) for value in coeffs):
        raise ValueError("contact scoring weights must be finite")

    n = int(boxes.shape[0])
    if n < 2:
        return ()
    left, bottom = boxes[:, 0], boxes[:, 1]
    right, top = left + boxes[:, 2], bottom + boxes[:, 3]
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]

    # Vectorized pair scan over the strict upper triangle (first < second).
    first_index, second_index = torch.triu_indices(n, n, offset=1)
    first_left, first_right = left[first_index], right[first_index]
    first_bottom, first_top = bottom[first_index], top[first_index]
    second_left, second_right = left[second_index], right[second_index]
    second_bottom, second_top = bottom[second_index], top[second_index]

    y_overlap = (torch.minimum(first_top, second_top) - torch.maximum(first_bottom, second_bottom)).clamp_min(0.0)
    x_overlap = (torch.minimum(first_right, second_right) - torch.maximum(first_left, second_left)).clamp_min(0.0)
    horizontal_gap = torch.minimum(
        (first_right - second_left).abs(), (second_right - first_left).abs()
    )
    vertical_gap = torch.minimum(
        (first_top - second_bottom).abs(), (second_top - first_bottom).abs()
    )

    horizontal = (y_overlap > 0.0) & (horizontal_gap <= tolerance)
    vertical = (x_overlap > 0.0) & (vertical_gap <= tolerance)
    candidates = horizontal | vertical
    if not bool(candidates.any()):
        return ()

    pair_index = torch.nonzero(candidates, as_tuple=False).reshape(-1)
    contacts: list[Contact] = []
    for offset in pair_index.tolist():
        first = int(first_index[offset])
        second = int(second_index[offset])
        if bool(horizontal[offset]):
            if abs(float(right[first] - left[second])) <= tolerance:
                sides: tuple[Side, Side] = (RIGHT, LEFT)
            else:
                sides = (LEFT, RIGHT)
            contacts.append(
                _contact(
                    first,
                    second,
                    sides,
                    float(y_overlap[offset]),
                    centers,
                    weights,
                    coeffs,
                )
            )
        if bool(vertical[offset]):
            if abs(float(top[first] - bottom[second])) <= tolerance:
                sides = (TOP, BOTTOM)
            else:
                sides = (BOTTOM, TOP)
            contacts.append(
                _contact(
                    first,
                    second,
                    sides,
                    float(x_overlap[offset]),
                    centers,
                    weights,
                    coeffs,
                )
            )
    return tuple(sorted(contacts, key=_contact_order))


def contact_tree_report(
    xywh: Any,
    groups: Any,
    *,
    net_weight: Any | None = None,
    tolerance: float,
    fail_on_disconnected: bool = True,
    contact_length_weight: float = 1.0,
    net_weight_weight: float = 1.0,
    distance_weight: float = 1.0e-6,
) -> ContactTreeReport:
    """Build deterministic weighted maximum spanning trees for each group."""

    boxes = _boxes(xywh)
    membership = _groups(groups, int(boxes.shape[0]))
    contacts = extract_contacts(
        boxes,
        net_weight=net_weight,
        tolerance=tolerance,
        contact_length_weight=contact_length_weight,
        net_weight_weight=net_weight_weight,
        distance_weight=distance_weight,
    )
    trees: list[ContactTree] = []
    disconnected: list[int] = []
    for group_index, row in enumerate(membership):
        members = tuple(
            int(i)
            for i in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()
        )
        if len(members) <= 1:
            trees.append(ContactTree(group_index, members, (), True))
            continue
        member_set = set(members)
        candidates = [
            contact
            for contact in contacts
            if contact.first in member_set and contact.second in member_set
        ]
        edges, connected = _maximum_spanning_tree(members, candidates)
        if not connected:
            disconnected.append(group_index)
        trees.append(ContactTree(group_index, members, edges, connected))

    report = ContactTreeReport(contacts, tuple(trees), tuple(disconnected))
    if fail_on_disconnected and disconnected:
        raise ValueError(f"disconnected contact graph for group(s): {tuple(disconnected)}")
    return report


def _contact(
    first: int,
    second: int,
    sides: tuple[Side, Side],
    length: float,
    centers: torch.Tensor,
    weights: torch.Tensor,
    coeffs: tuple[float, float, float],
) -> Contact:
    distance = float(torch.abs(centers[first] - centers[second]).sum().item())
    net = float(weights[first, second].item())
    score = float(coeffs[0]) * length + float(coeffs[1]) * net - float(coeffs[2]) * distance
    return Contact(first, second, sides[0], sides[1], length, distance, net, score)


def _maximum_spanning_tree(
    members: tuple[int, ...],
    candidates: list[Contact],
) -> tuple[tuple[Contact, ...], bool]:
    parent = {member: member for member in members}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    chosen: list[Contact] = []
    for contact in sorted(candidates, key=_tree_order):
        first_root = find(contact.first)
        second_root = find(contact.second)
        if first_root == second_root:
            continue
        parent[second_root] = first_root
        chosen.append(contact)
        if len(chosen) == len(members) - 1:
            break
    return tuple(chosen), len(chosen) == len(members) - 1


def _boxes(xywh: Any) -> torch.Tensor:
    boxes = torch.as_tensor(xywh, dtype=torch.float64, device="cpu")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("xywh must have shape [N, 4]")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("xywh must be finite")
    if not bool((boxes[:, 2:4] > 0.0).all()):
        raise ValueError("rectangle width and height must be positive")
    return boxes


def _weights(net_weight: Any | None, n: int) -> torch.Tensor:
    if net_weight is None:
        return torch.zeros((n, n), dtype=torch.float64)
    weights = torch.as_tensor(net_weight, dtype=torch.float64, device="cpu")
    if weights.shape != (n, n):
        raise ValueError("net_weight must have shape [N, N]")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("net_weight must be finite and non-negative")
    return weights


def _groups(groups: Any, n: int) -> torch.Tensor:
    membership = torch.as_tensor(groups, dtype=torch.bool, device="cpu")
    if membership.numel() == 0:
        return torch.zeros((0, n), dtype=torch.bool)
    if membership.ndim == 1:
        membership = membership.reshape(1, -1)
    if membership.ndim != 2 or membership.shape[1] != n:
        raise ValueError("groups must have shape [G, N] or [N]")
    return membership


def _check_tolerance(tolerance: float) -> None:
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")


def _overlap(a_min: torch.Tensor, a_max: torch.Tensor, b_min: torch.Tensor, b_max: torch.Tensor) -> float:
    return max(0.0, min(float(a_max), float(b_max)) - max(float(a_min), float(b_min)))


def _contact_order(contact: Contact) -> tuple[int, int, str, str]:
    return (contact.first, contact.second, contact.first_side, contact.second_side)


def _tree_order(contact: Contact) -> tuple[float, float, float, float, int, int, str, str]:
    return (
        -contact.score,
        -contact.length,
        -contact.net_weight,
        contact.center_distance,
        contact.first,
        contact.second,
        contact.first_side,
        contact.second_side,
    )
