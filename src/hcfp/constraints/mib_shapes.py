"""Hard-by-construction MIB shape resolution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


OFFICIAL_AREA_REL_TOL = 1.0e-2


@dataclass(frozen=True)
class MibGroupResolution:
    group: int
    members: tuple[int, ...]
    shape: tuple[float, float] | None
    compatible: bool
    anchor: int | None
    reason: str


@dataclass(frozen=True)
class MibShapeResolution:
    shapes: torch.Tensor
    groups: tuple[MibGroupResolution, ...]

    @property
    def incompatible_groups(self) -> tuple[MibGroupResolution, ...]:
        return tuple(group for group in self.groups if not group.compatible)


def resolve_mib_shapes(
    area: torch.Tensor,
    mib_membership: torch.Tensor,
    *,
    proposed_wh: torch.Tensor | None = None,
    hard_mask: torch.Tensor | None = None,
    hard_wh: torch.Tensor | None = None,
    rel_tol: float = OFFICIAL_AREA_REL_TOL,
) -> MibShapeResolution:
    """Resolve exact shared MIB shapes while preserving hard member shapes."""

    areas = torch.as_tensor(area, dtype=torch.float64, device="cpu").reshape(-1)
    if not bool(torch.isfinite(areas).all()) or bool((areas <= 0.0).any()):
        raise ValueError("area must be finite and positive")
    n = int(areas.numel())
    membership = torch.as_tensor(mib_membership, dtype=torch.bool, device="cpu")
    if membership.ndim != 2 or membership.shape[1] != n:
        raise ValueError("mib_membership must have shape [M,N]")
    shapes = _initial_shapes(areas, proposed_wh)
    hard = _hard_mask(hard_mask, n)
    hard_shapes = _hard_shapes(hard_wh, n)
    if hard_shapes is not None:
        shapes[hard] = hard_shapes[hard]

    reports: list[MibGroupResolution] = []
    for group_index, row in enumerate(membership):
        members = tuple(int(i) for i in torch.nonzero(row, as_tuple=False).reshape(-1).tolist())
        if len(members) < 2:
            continue
        group_hard = [index for index in members if bool(hard[index])]
        anchor = group_hard[0] if group_hard else None
        compatible = True
        reason = "resolved"
        member_index = list(members)
        if group_hard:
            shape = shapes[anchor].clone()
            if any(not bool(torch.equal(shapes[index], shape)) for index in group_hard[1:]):
                compatible = False
                reason = "conflicting hard shapes"
        else:
            shape = _soft_group_shape(areas[member_index], shapes[member_index], rel_tol)
            if shape is None:
                compatible = False
                reason = "empty area-tolerance intersection"

        if compatible and shape is not None and not _within_area_tolerance(areas[member_index], shape, rel_tol):
            compatible = False
            reason = "hard shape violates member area tolerance" if group_hard else "shape violates area tolerance"
        if compatible and shape is not None:
            for index in members:
                shapes[index] = shape

        reports.append(
            MibGroupResolution(
                group=group_index,
                members=members,
                shape=(float(shape[0]), float(shape[1])) if compatible and shape is not None else None,
                compatible=compatible,
                anchor=anchor,
                reason=reason,
            )
        )

    return MibShapeResolution(shapes=shapes.to(dtype=torch.float32), groups=tuple(reports))


def _initial_shapes(area: torch.Tensor, proposed_wh: torch.Tensor | None) -> torch.Tensor:
    if proposed_wh is None:
        side = torch.sqrt(area)
        return torch.stack((side, side), dim=1)
    shapes = torch.as_tensor(proposed_wh, dtype=torch.float64, device="cpu").clone()
    if shapes.shape != (area.numel(), 2):
        raise ValueError("proposed_wh must have shape [N,2]")
    if not bool(torch.isfinite(shapes).all()) or bool((shapes <= 0.0).any()):
        raise ValueError("proposed_wh must be finite with positive dimensions")
    return shapes


def _hard_mask(mask: torch.Tensor | None, n: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros(n, dtype=torch.bool)
    out = torch.as_tensor(mask, dtype=torch.bool, device="cpu").reshape(-1)
    if out.numel() != n:
        raise ValueError("hard_mask must have shape [N]")
    return out


def _hard_shapes(hard_wh: torch.Tensor | None, n: int) -> torch.Tensor | None:
    if hard_wh is None:
        return None
    shapes = torch.as_tensor(hard_wh, dtype=torch.float64, device="cpu")
    if shapes.shape != (n, 2):
        raise ValueError("hard_wh must have shape [N,2]")
    if not bool(torch.isfinite(shapes).all()) or bool((shapes <= 0.0).any()):
        raise ValueError("hard_wh must be finite with positive dimensions")
    return shapes


def _soft_group_shape(area: torch.Tensor, proposed: torch.Tensor, rel_tol: float) -> torch.Tensor | None:
    low = torch.max(area * (1.0 - rel_tol))
    high = torch.min(area * (1.0 + rel_tol))
    if float(low) > float(high):
        return None
    target_area = (low + high) * 0.5
    aspect = float(proposed[0, 0] / proposed[0, 1])
    width = torch.sqrt(target_area * aspect)
    return torch.stack((width, target_area / width))


def _within_area_tolerance(area: torch.Tensor, shape: torch.Tensor, rel_tol: float) -> bool:
    actual = shape[0] * shape[1]
    rel = torch.abs(actual - area) / torch.clamp(area, min=1.0e-300)
    return bool((rel <= rel_tol).all())
