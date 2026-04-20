from __future__ import annotations

from dataclasses import dataclass

from puzzleplace.actions import ActionPrimitive, TypedAction
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.roles import RoleLabel, WeakRoleEvidence, label_case_roles


@dataclass(slots=True)
class PseudoTrace:
    name: str
    actions: list[TypedAction]
    ordered_blocks: list[int]
    notes: list[str]


def _boundary_first(case: FloorSetCase) -> list[int]:
    def key(idx: int) -> tuple[int, int]:
        boundary = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item() != 0)
        fixed = int(case.constraints[idx, ConstraintColumns.FIXED].item() != 0 or case.constraints[idx, ConstraintColumns.PREPLACED].item() != 0)
        return (-fixed, -boundary)
    return sorted(range(case.block_count), key=key)


def _area_desc(case: FloorSetCase) -> list[int]:
    return sorted(range(case.block_count), key=lambda idx: float(case.area_targets[idx].item()), reverse=True)


def _connectivity_hub_first(case: FloorSetCase) -> list[int]:
    degrees = [0.0 for _ in range(case.block_count)]
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i < case.block_count:
            degrees[i] += float(weight)
        if j < case.block_count:
            degrees[j] += float(weight)
    for _pin, block, weight in case.p2b_edges.tolist():
        j = int(block)
        if j < case.block_count:
            degrees[j] += float(weight)
    return sorted(range(case.block_count), key=lambda idx: degrees[idx], reverse=True)


def _role_priority(case: FloorSetCase, roles: list[WeakRoleEvidence]) -> list[int]:
    priority = {
        RoleLabel.FIXED_ANCHOR: 0,
        RoleLabel.PREPLACED_ANCHOR: 1,
        RoleLabel.BOUNDARY_SEEKER: 2,
        RoleLabel.CONNECTIVITY_HUB: 3,
        RoleLabel.CLUSTER_MEMBER: 4,
        RoleLabel.MULTI_INSTANCE: 5,
        RoleLabel.AREA_FOLLOWER: 6,
    }
    by_index = {role.block_index: role.role for role in roles}
    return sorted(range(case.block_count), key=lambda idx: (priority[by_index[idx]], idx))


def _trace_from_order(case: FloorSetCase, name: str, order: list[int]) -> PseudoTrace:
    if case.target_positions is None:
        raise ValueError("Pseudo trajectories require case.target_positions as pseudo labels")
    actions: list[TypedAction] = []
    notes = [f"generated from final layout using strategy={name}"]
    for block_index in order:
        x, y, w, h = [float(v) for v in case.target_positions[block_index].tolist()]
        actions.append(
            TypedAction(
                primitive=ActionPrimitive.PLACE_ABSOLUTE,
                block_index=block_index,
                x=x,
                y=y,
                w=w,
                h=h,
                metadata={"strategy": name, "source": "pseudo_label_from_final_layout"},
            )
        )
        if bool(case.constraints[block_index, ConstraintColumns.FIXED].item()) or bool(
            case.constraints[block_index, ConstraintColumns.PREPLACED].item()
        ):
            actions.append(TypedAction(primitive=ActionPrimitive.FREEZE, block_index=block_index, metadata={"strategy": name}))
    return PseudoTrace(name=name, actions=actions, ordered_blocks=order, notes=notes)


def generate_pseudo_traces(case: FloorSetCase, *, max_traces: int = 4) -> list[PseudoTrace]:
    roles = label_case_roles(case)
    strategies: list[tuple[str, list[int]]] = [
        ("index_order", list(range(case.block_count))),
        ("boundary_first", _boundary_first(case)),
        ("area_desc", _area_desc(case)),
        ("hub_first", _connectivity_hub_first(case)),
        ("role_priority", _role_priority(case, roles)),
    ]
    traces: list[PseudoTrace] = []
    seen_orders: set[tuple[int, ...]] = set()
    for name, order in strategies:
        key = tuple(order)
        if key in seen_orders:
            continue
        seen_orders.add(key)
        traces.append(_trace_from_order(case, name, order))
        if len(traces) >= max_traces:
            break
    return traces
