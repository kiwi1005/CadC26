from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.research.move_library import grouping_violation_count, mib_violation_count
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame

RegionId = str


def box_center(box: Placement) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def block_area(box: Placement) -> float:
    return max(box[2], 0.0) * max(box[3], 0.0)


def build_region_grid(frame: PuzzleFrame, *, rows: int = 4, cols: int = 4) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    cell_w = frame.width / max(cols, 1)
    cell_h = frame.height / max(rows, 1)
    for row in range(rows):
        for col in range(cols):
            xmin = frame.xmin + col * cell_w
            ymin = frame.ymin + row * cell_h
            cells.append(
                {
                    "region_id": region_id(row, col),
                    "row": row,
                    "col": col,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmin + cell_w,
                    "ymax": ymin + cell_h,
                    "area": cell_w * cell_h,
                }
            )
    return cells


def region_id(row: int, col: int) -> RegionId:
    return f"r{row}_{col}"


def point_region(
    x: float,
    y: float,
    frame: PuzzleFrame,
    *,
    rows: int = 4,
    cols: int = 4,
) -> RegionId:
    col = int((x - frame.xmin) / max(frame.width, 1e-9) * cols)
    row = int((y - frame.ymin) / max(frame.height, 1e-9) * rows)
    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)
    return region_id(row, col)


def region_grid_distance(left: RegionId, right: RegionId) -> int:
    lr, lc = _region_coords(left)
    rr, rc = _region_coords(right)
    return abs(lr - rr) + abs(lc - rc)


def _region_coords(value: RegionId) -> tuple[int, int]:
    raw = value.removeprefix("r")
    row, col = raw.split("_", 1)
    return int(row), int(col)


def region_center(cell: dict[str, Any]) -> tuple[float, float]:
    return (float(cell["xmin"]) + float(cell["xmax"])) / 2.0, (
        float(cell["ymin"]) + float(cell["ymax"])
    ) / 2.0


def cell_overlap_area(box: Placement, cell: dict[str, Any]) -> float:
    x, y, w, h = box
    ix0 = max(x, float(cell["xmin"]))
    iy0 = max(y, float(cell["ymin"]))
    ix1 = min(x + w, float(cell["xmax"]))
    iy1 = min(y + h, float(cell["ymax"]))
    return max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)


def region_occupancy(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    rows: int = 4,
    cols: int = 4,
) -> dict[str, Any]:
    cells = build_region_grid(frame, rows=rows, cols=cols)
    rows_out: list[dict[str, Any]] = []
    for cell in cells:
        block_area_sum = 0.0
        fixed_area = 0.0
        preplaced_area = 0.0
        boundary_area = 0.0
        block_count = 0
        for idx, box in placements.items():
            overlap = cell_overlap_area(box, cell)
            if overlap <= 0.0:
                continue
            block_area_sum += overlap
            block_count += 1
            if bool(case.constraints[idx, ConstraintColumns.FIXED].item()):
                fixed_area += overlap
            if bool(case.constraints[idx, ConstraintColumns.PREPLACED].item()):
                preplaced_area += overlap
            if bool(case.constraints[idx, ConstraintColumns.BOUNDARY].item()):
                boundary_area += overlap
        capacity = float(cell["area"])
        utilization = block_area_sum / max(capacity, 1e-9)
        rows_out.append(
            {
                **cell,
                "block_area": block_area_sum,
                "block_count": block_count,
                "fixed_area": fixed_area,
                "preplaced_area": preplaced_area,
                "boundary_area": boundary_area,
                "unused_capacity": max(capacity - block_area_sum, 0.0),
                "overflow_area": max(block_area_sum - capacity, 0.0),
                "underflow_area": max(capacity - block_area_sum, 0.0),
                "utilization": utilization,
            }
        )
    utilizations = [float(row["utilization"]) for row in rows_out]
    return {
        "case_id": case.case_id,
        "grid": {"rows": rows, "cols": cols},
        "regions": rows_out,
        "max_utilization": max(utilizations, default=0.0),
        "min_utilization": min(utilizations, default=0.0),
        "utilization_spread": max(utilizations, default=0.0) - min(utilizations, default=0.0),
        "overflow_region_count": sum(int(value > 1.0) for value in utilizations),
    }


def pin_density_regions(
    case: FloorSetCase,
    frame: PuzzleFrame,
    *,
    rows: int = 4,
    cols: int = 4,
) -> dict[str, Any]:
    cells = {
        cell["region_id"]: {**cell, "pin_count": 0, "terminal_pull": 0.0}
        for cell in build_region_grid(frame, rows=rows, cols=cols)
    }
    for pin in case.pins_pos.tolist():
        x, y = float(pin[0]), float(pin[1])
        cells[point_region(x, y, frame, rows=rows, cols=cols)]["pin_count"] += 1
    for pin_idx, _block_idx, weight in case.p2b_edges.tolist():
        pin_i = int(pin_idx)
        if 0 <= pin_i < len(case.pins_pos):
            x, y = [float(v) for v in case.pins_pos[pin_i].tolist()]
            cells[point_region(x, y, frame, rows=rows, cols=cols)]["terminal_pull"] += abs(
                float(weight)
            )
    region_rows = []
    for row in cells.values():
        area = max(float(row["area"]), 1e-9)
        region_rows.append(
            {
                **row,
                "pin_density": float(row["pin_count"]) / area,
                "terminal_pull_density": float(row["terminal_pull"]) / area,
            }
        )
    return {
        "case_id": case.case_id,
        "grid": {"rows": rows, "cols": cols},
        "regions": sorted(region_rows, key=lambda row: row["region_id"]),
    }


def net_community_clusters(case: FloorSetCase) -> dict[str, Any]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(case.block_count)}
    weights = [abs(float(row[2])) for row in case.b2b_edges.tolist()]
    threshold = percentile(weights, 0.70) if weights else 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if abs(float(weight)) >= threshold:
            adjacency[i].add(j)
            adjacency[j].add(i)
    _connect_constraint_members(case, adjacency, ConstraintColumns.MIB)
    _connect_constraint_members(case, adjacency, ConstraintColumns.CLUSTER)
    components = connected_components(adjacency)
    rows: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(components):
        member_set = set(members)
        area = sum(float(case.area_targets[idx].item()) for idx in members)
        internal_degree = 0.0
        external_degree = 0.0
        for src, dst, weight in case.b2b_edges.tolist():
            i, j = int(src), int(dst)
            w = abs(float(weight))
            if i in member_set and j in member_set:
                internal_degree += w
            elif i in member_set or j in member_set:
                external_degree += w
        terminal_pull = 0.0
        for _pin_idx, block_idx, weight in case.p2b_edges.tolist():
            if int(block_idx) in member_set:
                terminal_pull += abs(float(weight))
        rows.append(
            {
                "cluster_id": cluster_id,
                "members": members,
                "size": len(members),
                "area": area,
                "internal_degree": internal_degree,
                "external_degree": external_degree,
                "external_terminal_pull": terminal_pull,
                "mib_member_count": sum(
                    int(bool(case.constraints[idx, ConstraintColumns.MIB].item()))
                    for idx in members
                ),
                "group_member_count": sum(
                    int(bool(case.constraints[idx, ConstraintColumns.CLUSTER].item()))
                    for idx in members
                ),
            }
        )
    return {
        "case_id": case.case_id,
        "edge_weight_threshold": threshold,
        "cluster_count": len(rows),
        "clusters": rows,
    }


def _connect_constraint_members(
    case: FloorSetCase,
    adjacency: dict[int, set[int]],
    column: ConstraintColumns,
) -> None:
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(case.block_count):
        group_id = int(float(case.constraints[idx, column].item()))
        if group_id > 0:
            groups[group_id].append(idx)
    for members in groups.values():
        for left, right in zip(members, members[1:], strict=False):
            adjacency[left].add(right)
            adjacency[right].add(left)


def connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    unseen = set(adjacency)
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        queue: deque[int] = deque([start])
        unseen.remove(start)
        members: list[int] = []
        while queue:
            node = queue.popleft()
            members.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(members))
    return sorted(components, key=lambda row: (-len(row), row[0]))


def block_region_assignment(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    clusters: dict[str, Any],
    *,
    rows: int = 4,
    cols: int = 4,
) -> dict[str, Any]:
    cluster_by_block = {
        int(member): int(cluster["cluster_id"])
        for cluster in clusters["clusters"]
        for member in cluster["members"]
    }
    assignments: list[dict[str, Any]] = []
    for idx, box in sorted(placements.items()):
        actual_x, actual_y = box_center(box)
        expected_x, expected_y = expected_block_centroid(case, idx, placements)
        actual_region = point_region(actual_x, actual_y, frame, rows=rows, cols=cols)
        expected_region = point_region(expected_x, expected_y, frame, rows=rows, cols=cols)
        assignments.append(
            {
                "block_id": idx,
                "cluster_id": cluster_by_block.get(idx, -1),
                "actual_center": [actual_x, actual_y],
                "expected_center": [expected_x, expected_y],
                "actual_region": actual_region,
                "expected_region": expected_region,
                "mismatch_grid_distance": region_grid_distance(actual_region, expected_region),
                "mismatch_distance_norm": math.hypot(actual_x - expected_x, actual_y - expected_y)
                / max(math.hypot(frame.width, frame.height), 1e-9),
                "is_mib": bool(case.constraints[idx, ConstraintColumns.MIB].item()),
                "is_grouping": bool(case.constraints[idx, ConstraintColumns.CLUSTER].item()),
                "is_boundary": bool(case.constraints[idx, ConstraintColumns.BOUNDARY].item()),
            }
        )
    cluster_rows: list[dict[str, Any]] = []
    for cluster in clusters["clusters"]:
        members = [row for row in assignments if row["block_id"] in set(cluster["members"])]
        actual_counts = Counter(str(row["actual_region"]) for row in members)
        expected_counts = Counter(str(row["expected_region"]) for row in members)
        cluster_rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "actual_region_count": len(actual_counts),
                "expected_region_count": len(expected_counts),
                "actual_region_entropy": normalized_entropy(actual_counts.values()),
                "expected_region_entropy": normalized_entropy(expected_counts.values()),
                "mean_mismatch_grid_distance": mean(
                    float(row["mismatch_grid_distance"]) for row in members
                ),
                "mib_member_count": cluster["mib_member_count"],
                "group_member_count": cluster["group_member_count"],
            }
        )
    mismatches = [float(row["mismatch_grid_distance"]) for row in assignments]
    return {
        "case_id": case.case_id,
        "grid": {"rows": rows, "cols": cols},
        "assignments": assignments,
        "clusters": cluster_rows,
        "mean_mismatch_grid_distance": mean(mismatches),
        "p90_mismatch_grid_distance": percentile(mismatches, 0.90),
        "large_mismatch_count": sum(int(value >= 2.0) for value in mismatches),
        "assignment_entropy": mean(float(row["actual_region_entropy"]) for row in cluster_rows),
        "max_cluster_spread_regions": max(
            (int(row["actual_region_count"]) for row in cluster_rows),
            default=0,
        ),
        "mib_group_spread_regions": max(
            (
                int(row["actual_region_count"])
                for row in cluster_rows
                if int(row["mib_member_count"]) + int(row["group_member_count"]) > 0
            ),
            default=0,
        ),
    }


def expected_block_centroid(
    case: FloorSetCase,
    block_id: int,
    placements: dict[int, Placement],
) -> tuple[float, float]:
    points: list[tuple[float, float, float]] = []
    for pin_idx, dst, weight in case.p2b_edges.tolist():
        if int(dst) == block_id and 0 <= int(pin_idx) < len(case.pins_pos):
            px, py = [float(v) for v in case.pins_pos[int(pin_idx)].tolist()]
            points.append((px, py, abs(float(weight))))
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        other = j if i == block_id else i if j == block_id else None
        if other is not None and other in placements:
            ox, oy = box_center(placements[other])
            points.append((ox, oy, abs(float(weight)) * 0.25))
    if not points:
        return box_center(placements[block_id])
    total = sum(max(weight, 1e-6) for _x, _y, weight in points)
    return (
        sum(x * max(weight, 1e-6) for x, _y, weight in points) / total,
        sum(y * max(weight, 1e-6) for _x, y, weight in points) / total,
    )


def free_space_fragmentation(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    rows: int = 12,
    cols: int = 12,
) -> dict[str, Any]:
    cells = build_region_grid(frame, rows=rows, cols=cols)
    occupied: dict[RegionId, float] = {}
    for cell in cells:
        occ_area = sum(cell_overlap_area(box, cell) for box in placements.values())
        occupied[cell["region_id"]] = occ_area / max(float(cell["area"]), 1e-9)
    empty = {rid for rid, occ in occupied.items() if occ < 0.15}
    components = empty_components(empty, rows=rows, cols=cols)
    empty_area = sum(float(cell["area"]) for cell in cells if cell["region_id"] in empty)
    largest = max((len(component) for component in components), default=0)
    total_empty_cells = max(len(empty), 1)
    sliver_count = sum(
        int(_occupied_neighbor_count(rid, occupied, rows=rows, cols=cols) >= 3) for rid in empty
    )
    fragmentation_score = (
        len(components) / max(rows * cols, 1)
        + (1.0 - largest / total_empty_cells)
        + sliver_count / max(total_empty_cells, 1)
    )
    return {
        "case_id": case.case_id,
        "grid": {"rows": rows, "cols": cols},
        "empty_cell_count": len(empty),
        "empty_area_ratio": empty_area / max(frame.area, 1e-9),
        "empty_component_count": len(components),
        "largest_empty_component_cell_fraction": largest / total_empty_cells,
        "sliver_cell_count": sliver_count,
        "fragmentation_score": fragmentation_score,
        "cell_occupancy": [
            {"region_id": rid, "occupancy": occupied[rid]} for rid in sorted(occupied)
        ],
    }


def empty_components(empty: set[RegionId], *, rows: int, cols: int) -> list[list[RegionId]]:
    unseen = set(empty)
    components: list[list[RegionId]] = []
    while unseen:
        start = min(unseen)
        queue: deque[RegionId] = deque([start])
        unseen.remove(start)
        members: list[RegionId] = []
        while queue:
            rid = queue.popleft()
            members.append(rid)
            row, col = _region_coords(rid)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor = region_id(nr, nc)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        components.append(sorted(members))
    return components


def _occupied_neighbor_count(
    rid: RegionId,
    occupied: dict[RegionId, float],
    *,
    rows: int,
    cols: int,
) -> int:
    row, col = _region_coords(rid)
    count = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            count += int(occupied.get(region_id(nr, nc), 0.0) >= 0.50)
    return count


def moved_region_count(
    baseline: dict[int, Placement],
    alternative: dict[int, Placement],
    frame: PuzzleFrame,
    *,
    rows: int = 4,
    cols: int = 4,
) -> int:
    regions: set[RegionId] = set()
    for idx, before in baseline.items():
        after = alternative.get(idx)
        if after is None:
            continue
        before_center = box_center(before)
        after_center = box_center(after)
        if (
            math.hypot(
                before_center[0] - after_center[0],
                before_center[1] - after_center[1],
            )
            > 1e-6
        ):
            regions.add(point_region(*box_center(before), frame, rows=rows, cols=cols))
            regions.add(point_region(*box_center(after), frame, rows=rows, cols=cols))
    return len(regions)


def repair_radius_metrics(
    case: FloorSetCase,
    baseline: dict[int, Placement],
    alternative: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[str, Any]:
    moved: list[int] = []
    displacements: list[float] = []
    for idx, before in baseline.items():
        after = alternative.get(idx)
        if after is None:
            continue
        bx, by = box_center(before)
        ax, ay = box_center(after)
        disp = math.hypot(ax - bx, ay - by)
        if disp > 1e-6 or abs(before[2] - after[2]) > 1e-6 or abs(before[3] - after[3]) > 1e-6:
            moved.append(idx)
            displacements.append(disp / max(math.sqrt(block_area(before)), 1e-9))
    return {
        "moved_block_count": len(moved),
        "moved_block_fraction": len(moved) / max(len(baseline), 1),
        "max_displacement_norm": max(displacements, default=0.0),
        "mean_displacement_norm": mean(displacements),
        "displacement_chain_length": moved_component_size(case, set(moved)),
        "affected_region_count": moved_region_count(baseline, alternative, frame),
        "grouping_violation_count": grouping_violation_count(case, alternative),
        "mib_violation_count": mib_violation_count(case, alternative),
    }


def moved_component_size(case: FloorSetCase, moved: set[int]) -> int:
    if not moved:
        return 0
    adjacency: dict[int, set[int]] = {idx: set() for idx in moved}
    for src, dst, _weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i in moved and j in moved:
            adjacency[i].add(j)
            adjacency[j].add(i)
    return max((len(component) for component in connected_components(adjacency)), default=0)


def normalized_entropy(values: Any) -> float:
    rows = [float(value) for value in values if float(value) > 0.0]
    total = sum(rows)
    if total <= 0.0 or len(rows) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in rows)
    return entropy / max(math.log(len(rows)), 1e-9)


def percentile(values: Any, q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    pos = (len(rows) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return rows[lo]
    return rows[lo] * (hi - pos) + rows[hi] * (pos - lo)


def mean(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / max(len(rows), 1)
