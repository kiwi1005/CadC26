from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

from puzzleplace.data import ConstraintColumns, FloorSetCase

Placement = tuple[float, float, float, float]
FrameVariant = Literal["tight", "medium", "loose", "pin_aspect", "square"]


@dataclass(frozen=True, slots=True)
class PuzzleFrame:
    """Internal sidecar construction board for Step6H candidate generation.

    The frame is not an official hard constraint.  It bounds the sidecar action
    space so HPWL cannot select candidates whose centers are reasonable while
    the rectangular body protrudes far outside the pin/preplaced region.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    density: float
    variant: str = "medium"
    relaxation: int = 0

    @property
    def width(self) -> float:
        return max(self.xmax - self.xmin, 0.0)

    @property
    def height(self) -> float:
        return max(self.ymax - self.ymin, 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def expanded(self, factor: float = 1.05) -> PuzzleFrame:
        cx, cy = self.center
        half_w = self.width * factor / 2.0
        half_h = self.height * factor / 2.0
        return replace(
            self,
            xmin=cx - half_w,
            ymin=cy - half_h,
            xmax=cx + half_w,
            ymax=cy + half_h,
            relaxation=self.relaxation + 1,
        )

    def contains_box(self, box: Placement, *, eps: float = 1e-6) -> bool:
        x, y, w, h = box
        return (
            x >= self.xmin - eps
            and y >= self.ymin - eps
            and x + w <= self.xmax + eps
            and y + h <= self.ymax + eps
        )

    def protrusion_distance(self, box: Placement) -> float:
        x, y, w, h = box
        return max(
            self.xmin - x,
            self.ymin - y,
            x + w - self.xmax,
            y + h - self.ymax,
            0.0,
        )


def _bbox_from_points(
    points: Iterable[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    rows = list(points)
    if not rows:
        return None
    return (
        min(x for x, _y in rows),
        min(y for _x, y in rows),
        max(x for x, _y in rows),
        max(y for _x, y in rows),
    )


def _bbox_from_placements(
    placements: Iterable[Placement],
) -> tuple[float, float, float, float] | None:
    rows = list(placements)
    if not rows:
        return None
    return (
        min(x for x, _y, _w, _h in rows),
        min(y for _x, y, _w, _h in rows),
        max(x + w for x, _y, w, _h in rows),
        max(y + h for _x, y, _w, h in rows),
    )


def _union_boxes(
    boxes: Iterable[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    rows = [box for box in boxes if box is not None]
    if not rows:
        return None
    return (
        min(box[0] for box in rows),
        min(box[1] for box in rows),
        max(box[2] for box in rows),
        max(box[3] for box in rows),
    )


def pin_box(case: FloorSetCase) -> tuple[float, float, float, float] | None:
    if case.pins_pos.numel() == 0:
        return None
    return _bbox_from_points((float(row[0]), float(row[1])) for row in case.pins_pos.tolist())


def preplaced_boxes(case: FloorSetCase) -> list[Placement]:
    if case.target_positions is None:
        return []
    rows: list[Placement] = []
    for idx in range(case.block_count):
        if not bool(case.constraints[idx, ConstraintColumns.PREPLACED].item()):
            continue
        x, y, w, h = [float(v) for v in case.target_positions[idx].tolist()]
        if x >= 0.0 and y >= 0.0 and w > 0.0 and h > 0.0:
            rows.append((x, y, w, h))
    return rows


def anchor_box(case: FloorSetCase) -> tuple[float, float, float, float] | None:
    return _union_boxes([pin_box(case), _bbox_from_placements(preplaced_boxes(case))])


def _area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _aspect(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 1.0
    width = max(box[2] - box[0], 1e-6)
    height = max(box[3] - box[1], 1e-6)
    return width / height


def estimate_virtual_puzzle_frame(
    case: FloorSetCase,
    *,
    target_density: float = 0.90,
    margin_factor: float = 1.05,
    aspect: float | None = None,
    min_aspect: float = 0.5,
    max_aspect: float = 2.0,
    variant: str = "medium",
) -> PuzzleFrame:
    """Estimate a virtual puzzle board from pins, preplaced anchors, and area.

    The frame is centered on the union of the terminal pin box and preplaced
    rectangles.  Its area is at least `total_block_area / target_density` and at
    least `anchor_box.area * margin_factor`, then widened/tallied as needed to
    fully contain all anchors.
    """

    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    pins = pin_box(case)
    anchors = anchor_box(case)
    if anchors is None:
        side = math.sqrt(total_area / max(target_density, 1e-6))
        anchors = (0.0, 0.0, side, side)
    center_x = (anchors[0] + anchors[2]) / 2.0
    center_y = (anchors[1] + anchors[3]) / 2.0

    target_area = max(_area(anchors) * margin_factor, total_area / max(target_density, 1e-6))
    raw_aspect = _aspect(pins if aspect is None else None) if aspect is None else aspect
    frame_aspect = min(max(raw_aspect, min_aspect), max_aspect)
    frame_w = math.sqrt(target_area * frame_aspect)
    frame_h = target_area / max(frame_w, 1e-6)
    frame_w = max(frame_w, anchors[2] - anchors[0])
    frame_h = max(frame_h, anchors[3] - anchors[1])

    xmin = center_x - frame_w / 2.0
    xmax = center_x + frame_w / 2.0
    ymin = center_y - frame_h / 2.0
    ymax = center_y + frame_h / 2.0

    # Numerical/saturation guard: after clamping aspect/size, translate just
    # enough to keep every pin/preplaced anchor inside the virtual board.
    if xmin > anchors[0]:
        xmax -= xmin - anchors[0]
        xmin = anchors[0]
    if xmax < anchors[2]:
        xmin += anchors[2] - xmax
        xmax = anchors[2]
    if ymin > anchors[1]:
        ymax -= ymin - anchors[1]
        ymin = anchors[1]
    if ymax < anchors[3]:
        ymin += anchors[3] - ymax
        ymax = anchors[3]

    return PuzzleFrame(
        xmin=float(xmin),
        ymin=float(ymin),
        xmax=float(xmax),
        ymax=float(ymax),
        density=float(target_density),
        variant=variant,
    )


def multistart_virtual_frames(case: FloorSetCase) -> list[PuzzleFrame]:
    """Return the Step6H frame portfolio: tight/medium/loose plus aspect variants."""

    specs: list[tuple[str, float, float | None]] = [
        ("tight", 0.95, None),
        ("medium", 0.90, None),
        ("loose", 0.85, None),
        ("pin_aspect", 0.90, _aspect(pin_box(case))),
        ("square", 0.90, 1.0),
    ]
    frames: list[PuzzleFrame] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for variant, density, aspect in specs:
        frame = estimate_virtual_puzzle_frame(
            case, target_density=density, aspect=aspect, variant=variant
        )
        key = (
            round(frame.xmin, 4),
            round(frame.ymin, 4),
            round(frame.xmax, 4),
            round(frame.ymax, 4),
            variant,
        )
        if key in seen:
            continue
        seen.add(key)
        frames.append(frame)
    return frames


def estimate_predicted_compact_hull(
    case: FloorSetCase,
    frame: PuzzleFrame,
    *,
    target_density: float = 0.90,
    margin_factor: float = 1.02,
    min_aspect: float = 0.5,
    max_aspect: float = 2.0,
) -> PuzzleFrame:
    """Estimate the compact final-bbox ownership hull inside a virtual frame.

    Step6J separates this predicted compact hull from the virtual frame.  The
    virtual frame remains a hard containment upper bound, while boundary blocks
    should generally own this smaller predicted hull edge.
    """

    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    anchors = anchor_box(case)
    pins = pin_box(case)
    target_area = total_area / max(target_density, 1e-6)
    if anchors is not None:
        target_area = max(target_area, _area(anchors) * margin_factor)
    target_area = min(target_area, frame.area)
    aspect = min(max(_aspect(pins), min_aspect), max_aspect)
    width = min(math.sqrt(target_area * aspect), frame.width)
    height = min(target_area / max(width, 1e-6), frame.height)
    width = min(max(width, (anchors[2] - anchors[0]) if anchors is not None else 0.0), frame.width)
    height = min(
        max(height, (anchors[3] - anchors[1]) if anchors is not None else 0.0),
        frame.height,
    )
    if anchors is not None:
        center_x = (anchors[0] + anchors[2]) / 2.0
        center_y = (anchors[1] + anchors[3]) / 2.0
    else:
        center_x, center_y = frame.center
    xmin = min(max(center_x - width / 2.0, frame.xmin), frame.xmax - width)
    ymin = min(max(center_y - height / 2.0, frame.ymin), frame.ymax - height)
    xmax = xmin + width
    ymax = ymin + height
    if anchors is not None:
        if xmin > anchors[0]:
            xmin = max(frame.xmin, anchors[0])
            xmax = xmin + width
        if xmax < anchors[2]:
            xmax = min(frame.xmax, anchors[2])
            xmin = xmax - width
        if ymin > anchors[1]:
            ymin = max(frame.ymin, anchors[1])
            ymax = ymin + height
        if ymax < anchors[3]:
            ymax = min(frame.ymax, anchors[3])
            ymin = ymax - height
    return PuzzleFrame(
        xmin=float(xmin),
        ymin=float(ymin),
        xmax=float(xmax),
        ymax=float(ymax),
        density=float(target_density),
        variant=f"predicted_hull:{frame.variant}",
        relaxation=frame.relaxation,
    )


def outside_area(box: Placement, frame: PuzzleFrame) -> float:
    x, y, w, h = box
    box_area = max(w, 0.0) * max(h, 0.0)
    ix0 = max(x, frame.xmin)
    iy0 = max(y, frame.ymin)
    ix1 = min(x + w, frame.xmax)
    iy1 = min(y + h, frame.ymax)
    inside = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    return max(box_area - inside, 0.0)


def frame_diagnostics(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[str, float | int | str]:
    """Compute Step6H protrusion diagnostics for a layout relative to a frame."""

    total_rect_area = sum(max(w, 0.0) * max(h, 0.0) for _x, _y, w, h in placements.values())
    outside_total = sum(outside_area(box, frame) for box in placements.values())
    violations = sum(0 if frame.contains_box(box) else 1 for box in placements.values())
    max_protrusion = max(
        (frame.protrusion_distance(box) for box in placements.values()), default=0.0
    )
    layout_box = _bbox_from_placements(placements.values())
    bbox_area = _area(layout_box)
    pins = pin_box(case)
    pin_area = _area(pins)
    hull_outliers = 0
    if pins is not None:
        px0, py0, px1, py1 = pins
        for x, y, w, h in placements.values():
            cx, cy = x + w / 2.0, y + h / 2.0
            if cx < px0 or cx > px1 or cy < py0 or cy > py1:
                hull_outliers += 1
    return {
        "frame_variant": frame.variant,
        "frame_density": frame.density,
        "frame_relaxation": frame.relaxation,
        "frame_area": frame.area,
        "outside_frame_area_ratio": outside_total / max(total_rect_area, 1e-6),
        "num_frame_violations": int(violations),
        "bbox_area_over_frame_area": bbox_area / max(frame.area, 1e-6),
        "bbox_area_over_pin_box_area": bbox_area / max(pin_area, 1e-6) if pin_area > 0 else 0.0,
        "max_protrusion_distance": float(max_protrusion),
        "num_hull_outlier_blocks": int(hull_outliers),
        **boundary_frame_metrics(case, placements, frame),
    }

def _boundary_edges(code: int) -> tuple[str, ...]:
    edges: list[str] = []
    if code & 1:
        edges.append("left")
    if code & 2:
        edges.append("right")
    if code & 8:
        edges.append("bottom")
    if code & 4:
        edges.append("top")
    return tuple(edges)


def boundary_frame_satisfied_edges(
    boundary_code: int,
    box: Placement,
    frame: PuzzleFrame,
    *,
    eps: float = 1e-4,
) -> tuple[int, int]:
    """Return `(satisfied_edges, required_edges)` for a box against a frame."""

    edges = _boundary_edges(int(boundary_code))
    if not edges:
        return 0, 0
    x, y, w, h = box
    checks = {
        "left": abs(x - frame.xmin) <= eps,
        "right": abs((x + w) - frame.xmax) <= eps,
        "bottom": abs(y - frame.ymin) <= eps,
        "top": abs((y + h) - frame.ymax) <= eps,
    }
    return sum(1 for edge in edges if checks[edge]), len(edges)


def boundary_frame_satisfaction_fraction(
    boundary_code: int,
    box: Placement,
    frame: PuzzleFrame,
    *,
    eps: float = 1e-4,
) -> float:
    satisfied, total = boundary_frame_satisfied_edges(boundary_code, box, frame, eps=eps)
    return 1.0 if total == 0 else float(satisfied) / float(total)


def boundary_frame_metrics(
    case: FloorSetCase,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
) -> dict[str, float | int]:
    total_edges = 0
    satisfied_edges = 0
    unsatisfied_blocks = 0
    boundary_blocks = 0
    for idx, box in placements.items():
        code = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if code == 0:
            continue
        boundary_blocks += 1
        satisfied, total = boundary_frame_satisfied_edges(code, box, frame)
        total_edges += total
        satisfied_edges += satisfied
        if total > 0 and satisfied < total:
            unsatisfied_blocks += 1
    return {
        "boundary_frame_blocks": int(boundary_blocks),
        "boundary_frame_satisfied_edges": int(satisfied_edges),
        "boundary_frame_total_edges": int(total_edges),
        "boundary_frame_satisfaction_rate": satisfied_edges / max(total_edges, 1),
        "boundary_frame_unsatisfied_blocks": int(unsatisfied_blocks),
    }


def _bbox_boundary_satisfied_edges(
    boundary_code: int,
    box: Placement,
    bbox: tuple[float, float, float, float],
    *,
    eps: float = 1e-4,
) -> tuple[int, int]:
    edges = _boundary_edges(int(boundary_code))
    if not edges:
        return 0, 0
    x, y, w, h = box
    xmin, ymin, xmax, ymax = bbox
    checks = {
        "left": abs(x - xmin) <= eps,
        "right": abs((x + w) - xmax) <= eps,
        "bottom": abs(y - ymin) <= eps,
        "top": abs((y + h) - ymax) <= eps,
    }
    return sum(1 for edge in edges if checks[edge]), len(edges)


def final_bbox_boundary_metrics(
    case: FloorSetCase,
    placements: dict[int, Placement],
) -> dict[str, float | int]:
    bbox = _bbox_from_placements(placements.values())
    if bbox is None:
        return {
            "final_bbox_boundary_satisfied_edges": 0,
            "final_bbox_boundary_total_edges": 0,
            "final_bbox_boundary_satisfaction_rate": 0.0,
            "final_bbox_boundary_unsatisfied_blocks": 0,
        }
    total_edges = 0
    satisfied_edges = 0
    unsatisfied_blocks = 0
    for idx, box in placements.items():
        code = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if code == 0:
            continue
        satisfied, total = _bbox_boundary_satisfied_edges(code, box, bbox)
        total_edges += total
        satisfied_edges += satisfied
        if total > 0 and satisfied < total:
            unsatisfied_blocks += 1
    return {
        "final_bbox_boundary_satisfied_edges": int(satisfied_edges),
        "final_bbox_boundary_total_edges": int(total_edges),
        "final_bbox_boundary_satisfaction_rate": satisfied_edges / max(total_edges, 1),
        "final_bbox_boundary_unsatisfied_blocks": int(unsatisfied_blocks),
    }


def boundary_commit_attribution(
    case: FloorSetCase,
    pre_placements: dict[int, Placement],
    post_placements: dict[int, Placement],
    frame: PuzzleFrame,
    predicted_hull: PuzzleFrame,
) -> list[dict[str, float | int | bool | str]]:
    pre_bbox = _bbox_from_placements(pre_placements.values())
    post_bbox = _bbox_from_placements(post_placements.values())
    pre_area = _area(pre_bbox)
    rows: list[dict[str, float | int | bool | str]] = []
    for idx, pre_box in sorted(pre_placements.items()):
        code = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if code == 0:
            continue
        post_box = post_placements.get(idx, pre_box)
        pre_sat, pre_total = (
            _bbox_boundary_satisfied_edges(code, pre_box, pre_bbox) if pre_bbox else (0, 0)
        )
        post_sat, post_total = (
            _bbox_boundary_satisfied_edges(code, post_box, post_bbox) if post_bbox else (0, 0)
        )
        frame_sat, frame_total = boundary_frame_satisfied_edges(code, pre_box, frame)
        hull_sat, hull_total = boundary_frame_satisfied_edges(code, pre_box, predicted_hull)
        without = {block: box for block, box in pre_placements.items() if block != idx}
        expansion = pre_area - _area(_bbox_from_placements(without.values()))
        px, py, _pw, _ph = pre_box
        qx, qy, _qw, _qh = post_box
        rows.append(
            {
                "block_id": int(idx),
                "required_boundary_code": int(code),
                "required_boundary_type": "+".join(_boundary_edges(code)),
                "is_corner_requirement": bool(bin(code).count("1") >= 2),
                "committed_to_virtual_frame_edge": bool(
                    frame_total > 0 and frame_sat == frame_total
                ),
                "satisfies_predicted_hull_edge": bool(
                    hull_total > 0 and hull_sat == hull_total
                ),
                "satisfies_final_bbox_edge_pre_repair": bool(
                    pre_total > 0 and pre_sat == pre_total
                ),
                "satisfies_final_bbox_edge_post_repair": bool(
                    post_total > 0 and post_sat == post_total
                ),
                "distance_to_virtual_frame_edge": frame.protrusion_distance(pre_box),
                "distance_to_predicted_hull_edge": predicted_hull.protrusion_distance(pre_box),
                "bbox_expansion_caused_by_block": float(expansion),
                "boundary_satisfaction_gain": (pre_sat / max(pre_total, 1))
                - (frame_sat / max(frame_total, 1)),
                "repair_displacement": abs(px - qx) + abs(py - qy),
            }
        )
    return rows


def repair_attribution_summary(
    case: FloorSetCase,
    pre_placements: dict[int, Placement],
    post_placements: dict[int, Placement],
) -> dict[str, float | int]:
    boundary_displacements: list[float] = []
    all_displacements: list[float] = []
    for idx, pre_box in pre_placements.items():
        post_box = post_placements.get(idx, pre_box)
        displacement = abs(pre_box[0] - post_box[0]) + abs(pre_box[1] - post_box[1])
        all_displacements.append(displacement)
        if int(case.constraints[idx, ConstraintColumns.BOUNDARY].item()) != 0:
            boundary_displacements.append(displacement)
    return {
        "pre_repair_bbox_area": _area(_bbox_from_placements(pre_placements.values())),
        "post_repair_bbox_area": _area(_bbox_from_placements(post_placements.values())),
        **{
            f"pre_repair_{key}": value
            for key, value in final_bbox_boundary_metrics(case, pre_placements).items()
        },
        **{
            f"post_repair_{key}": value
            for key, value in final_bbox_boundary_metrics(case, post_placements).items()
        },
        "repair_displacement_mean": sum(all_displacements) / max(len(all_displacements), 1),
        "repair_displacement_boundary_mean": sum(boundary_displacements)
        / max(len(boundary_displacements), 1),
    }
