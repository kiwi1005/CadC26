#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from puzzleplace.alternatives.shape_policy import ShapePolicyName
from puzzleplace.diagnostics.repair_radius import (
    changed_blocks,
    hard_summary,
    pareto_repair_selection,
    repair_failure_attribution,
    repair_radius_metrics,
)
from puzzleplace.experiments.shape_policy_replay import (
    evaluate_shape_policy_case,
    reconstruct_original_layout,
)
from puzzleplace.legalization import RepairMode, bounded_repair
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame, multistart_virtual_frames
from puzzleplace.train.dataset_bc import load_validation_cases

REPAIR_MODES: tuple[RepairMode, ...] = (
    "current_repair_baseline",
    "geometry_window_repair",
    "region_cell_repair",
    "graph_hop_repair",
    "macro_component_repair",
    "cascade_capped_repair",
    "rollback_to_original",
)
REPLAY_POLICIES: tuple[ShapePolicyName, ...] = (
    "role_aware_cap",
    "boundary_edge_slot_exception",
    "MIB_shape_master_regularized",
    "group_macro_aspect_regularized",
)
DEFAULT_LARGE_XL = [79, 51, 76, 99, 91]
DEFAULT_COMPARISON = [19, 24, 25]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _cheap_profile_layout(case: Any) -> dict[int, Placement]:
    placements: dict[int, Placement] = {}
    x = 0.0
    for idx in range(case.block_count):
        side = float(case.area_targets[idx].sqrt().item())
        placements[idx] = (x, 0.0, side, side)
        x += side + 1.0
    return placements


def _baseline_layout(
    case: Any,
    case_id: int,
    step6j: dict[str, Any],
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[dict[int, Placement], PuzzleFrame]:
    if case.block_count >= 50:
        frame = multistart_virtual_frames(case)[case_id % len(multistart_virtual_frames(case))]
        return _cheap_profile_layout(case), frame
    placements, frame, _family_usage = reconstruct_original_layout(
        case,
        case_id,
        step6j,
        boundary_commit_mode,
    )
    return placements, frame


def _focus_case_ids(
    step7e_rows: list[dict[str, Any]],
    step7d_rows: list[dict[str, Any]],
) -> list[int]:
    large_xl = [
        int(row["case_id"])
        for row in step7e_rows
        if row.get("size_bucket") in {"large", "xl"}
        and row.get("primary_failure") == "local_move_repair_radius_too_large"
    ] or DEFAULT_LARGE_XL
    by_winner = {int(row["case_id"]): str(row.get("winner_type")) for row in step7d_rows}
    comparison = [
        case_id
        for case_id, winner in by_winner.items()
        if "shape policy wins" in winner or "MIB/group policy wins" in winner
    ][:3]
    if len(comparison) < 3:
        comparison.extend(case_id for case_id in DEFAULT_COMPARISON if case_id not in comparison)
    ordered = []
    for case_id in [*large_xl, *comparison]:
        if case_id not in ordered:
            ordered.append(case_id)
    return ordered


def _source_candidate(
    rows: list[dict[str, Any]],
    layouts: dict[tuple[str, str], tuple[dict[int, Placement], PuzzleFrame]],
    *,
    case_bucket: str,
) -> tuple[dict[str, Any], dict[int, Placement], PuzzleFrame]:
    candidates = [row for row in rows if row["policy"] != "original_shape_policy"]
    if case_bucket in {"large", "xl"}:
        candidates = sorted(
            candidates,
            key=lambda row: (
                not bool(row["hard_feasible"]),
                float(row.get("disruption", 0.0)),
                float(row.get("frame_protrusion", 0.0)),
            ),
            reverse=True,
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda row: (
                float(row.get("boundary_delta", 0.0)),
                -float(row.get("hpwl_delta_norm", 0.0)),
            ),
            reverse=True,
        )
    row = candidates[0]
    layout, frame = layouts[(str(row["track"]), str(row["policy"]))]
    return row, layout, frame


def _run_case(
    *,
    case_id: int,
    case: Any,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, tuple[dict[int, Placement], PuzzleFrame]],
]:
    replay_rows, replay_layouts = evaluate_shape_policy_case(
        case=case,
        baseline=baseline,
        frame=frame,
        boundary_commit_mode=boundary_commit_mode,
        policies=("original_shape_policy", *REPLAY_POLICIES),
    )
    source_row, source_layout, source_frame = _source_candidate(
        replay_rows,
        replay_layouts,
        case_bucket=_size_bucket(case.block_count),
    )
    source_move_type = f"{source_row['track']}:{source_row['policy']}"
    candidates = {
        "case_id": case_id,
        "source_move_type": source_move_type,
        "source_policy": source_row["policy"],
        "source_track": source_row["track"],
        "changed_block_count": len(changed_blocks(baseline, source_layout)),
        "changed_block_fraction": len(changed_blocks(baseline, source_layout))
        / max(case.block_count, 1),
        "hard_summary": hard_summary(case, source_layout),
    }
    rows: list[dict[str, Any]] = []
    layout_by_mode: dict[str, tuple[dict[int, Placement], PuzzleFrame]] = {
        "before_repair": (source_layout, source_frame)
    }
    for mode in REPAIR_MODES:
        result = bounded_repair(
            case,
            baseline=baseline,
            candidate=source_layout,
            frame=source_frame,
            mode=mode,
            max_moved_fraction=0.35,
            max_affected_regions=8,
        )
        metrics = repair_radius_metrics(
            case,
            baseline=baseline,
            before_repair=source_layout,
            after_repair=result.placements,
            frame=source_frame,
            source_move_type=source_move_type,
            repair_mode=mode,
            repair_seed=result.repair_seed,
            repair_region=result.repair_region,
            repair_radius_exceeded=result.repair_radius_exceeded,
            runtime_estimate_ms=result.runtime_estimate_ms,
            reject_reason=result.reject_reason,
        )
        metrics["repair_seed_blocks"] = sorted(result.repair_seed)
        metrics["repair_region_blocks"] = sorted(result.repair_region)
        metrics["expansion_reasons"] = {
            str(idx): reasons for idx, reasons in sorted(result.expansion_reasons.items())
        }
        rows.append(metrics)
        layout_by_mode[mode] = (result.placements, source_frame)
    return rows, candidates, layout_by_mode


def _decision(selection_by_case: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    large_xl_rows = [row for row in rows if row["case_size_bucket"] in {"large", "xl"}]
    bounded_success = [
        row
        for row in large_xl_rows
        if row["repair_mode"]
        not in {"current_repair_baseline", "rollback_to_original"}
        and row["hard_feasible_after"]
        and not row["repair_radius_exceeded"]
        and float(row["moved_block_fraction"]) <= 0.35
        and int(row["affected_region_count"]) <= 8
        and float(row["hpwl_delta_norm"]) <= 0.20
        and float(row["bbox_delta_norm"]) <= 0.20
        and float(row["boundary_delta"]) >= -0.05
    ]
    if len(bounded_success) >= max(2, len({int(row["case_id"]) for row in large_xl_rows}) // 2):
        return "promote_bounded_repair_to_step7c"
    failure_labels = Counter(
        row.get("label")
        for row in _failed_attributions(rows)
        if row.get("case_size_bucket") in {"large", "xl"}
    )
    if failure_labels.get("macro_component_missing", 0) >= 2:
        return "pivot_to_macro_level_legalizer"
    if failure_labels.get("repair_window_too_small", 0) >= 3:
        return "pivot_to_region_replanner"
    if failure_labels.get("repair_window_too_large", 0) >= 3:
        return "pivot_to_move_generation_constraints"
    if not selection_by_case:
        return "inconclusive_due_to_surrogate_or_trace_gap"
    return "pivot_to_move_generation_constraints"


def _failed_attributions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = [
        row
        for row in rows
        if row["repair_mode"] not in {"current_repair_baseline", "rollback_to_original"}
        and (not row["hard_feasible_after"] or row["repair_radius_exceeded"])
    ]
    out = []
    for row in failed:
        attr = repair_failure_attribution(row, cap_fraction=0.35)
        attr["case_size_bucket"] = row["case_size_bucket"]
        out.append(attr)
    return out


def _write_decision_md(
    *,
    decision: str,
    case_ids: list[int],
    rows: list[dict[str, Any]],
    selection: dict[int, dict[str, Any]],
    failure_attrs: list[dict[str, Any]],
) -> str:
    large_xl = [row for row in rows if row["case_size_bucket"] in {"large", "xl"}]
    success = [
        row
        for row in large_xl
        if row["repair_mode"] not in {"current_repair_baseline", "rollback_to_original"}
        and row["hard_feasible_after"]
        and not row["repair_radius_exceeded"]
    ]
    return (
        "\n".join(
            [
                "# Step7F Bounded Repair Radius Study",
                "",
                f"Decision: `{decision}`",
                "",
                "## Coverage",
                "",
                f"- cases: `{case_ids}`",
                "- large/XL focus: `79, 51, 76, 99, 91`",
                "- scope: sidecar-only; bounded repair is not wired to contest runtime.",
                "",
                "## Large/XL bounded success count",
                "",
                f"`{len(success)} / {len({int(row['case_id']) for row in large_xl})}`",
                "",
                "## Pareto selection by case",
                "",
                "```json",
                json.dumps(selection, indent=2),
                "```",
                "",
                "## Failure attribution counts",
                "",
                "```json",
                json.dumps(dict(Counter(row['label'] for row in failure_attrs)), indent=2),
                "```",
                "",
                "## Interpretation",
                "",
                "- Hard-invalid bounded attempts are excluded from the Pareto front.",
                "- Radius-exceeded attempts are marked, not silently promoted.",
                "- If bounded repair mostly rolls back or exceeds radius, Step7G/Step7C should",
                "  constrain move generation before relying on iterative search.",
            ]
        )
        + "\n"
    )


def _render_visualizations(
    *,
    cases_by_id: dict[int, Any],
    layout_by_case: dict[int, dict[str, tuple[dict[int, Placement], PuzzleFrame]]],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7f_visualizations"
    if viz_dir.exists():
        for stale in viz_dir.glob("*.png"):
            stale.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    arrow_debug: list[dict[str, Any]] = []
    focus = [79, 51, 76, 99, 91, 19]
    rows_by_case_mode = {
        (int(row["case_id"]), str(row["repair_mode"])): row for row in rows
    }
    for case_id in focus:
        if case_id not in layout_by_case:
            continue
        modes = [
            "before_repair",
            "current_repair_baseline",
            "geometry_window_repair",
            "region_cell_repair",
            "graph_hop_repair",
            "macro_component_repair",
            "cascade_capped_repair",
            "rollback_to_original",
        ]
        for mode in modes:
            if mode not in layout_by_case[case_id]:
                continue
            placements, frame = layout_by_case[case_id][mode]
            row = rows_by_case_mode.get((case_id, mode), {})
            baseline = layout_by_case[case_id].get("rollback_to_original", (placements, frame))[0]
            path = viz_dir / f"case{case_id:03d}_{mode}.png"
            mode_debug = _plot_layout_triptych(
                plt,
                cases_by_id[case_id],
                before=baseline,
                after=placements,
                frame=frame,
                row=row,
                path=path,
                title=f"case {case_id} {mode}",
            )
            arrow_debug.extend(
                {"case_id": case_id, "repair_mode": mode, **entry} for entry in mode_debug
            )
            written.append(str(path))
    (viz_dir / "arrow_endpoint_debug.json").write_text(json.dumps(arrow_debug, indent=2))
    _write_viz_index(written, viz_dir)
    return written


def _plot_layout_triptych(
    plt: Any,
    case: Any,
    before: dict[int, Placement],
    after: dict[int, Placement],
    frame: PuzzleFrame,
    row: dict[str, Any],
    path: Path,
    title: str,
) -> list[dict[str, Any]]:

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    limits = _shared_axis_limits(frame, after)
    global_cascade = bool(
        float(row.get("moved_block_fraction", 0.0)) >= 0.80
        or int(row.get("affected_region_count", 0)) >= 12
        or bool(row.get("repair_radius_exceeded", False))
    )
    badge = "repair_global_cascade" if global_cascade else "bounded_repair"
    common_metrics = (
        f"{badge} | feas={row.get('hard_feasible_after', 'n/a')} "
        f"radius={row.get('repair_radius_exceeded', 'n/a')} "
        f"moved={float(row.get('moved_block_fraction', 0.0)):.2f} "
        f"regions={row.get('affected_region_count', 'n/a')} "
        f"protrude={float(row.get('frame_protrusion_after', 0.0)):.1f} "
        f"max_disp={float(row.get('max_displacement', 0.0)):.1f}"
    )
    fig.suptitle(f"{title}\n{common_metrics}", fontsize=10)
    _draw_layout_panel(
        axes[0],
        case,
        before,
        frame,
        limits,
        "before/original baseline",
        repair_region=set(),
        repair_seed=set(),
    )
    _draw_layout_panel(
        axes[1],
        case,
        after,
        frame,
        limits,
        "after repair",
        repair_region=set(row.get("repair_region_blocks", [])),
        repair_seed=set(row.get("repair_seed_blocks", [])),
    )
    debug = _draw_displacement_panel(
        axes[2],
        case,
        before,
        after,
        frame,
        limits,
        row,
        top_k=24,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return debug


def _draw_layout_panel(
    ax: Any,
    case: Any,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    limits: tuple[float, float, float, float],
    title: str,
    *,
    repair_region: set[int],
    repair_seed: set[int],
) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (frame.xmin, frame.ymin),
            frame.width,
            frame.height,
            facecolor="none",
            edgecolor="#f97316",
            linestyle="--",
            linewidth=1.0,
        )
    )
    for idx, box in sorted(placements.items()):
        color = _block_color(case, idx)
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                facecolor=color,
                edgecolor="#111827",
                linewidth=0.25,
                alpha=0.65,
            )
        )
        if idx in repair_region:
            ax.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    facecolor="none",
                    edgecolor="#f97316",
                    linewidth=0.9,
                    alpha=0.85,
                )
            )
        if idx in repair_seed:
            ax.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    facecolor="none",
                    edgecolor="#dc2626",
                    linewidth=1.2,
                    alpha=0.9,
                )
            )
    ax.set_title(title, fontsize=9)
    _apply_limits(ax, limits)


def _draw_displacement_panel(
    ax: Any,
    case: Any,
    before: dict[int, Placement],
    after: dict[int, Placement],
    frame: PuzzleFrame,
    limits: tuple[float, float, float, float],
    row: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    from matplotlib.patches import Rectangle

    for idx, box in sorted(after.items()):
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                facecolor=_block_color(case, idx),
                edgecolor="#111827",
                linewidth=0.2,
                alpha=0.35,
            )
        )
    moved = []
    for idx, before_box in before.items():
        if idx not in after:
            continue
        before_center = _center(before_box)
        after_center = _center(after[idx])
        distance = _distance(before_center, after_center)
        if distance > 1e-6:
            moved.append((distance, idx, before_center, after_center))
    moved.sort(reverse=True)
    selected = moved[:top_k]
    debug: list[dict[str, Any]] = []
    for distance, idx, before_center, after_center in selected:
        clipped_start, clipped_end = _normalized_arrow(
            before_center,
            after_center,
            limits,
            max_fraction=0.20,
        )
        debug.append(
            {
                "block_id": idx,
                "raw_before_center": [before_center[0], before_center[1]],
                "raw_after_center": [after_center[0], after_center[1]],
                "drawn_start": [clipped_start[0], clipped_start[1]],
                "drawn_end": [clipped_end[0], clipped_end[1]],
                "distance": distance,
                "after_inside_frame": frame.contains_box(after[idx]),
                "block_id_matched": True,
            }
        )
        ax.annotate(
            "",
            xy=clipped_end,
            xytext=clipped_start,
            arrowprops={
                "arrowstyle": "->",
                "color": "#7c3aed",
                "linewidth": 0.7,
                "alpha": 0.55,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=True,
        )
    _draw_affected_region_grid(ax, frame, after, before)
    max_raw = max((m[0] for m in moved), default=0.0)
    ax.set_title(
        "displacement top-k clipped/normalized\n"
        f"drawn={len(selected)} of {len(moved)} | max_raw={max_raw:.1f}",
        fontsize=9,
    )
    _apply_limits(ax, limits)
    return debug


def _block_color(case: Any, idx: int) -> str:
    color = "#e5e7eb"
    if bool(case.constraints[idx, 2].item()) or bool(case.constraints[idx, 3].item()):
        color = "#86efac"
    if bool(case.constraints[idx, 4].item()):
        color = "#93c5fd"
    return color


def _apply_limits(ax: Any, limits: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.3)


def _shared_axis_limits(
    frame: PuzzleFrame,
    after: dict[int, Placement],
) -> tuple[float, float, float, float]:
    xs = [frame.xmin, frame.xmax]
    ys = [frame.ymin, frame.ymax]
    for x, y, w, h in after.values():
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(xmax - xmin, ymax - ymin) * 0.04
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _center(box: Placement) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _normalized_arrow(
    before_center: tuple[float, float],
    after_center: tuple[float, float],
    limits: tuple[float, float, float, float],
    *,
    max_fraction: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax, ymin, ymax = limits
    width = max(xmax - xmin, 1e-9)
    height = max(ymax - ymin, 1e-9)
    max_len = ((width * width + height * height) ** 0.5) * max_fraction
    raw_dx = after_center[0] - before_center[0]
    raw_dy = after_center[1] - before_center[1]
    raw_len = max((raw_dx * raw_dx + raw_dy * raw_dy) ** 0.5, 1e-9)
    scale = min(max_len / raw_len, 1.0)
    end = _clip_point(after_center, limits)
    start = (end[0] - raw_dx * scale, end[1] - raw_dy * scale)
    return _clip_point(start, limits), end


def _clip_point(
    point: tuple[float, float],
    limits: tuple[float, float, float, float],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = limits
    return min(max(point[0], xmin), xmax), min(max(point[1], ymin), ymax)


def _draw_affected_region_grid(
    ax: Any,
    frame: PuzzleFrame,
    placements: dict[int, Placement],
    baseline: dict[int, Placement],
) -> None:
    from matplotlib.patches import Rectangle

    rows = cols = 4
    cell_w = frame.width / cols
    cell_h = frame.height / rows
    changed_regions: set[tuple[int, int]] = set()
    for idx, box in placements.items():
        if idx not in baseline:
            continue
        if not any(
            abs(left - right) > 1e-6
            for left, right in zip(baseline[idx], box, strict=False)
        ):
            continue
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        col = min(max(int((cx - frame.xmin) / max(frame.width, 1e-9) * cols), 0), cols - 1)
        row = min(max(int((cy - frame.ymin) / max(frame.height, 1e-9) * rows), 0), rows - 1)
        changed_regions.add((row, col))
    for row, col in changed_regions:
        ax.add_patch(
            Rectangle(
                (frame.xmin + col * cell_w, frame.ymin + row * cell_h),
                cell_w,
                cell_h,
                facecolor="#fde68a",
                edgecolor="#f59e0b",
                linewidth=0.5,
                alpha=0.18,
            )
        )


def _write_viz_index(paths: list[str], output_dir: Path) -> None:
    sections = []
    for path in paths:
        name = Path(path).name
        sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            f"<img src='{html.escape(name)}' style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    (output_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html><meta charset='utf-8'>",
                "<title>Step7F bounded repair visualizations</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
                "<h1>Step7F bounded repair visualizations</h1>",
                *sections,
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-limit", type=int, default=100)
    parser.add_argument(
        "--step7e-attribution",
        default="artifacts/research/step7e_failure_attribution.json",
    )
    parser.add_argument("--step7d-results", default="artifacts/research/step7d_replay_results.json")
    parser.add_argument(
        "--step6j",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step7e_rows = _load_json(Path(args.step7e_attribution), [])
    step7d_rows = _load_json(Path(args.step7d_results), [])
    case_ids = _focus_case_ids(step7e_rows, step7d_rows)
    cases = load_validation_cases(case_limit=max(max(case_ids) + 1, args.case_limit))
    cases_by_id = {idx: case for idx, case in enumerate(cases)}
    step6j = _load_json(Path(args.step6j), {})
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    all_rows: list[dict[str, Any]] = []
    repair_candidates: list[dict[str, Any]] = []
    layout_by_case: dict[int, dict[str, tuple[dict[int, Placement], PuzzleFrame]]] = {}
    for case_id in case_ids:
        case = cases_by_id[case_id]
        baseline, frame = _baseline_layout(case, case_id, step6j, boundary_commit_mode)
        rows, candidate, layouts = _run_case(
            case_id=case_id,
            case=case,
            baseline=baseline,
            frame=frame,
            boundary_commit_mode=boundary_commit_mode,
        )
        for row in rows:
            row["case_id"] = case_id
        all_rows.extend(rows)
        repair_candidates.append(candidate)
        layout_by_case[case_id] = layouts
    failure_attrs = _failed_attributions(all_rows)
    selection_by_case = {}
    for case_id in case_ids:
        case_rows = [row for row in all_rows if int(row["case_id"]) == case_id]
        selection_by_case[case_id] = pareto_repair_selection(case_rows)
    decision = _decision(selection_by_case, all_rows)
    visualizations = _render_visualizations(
        cases_by_id=cases_by_id,
        layout_by_case=layout_by_case,
        rows=all_rows,
        output_dir=output_dir,
    )
    decision_md = _write_decision_md(
        decision=decision,
        case_ids=case_ids,
        rows=all_rows,
        selection=selection_by_case,
        failure_attrs=failure_attrs,
    )
    radius_metrics = [
        {
            "case_id": row["case_id"],
            "repair_mode": row["repair_mode"],
            "moved_block_fraction": row["moved_block_fraction"],
            "affected_region_count": row["affected_region_count"],
            "repair_radius_exceeded": row["repair_radius_exceeded"],
            "hard_feasible_after": row["hard_feasible_after"],
        }
        for row in all_rows
    ]
    (output_dir / "step7f_repair_candidates.json").write_text(
        json.dumps(repair_candidates, indent=2)
    )
    (output_dir / "step7f_bounded_repair_results.json").write_text(
        json.dumps(all_rows, indent=2)
    )
    (output_dir / "step7f_repair_radius_metrics.json").write_text(
        json.dumps(radius_metrics, indent=2)
    )
    (output_dir / "step7f_failure_attribution.json").write_text(
        json.dumps(failure_attrs, indent=2)
    )
    (output_dir / "step7f_pareto_repair_selection.json").write_text(
        json.dumps(selection_by_case, indent=2)
    )
    (output_dir / "step7f_decision.md").write_text(decision_md)
    print(
        json.dumps(
            {
                "decision": decision,
                "case_ids": case_ids,
                "result_rows": len(all_rows),
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7f_decision.md"),
            },
            indent=2,
        )
    )


def _size_bucket(block_count: int) -> str:
    if block_count <= 40:
        return "small"
    if block_count <= 70:
        return "medium"
    if block_count <= 100:
        return "large"
    return "xl"


if __name__ == "__main__":
    main()
