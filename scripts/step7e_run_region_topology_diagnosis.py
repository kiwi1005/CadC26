#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from puzzleplace.alternatives.shape_policy import ShapePolicyName
from puzzleplace.diagnostics.placement_trace import candidate_ordering_trace
from puzzleplace.diagnostics.region_topology import (
    block_region_assignment,
    free_space_fragmentation,
    net_community_clusters,
    pin_density_regions,
    region_occupancy,
    repair_radius_metrics,
)
from puzzleplace.experiments.shape_policy_replay import (
    evaluate_shape_policy_case,
    reconstruct_original_layout,
)
from puzzleplace.research.move_library import layout_metrics
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame, multistart_virtual_frames
from puzzleplace.train.dataset_bc import load_validation_cases

STEP7D_DEFAULT_IDS = [2, 19, 24, 25, 79, 51, 76, 99, 91]
REPLAY_POLICIES: tuple[ShapePolicyName, ...] = (
    "original_shape_policy",
    "role_aware_cap",
    "boundary_edge_slot_exception",
    "MIB_shape_master_regularized",
    "group_macro_aspect_regularized",
)


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
) -> tuple[dict[int, Placement], PuzzleFrame, dict[str, int], str]:
    if case.block_count >= 50:
        frame = multistart_virtual_frames(case)[case_id % len(multistart_virtual_frames(case))]
        return (
            _cheap_profile_layout(case),
            frame,
            {"fast_surrogate_baseline": case.block_count},
            "reconstructed",
        )
    placements, frame, family_usage = reconstruct_original_layout(
        case,
        case_id,
        step6j,
        boundary_commit_mode,
    )
    return placements, frame, family_usage, "reconstructed"


def _repair_radius_audit(
    *,
    case: Any,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    boundary_commit_mode: BoundaryCommitMode,
) -> list[dict[str, Any]]:
    rows, layouts = evaluate_shape_policy_case(
        case=case,
        baseline=baseline,
        frame=frame,
        boundary_commit_mode=boundary_commit_mode,
        policies=REPLAY_POLICIES,
    )
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["track"]), str(row["policy"]))
        alternative, alt_frame = layouts[key]
        radius = repair_radius_metrics(case, baseline, alternative, alt_frame)
        violation_cause = "hard_feasible"
        if not bool(row["hard_feasible"]):
            violation_cause = "hard_invalid"
        elif float(row.get("frame_protrusion", 0.0)) > 1e-6:
            violation_cause = "frame_protrusion"
        elif float(row.get("boundary_delta", 0.0)) < -0.05:
            violation_cause = "boundary_regression"
        audit_rows.append(
            {
                "case_id": row["case_id"],
                "track": row["track"],
                "policy": row["policy"],
                "hard_feasible": row["hard_feasible"],
                "frame_protrusion": row["frame_protrusion"],
                "boundary_delta": row["boundary_delta"],
                "hpwl_delta_norm": row["hpwl_delta_norm"],
                "bbox_delta_norm": row["bbox_delta_norm"],
                "violation_cause": violation_cause,
                **radius,
            }
        )
    return audit_rows


def _failure_attribution(
    *,
    case_id: int,
    occupancy: dict[str, Any],
    assignment: dict[str, Any],
    fragmentation: dict[str, Any],
    trace: dict[str, Any],
    repair_audit: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "utilization_spread": occupancy["utilization_spread"],
        "overflow_region_count": occupancy["overflow_region_count"],
        "mean_mismatch_grid_distance": assignment["mean_mismatch_grid_distance"],
        "p90_mismatch_grid_distance": assignment["p90_mismatch_grid_distance"],
        "assignment_entropy": assignment["assignment_entropy"],
        "max_cluster_spread_regions": assignment["max_cluster_spread_regions"],
        "mib_group_spread_regions": assignment["mib_group_spread_regions"],
        "fragmentation_score": fragmentation["fragmentation_score"],
        "empty_component_count": fragmentation["empty_component_count"],
        "boundary_failure_rate": 1.0
        - float(baseline_metrics.get("boundary_satisfaction_rate", 1.0)),
        "hard_invalid_alternative_fraction": _mean(
            1.0 - float(row["hard_feasible"]) for row in repair_audit
        ),
        "max_moved_block_fraction": max(
            (float(row["moved_block_fraction"]) for row in repair_audit),
            default=0.0,
        ),
        "max_affected_region_count": max(
            (int(row["affected_region_count"]) for row in repair_audit),
            default=0,
        ),
        "trace_confidence": trace["trace_confidence"],
    }
    scores = {
        "region_capacity_mismatch": 0.0,
        "cluster_region_mismatch": 0.0,
        "candidate_ordering_failure": 0.0,
        "early_fragmentation": 0.0,
        "MIB_group_macro_failure": 0.0,
        "boundary_hull_ownership_failure": 0.0,
        "local_move_repair_radius_too_large": 0.0,
        "legalizer_capacity_failure": 0.0,
        "proxy_inconclusive": 0.0,
    }
    scores["region_capacity_mismatch"] = max(
        float(evidence["utilization_spread"]) - 0.45,
        float(evidence["overflow_region_count"]),
    )
    scores["cluster_region_mismatch"] = (
        float(evidence["mean_mismatch_grid_distance"]) / 2.0
        + float(evidence["assignment_entropy"]) * 0.5
        + float(evidence["max_cluster_spread_regions"]) / 8.0
    )
    scores["MIB_group_macro_failure"] = float(evidence["mib_group_spread_regions"]) / 4.0
    scores["early_fragmentation"] = (
        float(evidence["fragmentation_score"]) * 0.6
        + float(evidence["empty_component_count"]) / 24.0
    )
    if trace.get("first_major_region_mismatch") is not None:
        rank = int(trace["first_major_region_mismatch"]["rank"])
        scores["candidate_ordering_failure"] += max(0.0, 1.0 - rank / 32.0)
    if trace.get("first_macro_member_away") is not None:
        rank = int(trace["first_macro_member_away"]["rank"])
        scores["candidate_ordering_failure"] += max(0.0, 0.75 - rank / 48.0)
        scores["MIB_group_macro_failure"] += max(0.0, 0.75 - rank / 48.0)
    scores["boundary_hull_ownership_failure"] = max(
        0.0, float(evidence["boundary_failure_rate"]) - 0.45
    )
    scores["local_move_repair_radius_too_large"] = (
        float(evidence["max_moved_block_fraction"])
        + float(evidence["max_affected_region_count"]) / 8.0
    )
    scores["legalizer_capacity_failure"] = float(evidence["hard_invalid_alternative_fraction"])
    if trace["trace_confidence"] == "unavailable":
        scores["proxy_inconclusive"] = 1.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary = ranked[0][0] if ranked and ranked[0][1] > 0.0 else "proxy_inconclusive"
    secondary = [name for name, score in ranked[1:4] if score > 0.0]
    return {
        "case_id": case_id,
        "primary_failure": primary,
        "secondary_failures": secondary,
        "scores": scores,
        "evidence": evidence,
    }


def _architecture_decision(rows: list[dict[str, Any]]) -> str:
    large_xl = [row for row in rows if row.get("size_bucket") in {"large", "xl"}]
    target_rows = large_xl or rows
    counts = Counter(str(row["primary_failure"]) for row in target_rows)
    if counts.get("proxy_inconclusive", 0) >= max(1, len(target_rows) // 2):
        return "inconclusive_due_to_trace_gap"
    if counts.get("MIB_group_macro_failure", 0) >= max(2, counts.get("cluster_region_mismatch", 0)):
        return "pivot_to_macro_level_MIB_group_planner"
    if counts.get("cluster_region_mismatch", 0) >= max(2, counts.get("early_fragmentation", 0)):
        return "promote_cluster_first_region_planner"
    if counts.get("candidate_ordering_failure", 0) >= 2:
        return "pivot_to_candidate_ordering_policy"
    if counts.get("early_fragmentation", 0) >= 2:
        return "pivot_to_free_space_topology_generator"
    if (
        counts.get("local_move_repair_radius_too_large", 0)
        + counts.get("legalizer_capacity_failure", 0)
        >= 2
    ):
        return "pivot_to_large_scale_legalizer_repair"
    return "inconclusive_due_to_trace_gap"


def _write_decision_md(
    *,
    decision: str,
    case_ids: list[int],
    attribution_rows: list[dict[str, Any]],
    comparison_summary: dict[str, Any],
) -> str:
    counts = Counter(str(row["primary_failure"]) for row in attribution_rows)
    return (
        "\n".join(
            [
                "# Step7E Region / Topology Failure Diagnosis",
                "",
                f"Decision: `{decision}`",
                "",
                "## Coverage",
                "",
                f"- cases: `{case_ids}`",
                "- source: Step7D representative suite first; fallback only if",
                "  Step7D artifact is absent.",
                "- scope: sidecar-only diagnostics; no runtime integration or new placer.",
                "",
                "## Primary failure counts",
                "",
                "```json",
                json.dumps(dict(counts), indent=2),
                "```",
                "",
                "## Shape-worked vs large/XL no-safe-improvement comparison",
                "",
                "```json",
                json.dumps(comparison_summary, indent=2),
                "```",
                "",
                "## Interpretation",
                "",
                "- Step7E is diagnostic only and does not add region penalties or gates.",
                "- Trace confidence is reconstructed because exact Step6G placement-order",
                "  traces are not persisted.",
                "- Use the decision as the Step7F architecture direction, not as a runtime policy.",
            ]
        )
        + "\n"
    )


def _comparison_summary(
    attribution_rows: list[dict[str, Any]],
    step7d_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    shape_worked = []
    large_xl_failed = []
    for row in attribution_rows:
        case_id = int(row["case_id"])
        winner = str(step7d_results.get(case_id, {}).get("winner_type", "unknown"))
        target = {
            "case_id": case_id,
            "size_bucket": row["size_bucket"],
            "step7d_winner_type": winner,
            "primary_failure": row["primary_failure"],
            "key_evidence": row["evidence"],
        }
        if "shape policy wins" in winner or "MIB/group policy wins" in winner:
            shape_worked.append(target)
        if row["size_bucket"] in {"large", "xl"} and winner == "no safe improvement":
            large_xl_failed.append(target)
    return {
        "shape_policy_or_macro_worked_cases": shape_worked,
        "large_xl_no_safe_improvement_cases": large_xl_failed,
    }


def _render_visualizations(
    *,
    cases_by_id: dict[int, Any],
    layouts: dict[int, tuple[dict[int, Placement], PuzzleFrame]],
    assignments: dict[int, dict[str, Any]],
    occupancies: dict[int, dict[str, Any]],
    pin_density: dict[int, dict[str, Any]],
    clusters: dict[int, dict[str, Any]],
    fragmentation: dict[int, dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7e_visualizations"
    if viz_dir.exists():
        for stale in viz_dir.glob("*.png"):
            stale.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)
    focus_ids = [
        int(row["case_id"])
        for row in attribution_rows
        if row["size_bucket"] in {"large", "xl"}
        or row["primary_failure"] in {"cluster_region_mismatch", "MIB_group_macro_failure"}
    ][:6]
    written: list[str] = []
    for case_id in focus_ids:
        placements, frame = layouts[case_id]
        case = cases_by_id[case_id]
        written.append(
            _plot_region_scalar(
                plt,
                frame,
                occupancies[case_id]["regions"],
                "utilization",
                viz_dir / f"case{case_id:03d}_region_occupancy_heatmap.png",
                f"case {case_id} region occupancy",
            )
        )
        written.append(
            _plot_region_scalar(
                plt,
                frame,
                pin_density[case_id]["regions"],
                "pin_count",
                viz_dir / f"case{case_id:03d}_pin_density_overlay.png",
                f"case {case_id} pin density",
                placements=placements,
            )
        )
        written.append(
            _plot_cluster_overlay(
                plt,
                case,
                placements,
                frame,
                clusters[case_id],
                viz_dir / f"case{case_id:03d}_cluster_overlay.png",
            )
        )
        written.append(
            _plot_free_space(
                plt,
                frame,
                fragmentation[case_id],
                placements,
                viz_dir / f"case{case_id:03d}_free_space_fragmentation.png",
                f"case {case_id} free-space fragmentation",
            )
        )
        written.append(
            _plot_assignment_arrows(
                plt,
                frame,
                assignments[case_id],
                placements,
                viz_dir / f"case{case_id:03d}_assignment_mismatch_arrows.png",
                f"case {case_id} assignment mismatch",
            )
        )
    _write_viz_index(written, viz_dir)
    return written


def _plot_region_scalar(
    plt: Any,
    frame: PuzzleFrame,
    regions: list[dict[str, Any]],
    key: str,
    path: Path,
    title: str,
    *,
    placements: dict[int, Placement] | None = None,
) -> str:
    from matplotlib.patches import Rectangle

    values = [float(row.get(key, 0.0)) for row in regions]
    hi = max(values, default=1.0) or 1.0
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for row in regions:
        value = float(row.get(key, 0.0))
        ax.add_patch(
            Rectangle(
                (row["xmin"], row["ymin"]),
                row["xmax"] - row["xmin"],
                row["ymax"] - row["ymin"],
                facecolor=plt.cm.viridis(value / hi),
                edgecolor="#111827",
                alpha=0.65,
            )
        )
        cx = (row["xmin"] + row["xmax"]) / 2.0
        cy = (row["ymin"] + row["ymax"]) / 2.0
        ax.text(cx, cy, f"{value:.2f}", fontsize=7, ha="center", va="center")
    if placements:
        for box in placements.values():
            ax.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    facecolor="none",
                    edgecolor="#0f172a",
                    linewidth=0.35,
                    alpha=0.55,
                )
            )
    _finish_axes(ax, frame, title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_cluster_overlay(
    plt: Any,
    case: Any,
    placements: dict[int, Placement],
    frame: PuzzleFrame,
    clusters: dict[str, Any],
    path: Path,
) -> str:
    from matplotlib.patches import Rectangle

    color_by_block: dict[int, Any] = {}
    cmap = plt.get_cmap("tab20")
    for cluster in clusters["clusters"]:
        for member in cluster["members"]:
            color_by_block[int(member)] = cmap(int(cluster["cluster_id"]) % 20)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for idx, box in sorted(placements.items()):
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                facecolor=color_by_block.get(idx, "#e5e7eb"),
                edgecolor="#111827",
                linewidth=0.4,
                alpha=0.72,
            )
        )
        if bool(case.constraints[idx, 2].item()) or bool(case.constraints[idx, 3].item()):
            ax.text(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0, str(idx), fontsize=5)
    _finish_axes(ax, frame, f"case {case.case_id} cluster/community overlay")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_free_space(
    plt: Any,
    frame: PuzzleFrame,
    fragmentation: dict[str, Any],
    placements: dict[int, Placement],
    path: Path,
    title: str,
) -> str:
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    rows = int(fragmentation["grid"]["rows"])
    cols = int(fragmentation["grid"]["cols"])
    cell_w = frame.width / cols
    cell_h = frame.height / rows
    for item in fragmentation["cell_occupancy"]:
        rid = str(item["region_id"]).removeprefix("r")
        row, col = [int(v) for v in rid.split("_", 1)]
        occ = float(item["occupancy"])
        color = "#fee2e2" if occ < 0.15 else "#bfdbfe" if occ < 0.50 else "#1d4ed8"
        ax.add_patch(
            Rectangle(
                (frame.xmin + col * cell_w, frame.ymin + row * cell_h),
                cell_w,
                cell_h,
                facecolor=color,
                edgecolor="#94a3b8",
                linewidth=0.2,
                alpha=0.6,
            )
        )
    for box in placements.values():
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                facecolor="none",
                edgecolor="#111827",
                linewidth=0.3,
                alpha=0.5,
            )
        )
    _finish_axes(ax, frame, title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_assignment_arrows(
    plt: Any,
    frame: PuzzleFrame,
    assignment: dict[str, Any],
    placements: dict[int, Placement],
    path: Path,
    title: str,
) -> str:
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for box in placements.values():
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[2],
                box[3],
                facecolor="#e5e7eb",
                edgecolor="#111827",
                linewidth=0.25,
                alpha=0.45,
            )
        )
    for row in assignment["assignments"]:
        if int(row["mismatch_grid_distance"]) < 2:
            continue
        ax.annotate(
            "",
            xy=row["expected_center"],
            xytext=row["actual_center"],
            arrowprops={"arrowstyle": "->", "color": "#dc2626", "linewidth": 0.8, "alpha": 0.7},
        )
    _finish_axes(ax, frame, title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _finish_axes(ax: Any, frame: PuzzleFrame, title: str) -> None:
    pad = max(frame.width, frame.height) * 0.04
    ax.set_xlim(frame.xmin - pad, frame.xmax + pad)
    ax.set_ylim(frame.ymin - pad, frame.ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.3)


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
                "<title>Step7E region topology diagnostics</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
                "<h1>Step7E region topology diagnostics</h1>",
                *sections,
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-limit", type=int, default=100)
    parser.add_argument("--step7d-suite", default="artifacts/research/step7d_case_suite.json")
    parser.add_argument("--step7d-results", default="artifacts/research/step7d_replay_results.json")
    parser.add_argument(
        "--step6j",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = _load_json(Path(args.step7d_suite), {})
    selected_cases = suite.get("selected_cases", [])
    case_ids = [int(row["case_id"]) for row in selected_cases] or STEP7D_DEFAULT_IDS
    step7d_rows = _load_json(Path(args.step7d_results), [])
    step7d_by_case = {int(row["case_id"]): row for row in step7d_rows}
    step6j = _load_json(Path(args.step6j), {})
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    cases = load_validation_cases(case_limit=max(max(case_ids) + 1, args.case_limit))
    cases_by_id = {idx: case for idx, case in enumerate(cases)}

    region_rows: list[dict[str, Any]] = []
    pin_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    fragmentation_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    layouts: dict[int, tuple[dict[int, Placement], PuzzleFrame]] = {}
    assignment_by_case: dict[int, dict[str, Any]] = {}
    occupancy_by_case: dict[int, dict[str, Any]] = {}
    pin_by_case: dict[int, dict[str, Any]] = {}
    cluster_by_case: dict[int, dict[str, Any]] = {}
    frag_by_case: dict[int, dict[str, Any]] = {}

    for case_id in case_ids:
        case = cases_by_id[case_id]
        baseline, frame, family_usage, trace_confidence = _baseline_layout(
            case, case_id, step6j, boundary_commit_mode
        )
        baseline_metrics = layout_metrics(case, baseline, frame)
        clusters = net_community_clusters(case)
        occupancy = region_occupancy(case, baseline, frame)
        pin_density = pin_density_regions(case, frame)
        assignment = block_region_assignment(case, baseline, frame, clusters)
        fragmentation = free_space_fragmentation(case, baseline, frame)
        trace = candidate_ordering_trace(case, baseline, frame, assignment)
        trace["trace_confidence"] = trace_confidence
        trace["candidate_family_usage"] = family_usage
        repair_audit = _repair_radius_audit(
            case=case,
            baseline=baseline,
            frame=frame,
            boundary_commit_mode=boundary_commit_mode,
        )
        attribution = _failure_attribution(
            case_id=case_id,
            occupancy=occupancy,
            assignment=assignment,
            fragmentation=fragmentation,
            trace=trace,
            repair_audit=repair_audit,
            baseline_metrics=baseline_metrics,
        )
        attribution.update(
            {
                "size_bucket": _size_bucket(case.block_count),
                "step7d_winner_type": step7d_by_case.get(case_id, {}).get("winner_type"),
                "categories": next(
                    (
                        row.get("categories", [row.get("category")])
                        for row in selected_cases
                        if int(row["case_id"]) == case_id
                    ),
                    [],
                ),
            }
        )
        region_rows.append({"case_id": case_id, **occupancy})
        pin_rows.append({"case_id": case_id, **pin_density})
        cluster_rows.append({"case_id": case_id, **clusters})
        assignment_rows.append({"case_id": case_id, **assignment})
        fragmentation_rows.append({"case_id": case_id, **fragmentation})
        trace_rows.append({"case_id": case_id, **trace})
        repair_rows.append({"case_id": case_id, "alternatives": repair_audit})
        attribution_rows.append(attribution)
        layouts[case_id] = (baseline, frame)
        assignment_by_case[case_id] = assignment
        occupancy_by_case[case_id] = occupancy
        pin_by_case[case_id] = pin_density
        cluster_by_case[case_id] = clusters
        frag_by_case[case_id] = fragmentation

    decision = _architecture_decision(attribution_rows)
    comparison = _comparison_summary(attribution_rows, step7d_by_case)
    visualizations = _render_visualizations(
        cases_by_id=cases_by_id,
        layouts=layouts,
        assignments=assignment_by_case,
        occupancies=occupancy_by_case,
        pin_density=pin_by_case,
        clusters=cluster_by_case,
        fragmentation=frag_by_case,
        attribution_rows=attribution_rows,
        output_dir=output_dir,
    )
    decision_md = _write_decision_md(
        decision=decision,
        case_ids=case_ids,
        attribution_rows=attribution_rows,
        comparison_summary=comparison,
    )

    (output_dir / "step7e_region_occupancy.json").write_text(json.dumps(region_rows, indent=2))
    (output_dir / "step7e_pin_density_regions.json").write_text(json.dumps(pin_rows, indent=2))
    (output_dir / "step7e_net_community_clusters.json").write_text(
        json.dumps(cluster_rows, indent=2)
    )
    (output_dir / "step7e_block_region_assignment.json").write_text(
        json.dumps(assignment_rows, indent=2)
    )
    (output_dir / "step7e_free_space_fragmentation.json").write_text(
        json.dumps(fragmentation_rows, indent=2)
    )
    (output_dir / "step7e_candidate_ordering_trace.json").write_text(
        json.dumps(trace_rows, indent=2)
    )
    (output_dir / "step7e_repair_radius_audit.json").write_text(json.dumps(repair_rows, indent=2))
    (output_dir / "step7e_failure_attribution.json").write_text(
        json.dumps(attribution_rows, indent=2)
    )
    (output_dir / "step7e_decision.md").write_text(decision_md)
    print(
        json.dumps(
            {
                "decision": decision,
                "case_ids": case_ids,
                "primary_failure_counts": dict(
                    Counter(str(row["primary_failure"]) for row in attribution_rows)
                ),
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7e_decision.md"),
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


def _mean(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / max(len(rows), 1)


if __name__ == "__main__":
    main()
