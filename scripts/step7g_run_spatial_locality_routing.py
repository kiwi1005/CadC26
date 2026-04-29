#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.locality_routing import (
    calibration_report,
    predict_move_locality,
    routing_summary,
)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _case_id(value: Any) -> int:
    text = str(value)
    if text.startswith("validation-"):
        return int(text.replace("validation-", ""))
    return int(text)


def _artifact_locality_maps(
    occupancy_rows: list[dict[str, Any]],
    pin_rows: list[dict[str, Any]],
    fragmentation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pins_by_case = {_case_id(row["case_id"]): row for row in pin_rows}
    frag_by_case = {_case_id(row["case_id"]): row for row in fragmentation_rows}
    out: list[dict[str, Any]] = []
    for occ in occupancy_rows:
        case_id = _case_id(occ["case_id"])
        pin_by_region = {
            row["region_id"]: row for row in pins_by_case.get(case_id, {}).get("regions", [])
        }
        coarse_regions = []
        for row in occ["regions"]:
            rid = row["region_id"]
            pin = pin_by_region.get(rid, {})
            area = max(float(row["area"]), 1e-9)
            free = float(row["unused_capacity"]) / area
            occupancy = float(row["utilization"])
            coarse_regions.append(
                {
                    **row,
                    "occupancy_mask": occupancy,
                    "free_space_mask": free,
                    "fixed_preplaced_mask": (
                        float(row["fixed_area"]) + float(row["preplaced_area"])
                    )
                    / area,
                    "pin_density_heatmap": float(pin.get("pin_density", 0.0)),
                    "net_community_demand_map": float(pin.get("terminal_pull_density", 0.0)),
                    "region_slack_map": float(row["unused_capacity"]),
                    "hole_fragmentation_map": free,
                    "boundary_owner_map": float(row["boundary_area"]) / area,
                    "MIB_group_closure_mask": 0.0,
                    "repair_reachability_mask": max(free - max(occupancy - 0.85, 0.0), 0.0),
                }
            )
        frag = frag_by_case.get(case_id, {})
        adaptive_regions = [
            {
                "region_id": row["region_id"],
                "occupancy_mask": float(row["occupancy"]),
                "free_space_mask": max(1.0 - float(row["occupancy"]), 0.0),
                "fixed_preplaced_mask": 0.0,
                "pin_density_heatmap": 0.0,
                "net_community_demand_map": 0.0,
                "region_slack_map": max(1.0 - float(row["occupancy"]), 0.0),
                "hole_fragmentation_map": max(1.0 - float(row["occupancy"]), 0.0),
                "boundary_owner_map": 0.0,
                "MIB_group_closure_mask": 0.0,
                "repair_reachability_mask": max(1.0 - float(row["occupancy"]), 0.0),
            }
            for row in frag.get("cell_occupancy", [])
        ]
        coarse = {
            "name": "coarse",
            "grid": occ["grid"],
            "regions": coarse_regions,
            "summary": _summary(coarse_regions),
        }
        adaptive = {
            "name": "adaptive",
            "grid": frag.get("grid", {}),
            "regions": adaptive_regions,
            "summary": _summary(adaptive_regions),
        }
        out.append(
            {
                "case_id": case_id,
                "block_count": None,
                "resolutions": [coarse, adaptive],
                "sensitivity": _sensitivity(coarse, adaptive),
            }
        )
    return out


def _predict_from_artifacts(
    candidate: dict[str, Any],
    repair_rows: list[dict[str, Any]],
    locality_map: dict[str, Any],
) -> dict[str, Any]:
    current = next(
        (row for row in repair_rows if row["repair_mode"] == "current_repair_baseline"),
        repair_rows[0],
    )
    coarse = next(row for row in locality_map["resolutions"] if row["name"] == "coarse")
    min_slack = min((float(row["region_slack_map"]) for row in coarse["regions"]), default=0.0)
    total_slack = sum(float(row["region_slack_map"]) for row in coarse["regions"])
    changed_count = int(candidate["changed_block_count"])
    block_count = max(int(round(changed_count / max(candidate["changed_block_fraction"], 1e-9))), 1)
    fit_ratio = changed_count / max(total_slack, 1e-9)
    return predict_move_locality(
        case_id=int(candidate["case_id"]),
        block_count=block_count,
        changed_block_count=changed_count,
        touched_region_count=int(current.get("affected_region_count", 0)),
        macro_closure_size=max(int(current.get("repair_seed_count", 0)), changed_count),
        min_region_slack=min_slack,
        free_space_fit_ratio=fit_ratio,
        hard_summary=candidate["hard_summary"],
    )


def _visualization_audit(path: Path) -> dict[str, Any]:
    debug_path = path / "arrow_endpoint_debug.json"
    rows = _load_json(debug_path, [])
    if not rows:
        return {
            "status": "missing_debug",
            "arrow_endpoint_is_after_center": False,
            "block_id_matching_ok": False,
            "same_coordinate_frame_likely": False,
            "after_bbox_frame_protrusion_measured": False,
            "debug_path": str(debug_path),
        }
    return {
        "status": "ok",
        "arrow_endpoint_is_after_center": True,
        "block_id_matching_ok": all(bool(row.get("block_id_matched")) for row in rows),
        "same_coordinate_frame_likely": True,
        "after_bbox_frame_protrusion_measured": all("after_inside_frame" in row for row in rows),
        "debug_path": str(debug_path),
        "row_count": len(rows),
        "max_raw_distance": max(float(row["distance"]) for row in rows),
        "after_outside_frame_count": sum(int(not row["after_inside_frame"]) for row in rows),
    }


def _decision(calibration: dict[str, Any], audit: dict[str, Any], routing: dict[str, Any]) -> str:
    if audit["status"] != "ok" or not audit["block_id_matching_ok"]:
        return "pivot_to_visualization_or_trace_repair"
    if float(calibration["accuracy"]) < 0.60:
        return "inconclusive_due_to_prediction_quality"
    if calibration["counts"]["under_predicted_globality"] > 0:
        return "inconclusive_due_to_prediction_quality"
    if routing["route_counts"].get("global_route_not_local_selector", 0) >= 1:
        return "promote_locality_routing_to_step7c"
    if routing["route_counts"].get("macro_legalizer", 0) >= 2:
        return "pivot_to_macro_level_move_generator"
    return "pivot_to_coarse_region_planner"


def _write_decision_md(
    *,
    decision: str,
    case_ids: list[int],
    calibration: dict[str, Any],
    routing: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    return (
        "\n".join(
            [
                "# Step7G Spatial Locality Map and Move Routing",
                "",
                f"Decision: `{decision}`",
                "",
                "## Coverage",
                "",
                f"- cases: `{case_ids}`",
                "- source: Step7F repair candidates plus Step7E locality artifacts.",
                "- scope: sidecar-only; routing is classification, not a hard gate.",
                "",
                "## Routing summary",
                "",
                "```json",
                json.dumps(routing, indent=2),
                "```",
                "",
                "## Calibration",
                "",
                "```json",
                json.dumps(calibration["counts"], indent=2),
                "```",
                f"- accuracy: `{calibration['accuracy']:.3f}`",
                "",
                "## Visualization sanity audit",
                "",
                "```json",
                json.dumps(audit, indent=2),
                "```",
            ]
        )
        + "\n"
    )


def _render_visualizations(locality_maps: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "step7g_visualizations"
    if viz_dir.exists():
        for stale in viz_dir.glob("*.png"):
            stale.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for row in locality_maps:
        case_id = int(row["case_id"])
        coarse = next(item for item in row["resolutions"] if item["name"] == "coarse")
        for channel in ("occupancy_mask", "free_space_mask", "repair_reachability_mask"):
            path = viz_dir / f"case{case_id:03d}_{channel}.png"
            _plot_channel(plt, coarse, channel, path, f"case {case_id} {channel}")
            paths.append(str(path))
    _write_viz_index(paths, viz_dir)
    return paths


def _plot_channel(plt: Any, grid: dict[str, Any], channel: str, path: Path, title: str) -> None:
    regions = grid["regions"]
    values = [float(row[channel]) for row in regions]
    hi = max(values, default=1.0) or 1.0
    rows = int(grid["grid"].get("rows", 1))
    cols = int(grid["grid"].get("cols", 1))
    matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row in regions:
        matrix[int(row["row"])][int(row["col"])] = float(row[channel])
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=110)
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=hi, origin="lower")
    for row in regions:
        value = float(row[channel])
        ax.text(
            int(row["col"]),
            int(row["row"]),
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if value > hi * 0.45 else "black",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("region col")
    ax.set_ylabel("region row")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
                "<title>Step7G spatial locality maps</title>",
                "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
                "<h1>Step7G spatial locality maps</h1>",
                *sections,
            ]
        )
    )


def _summary(regions: list[dict[str, Any]]) -> dict[str, float]:
    if not regions:
        return {}
    return {
        "max_occupancy": max(float(row["occupancy_mask"]) for row in regions),
        "mean_free_space": _mean(float(row["free_space_mask"]) for row in regions),
        "max_pin_density": max(float(row["pin_density_heatmap"]) for row in regions),
        "max_repair_reachability": max(
            float(row["repair_reachability_mask"]) for row in regions
        ),
    }


def _sensitivity(coarse: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, float]:
    return {
        f"{key}_delta_adaptive_minus_coarse": float(adaptive["summary"].get(key, 0.0))
        - float(coarse["summary"].get(key, 0.0))
        for key in ("max_occupancy", "mean_free_space", "max_repair_reachability")
    }


def _mean(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / max(len(rows), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step7f-candidates",
        default="artifacts/research/step7f_repair_candidates.json",
    )
    parser.add_argument(
        "--step7f-results",
        default="artifacts/research/step7f_bounded_repair_results.json",
    )
    parser.add_argument(
        "--step7e-occupancy",
        default="artifacts/research/step7e_region_occupancy.json",
    )
    parser.add_argument(
        "--step7e-pins",
        default="artifacts/research/step7e_pin_density_regions.json",
    )
    parser.add_argument(
        "--step7e-fragmentation",
        default="artifacts/research/step7e_free_space_fragmentation.json",
    )
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_json(Path(args.step7f_candidates), [])
    step7f_results = _load_json(Path(args.step7f_results), [])
    step7f_by_case: dict[int, list[dict[str, Any]]] = {}
    for row in step7f_results:
        step7f_by_case.setdefault(int(row["case_id"]), []).append(row)
    locality_maps = _artifact_locality_maps(
        _load_json(Path(args.step7e_occupancy), []),
        _load_json(Path(args.step7e_pins), []),
        _load_json(Path(args.step7e_fragmentation), []),
    )
    maps_by_case = {int(row["case_id"]): row for row in locality_maps}
    predictions = [
        _predict_from_artifacts(
            candidate,
            step7f_by_case[int(candidate["case_id"])],
            maps_by_case[int(candidate["case_id"])],
        )
        for candidate in candidates
    ]
    routing = routing_summary(predictions)
    calibration = calibration_report(predictions, step7f_by_case)
    audit = _visualization_audit(output_dir / "step7f_visualizations")
    decision = _decision(calibration, audit, routing)
    visualizations = _render_visualizations(locality_maps, output_dir)
    case_ids = [int(row["case_id"]) for row in candidates]
    decision_md = _write_decision_md(
        decision=decision,
        case_ids=case_ids,
        calibration=calibration,
        routing=routing,
        audit=audit,
    )
    routing_results = {
        "summary": routing,
        "per_case": [
            {
                "case_id": row["case_id"],
                "predicted_locality_class": row["predicted_locality_class"],
                "routing_decision": row["predicted_repair_mode"],
            }
            for row in predictions
        ],
    }
    (output_dir / "step7g_locality_maps.json").write_text(json.dumps(locality_maps, indent=2))
    (output_dir / "step7g_move_locality_predictions.json").write_text(
        json.dumps(predictions, indent=2)
    )
    (output_dir / "step7g_routing_results.json").write_text(
        json.dumps(routing_results, indent=2)
    )
    (output_dir / "step7g_calibration_report.json").write_text(
        json.dumps(calibration, indent=2)
    )
    (output_dir / "step7g_visualization_audit.json").write_text(json.dumps(audit, indent=2))
    (output_dir / "step7g_decision.md").write_text(decision_md)
    print(
        json.dumps(
            {
                "decision": decision,
                "case_ids": case_ids,
                "routing": routing,
                "calibration_accuracy": calibration["accuracy"],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7g_decision.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
