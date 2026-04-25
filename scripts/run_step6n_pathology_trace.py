#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from puzzleplace.repair import finalize_layout
from puzzleplace.research.move_library import (
    MoveCandidate,
    apply_move,
    improvement_score,
    layout_metrics,
)
from puzzleplace.research.pathology import (
    METRIC_SEMANTICS,
    guard_calibration_candidates,
    hpwl_regression_per_boundary_gain,
    label_case_pathology,
    layout_pathology_metrics,
    normalized_move_row,
    pathology_delta,
    scale_coverage_report,
)
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.virtual_frame import (
    estimate_predicted_compact_hull,
    multistart_virtual_frames,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import _construct

Placement = tuple[float, float, float, float]


THRESHOLDS = {
    "suspicious_simple_compaction_hpwl_delta_raw": 3.0,
    "suspicious_simple_compaction_hpwl_delta_norm": 0.25,
    "spatial_balance_worsen_threshold": 0.10,
    "bbox_delta_norm_threshold": 0.02,
    "hpwl_regression_per_boundary_gain_threshold": 40.0,
    "safe_mode_bbox_worse_fraction": 0.05,
    "safe_mode_hpwl_worse_fraction": 0.25,
    "safe_mode_soft_worse_tolerance": 1,
}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _load_visualizer() -> ModuleType:
    path = Path("scripts/visualize_step6g_layouts.py")
    spec = importlib.util.spec_from_file_location("step6_visualizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reconstruct_original(
    case: Any,
    case_id: int,
    step6j: dict[str, Any],
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[dict[int, Placement], Any, dict[str, int], int, int]:
    frames = multistart_virtual_frames(case)
    runs_by_case = {int(row["case_id"]): row for row in step6j.get("runs", [])}
    if case_id in runs_by_case:
        run = runs_by_case[case_id]
        frame = frames[int(run["best_frame_index"])]
        seed = int(run["best_seed"])
        start = int(run["best_start"])
    else:
        frame = frames[case_id % len(frames)]
        seed = case_id % 3
        start = case_id % 5
    pre, family_usage, construction_frame, predicted_hull = _construct(
        case,
        seed,
        start,
        frame,
        boundary_commit_mode=boundary_commit_mode,
    )
    if construction_frame is None:
        construction_frame = frame
    if predicted_hull is None:
        predicted_hull = estimate_predicted_compact_hull(case, construction_frame)
    repair = finalize_layout(case, pre)
    post = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repair.positions)
    }
    return post, construction_frame, family_usage, seed, start


def _selected_eval_row(
    rows_by_case: dict[int, list[dict[str, Any]]],
    selection: dict[str, Any],
) -> dict[str, Any] | None:
    case_rows = rows_by_case.get(int(selection["case_id"]), [])
    selected_targets = tuple(int(value) for value in selection.get("selected_target_blocks", []))
    for row in case_rows:
        if row["move_type"] != selection["selected_move_type"]:
            continue
        if tuple(int(value) for value in row.get("target_blocks", [])) == selected_targets:
            return row
    return None


def _is_suspicious(row: dict[str, Any]) -> bool:
    return bool(
        row.get("selected_move_type") == "simple_compaction"
        and float(row.get("boundary_delta", 0.0)) > 0.0
        and (
            float(row.get("hpwl_delta", 0.0))
            > THRESHOLDS["suspicious_simple_compaction_hpwl_delta_raw"]
            or float(row.get("hpwl_delta_norm", 0.0))
            > THRESHOLDS["suspicious_simple_compaction_hpwl_delta_norm"]
        )
    )


def _best_rejected(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rejected = [row for row in rows if not row.get("accepted")]
    if not rejected:
        return None
    return max(
        rejected,
        key=lambda row: (
            improvement_score(row),
            float(row.get("boundary_delta", 0.0)),
            -float(row.get("generation_time_ms", 0.0) + row.get("eval_time_ms", 0.0)),
        ),
    )


def _apply_row(
    case: Any,
    original: dict[int, Placement],
    frame: Any,
    row: dict[str, Any],
) -> dict[int, Placement]:
    if row.get("move_type") == "original" or row.get("selected_move_type") == "original":
        return dict(original)
    move_type = str(row.get("move_type", row.get("selected_move_type")))
    target_values = row.get("target_blocks", row.get("selected_target_blocks", []))
    targets = tuple(int(value) for value in target_values)
    placements, _reasons, _count = apply_move(
        case,
        original,
        frame,
        MoveCandidate(move_type, targets, "step6n-trace"),
    )
    return placements


def _render_trace_images(
    *,
    visualizer: ModuleType,
    case: Any,
    case_id: int,
    frame: Any,
    original: dict[int, Placement],
    selected: dict[int, Placement],
    best_rejected: dict[int, Placement] | None,
    selection: dict[str, Any],
    rejected_row: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "original": original,
        "selected": selected,
    }
    if best_rejected is not None:
        rows["best_rejected"] = best_rejected
    paths: dict[str, str] = {}
    for stage, placements in rows.items():
        if stage == "original":
            metrics = {"move": "original"}
        elif stage == "selected":
            metrics = {
                "move": selection["selected_move_type"],
                "b_d": f"{float(selection['boundary_delta']):+.3f}",
                "hpwl_d": f"{float(selection['hpwl_delta']):+.3f}",
            }
        else:
            assert rejected_row is not None
            metrics = {
                "move": rejected_row["move_type"],
                "rej": ",".join(rejected_row.get("rejected_reason", [])[:2]),
                "b_d": f"{float(rejected_row['boundary_delta']):+.3f}",
            }
        path = output_dir / f"case{case_id:03d}_{stage}.png"
        visualizer.render_png(
            case=case,
            placements=placements,
            title=f"Step6N {stage} | case {case_id}",
            metrics=metrics,
            output_path=path,
            draw_nets=24,
            dpi=170,
            frame=frame,
        )
        paths[stage] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step6j",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
    )
    parser.add_argument("--step6m-report", default="artifacts/research/step6m_report.json")
    parser.add_argument(
        "--output",
        default="artifacts/research/step6n_case_pathology_report.json",
    )
    parser.add_argument(
        "--visualization-dir",
        default="artifacts/research/step6n_suspicious_visualizations",
    )
    args = parser.parse_args()

    step6j = _load_json(args.step6j)
    step6m = _load_json(args.step6m_report)
    selections = list(step6m["selected_alternatives"])
    eval_rows = list(step6m["move_library_eval"])
    profiles = list(step6m["case_profiles"])
    max_case = max(int(row["case_id"]) for row in selections)
    cases = load_validation_cases(case_limit=max_case + 1)
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    rows_by_case: dict[int, list[dict[str, Any]]] = {}
    for row in eval_rows:
        rows_by_case.setdefault(int(row["case_id"]), []).append(row)

    visualizer = _load_visualizer()
    pathology_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    placements_by_case: dict[int, dict[int, Placement]] = {}

    for selection in selections:
        case_id = int(selection["case_id"])
        case = cases[case_id]
        original, frame, family_usage, seed, start = _reconstruct_original(
            case,
            case_id,
            step6j,
            boundary_commit_mode,
        )
        placements_by_case[case_id] = original
        selected_eval = _selected_eval_row(rows_by_case, selection)
        normalized = normalized_move_row(selected_eval) if selected_eval is not None else {
            "hpwl_delta_norm": 0.0,
            "bbox_delta_norm": 0.0,
            "hpwl_delta_raw": selection.get("hpwl_delta", 0.0),
            "bbox_delta_raw": selection.get("bbox_delta", 0.0),
        }
        selected_layout = _apply_row(case, original, frame, selection)
        fallback_count = int(family_usage.get("fallback_append", 0))
        before_path = layout_pathology_metrics(case, original, frame, fallback_count=fallback_count)
        after_path = layout_pathology_metrics(
            case,
            selected_layout,
            frame,
            fallback_count=fallback_count,
        )
        enriched_selection = {
            **selection,
            "hpwl_delta_norm": normalized["hpwl_delta_norm"],
            "bbox_delta_norm": normalized["bbox_delta_norm"],
        }
        labels = label_case_pathology(
            enriched_selection,
            before_path,
            after_path,
            hpwl_delta_norm=float(normalized["hpwl_delta_norm"]),
            bbox_delta_norm=float(normalized["bbox_delta_norm"]),
        )
        path_delta = pathology_delta(before_path, after_path)
        suspicious = _is_suspicious(enriched_selection) or (
            "spatial_imbalance_after_compaction" in labels
            and selection["selected_move_type"] == "simple_compaction"
        )
        pathology_rows.append(
            {
                "case_id": case_id,
                "suite": selection.get("suite"),
                "selected_move_type": selection["selected_move_type"],
                "boundary_delta": selection["boundary_delta"],
                "hpwl_delta": selection["hpwl_delta"],
                "hpwl_delta_norm": normalized["hpwl_delta_norm"],
                "bbox_delta": selection["bbox_delta"],
                "bbox_delta_norm": normalized["bbox_delta_norm"],
                "soft_delta": selection["soft_delta"],
                "hpwl_regression_per_boundary_gain": hpwl_regression_per_boundary_gain(selection),
                "before_pathology": before_path,
                "after_pathology": after_path,
                "pathology_delta": path_delta,
                "pathology_labels": labels,
                "suspicious": suspicious,
                "seed": seed,
                "start": start,
            }
        )
        if suspicious:
            rejected = _best_rejected(rows_by_case.get(case_id, []))
            rejected_layout = (
                _apply_row(case, original, frame, rejected) if rejected is not None else None
            )
            image_paths = _render_trace_images(
                visualizer=visualizer,
                case=case,
                case_id=case_id,
                frame=frame,
                original=original,
                selected=selected_layout,
                best_rejected=rejected_layout,
                selection=enriched_selection,
                rejected_row=rejected,
                output_dir=Path(args.visualization_dir),
            )
            trace_rows.append(
                {
                    "case_id": case_id,
                    "thresholds_used": THRESHOLDS,
                    "original_metrics": layout_metrics(case, original, frame),
                    "selected_metrics": layout_metrics(case, selected_layout, frame),
                    "selected_move": enriched_selection,
                    "all_rejected_alternative_metrics": [
                        {
                            "move_type": row["move_type"],
                            "target_blocks": row["target_blocks"],
                            "after_metrics": row["after_metrics"],
                            "rejected_reason": row["rejected_reason"],
                            "boundary_delta": row["boundary_delta"],
                            "bbox_delta": row["bbox_delta"],
                            "hpwl_delta": row["hpwl_delta"],
                            "improvement_per_ms": row["improvement_per_ms"],
                        }
                        for row in rows_by_case.get(case_id, [])
                        if not row.get("accepted")
                    ],
                    "selected_reason": selection.get("selection_reason"),
                    "best_rejected_reason": None
                    if rejected is None
                    else rejected.get("rejected_reason"),
                    "best_rejected_move": rejected,
                    "move_runtime_cost": selection.get("runtime_ms"),
                    "pathology_labels": labels,
                    "image_paths": image_paths,
                }
            )

    scale = scale_coverage_report(profiles, selections, pathology_rows)
    guard_candidates = guard_calibration_candidates(pathology_rows)
    report = {
        "metric_semantics": METRIC_SEMANTICS,
        "case_pathology_rows": pathology_rows,
        "scale_coverage": scale,
        "selector_decision_trace": trace_rows,
        "guard_calibration_candidates": guard_candidates,
        "summary": {
            "case_count": len(pathology_rows),
            "suspicious_case_ids": [row["case_id"] for row in pathology_rows if row["suspicious"]],
            "pathology_label_counts": dict(
                Counter(label for row in pathology_rows for label in row["pathology_labels"])
            ),
            "large_xl_coverage_gap": scale["buckets"]["large"]["coverage_gap"]
            or scale["buckets"]["xl"]["coverage_gap"],
        },
        "step6n_gate_notes": {
            "metric_semantics_explicit": True,
            "selector_decision_trace_emitted": True,
            "normalized_deltas_available": True,
            "pathology_metrics_emitted": True,
            "guard_candidates_applied": False,
        },
    }
    summary = cast(dict[str, Any], report["summary"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    siblings = {
        "step6n_metric_semantics.json": METRIC_SEMANTICS,
        "step6n_selector_decision_trace.json": trace_rows,
        "step6n_case_pathology_report.json": report,
        "step6n_guard_calibration_candidates.json": guard_candidates,
    }
    for filename, payload in siblings.items():
        (output.parent / filename).write_text(json.dumps(payload, indent=2))
    (output.parent / "step6n_profile_summary.md").write_text(_profile_markdown(report))
    print(
        json.dumps(
            {
                "output": str(output),
                "suspicious_case_ids": summary["suspicious_case_ids"],
                "trace_count": len(trace_rows),
                "coverage_gap": scale["coverage_gap_note"],
            },
            indent=2,
        )
    )


def _profile_markdown(report: dict[str, Any]) -> str:
    labels = report["summary"]["pathology_label_counts"]
    suspicious = report["summary"]["suspicious_case_ids"]
    lines = [
        "# Step6N Metric Semantics + Pathology Summary",
        "",
        "## Metric semantics",
        "",
        "- `b_d > 0`: boundary satisfaction improved.",
        "- `hpwl_d > 0`: HPWL proxy regressed.",
        "- `bbox_d > 0`: bbox area increased.",
        "- `soft_d < 0`: fewer soft boundary violations.",
        "- normalized HPWL/bbox deltas divide by original HPWL/bbox.",
        "",
        "## Suspicious cases",
        "",
        f"`{suspicious}`",
        "",
        "## Pathology label counts",
        "",
        "```json",
        json.dumps(labels, indent=2),
        "```",
        "",
        "## Scale coverage",
        "",
        "```json",
        json.dumps(report["scale_coverage"], indent=2),
        "```",
        "",
        "## Guard candidates",
        "",
        "```json",
        json.dumps(report["guard_calibration_candidates"], indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
