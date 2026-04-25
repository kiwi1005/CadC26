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
from puzzleplace.research.move_library import MoveCandidate, apply_move, improvement_score
from puzzleplace.research.pathology import (
    layout_pathology_metrics,
    normalized_move_row,
    pathology_delta,
)
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.selector_replay import (
    STEP6O_GUARD_THRESHOLDS,
    SUSPICIOUS_CASE_IDS,
    compare_selection_sets,
    hpwl_regression_per_boundary_gain,
    select_guarded_case_alternative,
    spatial_balance_worsening,
    step6o_guard_reasons,
)
from puzzleplace.research.virtual_frame import (
    estimate_predicted_compact_hull,
    multistart_virtual_frames,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import _construct

Placement = tuple[float, float, float, float]


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
) -> tuple[dict[int, Placement], Any, dict[str, int]]:
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
    return post, construction_frame, family_usage


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
        MoveCandidate(move_type, targets, "step6o-replay"),
    )
    return placements


def _selected_eval_row(
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any] | None:
    targets = tuple(int(value) for value in selection.get("selected_target_blocks", []))
    for row in rows:
        if row["move_type"] != selection["selected_move_type"]:
            continue
        if tuple(int(value) for value in row.get("target_blocks", [])) == targets:
            return row
    return None


def _enrich_row_with_guard(
    *,
    case: Any,
    original: dict[int, Placement],
    frame: Any,
    family_usage: dict[str, int],
    row: dict[str, Any],
) -> dict[str, Any]:
    alternative = _apply_row(case, original, frame, row)
    fallback_count = int(family_usage.get("fallback_append", 0))
    before_path = layout_pathology_metrics(case, original, frame, fallback_count=fallback_count)
    after_path = layout_pathology_metrics(case, alternative, frame, fallback_count=fallback_count)
    path_delta = pathology_delta(before_path, after_path)
    reasons = step6o_guard_reasons(row, path_delta)
    normalized = normalized_move_row(row)
    return {
        **row,
        "hpwl_delta_norm": normalized["hpwl_delta_norm"],
        "bbox_delta_norm": normalized["bbox_delta_norm"],
        "hpwl_regression_per_boundary_gain": hpwl_regression_per_boundary_gain(row),
        "spatial_balance_worsening": spatial_balance_worsening(path_delta),
        "pathology_delta": path_delta,
        "step6o_guard_reasons": reasons,
        "guard_rejected": bool(reasons),
    }


def _suspicious_outcome_label(
    *,
    before_selection: dict[str, Any],
    after_selection: dict[str, Any],
    old_eval: dict[str, Any] | None,
) -> str:
    if old_eval is None or not old_eval.get("guard_rejected"):
        return "guard_keeps_because_benefit_justified"
    if after_selection["selected_move_type"] == "original":
        return "guard_reverts_to_original"
    if improvement_score(after_selection) < improvement_score(before_selection):
        return "guard_rejects_but_worse_alternative_selected"
    return "guard_correctly_rejects"


def _render_case_images(
    *,
    visualizer: ModuleType,
    case: Any,
    case_id: int,
    frame: Any,
    original: dict[int, Placement],
    before_layout: dict[int, Placement],
    after_layout: dict[int, Placement],
    before_selection: dict[str, Any],
    after_selection: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "original": (original, {"move": "original"}),
        "before_step6m": (
            before_layout,
            {
                "move": before_selection["selected_move_type"],
                "b_d": f"{float(before_selection['boundary_delta']):+.3f}",
                "hpwl_d": f"{float(before_selection['hpwl_delta']):+.3f}",
            },
        ),
        "after_step6o": (
            after_layout,
            {
                "move": after_selection["selected_move_type"],
                "b_d": f"{float(after_selection['boundary_delta']):+.3f}",
                "hpwl_d": f"{float(after_selection['hpwl_delta']):+.3f}",
            },
        ),
    }
    paths: dict[str, str] = {}
    for stage, (placements, metrics) in stages.items():
        path = output_dir / f"case{case_id:03d}_{stage}.png"
        visualizer.render_png(
            case=case,
            placements=placements,
            title=f"Step6O {stage} | case {case_id}",
            metrics=metrics,
            output_path=path,
            draw_nets=24,
            dpi=170,
            frame=frame,
        )
        paths[stage] = str(path)
    return paths


def _markdown(report: dict[str, Any]) -> str:
    compare = report["before_after_comparison"]
    summary = report["summary"]
    lines = [
        "# Step6O Guarded Selector Replay",
        "",
        "## Scope",
        "",
        "- Sidecar replay only; no runtime selector integration.",
        "- Replayed the same Step6M 40-case prefix.",
        "- Applied guards from Step6N: `hpwl_regression_per_boundary_gain > 40` and "
        "`spatial balance worsening > 0.10`.",
        "",
        "## Before / after",
        "",
        "```json",
        json.dumps(
            {
                "selected_move_counts_before": compare["selected_move_counts_before"],
                "selected_move_counts_after": compare["selected_move_counts_after"],
                "suspicious_selected_count_before": compare["suspicious_selected_count_before"],
                "suspicious_selected_count_after": compare["suspicious_selected_count_after"],
                "suspicious_simple_compaction_count_before": compare[
                    "suspicious_simple_compaction_count_before"
                ],
                "suspicious_simple_compaction_count_after": compare[
                    "suspicious_simple_compaction_count_after"
                ],
                "original_fallback_count_before": compare["original_fallback_count_before"],
                "original_fallback_count_after": compare["original_fallback_count_after"],
                "mean_deltas_before": compare["mean_deltas_before"],
                "mean_deltas_after": compare["mean_deltas_after"],
                "mean_delta_change": compare["mean_delta_change"],
            },
            indent=2,
        ),
        "```",
        "",
        "## Suspicious outcome counts",
        "",
        "```json",
        json.dumps(summary["suspicious_outcome_counts"], indent=2),
        "```",
        "",
        "## False rejection cases",
        "",
        f"`{summary['false_rejection_case_ids']}`",
        "",
        "## Decision",
        "",
        f"`{summary['decision']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step6j",
        default="artifacts/research/step6j_boundary_hull_ownership.json",
    )
    parser.add_argument("--step6m-report", default="artifacts/research/step6m_report.json")
    parser.add_argument(
        "--step6n-report",
        default="artifacts/research/step6n_case_pathology_report.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/research/step6o_guarded_selector_replay.json",
    )
    parser.add_argument(
        "--visualization-dir",
        default="artifacts/research/step6o_suspicious_visualizations",
    )
    args = parser.parse_args()

    step6j = _load_json(args.step6j)
    step6m = _load_json(args.step6m_report)
    step6n = _load_json(args.step6n_report)
    before_selections = list(step6m["selected_alternatives"])
    eval_rows = list(step6m["move_library_eval"])
    max_case = max(int(row["case_id"]) for row in before_selections)
    cases = load_validation_cases(case_limit=max_case + 1)
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    rows_by_case: dict[int, list[dict[str, Any]]] = {}
    for row in eval_rows:
        rows_by_case.setdefault(int(row["case_id"]), []).append(row)

    visualizer = _load_visualizer()
    guarded_eval_rows: list[dict[str, Any]] = []
    guarded_selections: list[dict[str, Any]] = []
    suspicious_traces: list[dict[str, Any]] = []
    guard_rejection_rows: list[dict[str, Any]] = []

    before_by_case = {int(row["case_id"]): row for row in before_selections}
    suspicious_ids = set(int(value) for value in step6n.get("summary", {}).get(
        "suspicious_case_ids",
        sorted(SUSPICIOUS_CASE_IDS),
    ))

    for case_id in sorted(rows_by_case):
        case = cases[case_id]
        original, frame, family_usage = _reconstruct_original(
            case,
            case_id,
            step6j,
            boundary_commit_mode,
        )
        enriched_rows = [
            _enrich_row_with_guard(
                case=case,
                original=original,
                frame=frame,
                family_usage=family_usage,
                row=row,
            )
            for row in rows_by_case[case_id]
        ]
        guard_count = sum(int(row["guard_rejected"]) for row in enriched_rows)
        enriched_rows = [
            {**row, "case_guard_rejected_count": guard_count} for row in enriched_rows
        ]
        guarded_eval_rows.extend(enriched_rows)
        guard_rejection_rows.extend(
            row for row in enriched_rows if row["accepted"] and row["guard_rejected"]
        )
        after_selection = select_guarded_case_alternative(enriched_rows, mode=str(step6m["mode"]))
        after_selection["case_id"] = case_id
        guarded_selections.append(after_selection)

        before_selection = before_by_case[case_id]
        old_eval = _selected_eval_row(enriched_rows, before_selection)
        if case_id in suspicious_ids:
            before_layout = _apply_row(case, original, frame, before_selection)
            after_layout = _apply_row(case, original, frame, after_selection)
            images = _render_case_images(
                visualizer=visualizer,
                case=case,
                case_id=case_id,
                frame=frame,
                original=original,
                before_layout=before_layout,
                after_layout=after_layout,
                before_selection=before_selection,
                after_selection=after_selection,
                output_dir=Path(args.visualization_dir),
            )
            suspicious_traces.append(
                {
                    "case_id": case_id,
                    "outcome": _suspicious_outcome_label(
                        before_selection=before_selection,
                        after_selection=after_selection,
                        old_eval=old_eval,
                    ),
                    "before_selection": before_selection,
                    "after_selection": after_selection,
                    "old_selected_guard_reasons": []
                    if old_eval is None
                    else old_eval["step6o_guard_reasons"],
                    "old_selected_hpwl_regression_per_boundary_gain": None
                    if old_eval is None
                    else old_eval["hpwl_regression_per_boundary_gain"],
                    "old_selected_spatial_balance_worsening": None
                    if old_eval is None
                    else old_eval["spatial_balance_worsening"],
                    "guarded_rejected_accepted_alternatives": [
                        {
                            "move_type": row["move_type"],
                            "target_blocks": row["target_blocks"],
                            "boundary_delta": row["boundary_delta"],
                            "bbox_delta": row["bbox_delta"],
                            "hpwl_delta": row["hpwl_delta"],
                            "soft_delta": row["soft_delta"],
                            "guard_reasons": row["step6o_guard_reasons"],
                        }
                        for row in enriched_rows
                        if row["accepted"] and row["guard_rejected"]
                    ],
                    "image_paths": images,
                }
            )

    comparison = compare_selection_sets(
        before_selections,
        guarded_selections,
        suspicious_case_ids=suspicious_ids,
    )
    outcome_counts = Counter(row["outcome"] for row in suspicious_traces)
    false_rejections = [
        int(row["case_id"])
        for row in guard_rejection_rows
        if int(row["case_id"]) not in suspicious_ids
        and row["move_type"] == before_by_case[int(row["case_id"])]["selected_move_type"]
    ]
    worse_alternative_count = outcome_counts.get(
        "guard_rejects_but_worse_alternative_selected",
        0,
    )
    if false_rejections:
        decision = "sidecar_guard_has_false_rejections_review_before_step6p"
    elif worse_alternative_count:
        decision = "guard_blocks_compaction_but_next_selector_tradeoff_needs_review"
    else:
        decision = "sidecar_guard_promising_needs_large_xl_replay"
    report_summary = {
        "suspicious_case_ids": sorted(suspicious_ids),
        "suspicious_outcome_counts": dict(outcome_counts),
        "false_rejection_case_ids": sorted(set(false_rejections)),
        "decision": decision,
    }
    report = {
        "mode": "sidecar_guarded_selector_replay",
        "inputs": {
            "step6m_report": args.step6m_report,
            "step6n_report": args.step6n_report,
            "case_count": len(before_selections),
        },
        "guard_thresholds": STEP6O_GUARD_THRESHOLDS,
        "guard_semantics": {
            "hpwl_regression_per_boundary_gain": "max(hpwl_delta, 0) / max(boundary_delta, eps)",
            "spatial_balance_worsening": "max(left_right_balance_delta, top_bottom_balance_delta)",
            "scope": "applied only to accepted simple_compaction alternatives in sidecar replay",
        },
        "before_after_comparison": comparison,
        "suspicious_case_traces": suspicious_traces,
        "guard_rejection_summary": {
            "accepted_alternatives_rejected_by_guard": len(guard_rejection_rows),
            "rejected_case_ids": sorted({int(row["case_id"]) for row in guard_rejection_rows}),
            "reason_counts": dict(
                Counter(
                    reason
                    for row in guard_rejection_rows
                    for reason in row["step6o_guard_reasons"]
                )
            ),
        },
        "summary": report_summary,
        "step6o_gate_notes": {
            "runtime_integration_changed": False,
            "large_xl_replay_changed": False,
            "repo_wide_ruff_debt_cleaned": False,
            "mib_or_group_moves_tuned": False,
            "guards_written_as_final_rules": False,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    (output.parent / "step6o_guarded_selected_alternatives.json").write_text(
        json.dumps(guarded_selections, indent=2)
    )
    (output.parent / "step6o_guarded_eval_rows.json").write_text(
        json.dumps(guarded_eval_rows, indent=2)
    )
    (output.parent / "step6o_profile_summary.md").write_text(_markdown(report))
    print(
        json.dumps(
            {
                "output": str(output),
                "suspicious_outcome_counts": dict(outcome_counts),
                "selected_move_counts_before": comparison["selected_move_counts_before"],
                "selected_move_counts_after": comparison["selected_move_counts_after"],
                "false_rejection_case_ids": report_summary["false_rejection_case_ids"],
                "decision": report_summary["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
