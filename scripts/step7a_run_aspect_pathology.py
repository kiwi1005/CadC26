#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from puzzleplace.diagnostics.aspect import (
    case_aspect_pathology,
    correlation_report,
    summarize_by_role,
    summarize_candidate_families,
)
from puzzleplace.repair import finalize_layout
from puzzleplace.research.move_library import MoveCandidate, apply_move, layout_metrics, size_bucket
from puzzleplace.research.pathology import layout_pathology_metrics
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.virtual_frame import (
    Placement,
    estimate_predicted_compact_hull,
    multistart_virtual_frames,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import _construct


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
    return post, construction_frame, dict(family_usage)


def _apply_representative(
    case: Any,
    original: dict[int, Placement],
    frame: Any,
    row: dict[str, Any],
) -> dict[int, Placement]:
    move_type = str(row.get("move_type", "original"))
    if move_type == "original":
        return dict(original)
    placements = dict(original)
    combo_parts = row.get("combo_parts")
    combo_targets = row.get("combo_part_targets")
    if isinstance(combo_parts, list) and isinstance(combo_targets, list):
        for part, targets in zip(combo_parts, combo_targets, strict=False):
            placements, _reasons, _count = apply_move(
                case,
                placements,
                frame,
                MoveCandidate(str(part), tuple(int(value) for value in targets), "step7a"),
            )
        return placements
    targets = tuple(int(value) for value in row.get("target_blocks", []))
    placements, _reasons, _count = apply_move(
        case,
        placements,
        frame,
        MoveCandidate(move_type, targets, "step7a"),
    )
    return placements


def _selected_representative(case_row: dict[str, Any], name: str) -> dict[str, Any]:
    reps = dict(case_row.get("representatives", {}))
    if name not in reps:
        return {
            "case_id": case_row["case_id"],
            "move_type": "original",
            "target_blocks": [],
            "boundary_delta": 0.0,
            "hpwl_delta": 0.0,
            "bbox_delta": 0.0,
            "objectives": {"hpwl_delta_norm": 0.0, "bbox_delta_norm": 0.0},
        }
    return dict(reps[name])


def _metric_row(
    case: Any,
    placements: dict[int, Placement],
    frame: Any,
    rep: dict[str, Any],
) -> dict[str, Any]:
    layout = layout_metrics(case, placements, frame)
    pathology = layout_pathology_metrics(case, placements, frame)
    return {
        "case_id": int(str(case.case_id).replace("validation-", "")),
        "selected_move_type": rep.get("move_type", "original"),
        "boundary_failure_rate": 1.0 - float(layout["boundary_satisfaction_rate"]),
        "bbox_delta_norm": float(rep.get("objectives", {}).get("bbox_delta_norm", 0.0)),
        "hpwl_delta_norm": float(rep.get("objectives", {}).get("hpwl_delta_norm", 0.0)),
        **pathology,
    }


def _markdown_summary(
    rows: list[dict[str, Any]],
    correlations: dict[str, Any],
    coverage: dict[str, Any],
) -> str:
    selected_counts = Counter(str(row["selected_move_type"]) for row in rows)
    worst = sorted(
        rows,
        key=lambda row: (
            float(row["post_move_aspect_stats"]["extreme_aspect_count_gt_1_5"]),
            float(row["post_move_aspect_stats"]["extreme_aspect_area_fraction_gt_1_5"]),
            float(row["p90_abs_log_aspect"]),
        ),
        reverse=True,
    )[:8]
    lines = [
        "# Step7A Aspect Pathology Correlation",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(coverage, indent=2),
        "```",
        "",
        "## Selected move counts",
        "",
        "```json",
        json.dumps(dict(selected_counts), indent=2),
        "```",
        "",
        "## Correlations",
        "",
        "```json",
        json.dumps(correlations, indent=2),
        "```",
        "",
        "## Worst aspect cases by threshold gt_1_5",
        "",
        *[
            (
                f"- case {row['case_id']}: move={row['selected_move_type']}, "
                f"n={row['n_blocks']}, p90={float(row['p90_abs_log_aspect']):.3f}, "
                f"gt1.5={row['post_move_aspect_stats']['extreme_aspect_count_gt_1_5']}, "
                f"gt2.0={row['extreme_aspect_count']}, "
                f"area1.5={float(row['post_move_aspect_stats']['extreme_aspect_area_fraction_gt_1_5']):.3f}"
            )
            for row in worst
        ],
        "",
        "## Notes",
        "",
        "- Step7A is diagnostic only; no runtime integration changed.",
        "- Candidate-family attribution is construction-family usage,",
        "  not per-block family causality.",
        "- Large/XL coverage is reported explicitly; default Step6P prefix is small/medium only.",
    ]
    return "\n".join(lines) + "\n"


def _render_worst_cases(
    *,
    rows: list[dict[str, Any]],
    case_payloads: dict[int, tuple[Any, dict[int, Placement], dict[int, Placement], Any]],
    output_dir: Path,
    limit: int,
    draw_nets: int,
    dpi: int,
) -> list[dict[str, str]]:
    visualizer = _load_visualizer()
    viz_dir = output_dir / "step7a_visualizations"
    if viz_dir.exists():
        for stale in viz_dir.glob("*.png"):
            stale.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, str]] = []
    ranked = sorted(
        rows,
        key=lambda item: (
            float(item["post_move_aspect_stats"]["extreme_aspect_count_gt_1_5"]),
            float(item["post_move_aspect_stats"]["extreme_aspect_area_fraction_gt_1_5"]),
            float(item["p90_abs_log_aspect"]),
        ),
        reverse=True,
    )
    for row in ranked[:limit]:
        case_id = int(row["case_id"])
        case, original, selected, frame = case_payloads[case_id]
        for stage, placements in (("original", original), ("selected", selected)):
            filename = f"case{case_id:03d}_{stage}_{row['selected_move_type']}.png"
            path = viz_dir / filename
            visualizer.render_png(
                case=case,
                placements=placements,
                title=f"Step7A {stage} | case {case_id}",
                metrics={
                    "move": row["selected_move_type"],
                    "p90": f"{float(row['p90_abs_log_aspect']):.3f}",
                    "extreme": int(row["extreme_aspect_count"]),
                },
                output_path=path,
                draw_nets=draw_nets,
                dpi=dpi,
                frame=frame,
            )
            cards.append(
                {
                    "case_id": str(case_id),
                    "stage": stage,
                    "image": str(path),
                    "selected_move_type": str(row["selected_move_type"]),
                }
            )
    _write_index(cards, viz_dir)
    return cards


def _write_index(cards: list[dict[str, str]], output_dir: Path) -> None:
    sections = []
    for card in cards:
        image_name = Path(card["image"]).name
        sections.append(
            "<section>"
            f"<h2>case {html.escape(card['case_id'])} / {html.escape(card['stage'])}</h2>"
            f"<p><code>{html.escape(card['selected_move_type'])}</code></p>"
            f"<img src='{html.escape(image_name)}' "
            "style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Step7A aspect visualizations</title>",
            "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
            "<h1>Step7A aspect visualizations</h1>",
            *sections,
        ]
    )
    (output_dir / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step6p-representatives",
        default="artifacts/research/step6p_selected_representatives.json",
    )
    parser.add_argument(
        "--step6j", default="artifacts/research/step6j_boundary_hull_ownership.json"
    )
    parser.add_argument("--representative", default="closest_to_ideal")
    parser.add_argument("--case-limit", type=int, default=40)
    parser.add_argument("--visualize-worst", type=int, default=8)
    parser.add_argument("--draw-nets", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step6p_rows = list(_load_json(args.step6p_representatives))
    step6j = _load_json(args.step6j) if Path(args.step6j).exists() else {}
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    cases = load_validation_cases(case_limit=args.case_limit)

    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    payloads: dict[int, tuple[Any, dict[int, Placement], dict[int, Placement], Any]] = {}
    for case_row in step6p_rows:
        case_id = int(case_row["case_id"])
        if case_id >= len(cases):
            continue
        case = cases[case_id]
        original, frame, family_usage = _reconstruct_original(
            case,
            case_id,
            step6j,
            boundary_commit_mode,
        )
        rep = _selected_representative(case_row, args.representative)
        selected = _apply_representative(case, original, frame, rep)
        row = case_aspect_pathology(
            case,
            pre_move_placements=original,
            post_move_placements=selected,
            selected_representative=args.representative,
            selected_move_type=str(rep.get("move_type", "original")),
            candidate_family_usage=family_usage,
        )
        row["case_id"] = case_id
        row["size_bucket"] = size_bucket(case.block_count)
        case_rows.append(row)
        metric_rows.append(_metric_row(case, selected, frame, rep))
        payloads[case_id] = (case, original, selected, frame)

    by_role = summarize_by_role(case_rows)
    by_family = summarize_candidate_families(case_rows)
    correlations = correlation_report(case_rows, metric_rows)
    bucket_counts = Counter(str(row["size_bucket"]) for row in case_rows)
    coverage = {
        "case_count": len(case_rows),
        "size_bucket_counts": dict(bucket_counts),
        "large_xl_coverage_status": "gap"
        if not (bucket_counts.get("large") or bucket_counts.get("xl"))
        else "present",
        "note": "Default Step6P prefix covers small/medium; run Step7D for large/XL replay.",
    }
    visualizations = _render_worst_cases(
        rows=case_rows,
        case_payloads=payloads,
        output_dir=output_dir,
        limit=args.visualize_worst,
        draw_nets=args.draw_nets,
        dpi=args.dpi,
    )
    report = {
        "representative": args.representative,
        "case_rows": case_rows,
        "metric_rows": metric_rows,
        "aspect_by_role": by_role,
        "aspect_by_candidate_family": by_family,
        "correlations": correlations,
        "coverage": coverage,
        "visualizations": visualizations,
        "decision": "diagnose_shape_policy_before_step7b",
    }
    (output_dir / "step7a_aspect_pathology.json").write_text(json.dumps(report, indent=2))
    (output_dir / "step7a_aspect_by_role.json").write_text(json.dumps(by_role, indent=2))
    (output_dir / "step7a_aspect_by_candidate_family.json").write_text(
        json.dumps(by_family, indent=2)
    )
    (output_dir / "step7a_aspect_correlation.md").write_text(
        _markdown_summary(case_rows, correlations, coverage)
    )
    print(
        json.dumps(
            {
                "output": str(output_dir / "step7a_aspect_pathology.json"),
                "case_count": len(case_rows),
                "coverage": coverage,
                "visualizations": len(visualizations),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
