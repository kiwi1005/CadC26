#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from puzzleplace.repair import finalize_layout
from puzzleplace.research.move_library import MoveCandidate, apply_move, layout_metrics
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
) -> tuple[dict[int, Placement], Any]:
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
    pre, _family_usage, construction_frame, predicted_hull = _construct(
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
    return post, construction_frame


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
                MoveCandidate(str(part), tuple(int(value) for value in targets), "step6p-viz"),
            )
        return placements
    targets = tuple(int(value) for value in row.get("target_blocks", []))
    placements, _reasons, _count = apply_move(
        case,
        placements,
        frame,
        MoveCandidate(move_type, targets, "step6p-viz"),
    )
    return placements


def _metrics(
    case: Any, placements: dict[int, Placement], frame: Any, row: dict[str, Any]
) -> dict[str, Any]:
    metrics = layout_metrics(case, placements, frame)
    return {
        "move": row.get("move_type", "original"),
        "b_d": f"{float(row.get('boundary_delta', 0.0)):+.3f}",
        "hpwl_n": f"{float(row.get('objectives', {}).get('hpwl_delta_norm', 0.0)):+.3f}",
        "bbox_n": f"{float(row.get('objectives', {}).get('bbox_delta_norm', 0.0)):+.3f}",
        "bbox": f"{float(metrics['bbox_area']):.1f}",
        "hpwl": f"{float(metrics['hpwl_proxy']):.2f}",
    }


def _write_index(cards: list[dict[str, str]], output_dir: Path) -> None:
    sections = []
    for card in cards:
        sections.append(
            "<section>"
            f"<h2>{html.escape(card['title'])}</h2>"
            f"<p><code>{html.escape(card['metrics'])}</code></p>"
            f"<p><a href='{html.escape(card['image'])}'>{html.escape(card['image'])}</a></p>"
            f"<img src='{html.escape(card['image'])}' "
            "style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Step6P Pareto visualizations</title>",
            "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
            "<h1>Step6P Pareto visualizations</h1>",
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
    parser.add_argument("--case-limit", type=int, default=40)
    parser.add_argument("--draw-nets", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--output-dir", default="artifacts/research/step6p_visualizations_png")
    parser.add_argument(
        "--representatives",
        nargs="+",
        default=["original", "closest_to_ideal", "min_disruption", "best_boundary", "best_hpwl"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = _load_visualizer()
    step6p_rows = list(_load_json(args.step6p_representatives))
    step6j = _load_json(args.step6j) if Path(args.step6j).exists() else {}
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    cases = load_validation_cases(case_limit=args.case_limit)
    cards: list[dict[str, str]] = []
    positions_dump: list[dict[str, Any]] = []

    for case_row in step6p_rows:
        case_id = int(case_row["case_id"])
        if case_id >= len(cases):
            continue
        case = cases[case_id]
        original, frame = _reconstruct_original(case, case_id, step6j, boundary_commit_mode)
        reps = dict(case_row.get("representatives", {}))
        reps["original"] = {
            "case_id": case_id,
            "move_type": "original",
            "target_blocks": [],
            "boundary_delta": 0.0,
            "hpwl_delta": 0.0,
            "bbox_delta": 0.0,
            "objectives": {"hpwl_delta_norm": 0.0, "bbox_delta_norm": 0.0},
        }
        for name in args.representatives:
            if name not in reps:
                continue
            rep = reps[name]
            placements = _apply_representative(case, original, frame, rep)
            metrics = _metrics(case, placements, frame, rep)
            filename = f"case{case_id:03d}_{name}_{rep['move_type']}.png"
            output_path = output_dir / filename
            visualizer.render_png(
                case=case,
                placements=placements,
                title=f"Step6P {name} | case {case_id}",
                metrics=metrics,
                output_path=output_path,
                draw_nets=args.draw_nets,
                dpi=args.dpi,
                frame=frame,
            )
            cards.append(
                {
                    "title": f"case {case_id} / {name} / {rep['move_type']}",
                    "image": filename,
                    "metrics": json.dumps(metrics, sort_keys=True),
                }
            )
            positions_dump.append(
                {
                    "case_id": case_id,
                    "representative": name,
                    "move_type": rep["move_type"],
                    "image": str(output_path),
                    "metrics": metrics,
                    "positions": {str(idx): list(box) for idx, box in sorted(placements.items())},
                }
            )

    _write_index(cards, output_dir)
    (output_dir / "positions.json").write_text(json.dumps(positions_dump, indent=2))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index": str(output_dir / "index.html"),
                "positions": str(output_dir / "positions.json"),
                "png_count": len(cards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
