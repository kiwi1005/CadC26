#!/usr/bin/env python3
"""Step7T strict-winner exact replay + visual sanity audit.

This script reconstructs the selected strict winners from
`step7t_active_soft_summary.json`, replays them through the same official-like
evaluator, and writes before/after PNGs with moved blocks highlighted.

It is deliberately sidecar-only: no runtime/finalizer/contest entrypoint code is
modified or imported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from puzzleplace.eval.official import evaluate_positions
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.step7q_fresh_metric_replay import (
    actual_delta,
    all_vector_nonregressing,
    strict_meaningful_winner,
)

Box = tuple[float, float, float, float]
DELTA_KEYS = (
    "official_like_cost_delta",
    "hpwl_delta",
    "bbox_area_delta",
    "soft_constraint_delta",
)


def apply_avnr(positions: list[Box], seed_block: int, target: list[float] | None) -> list[Box]:
    out = list(positions)
    if target is not None:
        out[seed_block] = tuple(float(v) for v in target)  # type: ignore[assignment]
    return out


def apply_moves(positions: list[Box], moves: list[dict[str, Any]]) -> list[Box]:
    out = list(positions)
    for move in moves:
        block_id = int(move["block_id"])
        x, y, w, h = out[block_id]
        out[block_id] = (
            x + float(move.get("dx", 0.0)),
            y + float(move.get("dy", 0.0)),
            w,
            h,
        )
    return out


def bbox_of(positions: list[Box]) -> tuple[float, float, float, float]:
    return (
        min(x for x, _, _, _ in positions),
        min(y for _, y, _, _ in positions),
        max(x + w for x, _, w, _ in positions),
        max(y + h for _, y, _, h in positions),
    )


def render_pair(
    before: list[Box],
    after: list[Box],
    *,
    case_id: int,
    candidate_id: str,
    moved_blocks: set[int],
    repaired_block: int | None,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    bx0, by0, bx1, by1 = bbox_of(before + after)
    pad = max((bx1 - bx0), (by1 - by0), 1.0) * 0.03
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, boxes, title in zip(axes, (before, after), ("before", "after"), strict=True):
        ax.set_title(f"case {case_id} {title}\\n{candidate_id}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(bx0 - pad, bx1 + pad)
        ax.set_ylim(by0 - pad, by1 + pad)
        ax.grid(True, linewidth=0.3, alpha=0.35)
        for idx, (x, y, w, h) in enumerate(boxes):
            if idx == repaired_block:
                edge = "#d62728"
                face = "#ff9896"
                lw = 2.4
                alpha = 0.45
            elif idx in moved_blocks:
                edge = "#ff7f0e"
                face = "#ffbb78"
                lw = 2.0
                alpha = 0.40
            else:
                edge = "#4c72b0"
                face = "#c6dbef"
                lw = 0.55
                alpha = 0.12
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=lw,
                    alpha=alpha,
                )
            )
            if idx in moved_blocks or idx == repaired_block:
                ax.text(x + w / 2.0, y + h / 2.0, str(idx), ha="center", va="center", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def replay_selected_winner(
    base_dir: Path,
    case_row: dict[str, Any],
    viz_dir: Path,
) -> dict[str, Any]:
    case_id = int(case_row["case_id"])
    seed_block = int(case_row["seed_block"])
    selected = case_row["selected_candidate"]
    cases = load_validation_cases(base_dir, [case_id])
    case = cases[case_id]
    original = [tuple(map(float, box)) for box in positions_from_case_targets(case)]
    before_positions = apply_avnr(original, seed_block, case_row.get("applied_avnr_target"))
    after_positions = apply_moves(before_positions, selected["moves"])

    before = evaluate_positions(case, before_positions, runtime=1.0)
    after = evaluate_positions(case, after_positions, runtime=1.0)
    hard = summarize_hard_legality(case, after_positions)
    delta = {key: float(value) for key, value in actual_delta(before, after).items()}
    hard_feasible = bool(hard.is_feasible) and bool(after["quality"].get("feasible"))
    avnr = all_vector_nonregressing(delta, hard_feasible)
    strict = strict_meaningful_winner(delta, hard_feasible)
    max_abs_delta_err = max(abs(delta[key] - float(selected[key])) for key in DELTA_KEYS)

    moved_blocks = {int(move["block_id"]) for move in selected["moves"]}
    repaired_block = int(selected.get("repaired_component", {}).get("block_id", -1))
    if repaired_block < 0:
        repaired_block = None
    png_path = viz_dir / f"case{case_id:03d}_{selected['candidate_id']}.png"
    render_pair(
        before_positions,
        after_positions,
        case_id=case_id,
        candidate_id=str(selected["candidate_id"]),
        moved_blocks=moved_blocks,
        repaired_block=repaired_block,
        output_path=png_path,
    )

    return {
        "case_id": case_id,
        "candidate_id": selected["candidate_id"],
        "moves": selected["moves"],
        "repaired_component": selected.get("repaired_component"),
        "hard_feasible": hard_feasible,
        "hard_summary": after["legality"],
        "all_vector_nonregressing": avnr,
        "strict_meaningful_winner": strict,
        "delta_exact": delta,
        "stored_delta": {key: float(selected[key]) for key in DELTA_KEYS},
        "stored_vs_exact_max_abs_delta_error": max_abs_delta_err,
        "before_quality": before["quality"],
        "after_quality": after["quality"],
        "after_soft_counts": {
            key: int(after["official"][key])
            for key in (
                "boundary_violations",
                "grouping_violations",
                "mib_violations",
                "total_soft_violations",
            )
        },
        "visualization_png": str(png_path),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Step7T Strict Winner Visual Sanity",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        f"- strict winners replayed: {summary['strict_winner_replayed_count']}",
        f"- exact strict winners: {summary['exact_strict_winner_count']}",
        f"- max stored/exact delta error: {summary['max_stored_vs_exact_delta_error']}",
        "",
        "| case | candidate | strict | ΔC | ΔH | ΔA | ΔS | PNG |",
        "|---:|---|:---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["records"]:
        delta = row["delta_exact"]
        lines.append(
            (
                "| {case_id} | `{candidate_id}` | {strict} | {dc:.8g} | "
                "{dh:.8g} | {da:.8g} | {ds:.8g} | `{png}` |"
            ).format(
                case_id=row["case_id"],
                candidate_id=row["candidate_id"],
                strict="yes" if row["strict_meaningful_winner"] else "no",
                dc=delta["official_like_cost_delta"],
                dh=delta["hpwl_delta"],
                da=delta["bbox_area_delta"],
                ds=delta["soft_constraint_delta"],
                png=row["visualization_png"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/research/step7t_active_soft_summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7t_visual_sanity.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("artifacts/research/step7t_visual_sanity.md"),
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=Path("artifacts/research/step7t_visual_sanity_png"),
    )
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    winners = [
        row
        for row in data.get("per_case", [])
        if row.get("selected_candidate", {}).get("strict_meaningful_winner")
    ]
    records = [replay_selected_winner(args.base_dir, row, args.viz_dir) for row in winners]
    max_err = max(
        (float(row["stored_vs_exact_max_abs_delta_error"]) for row in records),
        default=0.0,
    )
    exact_strict = sum(int(row["strict_meaningful_winner"]) for row in records)
    decision = (
        "strict_winner_visual_sanity_pass"
        if records and exact_strict == len(records) and max_err <= 1e-12
        else "strict_winner_visual_sanity_needs_followup"
    )
    summary = {
        "schema": "step7t_visual_sanity_v1",
        "decision": decision,
        "source_summary": str(args.summary),
        "strict_winner_replayed_count": len(records),
        "exact_strict_winner_count": exact_strict,
        "max_stored_vs_exact_delta_error": max_err,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_out.write_text(markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "strict_winner_replayed_count": len(records),
                "exact_strict_winner_count": exact_strict,
                "max_stored_vs_exact_delta_error": max_err,
            }
        )
    )


if __name__ == "__main__":
    main()
