#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from puzzleplace.actions import ActionExecutor, ExecutionState
from puzzleplace.alternatives.shape_policy import (
    SHAPE_POLICIES,
    ShapePolicyName,
    cap_for_block,
    mib_group_policy_summary,
    pareto_front,
    posthoc_shape_probe,
    select_shape_policy_representatives,
    shape_policy_eval_row,
)
from puzzleplace.repair import finalize_layout
from puzzleplace.research.puzzle_candidate_payload import (
    BoundaryCommitMode,
    build_puzzle_candidate_descriptors,
    heuristic_scores,
)
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    estimate_predicted_compact_hull,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from scripts.run_step6g_multistart_sidecar import _boundary_order_tier, _frame_fallback_box
from scripts.step7a_run_aspect_pathology import _apply_representative, _reconstruct_original

FOCUS_CASES = [29, 32, 33, 36, 39]


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


def _selected_representative(case_row: dict[str, Any], name: str) -> dict[str, Any]:
    reps = dict(case_row.get("representatives", {}))
    return dict(reps.get(name) or {"move_type": "original", "target_blocks": []})


def _construct_with_shape_policy(
    case: Any,
    seed: int,
    start: int,
    frame: PuzzleFrame | None,
    *,
    policy: ShapePolicyName,
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[dict[int, Placement], dict[str, int], PuzzleFrame | None]:
    state = ExecutionState()
    executor = ActionExecutor(case)
    family_usage: dict[str, int] = {}
    active_frame = frame
    predicted_hull = (
        estimate_predicted_compact_hull(case, active_frame) if active_frame is not None else None
    )
    order = list(range(case.block_count))
    variant = (seed + start) % 4
    if variant == 1:
        order = sorted(order, key=lambda idx: float(case.area_targets[idx].item()), reverse=True)
    elif variant == 2:
        order = order[start % len(order) :] + order[: start % len(order)]
    elif variant == 3:
        order = sorted(
            order,
            key=lambda idx: (
                -int(case.constraints[idx, 4].item() != 0),
                -float(case.area_targets[idx].item()),
            ),
        )
    if active_frame is not None:
        position = {idx: pos for pos, idx in enumerate(order)}
        order = sorted(order, key=lambda idx: (_boundary_order_tier(case, idx), position[idx]))
    for block_index in order:
        if block_index in state.placements:
            continue
        cap = cap_for_block(policy, case, block_index, state.placements)
        descriptors = build_puzzle_candidate_descriptors(
            case,
            state,
            remaining_blocks=[block_index],
            max_shape_bins=5,
            max_descriptors_per_block=24,
            virtual_frame=active_frame,
            predicted_hull=predicted_hull,
            frame_relaxation_steps=0,
            boundary_commit_mode=boundary_commit_mode,
            shape_log_aspect_limit=cap,
        )
        if not descriptors and active_frame is not None:
            for _relax in range(6):
                active_frame = active_frame.expanded(1.05)
                predicted_hull = estimate_predicted_compact_hull(case, active_frame)
                descriptors = build_puzzle_candidate_descriptors(
                    case,
                    state,
                    remaining_blocks=[block_index],
                    max_shape_bins=5,
                    max_descriptors_per_block=24,
                    virtual_frame=active_frame,
                    predicted_hull=predicted_hull,
                    frame_relaxation_steps=0,
                    boundary_commit_mode=boundary_commit_mode,
                    shape_log_aspect_limit=cap,
                )
                if descriptors:
                    break
        if not descriptors:
            side = float(case.area_targets[block_index].sqrt().item())
            if active_frame is None:
                x = max((px + pw for px, _py, pw, _ph in state.placements.values()), default=0.0)
                box = (x, 0.0, side, side)
            else:
                box, active_frame = _frame_fallback_box(
                    case, state.placements, block_index, active_frame
                )
            state.placements = {**state.placements, block_index: box}
            family_usage["fallback_append"] = family_usage.get("fallback_append", 0) + 1
            continue
        scores = heuristic_scores(descriptors)
        ranked = scores.argsort(descending=True).tolist()
        rank_offset = 0 if len(ranked) == 1 else (start + seed) % min(3, len(ranked))
        chosen = descriptors[int(ranked[rank_offset])]
        family_usage[chosen.candidate_family] = family_usage.get(chosen.candidate_family, 0) + 1
        executor.apply(state, chosen.action_token)
    return state.placements, family_usage, active_frame


def _construction_replay(
    case: Any,
    frame: PuzzleFrame,
    policy: ShapePolicyName,
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[dict[int, Placement], dict[str, int], PuzzleFrame]:
    pre, family_usage, active_frame = _construct_with_shape_policy(
        case,
        seed=0,
        start=0,
        frame=frame,
        policy=policy,
        boundary_commit_mode=boundary_commit_mode,
    )
    repair = finalize_layout(case, pre)
    post = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repair.positions)
    }
    return post, family_usage, active_frame or frame


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "track",
        "policy",
        "boundary_delta",
        "boundary_violation_delta",
        "hpwl_delta_norm",
        "bbox_delta_norm",
        "soft_delta",
        "aspect_pathology_score",
        "aspect_pathology_delta",
        "hole_fragmentation",
        "occupancy_ratio",
        "disruption",
        "hard_feasible",
        "frame_protrusion",
        "role_cap_count",
    )
    return {key: row[key] for key in keys if key in row}


def _candidate_family_impact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("track") != "construction_shape_policy_replay":
            continue
        by_policy[str(row["policy"])].update(Counter(row.get("candidate_family_usage", {})))
    original = by_policy.get("original_shape_policy", Counter())
    out: dict[str, Any] = {}
    for policy, usage in sorted(by_policy.items()):
        out[policy] = {
            "usage": dict(usage),
            "free_rect_usage_change": _usage_for(usage, "free_rect")
            - _usage_for(original, "free_rect"),
            "pin_pull_usage_change": _usage_for(usage, "pin_pull")
            - _usage_for(original, "pin_pull"),
            "boundary_usage_change": _usage_for(usage, "anchor:boundary")
            - _usage_for(original, "anchor:boundary"),
            "placed_block_usage_change": _usage_for(usage, "anchor:placed_block")
            - _usage_for(original, "anchor:placed_block"),
        }
    return out


def _usage_for(counter: Counter[str], needle: str) -> int:
    return sum(count for key, count in counter.items() if needle in key)


def _render_representatives(
    *,
    cases_by_id: dict[int, Any],
    layouts: dict[tuple[int, str, str], tuple[dict[int, Placement], PuzzleFrame]],
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, str]]:
    visualizer = _load_visualizer()
    viz_dir = output_dir / "step7b_visualizations"
    if viz_dir.exists():
        for stale in viz_dir.glob("*.png"):
            stale.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, str]] = []
    for row in selected:
        key = (int(row["case_id"]), str(row["track"]), str(row["policy"]))
        if key not in layouts:
            continue
        placements, frame = layouts[key]
        filename = f"case{key[0]:03d}_{row['track']}_{row['policy']}.png"
        path = viz_dir / filename
        visualizer.render_png(
            case=cases_by_id[key[0]],
            placements=placements,
            title=f"Step7B {row['policy']} | case {key[0]}",
            metrics={
                "track": row["track"],
                "aspect": f"{float(row['aspect_pathology_score']):.3f}",
                "hpwl_n": f"{float(row['hpwl_delta_norm']):+.3f}",
                "bbox_n": f"{float(row['bbox_delta_norm']):+.3f}",
            },
            output_path=path,
            draw_nets=24,
            dpi=170,
            frame=frame,
        )
        cards.append({"case_id": str(key[0]), "image": str(path), "policy": str(row["policy"])})
    _write_index(cards, viz_dir)
    return cards


def _write_index(cards: list[dict[str, str]], output_dir: Path) -> None:
    sections = []
    for card in cards:
        image_name = Path(card["image"]).name
        sections.append(
            "<section>"
            f"<h2>case {html.escape(card['case_id'])} / {html.escape(card['policy'])}</h2>"
            f"<img src='{html.escape(image_name)}' "
            "style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Step7B shape policy visualizations</title>",
            "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
            "<h1>Step7B shape policy visualizations</h1>",
            *sections,
        ]
    )
    (output_dir / "index.html").write_text(page)


def _summary_md(report: dict[str, Any]) -> str:
    lines = [
        "# Step7B Role-aware Shape Policy Smoke Test",
        "",
        "## Scope",
        "",
        "- Sidecar-only smoke test on Step7A focus cases.",
        "- Two tracks: posthoc shape probe and construction shape policy replay.",
        "- Original shape policy is retained as a Pareto candidate.",
        "- Large/XL coverage remains a known Step7D gap.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(report["coverage"], indent=2),
        "```",
        "",
        "## Representative counts",
        "",
        "```json",
        json.dumps(report["representative_counts"], indent=2),
        "```",
        "",
        "## Candidate family impact",
        "",
        "```json",
        json.dumps(report["candidate_family_impact"], indent=2)[:6000],
        "```",
        "",
        "## Decision",
        "",
        report["decision"],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*", default=[str(v) for v in FOCUS_CASES])
    parser.add_argument(
        "--step6p-representatives",
        default="artifacts/research/step6p_selected_representatives.json",
    )
    parser.add_argument(
        "--step6j", default="artifacts/research/step6j_boundary_hull_ownership.json"
    )
    parser.add_argument("--representative", default="closest_to_ideal")
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    case_ids = [int(value) for value in args.case_ids]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step6p_by_case = {int(row["case_id"]): row for row in _load_json(args.step6p_representatives)}
    step6j = _load_json(args.step6j) if Path(args.step6j).exists() else {}
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    cases = load_validation_cases(case_limit=max(case_ids) + 1)
    rows: list[dict[str, Any]] = []
    role_reasons: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []
    construction_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    selected_reps: list[dict[str, Any]] = []
    layouts: dict[tuple[int, str, str], tuple[dict[int, Placement], PuzzleFrame]] = {}
    cases_by_id: dict[int, Any] = {}
    mib_group_rows: list[dict[str, Any]] = []

    for case_id in case_ids:
        case = cases[case_id]
        cases_by_id[case_id] = case
        original, frame, original_family_usage = _reconstruct_original(
            case,
            case_id,
            step6j,
            boundary_commit_mode,
        )
        rep = _selected_representative(step6p_by_case[case_id], args.representative)
        baseline = _apply_representative(case, original, frame, rep)
        mib_group = {"case_id": case_id, **mib_group_policy_summary(case)}
        mib_group_rows.append(mib_group)
        case_eval_rows: list[dict[str, Any]] = []
        for policy in SHAPE_POLICIES:
            if policy == "original_shape_policy":
                alternative = dict(baseline)
                reasons: list[dict[str, Any]] = []
            else:
                alternative, reasons = posthoc_shape_probe(policy, case, baseline, frame)
            row = shape_policy_eval_row(
                case=case,
                policy=policy,
                track="posthoc_shape_probe",
                baseline=baseline,
                alternative=alternative,
                frame=frame,
                role_cap_reasons=reasons,
            )
            row["mib_group_summary"] = mib_group
            case_eval_rows.append(row)
            posthoc_rows.append(row)
            rows.append(row)
            layouts[(case_id, row["track"], policy)] = (alternative, frame)
            role_reasons.extend(
                {"case_id": case_id, "track": row["track"], "policy": policy, **r} for r in reasons
            )

            replay, family_usage, replay_frame = _construction_replay(
                case,
                frame,
                policy,
                boundary_commit_mode,
            )
            replay_row = shape_policy_eval_row(
                case=case,
                policy=policy,
                track="construction_shape_policy_replay",
                baseline=baseline,
                alternative=replay,
                frame=replay_frame,
                role_cap_reasons=[],
            )
            replay_row["candidate_family_usage"] = family_usage
            replay_row["original_candidate_family_usage"] = original_family_usage
            replay_row["mib_group_summary"] = mib_group
            case_eval_rows.append(replay_row)
            construction_rows.append(replay_row)
            rows.append(replay_row)
            layouts[(case_id, replay_row["track"], policy)] = (replay, replay_frame)
        front = pareto_front(case_eval_rows)
        reps = select_shape_policy_representatives(front)
        pareto_rows.append(
            {
                "case_id": case_id,
                "front": [_compact_row(row) for row in front],
                "representatives": {key: _compact_row(value) for key, value in reps.items()},
            }
        )
        selected_reps.extend(
            {"representative": key, **_compact_row(value)} for key, value in reps.items()
        )

    family_impact = _candidate_family_impact(construction_rows)
    representative_counts = dict(
        Counter(f"{row['representative']}:{row['track']}:{row['policy']}" for row in selected_reps)
    )
    visualization_rows = [
        row for row in selected_reps if row["representative"] == "closest_to_ideal"
    ]
    visualizations = _render_representatives(
        cases_by_id=cases_by_id,
        layouts=layouts,
        selected=visualization_rows,
        output_dir=output_dir,
    )
    coverage = {
        "focus_cases": case_ids,
        "case_count": len(case_ids),
        "block_count_range": [
            min(cases[idx].block_count for idx in case_ids),
            max(cases[idx].block_count for idx in case_ids),
        ],
        "current_smoke_scope": "21..60 validation prefix focus cases",
        "large_xl_coverage_status": "gap_until_step7d",
    }
    report = {
        "coverage": coverage,
        "shape_policies": list(SHAPE_POLICIES),
        "shape_policy_alternatives": rows,
        "posthoc_shape_probe": posthoc_rows,
        "construction_shape_replay": construction_rows,
        "pareto_shape_policy": pareto_rows,
        "role_cap_reasons": role_reasons,
        "candidate_family_impact": family_impact,
        "mib_group_summary": mib_group_rows,
        "representative_counts": representative_counts,
        "visualizations": visualizations,
        "decision": "posthoc_vs_construction_smoke_ready_for_review_before_step7c_or_step7d",
    }
    (output_dir / "step7b_shape_policy_alternatives.json").write_text(json.dumps(rows, indent=2))
    (output_dir / "step7b_posthoc_shape_probe.json").write_text(json.dumps(posthoc_rows, indent=2))
    (output_dir / "step7b_construction_shape_replay.json").write_text(
        json.dumps(construction_rows, indent=2)
    )
    (output_dir / "step7b_pareto_shape_policy.json").write_text(json.dumps(pareto_rows, indent=2))
    (output_dir / "step7b_role_cap_reasons.json").write_text(json.dumps(role_reasons, indent=2))
    (output_dir / "step7b_candidate_family_impact.json").write_text(
        json.dumps(family_impact, indent=2)
    )
    (output_dir / "step7b_shape_policy_summary.md").write_text(_summary_md(report))
    (output_dir / "step7b_report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(output_dir / "step7b_report.json"),
                "coverage": coverage,
                "alternatives": len(rows),
                "role_cap_reasons": len(role_reasons),
                "visualizations": len(visualizations),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
