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

from puzzleplace.alternatives.shape_policy import ShapePolicyName
from puzzleplace.diagnostics.case_profile import (
    build_case_profile,
)
from puzzleplace.experiments.representative_suite import select_representative_suite
from puzzleplace.experiments.shape_policy_replay import (
    evaluate_shape_policy_case,
    reconstruct_original_layout,
    shape_policy_pareto_representatives,
)
from puzzleplace.research.puzzle_candidate_payload import BoundaryCommitMode
from puzzleplace.research.virtual_frame import Placement, PuzzleFrame, multistart_virtual_frames
from puzzleplace.train.dataset_bc import load_validation_cases


def _load_json_if_exists(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return [] if file_path.suffix == ".json" else {}
    return json.loads(file_path.read_text())


def _load_visualizer() -> ModuleType:
    path = Path("scripts/visualize_step6g_layouts.py")
    spec = importlib.util.spec_from_file_location("step6_visualizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cheap_profile_layout(case: Any) -> dict[int, Placement]:
    placements: dict[int, Placement] = {}
    x = 0.0
    for idx in range(case.block_count):
        side = float(case.area_targets[idx].sqrt().item())
        placements[idx] = (x, 0.0, side, side)
        x += side + 1.0
    return placements


def _step6p_rep(case_id: int, rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        if int(row.get("case_id", -1)) == case_id:
            rep = row.get("representatives", {}).get("closest_to_ideal")
            if rep:
                return f"{rep.get('move_type')}"
    return None


def _step7b_rep(case_id: int, report: dict[str, Any]) -> str | None:
    for row in report.get("pareto_shape_policy", []):
        if int(row.get("case_id", -1)) == case_id:
            rep = row.get("representatives", {}).get("closest_to_ideal")
            if rep:
                return f"{rep.get('track')}:{rep.get('policy')}"
    return None


def _compact_alt(row: dict[str, Any], representative: str | None = None) -> dict[str, Any]:
    keys = (
        "case_id",
        "track",
        "policy",
        "boundary_delta",
        "boundary_violation_delta",
        "hpwl_delta_norm",
        "bbox_delta_norm",
        "aspect_pathology_score",
        "aspect_pathology_delta",
        "extreme_aspect_count_delta",
        "extreme_aspect_area_fraction_delta",
        "hole_fragmentation",
        "hole_fragmentation_delta",
        "disruption",
        "grouping_violation_delta",
        "mib_violation_delta",
        "runtime_estimate_ms",
        "hard_feasible",
        "frame_protrusion",
    )
    out = {key: row[key] for key in keys if key in row}
    out["boundary_failure_rate"] = 1.0 - float(
        row.get("after_metrics", {}).get("boundary_satisfaction_rate", 1.0)
    )
    out["candidate_family_usage_delta"] = row.get("candidate_family_usage_delta", {})
    if representative:
        out["representative"] = representative
    return out


def _candidate_family_delta(current: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(current) | set(baseline))
    return {key: int(current.get(key, 0)) - int(baseline.get(key, 0)) for key in keys}


def _winner_type(rep: dict[str, Any]) -> str:
    policy = str(rep.get("policy"))
    track = str(rep.get("track"))
    if policy == "original_shape_policy":
        return "original wins"
    if policy in {"MIB_shape_master_regularized", "group_macro_aspect_regularized"}:
        return "MIB/group policy wins"
    if track == "posthoc_shape_probe":
        return "posthoc wins"
    if track == "construction_shape_policy_replay":
        return "construction shape policy wins"
    return "no safe improvement"


def _failure_attribution(reps: dict[str, dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    closest = reps.get("closest_to_ideal", {})
    min_aspect = reps.get("min_aspect_pathology", {})
    if min_aspect and float(min_aspect.get("aspect_pathology_delta", 0.0)) < -0.05:
        if closest.get("policy") != "original_shape_policy":
            labels.append("shape policy helps")
        else:
            labels.append("topology/candidate ordering likely dominates")
    if float(min_aspect.get("hpwl_delta_norm", 0.0)) > 0.10:
        labels.append("shape policy hurts HPWL")
    if float(min_aspect.get("boundary_delta", 0.0)) < -0.05:
        labels.append("shape policy hurts boundary")
    if float(min_aspect.get("hole_fragmentation_delta", 0.0)) > 0.10:
        labels.append("shape policy creates holes")
    if not labels:
        labels.append(
            "no safe improvement"
            if closest.get("policy") == "original_shape_policy"
            else "shape policy helps"
        )
    return sorted(set(labels))


def _bucket_summary(
    results: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    by_case = {int(row["case_id"]): row for row in profiles}
    out: dict[str, Any] = {}
    for bucket in ("small", "medium", "large", "xl"):
        rows = [row for row in results if by_case[int(row["case_id"])]["size_bucket"] == bucket]
        out[bucket] = {
            "case_count": len(rows),
            "winner_type_counts": dict(Counter(row["winner_type"] for row in rows)),
            "mean_front_size": _mean(row["front_size"] for row in rows),
            "mean_closest_hpwl_delta_norm": _mean(
                row["representatives"].get("closest_to_ideal", {}).get("hpwl_delta_norm", 0.0)
                for row in rows
            ),
            "mean_closest_bbox_delta_norm": _mean(
                row["representatives"].get("closest_to_ideal", {}).get("bbox_delta_norm", 0.0)
                for row in rows
            ),
            "mean_closest_boundary_delta": _mean(
                row["representatives"].get("closest_to_ideal", {}).get("boundary_delta", 0.0)
                for row in rows
            ),
            "hard_invalid_rate": _mean(
                1.0 - float(row["hard_feasible_alternative_fraction"]) for row in rows
            ),
        }
    return out


def _pathology_result_summary(
    results: list[dict[str, Any]], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    by_case = {int(row["case_id"]): row for row in profiles}
    labels = sorted({label for row in profiles for label in row.get("pathology_labels", [])})
    out: dict[str, Any] = {}
    for label in labels:
        rows = [row for row in results if label in by_case[int(row["case_id"])]["pathology_labels"]]
        out[label] = {
            "case_count": len(rows),
            "case_ids": [row["case_id"] for row in rows],
            "winner_type_counts": dict(Counter(row["winner_type"] for row in rows)),
            "failure_attribution_counts": dict(
                Counter(label for row in rows for label in row["failure_attribution"])
            ),
        }
    return out


def _augment_pathology_labels_from_categories(profile: dict[str, Any]) -> None:
    """Keep requested suite categories visible in pathology-level outputs.

    The representative selector may use a same-bucket fallback when strict
    sparse/fragmented/aspect labels are not present in the raw profile. Step7D
    still needs explicit by-pathology coverage evidence, so the selected
    category is recorded as a pathology tag for reporting and visualization.
    """

    labels = set(profile.get("pathology_labels", []))
    for category in profile.get("categories", []):
        category_text = str(category)
        if "aspect" in category_text:
            labels.add("aspect-heavy")
        if "boundary" in category_text:
            labels.add("boundary-heavy")
        if "MIB/group" in category_text:
            labels.add("MIB/group-heavy")
        if "sparse" in category_text:
            labels.add("sparse")
        if "fragmented" in category_text:
            labels.add("fragmented")
    if len(labels) > 1:
        labels.discard("unclassified")
    profile["pathology_labels"] = sorted(labels)


def _decision(bucket_summary: dict[str, Any], suite_coverage: dict[str, Any]) -> str:
    if not suite_coverage.get("has_large") or not suite_coverage.get("has_xl"):
        return "inconclusive_due_to_coverage_or_artifact_gap"
    large_xl = [bucket_summary.get("large", {}), bucket_summary.get("xl", {})]
    construction_wins = sum(
        row.get("winner_type_counts", {}).get("construction shape policy wins", 0)
        for row in large_xl
    )
    total = sum(int(row.get("case_count", 0)) for row in large_xl)
    hpwl_side_effect = any(
        float(row.get("mean_closest_hpwl_delta_norm", 0.0)) > 0.10 for row in large_xl
    )
    boundary_side_effect = any(
        float(row.get("mean_closest_boundary_delta", 0.0)) < -0.05 for row in large_xl
    )
    if (
        total
        and construction_wins >= max(1, total // 2)
        and not hpwl_side_effect
        and not boundary_side_effect
    ):
        return "promote_shape_policy_to_step7c"
    return "pivot_to_region_topology_before_step7c"


def _write_decision_md(
    *,
    decision: str,
    coverage: dict[str, Any],
    bucket_summary: dict[str, Any],
    pathology_summary: dict[str, Any],
) -> str:
    return (
        "\n".join(
            [
                "# Step7D Representative Large/XL Coverage Replay",
                "",
                f"Decision: `{decision}`",
                "",
                "## Coverage",
                "",
                "```json",
                json.dumps(coverage, indent=2),
                "```",
                "",
                "## By size bucket",
                "",
                "```json",
                json.dumps(bucket_summary, indent=2),
                "```",
                "",
                "## By pathology",
                "",
                "```json",
                json.dumps(pathology_summary, indent=2),
                "```",
                "",
                "## Notes",
                "",
                "- Sidecar-only replay; no contest runtime integration changed.",
                "- Shape caps are alternatives, not gates.",
                "- If large/XL winners are not construction-shape policies,",
                "  Step7C should pivot toward region/topology diagnostics before iteration.",
            ]
        )
        + "\n"
    )


def _render_visualizations(
    *,
    results: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    layouts: dict[tuple[int, str, str], tuple[dict[int, Placement], PuzzleFrame]],
    cases_by_id: dict[int, Any],
    output_dir: Path,
) -> list[dict[str, str]]:
    visualizer = _load_visualizer()
    viz_dir = output_dir / "step7d_visualizations"
    if viz_dir.exists():
        for stale in viz_dir.glob("*.png"):
            stale.unlink()
    viz_dir.mkdir(parents=True, exist_ok=True)
    by_case_profile = {int(row["case_id"]): row for row in profiles}
    rendered_labels: set[str] = set()
    cards: list[dict[str, str]] = []
    for result in results:
        profile = by_case_profile[int(result["case_id"])]
        for label in profile.get("pathology_labels", []):
            if label in rendered_labels:
                continue
            key = _visualization_layout_key(result, layouts)
            if key not in layouts:
                continue
            placements, frame = layouts[key]
            filename = f"{label.replace('/', '_')}_case{key[0]:03d}_{key[2]}.png"
            path = viz_dir / filename
            visualizer.render_png(
                case=cases_by_id[key[0]],
                placements=placements,
                title=f"Step7D {label} | case {key[0]}",
                metrics={
                    "winner": result["winner_type"],
                    "policy": key[2],
                    "bucket": profile["size_bucket"],
                },
                output_path=path,
                draw_nets=24,
                dpi=170,
                frame=frame,
            )
            cards.append({"label": label, "case_id": str(key[0]), "image": str(path)})
            rendered_labels.add(label)
    _write_index(cards, viz_dir)
    return cards


def _visualization_layout_key(
    result: dict[str, Any],
    layouts: dict[tuple[int, str, str], tuple[dict[int, Placement], PuzzleFrame]],
) -> tuple[int, str, str]:
    case_id = int(result["case_id"])
    rep = result["representatives"].get("closest_to_ideal", {})
    preferred = (case_id, str(rep.get("track")), str(rep.get("policy")))
    if preferred in layouts:
        return preferred
    original = (case_id, "posthoc_shape_probe", "original_shape_policy")
    if original in layouts:
        return original
    return next((key for key in layouts if key[0] == case_id), preferred)


def _write_index(cards: list[dict[str, str]], output_dir: Path) -> None:
    sections = []
    for card in cards:
        image_name = Path(card["image"]).name
        sections.append(
            "<section>"
            f"<h2>{html.escape(card['label'])} / case {html.escape(card['case_id'])}</h2>"
            f"<img src='{html.escape(image_name)}' "
            "style='max-width:100%;border:1px solid #ddd'/>"
            "</section>"
        )
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Step7D representative replay visualizations</title>",
            "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
            "<h1>Step7D representative replay visualizations</h1>",
            *sections,
        ]
    )
    (output_dir / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-limit", type=int, default=100)
    parser.add_argument("--max-per-category", type=int, default=1)
    parser.add_argument(
        "--step6j", default="artifacts/research/step6j_boundary_hull_ownership.json"
    )
    parser.add_argument(
        "--step6p-representatives",
        default="artifacts/research/step6p_selected_representatives.json",
    )
    parser.add_argument("--step7b-report", default="artifacts/research/step7b_report.json")
    parser.add_argument("--output-dir", default="artifacts/research")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step6j = _load_json_if_exists(args.step6j) if Path(args.step6j).exists() else {}
    step6p_rows = _load_json_if_exists(args.step6p_representatives)
    step7b_report = _load_json_if_exists(args.step7b_report)
    boundary_commit_mode = cast(
        BoundaryCommitMode,
        str(step6j.get("boundary_commit_mode", "prefer_predicted_hull")),
    )
    cases = load_validation_cases(case_limit=args.case_limit)
    profiles: list[dict[str, Any]] = []
    cases_by_id: dict[int, Any] = {}
    for case_id, case in enumerate(cases):
        frame = multistart_virtual_frames(case)[case_id % len(multistart_virtual_frames(case))]
        cases_by_id[case_id] = case
        profiles.append(
            build_case_profile(
                case,
                _cheap_profile_layout(case),
                frame,
                candidate_family_usage={},
                selected_step6p_representative=_step6p_rep(case_id, step6p_rows),
                selected_step7b_representative=_step7b_rep(case_id, step7b_report),
            )
        )
    suite = select_representative_suite(profiles, max_per_category=args.max_per_category)
    selected_ids = [int(row["case_id"]) for row in suite["selected_cases"]]

    replay_results: list[dict[str, Any]] = []
    selected_profiles: list[dict[str, Any]] = []
    selected_profile_by_case: dict[int, dict[str, Any]] = {}
    layouts: dict[tuple[int, str, str], tuple[dict[int, Placement], PuzzleFrame]] = {}
    for suite_row in suite["selected_cases"]:
        case_id = int(suite_row["case_id"])
        case = cases[case_id]
        if case.block_count >= 50:
            frame = multistart_virtual_frames(case)[case_id % len(multistart_virtual_frames(case))]
            baseline = _cheap_profile_layout(case)
            baseline_family = {"fast_surrogate_baseline": case.block_count}
        else:
            baseline, frame, baseline_family = reconstruct_original_layout(
                case,
                case_id,
                step6j,
                boundary_commit_mode,
            )
        selected_profile = build_case_profile(
            case,
            baseline,
            frame,
            candidate_family_usage=baseline_family,
            selected_step6p_representative=_step6p_rep(case_id, step6p_rows),
            selected_step7b_representative=_step7b_rep(case_id, step7b_report),
        )
        selected_profile["categories"] = suite_row.get("categories", [suite_row.get("category")])
        _augment_pathology_labels_from_categories(selected_profile)
        selected_profiles.append(selected_profile)
        selected_profile_by_case[case_id] = selected_profile
        replay_policies: tuple[ShapePolicyName, ...] = (
            "original_shape_policy",
            "role_aware_cap",
            "boundary_edge_slot_exception",
            "MIB_shape_master_regularized",
            "group_macro_aspect_regularized",
        )
        rows, case_layouts = evaluate_shape_policy_case(
            case=case,
            baseline=baseline,
            frame=frame,
            boundary_commit_mode=boundary_commit_mode,
            policies=replay_policies,
        )
        for row in rows:
            current_usage = row.get("candidate_family_usage", baseline_family)
            row["candidate_family_usage_delta"] = _candidate_family_delta(
                current_usage,
                baseline_family,
            )
        for (track, policy), payload in case_layouts.items():
            layouts[(case_id, track, policy)] = payload
        reps = shape_policy_pareto_representatives(rows)
        compact_reps = {name: _compact_alt(row, name) for name, row in reps.items()}
        closest = compact_reps.get("closest_to_ideal", {})
        replay_results.append(
            {
                "case_id": case_id,
                "categories": selected_profile_by_case[case_id].get("categories", []),
                "alternatives": [_compact_alt(row) for row in rows],
                "representatives": compact_reps,
                "front_size": len([row for row in rows if row["hard_feasible"]]),
                "hard_feasible_alternative_fraction": _mean(
                    float(row["hard_feasible"]) for row in rows
                ),
                "winner_type": _winner_type(closest),
                "failure_attribution": _failure_attribution(compact_reps),
            }
        )

    bucket_summary = _bucket_summary(replay_results, selected_profiles)
    pathology_summary = _pathology_result_summary(replay_results, selected_profiles)
    decision = _decision(bucket_summary, suite["coverage"])
    visualizations = _render_visualizations(
        results=replay_results,
        profiles=selected_profiles,
        layouts=layouts,
        cases_by_id=cases_by_id,
        output_dir=output_dir,
    )
    decision_md = _write_decision_md(
        decision=decision,
        coverage=suite["coverage"],
        bucket_summary=bucket_summary,
        pathology_summary=pathology_summary,
    )

    (output_dir / "step7d_case_suite.json").write_text(json.dumps(suite, indent=2))
    (output_dir / "step7d_case_profiles.json").write_text(json.dumps(selected_profiles, indent=2))
    (output_dir / "step7d_replay_results.json").write_text(json.dumps(replay_results, indent=2))
    (output_dir / "step7d_bucket_summary.json").write_text(json.dumps(bucket_summary, indent=2))
    (output_dir / "step7d_pathology_summary.json").write_text(
        json.dumps(pathology_summary, indent=2)
    )
    (output_dir / "step7d_decision.md").write_text(decision_md)
    print(
        json.dumps(
            {
                "decision": decision,
                "selected_case_ids": selected_ids,
                "coverage": suite["coverage"],
                "visualizations": len(visualizations),
                "output": str(output_dir / "step7d_decision.md"),
            },
            indent=2,
        )
    )


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / max(len(vals), 1)


if __name__ == "__main__":
    main()
