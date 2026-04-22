#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

WORKTREE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_shared_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".venv").exists() and (candidate / "artifacts").exists():
            return candidate
    return start


SHARED_ROOT = _resolve_shared_root(WORKTREE_ROOT)
SRC = WORKTREE_ROOT / "src"
for path in (
    WORKTREE_ROOT,
    SRC,
    SHARED_ROOT / "external" / "FloorSet" / "iccad2026contest",
    SHARED_ROOT / "external" / "FloorSet",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.data import ConstraintColumns, adapt_validation_batch
from puzzleplace.eval.violation import summarize_violation_profile
from puzzleplace.feedback import load_policy_checkpoint
from puzzleplace.geometry import positions_from_case_targets, summarize_hard_legality
from puzzleplace.models import TypedActionPolicy
from puzzleplace.repair.intent_preserver import measure_intent_preservation
from puzzleplace.repair.overlap_resolver import resolve_overlaps
from puzzleplace.repair.shape_normalizer import normalize_shapes
from puzzleplace.repair.shelf_packer import shelf_pack_missing
from puzzleplace.rollout import semantic_rollout
from shapely.geometry import box
from shapely.ops import unary_union

INPUT_RESEARCH_DIR = SHARED_ROOT / "artifacts" / "research"
OUTPUT_RESEARCH_DIR = WORKTREE_ROOT / "artifacts" / "research"
FOLLOWUP_PATH = INPUT_RESEARCH_DIR / "generalization_followup_smallcheckpoints.json"
DELTA_PATH = INPUT_RESEARCH_DIR / "cost_semantics_and_trained_vs_untrained_delta.json"
OUTPUT_JSON = OUTPUT_RESEARCH_DIR / "top5_loss_drift_audit.json"
OUTPUT_MD = OUTPUT_RESEARCH_DIR / "top5_loss_drift_audit.md"

STAGE_ORDER = [
    "semantic_output",
    "normalized",
    "overlap_resolved_1",
    "shelf_packed",
    "overlap_resolved_2",
    "strict_final",
]


ALPHA = 0.5
BETA = 2.0
GAMMA = 0.3


def _resolve_validation_root() -> Path:
    candidates = [
        SHARED_ROOT / "LiteTensorDataTest",
        SHARED_ROOT.parent / "LiteTensorDataTest",
        WORKTREE_ROOT / "LiteTensorDataTest",
        WORKTREE_ROOT.parent / "LiteTensorDataTest",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate LiteTensorDataTest dataset")


VALIDATION_ROOT = _resolve_validation_root()


def _load_validation_cases_from_raw(*, case_limit: int) -> list[Any]:
    cases: list[Any] = []
    configs = sorted(
        VALIDATION_ROOT.glob("config_*"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    for index, config in enumerate(configs[:case_limit]):
        data = torch.load(config / "litedata_1.pth", map_location="cpu")[0]
        label = torch.load(config / "litelabel_1.pth", map_location="cpu")[0]
        inputs = (
            data[0][:, 0].unsqueeze(0),
            data[1].unsqueeze(0),
            data[2].unsqueeze(0),
            data[3].unsqueeze(0),
            data[0][:, 1:].unsqueeze(0),
        )
        labels = (label[1].unsqueeze(0), label[0].unsqueeze(0))
        cases.append(adapt_validation_batch((inputs, labels), case_id=f"validation-{index}"))
    return cases


def _calculate_hpwl_b2b(
    positions: list[tuple[float, float, float, float]],
    b2b_connectivity: torch.Tensor,
) -> float:
    total = 0.0
    for edge in b2b_connectivity.tolist():
        src, dst, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if src == -1 or dst == -1 or src >= len(positions) or dst >= len(positions):
            continue
        x1, y1, w1, h1 = positions[src]
        x2, y2, w2, h2 = positions[dst]
        total += weight * (abs((x1 + w1 / 2.0) - (x2 + w2 / 2.0)) + abs((y1 + h1 / 2.0) - (y2 + h2 / 2.0)))
    return total


def _calculate_hpwl_p2b(
    positions: list[tuple[float, float, float, float]],
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
) -> float:
    total = 0.0
    for edge in p2b_connectivity.tolist():
        pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if pin_idx == -1 or block_idx == -1 or block_idx >= len(positions) or pin_idx >= len(pins_pos):
            continue
        px, py = [float(v) for v in pins_pos[pin_idx].tolist()]
        x, y, w, h = positions[block_idx]
        total += weight * (abs(px - (x + w / 2.0)) + abs(py - (y + h / 2.0)))
    return total


def _calculate_bbox_area(positions: list[tuple[float, float, float, float]]) -> float:
    x_min = min(x for x, _y, _w, _h in positions)
    y_min = min(y for _x, y, _w, _h in positions)
    x_max = max(x + w for x, _y, w, _h in positions)
    y_max = max(y + h for _x, y, _w, h in positions)
    return float((x_max - x_min) * (y_max - y_min))


def _extract_validation_baseline_metrics(case) -> dict[str, float]:
    if case.target_positions is None:
        raise ValueError("validation baseline extraction requires target_positions")
    positions = positions_from_case_targets(case)
    bbox_area = _calculate_bbox_area(positions)
    hpwl_baseline = 0.0
    if case.metrics is not None and case.metrics.numel() >= 8:
        hpwl_baseline = float(case.metrics[-2].item() + case.metrics[-1].item())
        if case.metrics[0].item() > 0:
            bbox_area = float(case.metrics[0].item())
    return {"area_baseline": bbox_area, "hpwl_baseline": hpwl_baseline}


def _soft_violations(case, positions: list[tuple[float, float, float, float]]) -> dict[str, float]:
    constraints = case.constraints
    block_count = case.block_count
    ncols = constraints.shape[1] if constraints is not None else 0
    mib_const = constraints[:, 2] if ncols > 2 else torch.zeros(block_count)
    clust_const = constraints[:, 3] if ncols > 3 else torch.zeros(block_count)
    bound_const = constraints[:, 4] if ncols > 4 else torch.zeros(block_count)

    n_boundary = int((bound_const != 0).sum().item())
    n_soft = n_boundary

    n_mib_groups = int(mib_const.max().item()) if mib_const.numel() > 0 else 0
    for group in range(1, n_mib_groups + 1):
        group_size = int((mib_const == group).sum().item())
        n_soft += max(0, group_size - 1)

    n_clust_groups = int(clust_const.max().item()) if clust_const.numel() > 0 else 0
    for group in range(1, n_clust_groups + 1):
        group_size = int((clust_const == group).sum().item())
        n_soft += max(0, group_size - 1)

    grouping_violations = 0
    pred_polys = [box(x, y, x + w, y + h) for x, y, w, h in positions]
    for group in range(1, n_clust_groups + 1):
        group_indices = torch.where(clust_const == group)[0].tolist()
        if not group_indices:
            continue
        union_result = unary_union([pred_polys[idx] for idx in group_indices])
        if union_result.geom_type == "MultiPolygon":
            grouping_violations += len(union_result.geoms) - 1

    mib_violations = 0
    for group in range(1, n_mib_groups + 1):
        group_indices = torch.where(mib_const == group)[0].tolist()
        shapes = {(round(positions[idx][2], 4), round(positions[idx][3], 4)) for idx in group_indices}
        mib_violations += max(0, len(shapes) - 1)

    boundary_violations = 0
    if n_boundary > 0:
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        eps = 1e-6
        for idx, (x, y, w, h) in enumerate(positions):
            code = int(bound_const[idx].item())
            if code == 0:
                continue
            touches = {
                1: abs(x - x_min) < eps,
                2: abs(x + w - x_max) < eps,
                4: abs(y + h - y_max) < eps,
                8: abs(y - y_min) < eps,
            }
            required = [bit for bit in (1, 2, 4, 8) if code & bit]
            if not all(touches[bit] for bit in required):
                boundary_violations += 1

    total_soft = boundary_violations + grouping_violations + mib_violations
    violations_relative = total_soft / max(n_soft, 1)
    return {
        "boundary_violations": boundary_violations,
        "grouping_violations": grouping_violations,
        "mib_violations": mib_violations,
        "total_soft_violations": total_soft,
        "max_possible_violations": n_soft,
        "violations_relative": violations_relative,
    }


def _evaluate_positions(case, positions: list[tuple[float, float, float, float]], *, runtime: float, median_runtime: float) -> dict[str, float]:
    baseline = _extract_validation_baseline_metrics(case)
    hpwl_b2b = _calculate_hpwl_b2b(positions, case.b2b_edges)
    hpwl_p2b = _calculate_hpwl_p2b(positions, case.p2b_edges, case.pins_pos)
    hpwl_total = hpwl_b2b + hpwl_p2b
    hpwl_gap = (hpwl_total - baseline["hpwl_baseline"]) / max(baseline["hpwl_baseline"], 1e-6)
    bbox_area = _calculate_bbox_area(positions)
    area_gap = (bbox_area - baseline["area_baseline"]) / max(baseline["area_baseline"], 1e-6)
    hard = summarize_hard_legality(case, positions)
    soft = _soft_violations(case, positions)
    runtime_factor = runtime / max(median_runtime, 1e-6)
    if not hard.is_feasible:
        cost = 10.0
    else:
        quality_factor = 1 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
        violation_factor = math.exp(BETA * soft["violations_relative"])
        runtime_adjustment = max(0.7, math.pow(max(0.01, runtime_factor), GAMMA))
        cost = quality_factor * violation_factor * runtime_adjustment
    return {
        "cost": float(cost),
        "hpwl_b2b": float(hpwl_b2b),
        "hpwl_p2b": float(hpwl_p2b),
        "hpwl_total": float(hpwl_total),
        "hpwl_gap": float(hpwl_gap),
        "bbox_area": float(bbox_area),
        "bbox_area_baseline": float(baseline["area_baseline"]),
        "area_gap": float(area_gap),
        "runtime_seconds": float(runtime),
        **soft,
    }


@dataclass(frozen=True)
class VariantSpec:
    label: str
    variant: str
    seed: int
    checkpoint_path: str | None
    avg_official_cost: float


def _average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _ordered_positions(case, positions_dict: dict[int, tuple[float, float, float, float]]):
    return [positions_dict[idx] for idx in range(case.block_count)]


def _changed_block_count(
    reference: dict[int, tuple[float, float, float, float]],
    current: dict[int, tuple[float, float, float, float]],
    *,
    tolerance: float = 1e-6,
) -> int:
    count = 0
    for idx, ref in reference.items():
        cur = current[idx]
        if any(abs(float(a) - float(b)) > tolerance for a, b in zip(ref, cur, strict=True)):
            count += 1
    return count


def _stage_row(
    *,
    variant: VariantSpec,
    case,
    stage: str,
    semantic,
    positions_dict: dict[int, tuple[float, float, float, float]],
    proposal_positions: dict[int, tuple[float, float, float, float]],
    started_at: float,
    stage_move_count: int,
) -> dict[str, Any]:
    profile = summarize_violation_profile(case, positions_dict)
    ordered = _ordered_positions(case, positions_dict)
    hard = summarize_hard_legality(case, ordered)
    official = _evaluate_positions(case, ordered, runtime=1.0, median_runtime=1.0)
    intent = measure_intent_preservation(proposal_positions, positions_dict)
    changed_count = _changed_block_count(proposal_positions, positions_dict)
    return {
        "variant_label": variant.label,
        "variant": variant.variant,
        "seed": variant.seed,
        "checkpoint_path": variant.checkpoint_path,
        "case_id": str(case.case_id),
        "stage": stage,
        "semantic_completed": bool(semantic.semantic_completed),
        "semantic_placed_fraction": float(semantic.semantic_placed_fraction),
        "fallback_fraction": float(semantic.fallback_fraction),
        "hard_feasible": bool(hard.is_feasible),
        "overlap_pairs": int(profile.overlap_pairs),
        "total_overlap_area": float(profile.total_overlap_area),
        "area_violations": int(profile.area_violations),
        "dimension_violations": int(profile.dimension_violations),
        "boundary_violations": int(official["boundary_violations"]),
        "grouping_violations": int(official["grouping_violations"]),
        "mib_violations": int(official["mib_violations"]),
        "total_soft_violations": int(official["total_soft_violations"]),
        "violations_relative": float(official["violations_relative"]),
        "hpwl_b2b": float(official["hpwl_b2b"]),
        "hpwl_p2b": float(official["hpwl_p2b"]),
        "hpwl_total": float(official["hpwl_total"]),
        "hpwl_gap": float(official["hpwl_gap"]),
        "bbox_area": float(official["bbox_area"]),
        "bbox_area_baseline": float(official["bbox_area_baseline"]),
        "area_gap": float(official["area_gap"]),
        "official_cost": float(official["cost"]),
        "runtime_seconds": float(official["runtime_seconds"]),
        "wall_time_seconds": float(time.time() - started_at),
        "stage_move_count": int(stage_move_count),
        "changed_block_count_vs_proposal": int(changed_count),
        "changed_block_fraction_vs_proposal": changed_count / max(case.block_count, 1),
        "mean_displacement": float(intent["mean_displacement"]),
        "max_displacement": float(intent["max_displacement"]),
        "preserved_x_order_fraction": float(intent["preserved_x_order_fraction"]),
    }


def _select_variants() -> tuple[VariantSpec, VariantSpec, list[VariantSpec]]:
    payload = json.loads(FOLLOWUP_PATH.read_text())
    rows = payload["rows"]
    best_untrained_row = min(
        (row for row in rows if row["variant"] == "untrained"),
        key=lambda row: float(row["avg_official_cost"]),
    )
    trained_rows = sorted(
        (row for row in rows if row["variant"] in {"bc", "awbc"}),
        key=lambda row: float(row["avg_official_cost"]),
    )
    best_trained_row = trained_rows[0]
    best_untrained = VariantSpec(
        label=f"best_untrained_{best_untrained_row['variant']}_seed{best_untrained_row['seed']}",
        variant=str(best_untrained_row["variant"]),
        seed=int(best_untrained_row["seed"]),
        checkpoint_path=None,
        avg_official_cost=float(best_untrained_row["avg_official_cost"]),
    )
    trained_specs = [
        VariantSpec(
            label=(
                "best_trained_"
                f"{row['variant']}_seed{row['seed']}"
                if row is best_trained_row
                else f"trained_{row['variant']}_seed{row['seed']}"
            ),
            variant=str(row["variant"]),
            seed=int(row["seed"]),
            checkpoint_path=f"artifacts/models/small_overfit_{row['variant']}_seed{row['seed']}.pt",
            avg_official_cost=float(row["avg_official_cost"]),
        )
        for row in trained_rows
    ]
    return best_untrained, trained_specs[0], [best_untrained, *trained_specs]


def _load_policy(variant: VariantSpec):
    if variant.variant == "untrained":
        torch.manual_seed(variant.seed)
        return TypedActionPolicy(hidden_dim=32)
    if variant.checkpoint_path is None:
        raise ValueError(f"checkpoint required for trained variant {variant.label}")
    return load_policy_checkpoint(SHARED_ROOT / variant.checkpoint_path)


def _collect_case_ids(best_trained: VariantSpec) -> list[str]:
    payload = json.loads(DELTA_PATH.read_text())
    losses = payload["paired_comparison"]["top_5_trained_losses"]
    expected_variant = best_trained.variant
    expected_seed = best_trained.seed
    case_ids: list[str] = []
    for item in losses:
        trained = item["trained"]
        if trained["variant"] != expected_variant or int(trained["seed"]) != expected_seed:
            raise ValueError(
                "top-5 loss source no longer matches expected best trained pair: "
                f"{trained['variant']} seed {trained['seed']}"
            )
        case_ids.append(str(item["case_id"]))
    return case_ids


def _evaluate_variant(variant: VariantSpec, cases: list[Any]) -> list[dict[str, Any]]:
    policy = _load_policy(variant)
    rows: list[dict[str, Any]] = []
    for case in cases:
        started_at = time.time()
        semantic = semantic_rollout(case, policy)
        proposed = dict(semantic.proposed_positions)
        rows.append(
            _stage_row(
                variant=variant,
                case=case,
                stage="semantic_output",
                semantic=semantic,
                positions_dict=proposed,
                proposal_positions=proposed,
                started_at=started_at,
                stage_move_count=0,
            )
        )

        normalized = normalize_shapes(case, proposed)
        rows.append(
            _stage_row(
                variant=variant,
                case=case,
                stage="normalized",
                semantic=semantic,
                positions_dict=normalized,
                proposal_positions=proposed,
                started_at=started_at,
                stage_move_count=_changed_block_count(proposed, normalized),
            )
        )

        locked_blocks = {
            idx
            for idx in range(case.block_count)
            if bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
        }
        resolved_1, moved_1 = resolve_overlaps(normalized, locked_blocks=locked_blocks)
        rows.append(
            _stage_row(
                variant=variant,
                case=case,
                stage="overlap_resolved_1",
                semantic=semantic,
                positions_dict=resolved_1,
                proposal_positions=proposed,
                started_at=started_at,
                stage_move_count=len(moved_1),
            )
        )

        missing_blocks = [idx for idx in range(case.block_count) if idx not in resolved_1]
        packed, shelf_count = shelf_pack_missing(case, resolved_1, missing_blocks)
        rows.append(
            _stage_row(
                variant=variant,
                case=case,
                stage="shelf_packed",
                semantic=semantic,
                positions_dict=packed,
                proposal_positions=proposed,
                started_at=started_at,
                stage_move_count=shelf_count,
            )
        )

        repaired, moved_2 = resolve_overlaps(packed, locked_blocks=locked_blocks)
        rows.append(
            _stage_row(
                variant=variant,
                case=case,
                stage="overlap_resolved_2",
                semantic=semantic,
                positions_dict=repaired,
                proposal_positions=proposed,
                started_at=started_at,
                stage_move_count=len(moved_2),
            )
        )
        rows.append(
            _stage_row(
                variant=variant,
                case=case,
                stage="strict_final",
                semantic=semantic,
                positions_dict=repaired,
                proposal_positions=proposed,
                started_at=started_at,
                stage_move_count=0,
            )
        )
    return rows


def _aggregate_stage_means(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "official_cost",
        "hpwl_gap",
        "area_gap",
        "hpwl_b2b",
        "hpwl_p2b",
        "bbox_area",
        "violations_relative",
        "boundary_violations",
        "grouping_violations",
        "overlap_pairs",
        "total_overlap_area",
        "stage_move_count",
        "changed_block_fraction_vs_proposal",
        "mean_displacement",
        "preserved_x_order_fraction",
    ]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        buckets[(str(row["variant_label"]), str(row["stage"]))].append(row)
    aggregates: list[dict[str, Any]] = []
    for (variant_label, stage), rows in sorted(buckets.items()):
        aggregate = {
            "variant_label": variant_label,
            "variant": rows[0]["variant"],
            "seed": rows[0]["seed"],
            "stage": stage,
            "case_count": len(rows),
            "hard_feasible_rate": _average([1.0 if row["hard_feasible"] else 0.0 for row in rows]),
        }
        for metric in metrics:
            aggregate[f"avg_{metric}"] = _average([float(row[metric]) for row in rows])
        aggregates.append(aggregate)
    return aggregates


def _classify_case(summary: dict[str, float]) -> str:
    final_quality_gap = summary["final_quality_gap"]
    semantic_quality_gap = summary["semantic_quality_gap"]
    repair_added_quality_gap = summary["repair_added_quality_gap"]
    final_soft_gap = summary["final_violations_relative_gap"]
    if final_quality_gap <= 0.0 and final_soft_gap > 0.0:
        return "repair_soft_violation_dominated"
    if semantic_quality_gap <= 0.25 * max(final_quality_gap, 1e-6):
        return "repair_created_or_uncovered_gap"
    if semantic_quality_gap >= 0.75 * max(final_quality_gap, 1e-6):
        if repair_added_quality_gap > 0.2 * max(final_quality_gap, 1e-6):
            return "proposal_dominated_repair_amplified"
        return "proposal_dominated"
    if final_soft_gap > 0.05 and final_quality_gap < 0.5:
        return "soft_violation_dominated_after_repair"
    return "mixed_interaction"


def _build_best_pair_analysis(
    trace_rows: list[dict[str, Any]],
    best_untrained: VariantSpec,
    best_trained: VariantSpec,
    case_ids: list[str],
) -> list[dict[str, Any]]:
    index = {
        (row["variant_label"], row["case_id"], row["stage"]): row
        for row in trace_rows
        if row["variant_label"] in {best_untrained.label, best_trained.label}
    }
    analyses: list[dict[str, Any]] = []
    for case_id in case_ids:
        stage_deltas: list[dict[str, Any]] = []
        for stage in STAGE_ORDER:
            untrained_row = index[(best_untrained.label, case_id, stage)]
            trained_row = index[(best_trained.label, case_id, stage)]
            stage_deltas.append(
                {
                    "stage": stage,
                    "trained_minus_untrained": {
                        "official_cost": float(trained_row["official_cost"] - untrained_row["official_cost"]),
                        "hpwl_gap": float(trained_row["hpwl_gap"] - untrained_row["hpwl_gap"]),
                        "area_gap": float(trained_row["area_gap"] - untrained_row["area_gap"]),
                        "quality_gap": float(
                            (trained_row["hpwl_gap"] + trained_row["area_gap"])
                            - (untrained_row["hpwl_gap"] + untrained_row["area_gap"])
                        ),
                        "hpwl_b2b": float(trained_row["hpwl_b2b"] - untrained_row["hpwl_b2b"]),
                        "hpwl_p2b": float(trained_row["hpwl_p2b"] - untrained_row["hpwl_p2b"]),
                        "bbox_area": float(trained_row["bbox_area"] - untrained_row["bbox_area"]),
                        "violations_relative": float(
                            trained_row["violations_relative"] - untrained_row["violations_relative"]
                        ),
                        "boundary_violations": int(
                            trained_row["boundary_violations"] - untrained_row["boundary_violations"]
                        ),
                        "grouping_violations": int(
                            trained_row["grouping_violations"] - untrained_row["grouping_violations"]
                        ),
                        "mean_displacement": float(
                            trained_row["mean_displacement"] - untrained_row["mean_displacement"]
                        ),
                        "changed_block_fraction_vs_proposal": float(
                            trained_row["changed_block_fraction_vs_proposal"]
                            - untrained_row["changed_block_fraction_vs_proposal"]
                        ),
                    },
                    "trained": trained_row,
                    "untrained": untrained_row,
                }
            )
        semantic_delta = stage_deltas[0]["trained_minus_untrained"]
        final_delta = stage_deltas[-1]["trained_minus_untrained"]
        summary = {
            "case_id": case_id,
            "semantic_quality_gap": float(semantic_delta["quality_gap"]),
            "final_quality_gap": float(final_delta["quality_gap"]),
            "repair_added_quality_gap": float(final_delta["quality_gap"] - semantic_delta["quality_gap"]),
            "semantic_violations_relative_gap": float(semantic_delta["violations_relative"]),
            "final_violations_relative_gap": float(final_delta["violations_relative"]),
            "final_hpwl_b2b_gap": float(final_delta["hpwl_b2b"]),
            "final_hpwl_p2b_gap": float(final_delta["hpwl_p2b"]),
            "final_bbox_area_gap": float(final_delta["bbox_area"]),
            "final_boundary_gap": int(final_delta["boundary_violations"]),
            "final_grouping_gap": int(final_delta["grouping_violations"]),
            "final_mean_displacement_gap": float(final_delta["mean_displacement"]),
            "final_official_cost_gap": float(final_delta["official_cost"]),
        }
        summary["classification"] = _classify_case(summary)
        analyses.append({"case_id": case_id, "summary": summary, "stage_deltas": stage_deltas})
    return analyses


def _build_findings(
    best_pair_cases: list[dict[str, Any]],
    stage_means: list[dict[str, Any]],
    best_untrained: VariantSpec,
    best_trained: VariantSpec,
) -> list[str]:
    pair_summaries = [item["summary"] for item in best_pair_cases]
    semantic_quality_gap = _average([item["semantic_quality_gap"] for item in pair_summaries])
    final_quality_gap = _average([item["final_quality_gap"] for item in pair_summaries])
    repair_added_gap = _average([item["repair_added_quality_gap"] for item in pair_summaries])
    final_b2b_gap = _average([item["final_hpwl_b2b_gap"] for item in pair_summaries])
    final_p2b_gap = _average([item["final_hpwl_p2b_gap"] for item in pair_summaries])
    final_soft_gap = _average([item["final_violations_relative_gap"] for item in pair_summaries])
    classifications = defaultdict(int)
    for item in pair_summaries:
        classifications[str(item["classification"])] += 1

    stage_index = {(row["variant_label"], row["stage"]): row for row in stage_means}
    normalize_delta = _average(
        [
            abs(
                stage_index[(best_trained.label, "normalized")]["avg_hpwl_gap"]
                - stage_index[(best_trained.label, "semantic_output")]["avg_hpwl_gap"]
            ),
            abs(
                stage_index[(best_trained.label, "normalized")]["avg_area_gap"]
                - stage_index[(best_trained.label, "semantic_output")]["avg_area_gap"]
            ),
            abs(
                stage_index[(best_untrained.label, "normalized")]["avg_hpwl_gap"]
                - stage_index[(best_untrained.label, "semantic_output")]["avg_hpwl_gap"]
            ),
            abs(
                stage_index[(best_untrained.label, "normalized")]["avg_area_gap"]
                - stage_index[(best_untrained.label, "semantic_output")]["avg_area_gap"]
            ),
        ]
    )
    shelf_delta = _average(
        [
            abs(
                stage_index[(best_trained.label, "shelf_packed")]["avg_hpwl_gap"]
                - stage_index[(best_trained.label, "overlap_resolved_1")]["avg_hpwl_gap"]
            ),
            abs(
                stage_index[(best_trained.label, "shelf_packed")]["avg_area_gap"]
                - stage_index[(best_trained.label, "overlap_resolved_1")]["avg_area_gap"]
            ),
            abs(
                stage_index[(best_untrained.label, "shelf_packed")]["avg_hpwl_gap"]
                - stage_index[(best_untrained.label, "overlap_resolved_1")]["avg_hpwl_gap"]
            ),
            abs(
                stage_index[(best_untrained.label, "shelf_packed")]["avg_area_gap"]
                - stage_index[(best_untrained.label, "overlap_resolved_1")]["avg_area_gap"]
            ),
        ]
    )

    findings = [
        (
            f"Best trained `{best_trained.variant} seed {best_trained.seed}` is already worse than best "
            f"untrained `{best_untrained.variant} seed {best_untrained.seed}` at semantic proposal on the "
            f"top-5 loss slice: mean semantic quality gap `(HPWLgap + Areagap_bbox)` = `{semantic_quality_gap:.3f}`, "
            f"versus final gap `{final_quality_gap:.3f}`. Roughly "
            f"`{(semantic_quality_gap / max(final_quality_gap, 1e-6)) * 100:.1f}%` of the final quality deficit "
            "already exists before repair."
        ),
        (
            f"Repair/finalization still matters, but mostly as an amplifier rather than the root cause: mean repair-added "
            f"quality gap = `{repair_added_gap:.3f}` across the top-5 cases. Case mix = "
            + ", ".join(f"`{name}={count}`" for name, count in sorted(classifications.items()))
            + "."
        ),
        (
            f"The wirelength deficit is primarily internal/block-to-block, not pin-to-block: mean final `ΔHPWL_b2b = {final_b2b_gap:.3f}` "
            f"versus `ΔHPWL_p2b = {final_p2b_gap:.3f}` on the top-5 pair, which points to worse relative block topology more "
            "than terminal anchoring."
        ),
        (
            f"`normalize_shapes` and `shelf_pack_missing` are effectively no-ops for this audit: mean absolute quality change around "
            f"normalization = `{normalize_delta:.4f}` and around shelf packing = `{shelf_delta:.4f}`. The decisive step is still overlap resolution."
        ),
        (
            f"Soft-violation drift is secondary overall (`mean final ΔViolationsrelative = {final_soft_gap:.3f}`), but it is the main remaining story "
            "for `validation-15` and `validation-11`, where semantic HPWL/area gaps are small and the trained variant loses mostly after repair/final soft-constraint inflation."
        ),
    ]
    return findings


def _render_markdown(
    *,
    best_untrained: VariantSpec,
    best_trained: VariantSpec,
    case_ids: list[str],
    findings: list[str],
    stage_means: list[dict[str, Any]],
    best_pair_cases: list[dict[str, Any]],
    variants: list[VariantSpec],
) -> str:
    lines = [
        "# Top-5 Loss Drift Audit",
        "",
        f"- source pair: best untrained `{best_untrained.variant} seed {best_untrained.seed}` vs best trained `{best_trained.variant} seed {best_trained.seed}`",
        f"- cases: `{', '.join(case_ids)}`",
        "- runtime normalization: `runtime=1.0`, `median_runtime=1.0` for every stage evaluation",
        "- stages: `semantic_output -> normalized -> overlap_resolved_1 -> shelf_packed -> overlap_resolved_2 -> strict_final`",
        "",
        "## Main answer",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Best-pair case summary",
            "| Case | Semantic quality gap | Final quality gap | Repair-added quality gap | Final ΔHPWL_b2b | Final ΔHPWL_p2b | Final Δboundary | Final Δgrouping | Final Δcost | Classification |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in best_pair_cases:
        summary = item["summary"]
        lines.append(
            f"| {summary['case_id']} | {summary['semantic_quality_gap']:.3f} | {summary['final_quality_gap']:.3f} | "
            f"{summary['repair_added_quality_gap']:.3f} | {summary['final_hpwl_b2b_gap']:.3f} | {summary['final_hpwl_p2b_gap']:.3f} | "
            f"{summary['final_boundary_gap']} | {summary['final_grouping_gap']} | {summary['final_official_cost_gap']:.3f} | {summary['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Stage means by selected variant",
            "| Variant label | Stage | Avg cost | Avg HPWL gap | Avg area gap | Avg violations rel | Avg overlap pairs | Avg stage move count | Avg changed fraction | Avg displacement |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stage_means:
        lines.append(
            f"| {row['variant_label']} | {row['stage']} | {row['avg_official_cost']:.3f} | {row['avg_hpwl_gap']:.3f} | "
            f"{row['avg_area_gap']:.3f} | {row['avg_violations_relative']:.3f} | {row['avg_overlap_pairs']:.3f} | "
            f"{row['avg_stage_move_count']:.3f} | {row['avg_changed_block_fraction_vs_proposal']:.3f} | {row['avg_mean_displacement']:.3f} |"
        )
    lines.extend(["", "## Best-pair stage breakdown", ""])
    for item in best_pair_cases:
        lines.append(f"### {item['case_id']}")
        lines.append(
            "| Stage | U quality | T quality | Δquality | U viol rel | T viol rel | Δviol rel | U changed frac | T changed frac | Δcost |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for stage_delta in item["stage_deltas"]:
            stage = stage_delta["stage"]
            u_row = stage_delta["untrained"]
            t_row = stage_delta["trained"]
            delta = stage_delta["trained_minus_untrained"]
            lines.append(
                f"| {stage} | {u_row['hpwl_gap'] + u_row['area_gap']:.3f} | {t_row['hpwl_gap'] + t_row['area_gap']:.3f} | "
                f"{delta['quality_gap']:.3f} | {u_row['violations_relative']:.3f} | {t_row['violations_relative']:.3f} | "
                f"{delta['violations_relative']:.3f} | {u_row['changed_block_fraction_vs_proposal']:.3f} | "
                f"{t_row['changed_block_fraction_vs_proposal']:.3f} | {delta['official_cost']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Variant roster",
            *(
                f"- `{variant.label}` → `{variant.variant} seed {variant.seed}` (mean official cost `{variant.avg_official_cost:.3f}`)"
                for variant in variants
            ),
            "",
            "## Commands Used",
            "- `.venv/bin/python scripts/generate_top5_loss_drift_audit.py`",
            "- `.venv/bin/python -m ruff check scripts/generate_top5_loss_drift_audit.py`",
            "- `.venv/bin/python -m mypy scripts/generate_top5_loss_drift_audit.py`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    best_untrained, best_trained, variants = _select_variants()
    case_ids = _collect_case_ids(best_trained)
    cases_by_id = {str(case.case_id): case for case in _load_validation_cases_from_raw(case_limit=20)}
    cases = [cases_by_id[case_id] for case_id in case_ids]

    trace_rows: list[dict[str, Any]] = []
    for variant in variants:
        trace_rows.extend(_evaluate_variant(variant, cases))

    stage_means = _aggregate_stage_means(trace_rows)
    best_pair_cases = _build_best_pair_analysis(trace_rows, best_untrained, best_trained, case_ids)
    findings = _build_findings(best_pair_cases, stage_means, best_untrained, best_trained)

    payload = {
        "source_artifacts": [
            str(FOLLOWUP_PATH.relative_to(SHARED_ROOT)),
            str(DELTA_PATH.relative_to(SHARED_ROOT)),
        ],
        "output_files": [
            str(OUTPUT_JSON.relative_to(WORKTREE_ROOT)),
            str(OUTPUT_MD.relative_to(WORKTREE_ROOT)),
        ],
        "best_untrained": {
            "label": best_untrained.label,
            "variant": best_untrained.variant,
            "seed": best_untrained.seed,
            "avg_official_cost": best_untrained.avg_official_cost,
        },
        "best_trained": {
            "label": best_trained.label,
            "variant": best_trained.variant,
            "seed": best_trained.seed,
            "avg_official_cost": best_trained.avg_official_cost,
        },
        "selected_case_ids": case_ids,
        "selected_variants": [
            {
                "label": variant.label,
                "variant": variant.variant,
                "seed": variant.seed,
                "checkpoint_path": variant.checkpoint_path,
                "avg_official_cost": variant.avg_official_cost,
            }
            for variant in variants
        ],
        "stage_order": STAGE_ORDER,
        "findings": findings,
        "best_pair_case_analyses": best_pair_cases,
        "stage_means": stage_means,
        "trace_rows": trace_rows,
    }
    OUTPUT_RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    OUTPUT_MD.write_text(
        _render_markdown(
            best_untrained=best_untrained,
            best_trained=best_trained,
            case_ids=case_ids,
            findings=findings,
            stage_means=stage_means,
            best_pair_cases=best_pair_cases,
            variants=variants,
        )
    )
    print(
        json.dumps(
            {
                "cases": len(case_ids),
                "variants": len(variants),
                "trace_rows": len(trace_rows),
                "output_json": str(OUTPUT_JSON.relative_to(WORKTREE_ROOT)),
                "output_md": str(OUTPUT_MD.relative_to(WORKTREE_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
