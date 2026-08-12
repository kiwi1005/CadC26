#!/usr/bin/env python3
"""Measure gold B*-Tree topology with runtime-available shapes and the exact tail."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_hcfp_topology_heldout import _collect_heldout, _load_training_exclusion  # noqa: E402
from hcfp.analytic import (  # noqa: E402
    AnalyticConfig,
    select_device,
    solve_case_from_population_with_telemetry,
    to_official_placements,
)
from hcfp.benchmark import uncapped_objective  # noqa: E402
from hcfp.btree import (  # noqa: E402
    BStarTree,
    contact_aware_vertical_orders,
    decode_btree_logits,
    local_tree_variants,
)
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.constraints.mib_shapes import resolve_mib_shapes  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.geometry import bbox_area_tensor, centers_from_xywh  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    _post_tail_group_repair,
    analyze_case_with_checkpoint,
    select_official_from_analysis,
)
from hcfp.outline_inference import infer_outline_hypotheses  # noqa: E402
from hcfp.verify import compute_total_score, exact_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report")
    parser.add_argument("--skip-training-exclusion", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-limit", type=int, default=16)
    parser.add_argument("--heldout-seed", type=int, default=1)
    parser.add_argument("--heldout-max-layouts-per-file", type=int, default=1)
    parser.add_argument("--min-blocks", type=int, default=106)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--max-preplaced", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--topology-seeds", type=int, default=16)
    parser.add_argument("--constraint-seeds", type=int, default=16)
    parser.add_argument("--treemap-seeds", type=int, default=1)
    parser.add_argument("--shape-candidates", type=int, default=8)
    parser.add_argument("--contact-order-variants", type=int, default=0)
    parser.add_argument("--local-tree-variants", type=int, default=0)
    parser.add_argument("--tree-source", choices=("gold", "model"), default="gold")
    parser.add_argument("--btree-dynamics-steps", type=int, default=16)
    parser.add_argument("--projection-steps", type=int, default=24)
    parser.add_argument("--direction-beam", type=int, default=4)
    parser.add_argument("--no-candidate-funnel-proxy", action="store_true")
    args = parser.parse_args(argv)
    if args.heldout_limit <= 0 or args.shape_candidates <= 0:
        parser.error("heldout and shape candidate counts must be positive")
    if not 0 <= args.contact_order_variants <= 3:
        parser.error("contact-order-variants must be in [0,3]")
    if args.local_tree_variants < 0:
        parser.error("local-tree-variants must be non-negative")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not args.no_candidate_funnel_proxy:
        os.environ["HCFP_CANDIDATE_FUNNEL_PROXY"] = "1"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.heldout_seed)
    checkpoint = Path(args.checkpoint).resolve()
    model, metadata = load_checkpoint(
        checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    checkpoint_hash = str(metadata["state_hash"])
    training_report = Path(
        args.training_report or f"{checkpoint}.training.json"
    ).resolve()
    if args.skip_training_exclusion:
        training_payload = json.loads(training_report.read_text(encoding="utf-8"))
        exclude_ids: set[str] = set()
        exclude_provenance = {
            "skipped": True,
            "reason": "explicit experiment-mode override after stream hash drift",
        }
        contract = {"sampling": str(training_payload["sampling"])}
    else:
        exclude_ids, exclude_provenance, contract = _load_training_exclusion(
            training_report,
            root=args.floorset_lite_root,
            checkpoint=checkpoint,
            checkpoint_hash=checkpoint_hash,
            checkpoint_config=metadata["config"],
        )
    collection_limit = (
        args.heldout_limit
        if args.max_preplaced < 0
        else args.heldout_limit * 20
    )
    heldout_pool, split_provenance = _collect_heldout(
        args.floorset_lite_root,
        exclude_ids=exclude_ids,
        exclude_provenance=exclude_provenance,
        heldout_limit=collection_limit,
        heldout_seed=args.heldout_seed,
        heldout_max_layouts_per_file=args.heldout_max_layouts_per_file,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        score_aware=str(contract["sampling"]) == "score-aware",
    )
    heldout = [
        row
        for row in heldout_pool
        if args.max_preplaced < 0
        or int(row[0].case.preplaced_mask.sum()) <= args.max_preplaced
    ][: args.heldout_limit]
    if len(heldout) != args.heldout_limit:
        raise RuntimeError(
            f"preplaced filter kept {len(heldout)} cases, expected {args.heldout_limit}"
        )
    device = select_device(args.device)
    model = model.to(device=device).eval()
    current_config = LearnedConfig(
        analytic=AnalyticConfig(
            dynamics=DynamicsConfig(population=args.population, steps=0),
            projection_iterations=args.projection_steps,
            direction_beam=args.direction_beam,
        ),
        seed=6501,
        topology_seeds=args.topology_seeds,
        constraint_seeds=args.constraint_seeds,
        treemap_seeds=args.treemap_seeds,
    )
    btree_config = AnalyticConfig(
        dynamics=DynamicsConfig(
            population=args.shape_candidates,
            steps=args.btree_dynamics_steps,
        ),
        projection_iterations=args.projection_steps,
        direction_beam=args.direction_beam,
    )
    cases = [
        _audit_sample(
            sample,
            source,
            _tree_edges(args.floorset_lite_root, sample.sample_id, sample.case.n),
            model,
            args.tree_source,
            checkpoint,
            device,
            current_config,
            btree_config,
            args.population,
            args.shape_candidates,
            args.contact_order_variants,
            args.local_tree_variants,
        )
        for sample, source in heldout
    ]
    report = {
        "schema_version": 1,
        "config": vars(args),
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_hash": checkpoint_hash,
            "training_report": str(training_report),
            "training_exclusion": exclude_provenance,
            "heldout": split_provenance,
        },
        "summary": _summary(cases),
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _audit_sample(
    sample: Any,
    source: dict[str, Any],
    gold_edges: torch.Tensor,
    model: Any,
    tree_source: str,
    checkpoint: Path,
    device: torch.device,
    current_config: LearnedConfig,
    btree_config: AnalyticConfig,
    analytic_population: int,
    shape_candidates: int,
    contact_order_variants: int,
    local_tree_variant_count: int,
) -> dict[str, Any]:
    case = sample.case.to(device=device, dtype=torch.float32)
    edges = gold_edges
    if tree_source == "model":
        with torch.inference_mode():
            output = model(case, population=1)
        if output.btree_root_logits is None or output.btree_edge_logits is None:
            raise ValueError("tree-source=model requires a btree-enabled checkpoint")
        edges = decode_btree_logits(
            output.btree_root_logits,
            output.btree_edge_logits,
        ).edges()
    tree = BStarTree.from_edges(edges, case.n)
    gold_tree = BStarTree.from_edges(gold_edges, case.n)
    predicted_rows = {tuple(row) for row in edges.tolist()}
    gold_rows = {tuple(row) for row in gold_edges.tolist()}
    analysis = analyze_case_with_checkpoint(case, checkpoint, current_config)
    current_positions = select_official_from_analysis(
        source,
        case,
        analysis,
        config=current_config,
        device=device,
    )
    current = _metrics(sample, source, current_positions)
    population, shape_records = _runtime_shape_population(
        case,
        tree,
        analysis,
        analytic_population=analytic_population,
        count=shape_candidates,
        contact_order_variants=contact_order_variants,
        local_tree_variant_count=local_tree_variant_count,
    )
    actual_candidates = int(population.shape[0])
    btree_config = AnalyticConfig(
        dynamics=DynamicsConfig(
            population=actual_candidates,
            steps=btree_config.dynamics.steps,
        ),
        projection_iterations=btree_config.projection_iterations,
        direction_beam=btree_config.direction_beam,
    )
    tail = solve_case_from_population_with_telemetry(
        case,
        population.to(device=device, dtype=torch.float32),
        btree_config,
    )
    candidates = []
    stages = (
        ("initial", 1),
        ("post_relax", 1 + actual_candidates),
    )
    for stage, start in stages:
        for offset in range(actual_candidates):
            index = start + offset
            positions = to_official_placements(
                source,
                case,
                tail.projected_candidates[index].detach().cpu(),
            )
            positions = _post_tail_group_repair(source, case, positions)
            candidates.append(
                {
                    "candidate_index": offset,
                    "candidate_stage": stage,
                    **shape_records[offset],
                    **_metrics(sample, source, positions),
                    "raw_overlap_pairs": _overlap_pairs(tail.raw_candidates[index].detach().cpu()),
                    "projected_overlap_area": float(
                        tail.telemetry.projected_overlap[index].detach().cpu()
                    ),
                    "projection_ok": bool(tail.telemetry.projection_ok[index].detach().cpu()),
                    "internal_hard_feasible": bool(
                        tail.telemetry.hard_feasible[index].detach().cpu()
                    ),
                    "projection_displacement": float(
                        tail.telemetry.projection_displacement[index].detach().cpu()
                    ),
                }
            )
    oracle = min(
        (row for row in candidates if row["hard_feasible"]),
        key=lambda row: (row["uncapped_objective"], row["candidate_index"]),
        default=None,
    )
    topology_only = min(
        (
            row
            for row in candidates
            if row["hard_feasible"]
            and row["vertical_order_source"] == "base"
            and row["tree_variant_source"] == "base"
        ),
        key=lambda row: (row["uncapped_objective"], row["candidate_index"]),
        default=None,
    )
    base_tree_oracle = min(
        (
            row
            for row in candidates
            if row["hard_feasible"] and row["tree_variant_source"] == "base"
        ),
        key=lambda row: (row["uncapped_objective"], row["candidate_index"]),
        default=None,
    )
    gold_packed = tree.pack(sample.labels.rectangles[:, 2:4])
    gold_order = torch.argsort(
        centers_from_xywh(sample.labels.rectangles)[:, 1],
        stable=True,
    )
    gold_x_compacted = tree.pack_x_compacted(
        sample.labels.rectangles[:, 2:4],
        gold_order,
        torch.zeros(case.n, dtype=torch.bool),
        sample.labels.rectangles,
    )
    gold_area = float(bbox_area_tensor(sample.labels.rectangles))
    gold_tree_area = float(bbox_area_tensor(gold_packed))
    return {
        "sample_id": sample.sample_id,
        "block_count": case.n,
        "preplaced_count": int(case.preplaced_mask.sum()),
        "fixed_count": int(case.fixed_mask.sum()),
        "tree_source": tree_source,
        "tree_root_correct": tree.root == gold_tree.root,
        "tree_edge_accuracy": len(predicted_rows & gold_rows) / max(len(gold_rows), 1),
        "oracle0_gold_shape": {
            "overlap_pairs": _overlap_pairs(gold_packed),
            "bbox_area_relative_error": abs(gold_tree_area - gold_area) / max(gold_area, 1.0e-12),
            "x_compacted_overlap_pairs": _overlap_pairs(gold_x_compacted),
            "x_compacted_bbox_area_relative_error": abs(
                float(bbox_area_tensor(gold_x_compacted)) - gold_area
            )
            / max(gold_area, 1.0e-12),
        },
        "current": current,
        "btree_topology_only_oracle": topology_only,
        "btree_base_tree_oracle": base_tree_oracle,
        "btree_oracle": oracle,
        "btree_candidates": candidates,
        "unique_win": oracle is not None
        and float(oracle["uncapped_objective"]) < float(current["uncapped_objective"]) - 1.0e-9,
    }


def _runtime_shape_population(
    case: Any,
    tree: BStarTree,
    analysis: Any,
    *,
    analytic_population: int,
    count: int,
    contact_order_variants: int,
    local_tree_variant_count: int,
) -> tuple[torch.Tensor, tuple[dict[str, Any], ...]]:
    initial_count = int(analysis.result.candidate_count)
    raw = analysis.analytic.raw_candidates.detach().cpu()
    start = 1 + analytic_population
    stop = 1 + initial_count
    learned = raw[start:stop]
    fallback_order = torch.arange(case.n)
    if learned.numel():
        fallback_order = torch.argsort(
            centers_from_xywh(learned[0])[:, 1],
            stable=True,
        )
    proposals: list[tuple[torch.Tensor | None, torch.Tensor]] = [
        (None, fallback_order)
    ]
    if learned.numel():
        indices = torch.linspace(0, len(learned) - 1, max(1, min(len(learned), count))).round().long().tolist()
        proposals.extend(
            (
                learned[index, :, 2:4],
                torch.argsort(
                    centers_from_xywh(learned[index])[:, 1],
                    stable=True,
                ),
            )
            for index in dict.fromkeys(indices)
        )
    hypotheses = infer_outline_hypotheses(case.to(device="cpu", dtype=torch.float32))
    pool = []
    for proposal_index, (proposal, vertical_order) in enumerate(proposals):
        for shift in (-0.7, 0.0, 0.7):
            dims, mib_groups = _resolved_dimensions(case, proposal, shift)
            for outline in hypotheses:
                packed = tree.pack_x_compacted(
                    dims,
                    vertical_order,
                    case.preplaced_mask,
                    case.target,
                    origin=(outline.x_left, outline.y_bottom),
                )
                area = float(bbox_area_tensor(packed))
                width = float((packed[:, 0] + packed[:, 2]).amax() - packed[:, 0].amin())
                height = float((packed[:, 1] + packed[:, 3]).amax() - packed[:, 1].amin())
                target_width = outline.x_right - outline.x_left
                target_height = outline.y_top - outline.y_bottom
                outline_error = (
                    abs(math.log(max(width / height, 1.0e-12) / max(target_width / target_height, 1.0e-12)))
                    + abs(math.log(max(area, 1.0e-12) / max(target_width * target_height, 1.0e-12)))
                )
                pool.append(
                    (
                        (outline_error, area, proposal_index, shift, outline.hypothesis_id),
                        packed,
                        dims,
                        vertical_order,
                        outline,
                        {
                            "shape_source": "square" if proposal is None else f"learned_{proposal_index - 1}",
                            "global_log_aspect_shift": shift,
                            "outline_hypothesis": outline.hypothesis_id,
                            "outline_error": outline_error,
                            "raw_bbox_area": area,
                            "mib_constructed_groups": mib_groups,
                            "vertical_order_source": "base",
                            "tree_variant_source": "base",
                        },
                    )
                )
    selected = sorted(pool, key=lambda item: item[0])[:count]
    if len(selected) < count:
        raise RuntimeError("not enough B*-Tree shape candidates")
    expanded = []
    for _, packed, dims, vertical_order, outline, record in selected:
        order_variants = contact_aware_vertical_orders(
            vertical_order,
            case.boundary_bits,
            case.group_membership,
        )[: 1 + contact_order_variants]
        tree_variants = (("base", tree),) + local_tree_variants(
            tree,
            case.boundary_bits,
            case.group_membership,
            limit=local_tree_variant_count,
        )
        for tree_name, candidate_tree in tree_variants:
            for order_name, order in order_variants:
                candidate = (
                    packed
                    if tree_name == "base" and order_name == "base"
                    else candidate_tree.pack_x_compacted(
                        dims,
                        order,
                        case.preplaced_mask,
                        case.target,
                        origin=(outline.x_left, outline.y_bottom),
                    )
                )
                expanded.append(
                    (
                        candidate,
                        {
                            **record,
                            "vertical_order_source": order_name,
                            "tree_variant_source": tree_name,
                            "raw_bbox_area": float(bbox_area_tensor(candidate)),
                        },
                    )
                )
    return (
        torch.stack([item[0] for item in expanded]).float(),
        tuple(item[1] for item in expanded),
    )


def _resolved_dimensions(
    case: Any,
    proposed: torch.Tensor | None,
    shift: float,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    area = case.area.detach().cpu().double()
    if proposed is None:
        ratio = torch.ones_like(area)
    else:
        proposed64 = torch.as_tensor(proposed, dtype=torch.float64, device="cpu")
        ratio = proposed64[:, 0] / proposed64[:, 1]
    hard = (case.fixed_mask | case.preplaced_mask).detach().cpu()
    ratio[~hard] *= math.exp(shift)
    width = torch.sqrt(area * ratio.clamp_min(1.0e-12))
    dims = torch.stack((width, area / width), dim=1)
    hard_wh = dims.clone()
    hard_wh[hard] = case.target.detach().cpu().double()[hard, 2:4]
    dims[hard] = hard_wh[hard]
    resolution = resolve_mib_shapes(
        area,
        case.mib_membership,
        proposed_wh=dims,
        hard_mask=hard,
        hard_wh=hard_wh,
    )
    groups = tuple(group.group for group in resolution.groups if group.compatible)
    return resolution.shapes.double(), groups


def _metrics(sample: Any, source: dict[str, Any], positions: Any) -> dict[str, Any]:
    metrics = exact_metrics(
        source,
        positions,
        baseline_hpwl=float(sample.labels.baseline_hpwl),
        baseline_area=float(sample.labels.baseline_area),
    )
    objective = uncapped_objective(metrics.hpwl_gap, metrics.area_gap, metrics.soft.total)
    return {
        "hard_feasible": bool(metrics.verification.feasible),
        "overlap_pairs": len(metrics.verification.overlap_pairs),
        "area_bad_count": len(metrics.verification.area_bad),
        "fixed_bad_count": len(metrics.verification.fixed_bad),
        "preplaced_bad_count": len(metrics.verification.preplaced_bad),
        "official_capped_cost": float(metrics.cost),
        "uncapped_objective": objective,
        "cap_margin": math.log(10.0) - math.log(objective) if metrics.verification.feasible else None,
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "violations_relative": float(metrics.soft.total),
        "boundary_violations": int(metrics.soft.raw_boundary),
        "grouping_violations": int(metrics.soft.raw_grouping),
        "mib_violations": int(metrics.soft.raw_mib),
    }


def _tree_edges(root: str | Path, sample_id: str, block_count: int) -> torch.Tensor:
    relative, raw_index = sample_id.rsplit(":", 1)
    base = Path(root).resolve()
    layout_root = base if base.name == "floorset_lite" else base / "floorset_lite"
    payload = torch.load(layout_root / relative, map_location="cpu", weights_only=True)
    return payload[4][int(raw_index)][: block_count - 1]


def _overlap_pairs(boxes: torch.Tensor) -> int:
    left = boxes[:, 0]
    bottom = boxes[:, 1]
    right = left + boxes[:, 2]
    top = bottom + boxes[:, 3]
    overlap = (
        (left[:, None] < right[None, :] - 1.0e-9)
        & (right[:, None] > left[None, :] + 1.0e-9)
        & (bottom[:, None] < top[None, :] - 1.0e-9)
        & (top[:, None] > bottom[None, :] + 1.0e-9)
    )
    return int(torch.triu(overlap, diagonal=1).sum())


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    current_cost = [float(row["current"]["official_capped_cost"]) for row in cases]
    oracle_cost = [
        float(row["btree_oracle"]["official_capped_cost"])
        if row["btree_oracle"] is not None
        else 10.0
        for row in cases
    ]
    blocks = [int(row["block_count"]) for row in cases]
    wins = sum(bool(row["unique_win"]) for row in cases)
    feasible = sum(row["btree_oracle"] is not None for row in cases)
    contact_wins = sum(
        row["btree_base_tree_oracle"] is not None
        and row["btree_topology_only_oracle"] is not None
        and float(row["btree_base_tree_oracle"]["uncapped_objective"])
        < float(row["btree_topology_only_oracle"]["uncapped_objective"]) - 1.0e-9
        for row in cases
    )
    local_tree_wins = sum(
        row["btree_oracle"] is not None
        and row["btree_base_tree_oracle"] is not None
        and float(row["btree_oracle"]["uncapped_objective"])
        < float(row["btree_base_tree_oracle"]["uncapped_objective"]) - 1.0e-9
        for row in cases
    )
    return {
        "case_count": len(cases),
        "btree_feasible_oracle_count": feasible,
        "btree_unique_win_count": wins,
        "contact_order_win_count": contact_wins,
        "local_tree_win_count": local_tree_wins,
        "current_below_cap": sum(value < 9.999999 for value in current_cost),
        "btree_below_cap": sum(value < 9.999999 for value in oracle_cost),
        "current_weighted_cost": compute_total_score(current_cost, blocks),
        "btree_weighted_cost": compute_total_score(oracle_cost, blocks),
        "oracle0_overlap_free_count": sum(
            row["oracle0_gold_shape"]["overlap_pairs"] == 0 for row in cases
        ),
        "tree_root_accuracy": sum(bool(row["tree_root_correct"]) for row in cases)
        / len(cases),
        "tree_edge_accuracy": sum(float(row["tree_edge_accuracy"]) for row in cases)
        / len(cases),
        "decision": "KEEP" if wins else "REJECT",
    }


if __name__ == "__main__":
    raise SystemExit(main())
