#!/usr/bin/env python3
"""Audit topology candidates on a disjoint FloorSet-Lite training holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import AnalyticConfig, select_device  # noqa: E402
from hcfp.benchmark import (  # noqa: E402
    candidate_oracles,
    candidate_source_layout,
    select_candidate_oracle,
    summarize_attribution_cases,
    summarize_candidate_types,
    uncapped_objective,
)
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import DataSample, file_sha256  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.floorset_lite import (  # noqa: E402
    iter_floorset_lite,
    iter_floorset_lite_with_source,
)
from hcfp.learned import LearnedConfig, analyze_case_with_checkpoint  # noqa: E402
from hcfp.verify import (  # noqa: E402
    bbox_area,
    soft_violation_normalized,
    total_hpwl,
    verify,
)


def main(argv: list[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="FloorSet-Lite training root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--training-report",
        help="defaults to <checkpoint>.training.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-limit", type=int, default=16)
    parser.add_argument("--heldout-seed", type=int, default=1)
    parser.add_argument(
        "--exclude-train-limit",
        type=int,
        help="optional assertion against the training report source limit",
    )
    parser.add_argument(
        "--exclude-train-seed",
        type=int,
        help="optional assertion against the training report seed",
    )
    parser.add_argument("--heldout-max-layouts-per-file", type=int, default=1)
    parser.add_argument("--min-blocks", type=int, default=106)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--topology-seeds", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sampling",
        choices=("uniform", "score-aware"),
        help="optional assertion against the training report sampling mode",
    )
    parser.add_argument("--dynamics-steps", type=int, default=0)
    parser.add_argument("--projection-steps", type=int, default=4)
    parser.add_argument("--direction-beam", type=int, default=1)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--tail-topk", type=int)
    args = parser.parse_args(command_args)
    _validate_args(args)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.heldout_seed)
    device = select_device(args.device)
    checkpoint = Path(args.checkpoint).resolve()
    _, checkpoint_metadata = load_checkpoint(
        checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    checkpoint_hash = str(checkpoint_metadata["state_hash"])
    training_report = Path(
        args.training_report or f"{checkpoint}.training.json"
    ).resolve()
    exclude_ids, exclude_provenance, training_contract = _load_training_exclusion(
        training_report,
        root=args.root,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        checkpoint_config=checkpoint_metadata["config"],
        asserted_seed=args.exclude_train_seed,
        asserted_limit=args.exclude_train_limit,
        asserted_sampling=args.sampling,
    )
    training_sampling = str(training_contract["sampling"])
    training_seed = int(training_contract["seed"])
    if args.heldout_seed == training_seed:
        raise ValueError("heldout and consumed training stream seeds must differ")
    config = LearnedConfig(
        analytic=AnalyticConfig(
            dynamics=DynamicsConfig(
                population=args.population,
                steps=args.dynamics_steps,
            ),
            projection_iterations=args.projection_steps,
            direction_beam=args.direction_beam,
        ),
        flow_steps=args.flow_steps,
        tail_topk=args.tail_topk,
        seed=args.flow_seed,
        topology_seeds=args.topology_seeds,
    )
    heldout, split_provenance = _collect_heldout(
        args.root,
        exclude_ids=exclude_ids,
        exclude_provenance=exclude_provenance,
        heldout_limit=args.heldout_limit,
        heldout_seed=args.heldout_seed,
        heldout_max_layouts_per_file=args.heldout_max_layouts_per_file,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        score_aware=training_sampling == "score-aware",
    )
    cases = [
        _audit_sample(
            index,
            sample,
            checkpoint,
            checkpoint_hash,
            device,
            config,
            args.population,
            args.topology_seeds,
        )
        for index, (sample, _source) in enumerate(heldout)
    ]
    sample_ids = [sample.sample_id for sample, _source in heldout]
    report = {
        "schema_version": 1,
        "command": ["scripts/audit_hcfp_topology_heldout.py", *command_args],
        "config": {
            "root": str(Path(args.root).resolve()),
            "checkpoint": str(checkpoint),
            "training_report": str(training_report),
            "output": str(Path(args.output).resolve()),
            "heldout_limit": args.heldout_limit,
            "heldout_seed": args.heldout_seed,
            "exclude_train_limit": args.exclude_train_limit,
            "exclude_train_seed": args.exclude_train_seed,
            "heldout_max_layouts_per_file": args.heldout_max_layouts_per_file,
            "min_blocks": args.min_blocks,
            "max_blocks": args.max_blocks,
            "population": args.population,
            "topology_seeds": args.topology_seeds,
            "device": str(device),
            "sampling": training_sampling,
            "dynamics_steps": args.dynamics_steps,
            "projection_steps": args.projection_steps,
            "direction_beam": args.direction_beam,
            "flow_steps": args.flow_steps,
            "flow_seed": args.flow_seed,
            "tail_topk": args.tail_topk,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "file_sha256": file_sha256(checkpoint),
            "state_hash": checkpoint_hash,
            "normalization": checkpoint_metadata["normalization"],
        },
        "sampling": {
            "source": "FloorSet-Lite training",
            "mode": training_sampling,
            "training_report": {
                "path": str(training_report),
                "sha256": file_sha256(training_report),
            },
            "heldout_max_layouts_per_file": args.heldout_max_layouts_per_file,
            **split_provenance,
            "heldout": {
                **split_provenance["heldout"],
                "sample_ids": sample_ids,
                "sample_id_sha256": _sample_id_hash(sample_ids),
            },
        },
        "cases": cases,
        "summary": _summary(cases),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "heldout_limit",
        "heldout_max_layouts_per_file",
        "population",
        "topology_seeds",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.exclude_train_limit is not None and args.exclude_train_limit <= 0:
        raise ValueError("--exclude-train-limit must be positive")
    if args.min_blocks <= 0 or args.max_blocks < args.min_blocks:
        raise ValueError("--min-blocks/--max-blocks must define a positive range")
    if args.dynamics_steps < 0 or args.projection_steps <= 0 or args.direction_beam <= 0:
        raise ValueError("candidate search step and beam counts are invalid")
    if args.flow_steps < 0:
        raise ValueError("--flow-steps must be non-negative")
    if args.tail_topk is not None and not 0 < args.tail_topk <= args.population:
        raise ValueError("--tail-topk must be in [1, population]")


def _load_training_exclusion(
    training_report: str | Path,
    *,
    root: str | Path,
    checkpoint: str | Path,
    checkpoint_hash: str,
    checkpoint_config: dict[str, Any],
    asserted_seed: int | None = None,
    asserted_limit: int | None = None,
    asserted_sampling: str | None = None,
) -> tuple[set[str], dict[str, Any], dict[str, Any]]:
    path = Path(training_report).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError("training report schema does not record consumed stream IDs")
    if str(payload.get("checkpoint_hash")) != checkpoint_hash:
        raise ValueError("training report checkpoint hash mismatch")
    if Path(str(payload.get("checkpoint", ""))).resolve() != Path(checkpoint).resolve():
        raise ValueError("training report checkpoint path mismatch")
    if payload.get("model_config") != checkpoint_config:
        raise ValueError("training report model config mismatch")
    if not isinstance(payload.get("command"), list):
        raise ValueError("training report command is missing")
    contract = payload.get("direct_floorset_lite_stream")
    if not isinstance(contract, dict):
        raise ValueError("training report has no direct FloorSet-Lite stream contract")
    if str(contract.get("checkpoint_hash")) != checkpoint_hash:
        raise ValueError("training stream checkpoint hash mismatch")

    resolved_root = str(Path(root).resolve())
    if str(Path(str(contract.get("root", ""))).resolve()) != resolved_root:
        raise ValueError("training stream root mismatch")
    sampling = str(contract.get("sampling"))
    if sampling not in {"uniform", "score-aware"}:
        raise ValueError("training stream sampling mode is invalid")
    seed = int(contract.get("seed"))
    source_limit_value = contract.get("source_limit")
    source_limit = None if source_limit_value is None else int(source_limit_value)
    if source_limit is not None and source_limit <= 0:
        raise ValueError("training stream source limit is invalid")
    if contract.get("max_layouts_per_file") is not None:
        raise ValueError("training stream file cap does not match train_hcfp.py")
    if asserted_seed is not None and asserted_seed != seed:
        raise ValueError("manual exclude seed does not match training report")
    if asserted_limit is not None and asserted_limit != source_limit:
        raise ValueError("manual exclude limit does not match training report")
    if asserted_sampling is not None and asserted_sampling != sampling:
        raise ValueError("manual sampling mode does not match training report")

    consumed_count = int(contract.get("consumed_count", -1))
    if consumed_count <= 0 or int(payload.get("steps", -1)) != consumed_count:
        raise ValueError("training stream consumed count mismatch")
    if int(contract.get("ordered_sample_id_count", -1)) != consumed_count:
        raise ValueError("training stream ordered sample count mismatch")
    ordered_ids = _reconstruct_consumed_sample_ids(
        resolved_root,
        source_limit=source_limit,
        seed=seed,
        score_aware=sampling == "score-aware",
        consumed_count=consumed_count,
    )
    ordered_hash = _sample_id_hash(ordered_ids)
    if ordered_hash != str(contract.get("ordered_sample_id_sha256")):
        raise ValueError("training stream ordered sample ID hash mismatch")
    unique_ids = sorted(set(ordered_ids))
    unique_hash = _sample_id_hash(unique_ids)
    if int(contract.get("unique_sample_id_count", -1)) != len(unique_ids):
        raise ValueError("training stream unique sample count mismatch")
    if unique_hash != str(contract.get("unique_sample_id_sha256")):
        raise ValueError("training stream unique sample ID hash mismatch")
    provenance = {
        "training_report": str(path),
        "training_report_sha256": file_sha256(path),
        "root": resolved_root,
        "sampling": sampling,
        "seed": seed,
        "source_limit": source_limit,
        "consumed_count": consumed_count,
        "ordered_sample_id_count": consumed_count,
        "ordered_sample_id_sha256": ordered_hash,
        "unique_sample_id_count": len(unique_ids),
        "unique_sample_id_sha256": unique_hash,
        "count": len(unique_ids),
        "sample_id_sha256": unique_hash,
        "checkpoint_hash": checkpoint_hash,
    }
    return set(unique_ids), provenance, dict(contract)


def _reconstruct_consumed_sample_ids(
    root: str | Path,
    *,
    source_limit: int | None,
    seed: int,
    score_aware: bool,
    consumed_count: int,
) -> list[str]:
    if consumed_count <= 0:
        raise ValueError("consumed_count must be positive")
    sample_ids: list[str] = []
    while len(sample_ids) < consumed_count:
        produced = 0
        for sample in iter_floorset_lite(
            root,
            limit=source_limit,
            seed=seed,
            score_aware=score_aware,
        ):
            sample_ids.append(sample.sample_id)
            produced += 1
            if len(sample_ids) >= consumed_count:
                break
        if produced == 0:
            raise RuntimeError("training stream reconstruction produced no samples")
    return sample_ids


def _collect_heldout(
    root: str | Path,
    *,
    exclude_ids: set[str],
    exclude_provenance: dict[str, Any],
    heldout_limit: int,
    heldout_seed: int,
    heldout_max_layouts_per_file: int,
    min_blocks: int,
    max_blocks: int,
    score_aware: bool,
) -> tuple[list[tuple[DataSample, dict[str, Any]]], dict[str, Any]]:
    selected: list[tuple[DataSample, dict[str, Any]]] = []
    heldout_ids: set[str] = set()
    examined = overlap_filtered = block_filtered = 0
    for sample, source in iter_floorset_lite_with_source(
        root,
        limit=None,
        seed=heldout_seed,
        score_aware=score_aware,
        max_layouts_per_file=heldout_max_layouts_per_file,
    ):
        examined += 1
        if sample.sample_id in heldout_ids:
            raise RuntimeError("heldout stream produced duplicate sample IDs")
        heldout_ids.add(sample.sample_id)
        if sample.sample_id in exclude_ids:
            overlap_filtered += 1
            continue
        if not min_blocks <= sample.case.n <= max_blocks:
            block_filtered += 1
            continue
        selected.append((sample, source))
        if len(selected) >= heldout_limit:
            break
    if len(selected) != heldout_limit:
        raise RuntimeError(
            f"heldout stream produced {len(selected)} disjoint in-range samples, "
            f"expected {heldout_limit}"
        )
    selected_ids = [sample.sample_id for sample, _source in selected]
    source_files = [sample_id.rsplit(":", 1)[0] for sample_id in selected_ids]
    overlap = exclude_ids.intersection(selected_ids)
    if overlap:
        raise RuntimeError(f"heldout split overlaps exclude training IDs: {sorted(overlap)}")
    return selected, {
        "exclude_training": dict(exclude_provenance),
        "heldout": {
            "seed": heldout_seed,
            "requested_count": heldout_limit,
            "count": len(selected),
            "examined_count": examined,
            "overlap_filtered_count": overlap_filtered,
            "block_filtered_count": block_filtered,
            "exclude_overlap_count": 0,
            "source_file_count": len(set(source_files)),
            "source_file_sha256": _sample_id_hash(sorted(set(source_files))),
            "max_layouts_per_file": heldout_max_layouts_per_file,
            "min_blocks": min_blocks,
            "max_blocks": max_blocks,
        },
    }


def _audit_sample(
    index: int,
    sample: DataSample,
    checkpoint: Path,
    checkpoint_hash: str,
    device: torch.device,
    config: LearnedConfig,
    population: int,
    topology_seeds: int,
) -> dict[str, Any]:
    case = sample.case.to(device=device, dtype=torch.float32)
    analysis = analyze_case_with_checkpoint(case, checkpoint, config)
    _validate_topology_result(sample.sample_id, analysis, checkpoint_hash, topology_seeds)
    learned_count = analysis.result.candidate_count - population
    sources = candidate_source_layout(population, learned_count)
    raw = analysis.analytic.raw_candidates
    projected = analysis.analytic.projected_candidates
    if len(sources) != raw.shape[0] or raw.shape != projected.shape:
        raise RuntimeError(
            f"sample {sample.sample_id}: source layout {len(sources)} does not match "
            f"raw/projected candidates {tuple(raw.shape)}/{tuple(projected.shape)}"
        )
    snapshot = analysis.analytic.incumbent_snapshot
    topology_indices = _topology_indices(snapshot.get("topology_seed_sources"))
    if len(topology_indices) != 2 * topology_seeds:
        raise RuntimeError(
            f"sample {sample.sample_id}: topology provenance names "
            f"{len(topology_indices)} candidates, expected {2 * topology_seeds}"
        )
    baseline_area = float(sample.labels.baseline_area)
    baseline_hpwl = float(sample.labels.baseline_hpwl)
    raw_records = _candidate_records(
        case,
        raw,
        sources,
        topology_indices,
        baseline_area,
        baseline_hpwl,
    )
    projected_records = _candidate_records(
        case,
        projected,
        sources,
        topology_indices,
        baseline_area,
        baseline_hpwl,
    )
    incumbent_index, incumbent_source = _incumbent_source(
        snapshot.get("exact_source"),
        sources,
    )
    incumbent = _candidate_record(
        case,
        analysis.result.selected,
        incumbent_index,
        incumbent_source,
        topology_indices,
        baseline_area,
        baseline_hpwl,
    )
    return {
        "test_id": index,
        "sample_id": sample.sample_id,
        "block_count": case.n,
        "baseline": {"hpwl": baseline_hpwl, "area": baseline_area},
        "candidate_layout": {
            "population": population,
            "learned_count": learned_count,
            "topology_count": topology_seeds,
            "candidate_count": len(sources),
        },
        "topology_provenance": {
            key: value
            for key, value in snapshot.items()
            if str(key).startswith("topology_")
        },
        "raw": {"candidates": raw_records, "oracles": _oracles(raw_records)},
        "post_bdp": {
            "candidates": projected_records,
            "oracles": _oracles(projected_records),
        },
        "incumbent": incumbent,
    }


def _validate_topology_result(
    sample_id: str,
    analysis: Any,
    checkpoint_hash: str,
    requested_count: int,
) -> None:
    if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash != checkpoint_hash:
        reason = analysis.result.failure_reason or "checkpoint was not used"
        raise RuntimeError(f"sample {sample_id}: {reason}")
    produced = int(analysis.result.topology_seed_count)
    if produced != requested_count:
        reason = analysis.analytic.incumbent_snapshot.get(
            "topology_seed_failure_reason",
            "topology decode produced an incomplete candidate set",
        )
        raise RuntimeError(
            f"sample {sample_id}: requested {requested_count} topology seeds, "
            f"produced {produced}: {reason}"
        )


def _candidate_records(
    case: Any,
    boxes: torch.Tensor,
    sources: tuple[str, ...],
    topology_indices: frozenset[int],
    baseline_area: float,
    baseline_hpwl: float,
) -> list[dict[str, Any]]:
    return [
        _candidate_record(
            case,
            candidate,
            index,
            source,
            topology_indices,
            baseline_area,
            baseline_hpwl,
        )
        for index, (source, candidate) in enumerate(zip(sources, boxes, strict=True))
    ]


def _candidate_record(
    case: Any,
    candidate: torch.Tensor,
    index: int,
    source: str,
    topology_indices: frozenset[int],
    baseline_area: float,
    baseline_hpwl: float,
) -> dict[str, Any]:
    boxes = candidate.detach().to(device="cpu", dtype=torch.float32)
    verification = verify(case, boxes)
    soft = soft_violation_normalized(case, boxes)
    hpwl = total_hpwl(case, boxes)
    layout_area = bbox_area(boxes)
    if bool(getattr(case, "normalized", False)):
        layout_area *= float(case.scale) ** 2
    hpwl_gap = (hpwl - baseline_hpwl) / max(baseline_hpwl, 1.0e-6)
    area_gap = (layout_area - baseline_area) / max(baseline_area, 1.0e-6)
    objective = uncapped_objective(hpwl_gap, area_gap, soft.total)
    return {
        "candidate_index": int(index),
        "source": source,
        "candidate_type": (
            "topology"
            if index in topology_indices
            else "fallback"
            if index == 0
            else "learned_residual"
            if source.startswith("learned_")
            else "analytic"
        ),
        "hard_feasible": verification.feasible,
        "overlap_violations": len(verification.overlap_pairs),
        "overlap_pairs": [list(pair) for pair in verification.overlap_pairs],
        "area_violations": len(verification.area_bad),
        "area_bad_blocks": list(verification.area_bad),
        "fixed_violations": len(verification.fixed_bad),
        "fixed_bad_blocks": list(verification.fixed_bad),
        "preplaced_violations": len(verification.preplaced_bad),
        "preplaced_bad_blocks": list(verification.preplaced_bad),
        "hpwl_total": hpwl,
        "bbox_area": layout_area,
        "hpwl_gap": hpwl_gap,
        "area_gap": area_gap,
        "boundary_violations": soft.raw_boundary,
        "grouping_violations": soft.raw_grouping,
        "mib_violations": soft.raw_mib,
        "total_soft_violations": soft.raw_total,
        "max_possible_violations": soft.maximum,
        "violations_relative": soft.total,
        "official_capped_cost": None,
        "uncapped_objective": objective,
    }


def _topology_indices(value: object) -> frozenset[int]:
    indices = set()
    for source in value if isinstance(value, (list, tuple)) else ():
        text = str(source)
        if text.startswith("candidate_"):
            indices.add(int(text.removeprefix("candidate_")))
    return frozenset(indices)


def _oracles(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    result = candidate_oracles(candidates)
    result["topology"] = select_candidate_oracle(
        [row for row in candidates if row["candidate_type"] == "topology"]
    )
    result["learned_residual"] = select_candidate_oracle(
        [row for row in candidates if row["candidate_type"] == "learned_residual"]
    )
    return result


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    attribution = summarize_attribution_cases(cases)
    candidate_types = summarize_candidate_types(cases)
    return {
        "cases": len(cases),
        "topology_vs_analytic_weighted_gain": {
            stage: candidate_types[stage]["topology_vs_analytic"][
                "weighted_mean_topology_oracle_gain"
            ]
            for stage in ("raw", "post_bdp")
        },
        "hard_feasibility": {
            stage: {
                "candidate_count": attribution[stage]["candidate_count"],
                "hard_feasible_candidates": attribution[stage]["hard_feasible_candidates"],
                "rate": attribution[stage]["hard_feasibility_rate"],
                "candidate_count_by_type": candidate_types[stage]["candidate_count_by_type"],
                "hard_feasible_by_type": candidate_types[stage]["hard_feasible_by_type"],
            }
            for stage in ("raw", "post_bdp")
        },
        "selected_vs_analytic": _selected_vs_analytic(cases),
        "attribution": attribution,
        "candidate_types": candidate_types,
    }


def _selected_vs_analytic(cases: list[dict[str, Any]]) -> dict[str, Any]:
    gains: list[tuple[int, float]] = []
    selected_better = analytic_better = tied = 0
    for case in cases:
        analytic = case["post_bdp"]["oracles"]["analytic"]
        selected = case["incumbent"]
        if analytic is None or not bool(selected["hard_feasible"]):
            continue
        gain = float(analytic["uncapped_objective"]) - float(
            selected["uncapped_objective"]
        )
        gains.append((int(case["block_count"]), gain))
        if gain > 1.0e-9:
            selected_better += 1
        elif gain < -1.0e-9:
            analytic_better += 1
        else:
            tied += 1
    if not gains:
        return {
            "comparable_cases": 0,
            "selected_better_cases": 0,
            "analytic_better_cases": 0,
            "tied_cases": 0,
            "mean_selected_gain": None,
            "weighted_mean_selected_gain": None,
        }
    max_blocks = max(blocks for blocks, _gain in gains)
    weights = [math.exp((blocks - max_blocks) / 12.0) for blocks, _gain in gains]
    return {
        "comparable_cases": len(gains),
        "selected_better_cases": selected_better,
        "analytic_better_cases": analytic_better,
        "tied_cases": tied,
        "mean_selected_gain": sum(gain for _blocks, gain in gains) / len(gains),
        "weighted_mean_selected_gain": sum(
            gain * weight for (_blocks, gain), weight in zip(gains, weights, strict=True)
        )
        / sum(weights),
    }


def _incumbent_source(source: object, sources: tuple[str, ...]) -> tuple[int, str]:
    value = str(source)
    if value == "fallback":
        return 0, "fallback"
    if not value.startswith("candidate_"):
        raise RuntimeError(f"unknown exact incumbent source: {value}")
    index = int(value.removeprefix("candidate_"))
    if not 0 <= index < len(sources):
        raise RuntimeError(f"exact incumbent index {index} is outside candidate layout")
    return index, sources[index]


def _sample_id_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
