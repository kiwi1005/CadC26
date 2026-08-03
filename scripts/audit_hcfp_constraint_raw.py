#!/usr/bin/env python3
"""Audit constraint seeds with exact raw replay and the pinned official evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_hcfp_topology_heldout import (  # noqa: E402
    _collect_heldout,
    _load_training_exclusion,
    _sample_id_hash,
    _validate_args as _validate_heldout_args,
)
from benchmark_hcfp import _load_evaluator  # noqa: E402
from hcfp.analytic import (  # noqa: E402
    AnalyticConfig,
    select_device,
    to_official_placements,
)
from hcfp.benchmark import candidate_source_layout, percentile  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.constraints.raw_repair import repair_raw_constraints  # noqa: E402
from hcfp.data import DataSample, file_sha256  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    analyze_case_with_checkpoint,
    select_official_from_analysis,
)
from hcfp.projection import ComponentBDPConfig  # noqa: E402
from hcfp.reference import OFFICIAL_FLOORSET_V10  # noqa: E402
from hcfp.score_attribution import attribute_score  # noqa: E402
from hcfp.verify import overlap_pairs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(command_args)
    _validate_args(args)
    solver = _solver_provenance()
    if args.require_clean_solver and not solver["clean"]:
        raise RuntimeError(
            "--require-clean-solver requires a clean source worktree; "
            f"status_sha256={solver['status_sha256']}"
        )

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
    training_seed = int(training_contract["seed"])
    if args.heldout_seed == training_seed:
        raise ValueError("heldout and consumed training stream seeds must differ")
    training_sampling = str(training_contract["sampling"])
    heldout, split_provenance = _collect_heldout(
        args.root,
        exclude_ids=exclude_ids,
        exclude_provenance=exclude_provenance,
        heldout_limit=args.heldout_start + args.heldout_limit,
        heldout_seed=args.heldout_seed,
        heldout_max_layouts_per_file=args.heldout_max_layouts_per_file,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        score_aware=training_sampling == "score-aware",
    )
    heldout = heldout[args.heldout_start :]
    config = _learned_config(args)
    data_path = Path(args.data_path).resolve()
    evaluator_module = _load_evaluator(data_path)
    cases = []
    for index, (sample, source) in enumerate(
        heldout, start=args.heldout_start
    ):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        case = _audit_sample(
            evaluator_module,
            index,
            sample,
            source,
            checkpoint,
            checkpoint_hash,
            device,
            config,
            args.population,
            args.topology_seeds,
            args.constraint_seeds,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        case["runtime_seconds"] = time.perf_counter() - started
        cases.append(case)
    sample_ids = [sample.sample_id for sample, _source in heldout]
    evaluator_path = data_path / OFFICIAL_FLOORSET_V10.evaluator_path
    report = {
        "schema_version": 5,
        "command": ["scripts/audit_hcfp_constraint_raw.py", *command_args],
        "solver": solver,
        "config": {
            "root": str(Path(args.root).resolve()),
            "data_path": str(data_path),
            "checkpoint": str(checkpoint),
            "training_report": str(training_report),
            "output": str(Path(args.output).resolve()),
            "heldout_limit": args.heldout_limit,
            "heldout_start": args.heldout_start,
            "heldout_seed": args.heldout_seed,
            "exclude_train_limit": args.exclude_train_limit,
            "exclude_train_seed": args.exclude_train_seed,
            "heldout_max_layouts_per_file": args.heldout_max_layouts_per_file,
            "min_blocks": args.min_blocks,
            "max_blocks": args.max_blocks,
            "population": args.population,
            "topology_seeds": args.topology_seeds,
            "constraint_seeds": args.constraint_seeds,
            "device": str(device),
            "sampling": training_sampling,
            "dynamics_steps": args.dynamics_steps,
            "projection_steps": args.projection_steps,
            "direction_beam": args.direction_beam,
            "component_bdp": args.component_bdp,
            "component_beam": args.component_beam,
            "component_limit": args.component_limit,
            "component_uncertain_pairs": args.component_uncertain_pairs,
            "component_sweeps": args.component_sweeps,
            "component_reset_limit": args.component_reset_limit,
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
        "evaluation": {
            "mode": "pinned official-v10 evaluator on exact raw coordinates",
            "official_raw_replay": True,
            "evaluator_path": str(evaluator_path),
            "evaluator_sha256": file_sha256(evaluator_path),
            "evaluator_commit": OFFICIAL_FLOORSET_V10.commit,
        },
        "sampling": {
            "source": "FloorSet-Lite training",
            "mode": training_sampling,
            "training_report": {
                "path": str(training_report),
                "sha256": file_sha256(training_report),
            },
            **split_provenance,
            "heldout": {
                **split_provenance["heldout"],
                "selection_start": args.heldout_start,
                "requested_count": args.heldout_limit,
                "count": len(sample_ids),
                "sample_ids": sample_ids,
                "sample_id_sha256": _sample_id_hash(sample_ids),
            },
        },
        "cases": cases,
        "summary": _summary(cases),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="FloorSet-Lite training root")
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-clean-solver",
        action="store_true",
        help="fail unless the source worktree is clean before the audit",
    )
    parser.add_argument("--heldout-limit", type=int, default=16)
    parser.add_argument("--heldout-start", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=1)
    parser.add_argument("--exclude-train-limit", type=int)
    parser.add_argument("--exclude-train-seed", type=int)
    parser.add_argument("--heldout-max-layouts-per-file", type=int, default=1)
    parser.add_argument("--min-blocks", type=int, default=106)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--topology-seeds", type=int, default=16)
    parser.add_argument("--constraint-seeds", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling", choices=("uniform", "score-aware"))
    parser.add_argument("--dynamics-steps", type=int, default=0)
    parser.add_argument("--projection-steps", type=int, default=4)
    parser.add_argument("--direction-beam", type=int, default=1)
    parser.add_argument("--component-bdp", action="store_true")
    parser.add_argument("--component-beam", type=int, default=4)
    parser.add_argument("--component-limit", type=int, default=24)
    parser.add_argument("--component-uncertain-pairs", type=int, default=6)
    parser.add_argument("--component-sweeps", type=int, default=4)
    parser.add_argument("--component-reset-limit", type=int, default=2)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--tail-topk", type=int)
    return parser


def _solver_provenance() -> dict[str, Any]:
    commit = _git_bytes("rev-parse", "HEAD").decode().strip()
    status = _git_bytes(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff = _git_bytes("diff", "--binary", "HEAD", "--", ".")
    fingerprint = hashlib.sha256(status + b"\0" + diff).hexdigest()
    return {
        "repository": str(ROOT),
        "commit": commit,
        "clean": not status,
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "workspace_fingerprint": fingerprint,
    }


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(ROOT), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _validate_args(args: argparse.Namespace) -> None:
    _validate_heldout_args(args)
    if args.heldout_start < 0:
        raise ValueError("--heldout-start must be non-negative")
    if args.constraint_seeds <= 0:
        raise ValueError(
            "--constraint-seeds must be positive for a raw constraint audit"
        )


def _learned_config(args: argparse.Namespace) -> LearnedConfig:
    return LearnedConfig(
        analytic=AnalyticConfig(
            dynamics=DynamicsConfig(
                population=args.population,
                steps=args.dynamics_steps,
            ),
            projection_iterations=args.projection_steps,
            direction_beam=args.direction_beam,
            component_bdp=ComponentBDPConfig(
                enabled=args.component_bdp,
                beam_width=args.component_beam,
                component_limit=args.component_limit,
                max_uncertain_pairs=args.component_uncertain_pairs,
                outer_sweeps=args.component_sweeps,
                reset_limit=args.component_reset_limit,
            ),
        ),
        flow_steps=args.flow_steps,
        tail_topk=args.tail_topk,
        seed=args.flow_seed,
        topology_seeds=args.topology_seeds,
        constraint_seeds=args.constraint_seeds,
    )


def _audit_sample(
    evaluator_module: Any,
    index: int,
    sample: DataSample,
    source: dict[str, Any],
    checkpoint: Path,
    checkpoint_hash: str,
    device: torch.device,
    config: LearnedConfig,
    population: int,
    topology_seeds: int,
    constraint_seeds: int,
) -> dict[str, Any]:
    case = sample.case.to(device=device, dtype=torch.float32)
    analysis = analyze_case_with_checkpoint(case, checkpoint, config)
    _validate_analysis(
        sample.sample_id,
        analysis,
        checkpoint_hash,
        topology_seeds,
        constraint_seeds,
    )
    learned_count = int(analysis.result.candidate_count) - population
    sources = candidate_source_layout(population, learned_count)
    raw = analysis.analytic.raw_candidates
    projected = analysis.analytic.projected_candidates
    if len(sources) != raw.shape[0] or raw.shape != projected.shape:
        raise RuntimeError(
            f"sample {sample.sample_id}: source layout {len(sources)} does not match "
            f"raw/projected candidates {tuple(raw.shape)}/{tuple(projected.shape)}"
        )
    snapshot = analysis.analytic.incumbent_snapshot
    topology_indices = _seed_indices(
        sample.sample_id,
        snapshot,
        "topology_seed_sources",
        2 * topology_seeds,
    )
    constraint_indices = _seed_indices(
        sample.sample_id,
        snapshot,
        "constraint_seed_sources",
        2 * constraint_seeds,
    )
    records = _constraint_records(
        sample.sample_id,
        snapshot,
        constraint_indices,
    )
    invalid = (topology_indices | constraint_indices) - frozenset(range(len(sources)))
    if invalid:
        raise RuntimeError(
            f"sample {sample.sample_id}: seed provenance indices are outside "
            f"candidate layout: {sorted(invalid)}"
        )
    if topology_indices & constraint_indices:
        raise RuntimeError(
            f"sample {sample.sample_id}: topology and constraint provenance overlap"
        )
    metric_args = _metric_args(sample, source)
    raw_records, projected_records = _candidate_pair_records(
        evaluator_module,
        source,
        case,
        raw,
        projected,
        sources,
        topology_indices,
        constraint_indices,
        records,
        metric_args,
    )
    telemetry = analysis.analytic.telemetry
    false_fast_gate = telemetry.projection_ok & ~telemetry.hard_feasible
    if bool(false_fast_gate.any().item()):
        indices = torch.nonzero(
            false_fast_gate, as_tuple=False
        ).reshape(-1).tolist()
        raise RuntimeError(
            f"sample {sample.sample_id}: projection_ok disagrees with exact "
            f"normalized feasibility at candidates {indices}"
        )
    projection_fields = {
        "initial_pair_count": telemetry.projection_initial_pairs,
        "final_pair_count": telemetry.projection_final_pairs,
        "component_rebuilds": telemetry.projection_component_rebuilds,
        "new_pairs_detected": telemetry.projection_new_pairs,
        "reset_count": telemetry.projection_resets,
        "beam_states_evaluated": telemetry.projection_beam_states,
        "max_component_size": telemetry.projection_max_component_size,
    }
    for candidate_index, row in enumerate(projected_records):
        row["projection"] = {
            name: int(values[candidate_index].detach().cpu().item())
            for name, values in projection_fields.items()
        }
        row["projection"]["normalized_fp64_final_pairs"] = row[
            "projection"
        ]["final_pair_count"]
    selected_positions = select_official_from_analysis(
        source,
        case,
        analysis,
        config=config,
        device=device,
    )
    selected = _evaluate_positions(
        evaluator_module,
        selected_positions,
        candidate_index=-1,
        source_name="runtime_final",
        candidate_type="selected",
        constraint_kind=None,
        stage="runtime_final",
        projection_displacement=None,
        repair=None,
        metric_args=metric_args,
    )
    if not selected["hard_feasible"]:
        raise RuntimeError(
            f"sample {sample.sample_id}: runtime final output is infeasible"
        )
    selected["candidate_matches"] = [
        {
            key: row[key]
            for key in (
                "candidate_index",
                "source",
                "candidate_type",
                "constraint_kind",
                "stage",
            )
        }
        for row in (*raw_records, *projected_records)
        if row["placement_sha256"] == selected["placement_sha256"]
    ]
    return {
        "test_id": index,
        "sample_id": sample.sample_id,
        "block_count": case.n,
        "baseline": {
            "hpwl": float(sample.labels.baseline_hpwl),
            "area": float(sample.labels.baseline_area),
        },
        "candidate_layout": {
            "population": population,
            "learned_count": learned_count,
            "topology_count": topology_seeds,
            "constraint_count": constraint_seeds,
            "candidate_count": len(sources),
        },
        "selection_provenance": {
            key: snapshot.get(key)
            for key in (
                "exact_source",
                "analytic_exact_source",
                "analytic_fast_source",
            )
        },
        "topology_provenance": {
            key: value
            for key, value in snapshot.items()
            if str(key).startswith("topology_")
        },
        "constraint_provenance": {
            key: value
            for key, value in snapshot.items()
            if str(key).startswith("constraint_")
        },
        "raw": {"candidates": raw_records, "oracles": _oracles(raw_records)},
        "post_bdp": {
            "candidates": projected_records,
            "oracles": _oracles(projected_records),
        },
        "selected": selected,
    }


def _validate_analysis(
    sample_id: str,
    analysis: Any,
    checkpoint_hash: str,
    topology_seeds: int,
    constraint_seeds: int,
) -> None:
    result = analysis.result
    if not result.used_checkpoint or result.checkpoint_hash != checkpoint_hash:
        reason = result.failure_reason or "checkpoint was not used"
        raise RuntimeError(f"sample {sample_id}: {reason}")
    for name, requested, produced, reason_key in (
        (
            "topology",
            topology_seeds,
            int(result.topology_seed_count),
            "topology_seed_failure_reason",
        ),
        (
            "constraint",
            constraint_seeds,
            int(result.constraint_seed_count),
            "constraint_seed_failure_reason",
        ),
    ):
        if produced != requested:
            reason = analysis.analytic.incumbent_snapshot.get(
                reason_key,
                f"{name} construction produced an incomplete candidate set",
            )
            raise RuntimeError(
                f"sample {sample_id}: requested {requested} {name} seeds, "
                f"produced {produced}: {reason}"
            )


def _seed_indices(
    sample_id: str,
    snapshot: dict[str, Any],
    key: str,
    expected_count: int,
) -> frozenset[int]:
    raw = snapshot.get(key)
    if not isinstance(raw, (tuple, list)) or len(raw) != expected_count:
        actual = len(raw) if isinstance(raw, (tuple, list)) else 0
        raise RuntimeError(
            f"sample {sample_id}: {key} names {actual} candidates, "
            f"expected {expected_count}"
        )
    indices = [_candidate_index(value) for value in raw]
    if any(value is None for value in indices) or len(set(indices)) != expected_count:
        raise RuntimeError(f"sample {sample_id}: {key} is malformed or duplicated")
    return frozenset(int(value) for value in indices if value is not None)


def _constraint_records(
    sample_id: str,
    snapshot: dict[str, Any],
    expected_indices: frozenset[int],
) -> dict[int, dict[str, object]]:
    raw = snapshot.get("constraint_seed_provenance")
    if not isinstance(raw, (tuple, list)) or len(raw) != len(expected_indices):
        actual = len(raw) if isinstance(raw, (tuple, list)) else 0
        raise RuntimeError(
            f"sample {sample_id}: constraint provenance has {actual} records, "
            f"expected {len(expected_indices)}"
        )
    records: dict[int, dict[str, object]] = {}
    for record in raw:
        if not isinstance(record, dict):
            raise RuntimeError(
                f"sample {sample_id}: constraint provenance is malformed"
            )
        index = _candidate_index(record.get("source"))
        if index is None or index in records:
            raise RuntimeError(
                f"sample {sample_id}: constraint provenance source is malformed or duplicated"
            )
        records[index] = record
    if frozenset(records) != expected_indices:
        raise RuntimeError(
            f"sample {sample_id}: constraint provenance sources do not match "
            "constraint_seed_sources"
        )
    return records


def _candidate_index(value: object) -> int | None:
    text = str(value)
    if not text.startswith("candidate_"):
        return None
    try:
        index = int(text.removeprefix("candidate_"))
    except ValueError:
        return None
    return index if index >= 0 else None


def _metric_args(
    sample: DataSample,
    source: dict[str, Any],
) -> tuple[Any, ...]:
    baseline = {
        "hpwl_baseline": float(sample.labels.baseline_hpwl),
        "area_baseline": float(sample.labels.baseline_area),
    }
    return (
        baseline,
        source["constraints"],
        source["b2b_connectivity"],
        source["p2b_connectivity"],
        source["pins_pos"],
        source["area_targets"],
        source["target_positions"],
    )


def _candidate_pair_records(
    evaluator_module: Any,
    source: dict[str, Any],
    case: Any,
    raw: torch.Tensor,
    projected: torch.Tensor,
    sources: tuple[str, ...],
    topology_indices: frozenset[int],
    constraint_indices: frozenset[int],
    constraint_records: dict[int, dict[str, object]],
    metric_args: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    projected_rows: list[dict[str, Any]] = []
    for index, (source_name, raw_box, projected_box) in enumerate(
        zip(sources, raw, projected, strict=True)
    ):
        candidate_type = _candidate_type(
            index,
            source_name,
            topology_indices,
            constraint_indices,
        )
        raw_positions = to_official_placements(source, case, raw_box.detach().cpu())
        projected_positions = to_official_placements(
            source, case, projected_box.detach().cpu()
        )
        raw_denormal_overlap = len(overlap_pairs(raw_positions))
        projected_denormal_overlap = len(overlap_pairs(projected_positions))
        raw_repair = projected_repair = None
        constraint_kind = None
        if candidate_type == "constraint":
            record = constraint_records[index]
            constraint_kind = str(record.get("kind", "unknown"))
            raw_repair = repair_raw_constraints(source, raw_positions, record)
            projected_repair = repair_raw_constraints(
                source, projected_positions, record
            )
            raw_positions = raw_repair.placements
            projected_positions = projected_repair.placements
        displacement = _projection_displacement(raw_positions, projected_positions)
        raw_row = _evaluate_positions(
            evaluator_module,
            raw_positions,
            candidate_index=index,
            source_name=source_name,
            candidate_type=candidate_type,
            constraint_kind=constraint_kind,
            stage="raw",
            projection_displacement=displacement,
            repair=raw_repair,
            metric_args=metric_args,
        )
        projected_row = _evaluate_positions(
            evaluator_module,
            projected_positions,
            candidate_index=index,
            source_name=source_name,
            candidate_type=candidate_type,
            constraint_kind=constraint_kind,
            stage="post_bdp",
            projection_displacement=displacement,
            repair=projected_repair,
            metric_args=metric_args,
        )
        raw_row["post_denormal_exact_pairs"] = raw_denormal_overlap
        raw_row["post_repair_exact_pairs"] = len(overlap_pairs(raw_positions))
        projected_row[
            "post_denormal_exact_pairs"
        ] = projected_denormal_overlap
        projected_row["post_repair_exact_pairs"] = len(
            overlap_pairs(projected_positions)
        )
        for row in (raw_row, projected_row):
            if row["overlap_violations"] != row["post_repair_exact_pairs"]:
                raise RuntimeError(
                    "internal exact overlap count disagrees with official evaluator"
                )
        raw_rows.append(raw_row)
        projected_rows.append(projected_row)
    return raw_rows, projected_rows


def _candidate_type(
    index: int,
    source_name: str,
    topology_indices: frozenset[int],
    constraint_indices: frozenset[int],
) -> str:
    if index in constraint_indices:
        return "constraint"
    if index in topology_indices:
        return "topology"
    if index == 0:
        return "fallback"
    if source_name.startswith("learned_"):
        return "learned_residual"
    return "analytic"


def _projection_displacement(raw: Any, projected: Any) -> float:
    raw_boxes = torch.as_tensor(raw, dtype=torch.float64, device="cpu")
    projected_boxes = torch.as_tensor(projected, dtype=torch.float64, device="cpu")
    if raw_boxes.shape != projected_boxes.shape or raw_boxes.ndim != 2:
        raise ValueError("raw/projected placements must have the same [N,4] shape")
    return float(
        torch.linalg.vector_norm(projected_boxes[:, :2] - raw_boxes[:, :2], dim=1).sum()
    )


def _evaluate_positions(
    evaluator_module: Any,
    positions: Any,
    *,
    candidate_index: int,
    source_name: str,
    candidate_type: str,
    constraint_kind: str | None,
    stage: str,
    projection_displacement: float | None,
    repair: Any | None,
    metric_args: tuple[Any, ...],
) -> dict[str, Any]:
    rows = [tuple(float(value) for value in row) for row in positions]
    metrics = evaluator_module.evaluate_solution(
        {"positions": rows, "runtime": 1.0},
        *metric_args,
        median_runtime=1.0,
    )
    score = attribute_score(
        float(metrics.hpwl_gap),
        float(metrics.area_gap),
        boundary_violations=int(metrics.boundary_violations),
        grouping_violations=int(metrics.grouping_violations),
        mib_violations=int(metrics.mib_violations),
        max_possible_violations=int(metrics.max_possible_violations),
        hard_feasible=bool(metrics.is_feasible),
    )
    if not math.isclose(
        score.official_capped_cost,
        float(metrics.cost),
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError("official evaluator cost does not match score attribution")
    return {
        "candidate_index": candidate_index,
        "source": source_name,
        "candidate_type": candidate_type,
        "constraint_kind": constraint_kind,
        "stage": stage,
        "placement_sha256": _placement_sha256(rows),
        "hard_feasible": bool(metrics.is_feasible),
        "overlap_violations": int(metrics.overlap_violations),
        "area_violations": int(metrics.area_violations),
        "dimension_violations": int(metrics.dimension_violations),
        "fixed_violations": int(metrics.fixed_violations),
        "preplaced_violations": int(metrics.preplaced_violations),
        "hpwl_total": float(metrics.hpwl_total),
        "bbox_area": float(metrics.bbox_area),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "total_soft_violations": int(metrics.total_soft_violations),
        "max_possible_violations": int(metrics.max_possible_violations),
        "violations_relative": float(metrics.violations_relative),
        "uncapped_cost": score.uncapped_cost,
        "log_uncapped_cost": score.log_uncapped_cost,
        "cap_margin": score.cap_margin,
        "is_capped": score.is_capped,
        "required_soft_fixes_to_uncap": score.required_soft_fixes_to_uncap,
        "required_quality_gap_to_uncap": score.required_quality_gap_to_uncap,
        "blocker_classification": score.blocker_classification,
        "official_capped_cost": float(metrics.cost),
        "projection_displacement": projection_displacement,
        "raw_constraint_repair": (
            {
                name: int(getattr(repair, name))
                for name in (
                    "group_edges_applied",
                    "group_edges_rejected",
                    "boundary_blocks_applied",
                    "boundary_blocks_rejected",
                )
            }
            if repair is not None
            else None
        ),
    }


def _placement_sha256(rows: list[tuple[float, ...]]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _oracles(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _oracle(candidates),
        "analytic": _oracle(
            candidates,
            candidate_types=frozenset(("fallback", "analytic")),
        ),
        "topology": _oracle(candidates, candidate_types=frozenset(("topology",))),
        "constraint": _oracle(
            candidates,
            candidate_types=frozenset(("constraint",)),
        ),
    }


def _oracle(
    candidates: list[dict[str, Any]],
    *,
    candidate_types: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in candidates
        if bool(row["hard_feasible"])
        and (candidate_types is None or str(row["candidate_type"]) in candidate_types)
    ]
    if not eligible:
        return None
    return dict(
        min(
            eligible,
            key=lambda row: (
                float(row["log_uncapped_cost"]),
                int(row["candidate_index"]),
            ),
        )
    )


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "hard_feasibility": {
            stage: _hard_feasibility(cases, stage) for stage in ("raw", "post_bdp")
        },
        "oracle": {
            stage: {
                candidate_type: _oracle_aggregate(cases, stage, candidate_type)
                for candidate_type in ("analytic", "topology", "constraint")
            }
            for stage in ("raw", "post_bdp")
        },
        "topology_vs_constraint": {
            stage: _oracle_gain(cases, stage, "topology", "constraint")
            for stage in ("raw", "post_bdp")
        },
        "selected_vs_analytic": _selected_vs_analytic(cases),
        "projection_displacement": _displacement_summary(cases),
        "runtime": _runtime_summary(cases),
    }


def _runtime_summary(cases: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = sorted(float(case["runtime_seconds"]) for case in cases)
    return {
        "case_count": len(values),
        "total": sum(values),
        "mean": _mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "maximum": values[-1] if values else None,
    }


def _hard_feasibility(cases: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    rows = [candidate for case in cases for candidate in case[stage]["candidates"]]
    types = ("fallback", "analytic", "learned_residual", "topology", "constraint")
    return {
        "candidate_count": len(rows),
        "hard_feasible_candidates": sum(bool(row["hard_feasible"]) for row in rows),
        "candidate_count_by_type": {
            name: sum(row["candidate_type"] == name for row in rows) for name in types
        },
        "hard_feasible_by_type": {
            name: sum(
                row["candidate_type"] == name and bool(row["hard_feasible"])
                for row in rows
            )
            for name in types
        },
    }


def _oracle_aggregate(
    cases: list[dict[str, Any]],
    stage: str,
    candidate_type: str,
) -> dict[str, Any]:
    values = [
        (int(case["block_count"]), case[stage]["oracles"].get(candidate_type))
        for case in cases
    ]
    available = [(blocks, row) for blocks, row in values if row is not None]
    return {
        "available_cases": len(available),
        "mean_log_uncapped_cost": _mean(
            [float(row["log_uncapped_cost"]) for _blocks, row in available]
        ),
        "weighted_mean_log_uncapped_cost": _weighted_mean(
            [(blocks, float(row["log_uncapped_cost"])) for blocks, row in available]
        ),
        **{
            f"total_{name}": sum(int(row[name]) for _blocks, row in available)
            for name in (
                "boundary_violations",
                "grouping_violations",
                "mib_violations",
            )
        },
    }


def _oracle_gain(
    cases: list[dict[str, Any]],
    stage: str,
    baseline_type: str,
    candidate_type: str,
) -> dict[str, Any]:
    gains: list[tuple[int, float]] = []
    candidate_better = baseline_better = tied = 0
    for case in cases:
        oracles = case[stage]["oracles"]
        baseline = oracles.get(baseline_type)
        candidate = oracles.get(candidate_type)
        if baseline is None or candidate is None:
            continue
        gain = float(baseline["log_uncapped_cost"]) - float(
            candidate["log_uncapped_cost"]
        )
        gains.append((int(case["block_count"]), gain))
        if gain > 1.0e-9:
            candidate_better += 1
        elif gain < -1.0e-9:
            baseline_better += 1
        else:
            tied += 1
    return {
        "comparable_cases": len(gains),
        f"{candidate_type}_better_cases": candidate_better,
        f"{baseline_type}_better_cases": baseline_better,
        "tied_cases": tied,
        f"mean_{candidate_type}_j_gain": _mean([gain for _blocks, gain in gains]),
        f"weighted_mean_{candidate_type}_j_gain": _weighted_mean(gains),
    }


def _selected_vs_analytic(cases: list[dict[str, Any]]) -> dict[str, Any]:
    gains: list[tuple[int, float]] = []
    selected_better = analytic_better = tied = 0
    for case in cases:
        analytic = case["post_bdp"]["oracles"].get("analytic")
        selected = case["selected"]
        if analytic is None or not bool(selected["hard_feasible"]):
            continue
        gain = float(analytic["log_uncapped_cost"]) - float(
            selected["log_uncapped_cost"]
        )
        gains.append((int(case["block_count"]), gain))
        if gain > 1.0e-9:
            selected_better += 1
        elif gain < -1.0e-9:
            analytic_better += 1
        else:
            tied += 1
    return {
        "comparable_cases": len(gains),
        "selected_better_cases": selected_better,
        "analytic_better_cases": analytic_better,
        "tied_cases": tied,
        "mean_selected_j_gain": _mean([gain for _blocks, gain in gains]),
        "weighted_mean_selected_j_gain": _weighted_mean(gains),
    }


def _displacement_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for candidate_type in ("topology", "constraint"):
        paired = [
            (int(case["block_count"]), raw, projected)
            for case in cases
            for raw, projected in zip(
                case["raw"]["candidates"],
                case["post_bdp"]["candidates"],
                strict=True,
            )
            if raw["candidate_type"] == candidate_type
        ]
        values = [
            (blocks, float(projected["projection_displacement"]))
            for blocks, _raw, projected in paired
        ]
        feasible = [
            (blocks, float(projected["projection_displacement"]))
            for blocks, _raw, projected in paired
            if bool(projected["hard_feasible"])
        ]
        summaries[candidate_type] = {
            "candidate_count": len(values),
            "mean": _mean([value for _blocks, value in values]),
            "weighted_mean": _weighted_mean(values),
            "post_bdp_hard_feasible_count": len(feasible),
            "post_bdp_hard_feasible_mean": _mean(
                [value for _blocks, value in feasible]
            ),
            "post_bdp_hard_feasible_weighted_mean": _weighted_mean(feasible),
            "newly_hard_feasible_count": sum(
                not bool(raw["hard_feasible"])
                and bool(projected["hard_feasible"])
                for _blocks, raw, projected in paired
            ),
            "hard_feasible_regression_count": sum(
                bool(raw["hard_feasible"])
                and not bool(projected["hard_feasible"])
                for _blocks, raw, projected in paired
            ),
            "no_commit_count": sum(value == 0.0 for _blocks, value in values),
        }
    topology = summaries["topology"]
    constraint = summaries["constraint"]
    summaries["constraint_minus_topology"] = {
        "mean": _difference(constraint["mean"], topology["mean"]),
        "weighted_mean": _difference(
            constraint["weighted_mean"], topology["weighted_mean"]
        ),
    }
    return summaries


def _difference(first: object, second: object) -> float | None:
    if first is None or second is None:
        return None
    return float(first) - float(second)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _weighted_mean(values: list[tuple[int, float]]) -> float | None:
    if not values:
        return None
    max_blocks = max(blocks for blocks, _value in values)
    weights = [math.exp((blocks - max_blocks) / 12.0) for blocks, _value in values]
    return sum(
        value * weight for (_blocks, value), weight in zip(values, weights, strict=True)
    ) / sum(weights)


if __name__ == "__main__":
    raise SystemExit(main())
