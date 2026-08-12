#!/usr/bin/env python3
"""Audit HCFP ranker counterfactuals without altering runtime output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_hcfp_topology_heldout import _collect_heldout  # noqa: E402
from benchmark_hcfp import _load_evaluator  # noqa: E402
from hcfp.analytic import AnalyticConfig  # noqa: E402
from hcfp.case import from_official  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import DataSample, extract_labels, file_sha256, sample_from_fixture  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    analyze_case_with_checkpoint,
    select_official_from_analysis,
)
from hcfp.projection import ComponentBDPConfig  # noqa: E402
from hcfp.reference import OFFICIAL_FLOORSET_V10  # noqa: E402
from audit_hcfp_constraint_raw import _placement_sha256  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(command)
    _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    data_path = Path(args.data_path)
    evaluator_path = data_path / OFFICIAL_FLOORSET_V10.evaluator_path
    expected_hash = args.evaluator_sha256 or OFFICIAL_FLOORSET_V10.evaluator_sha256
    actual_hash = file_sha256(evaluator_path)
    if actual_hash != expected_hash:
        raise ValueError("evaluator hash mismatch")
    evaluator = _load_evaluator(data_path)
    checkpoint = Path(args.checkpoint)
    _model, checkpoint_metadata = load_checkpoint(
        checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    cases = _load_case_sources(args, evaluator)
    if not cases:
        raise ValueError("counterfactual audit selected no cases")
    case_ids = [sample.sample_id for sample, _source in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs")
    seeds = _parse_seeds(args.seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seeds")
    config_payload = _config_payload(args, checkpoint, actual_hash, seeds, case_ids)
    config_hash = _stable_hash(config_payload)
    previous = _load_previous(output, config_hash) if args.resume else {}
    rows: list[dict[str, Any]] = list(previous.values())
    provenance = _provenance(command, checkpoint, checkpoint_metadata, evaluator_path)
    expected_rows = len(cases) * len(seeds)

    def persist(status: str) -> None:
        report = {
            "schema_version": 1,
            "mode": "ranker_counterfactual_audit_only",
            "status": status,
            "production_output_altered": False,
            "config_hash": config_hash,
            "config": config_payload,
            "provenance": provenance,
            "cases": sorted(rows, key=lambda row: (row["case_id"], row["seed"])),
            "summary": _summary(rows, expected_rows=expected_rows),
        }
        _atomic_write_json(output, report)

    persist("in_progress")
    for sample, source in cases:
        for seed in seeds:
            key = _resume_key(sample.sample_id, seed, config_hash)
            if key in previous:
                continue
            started = time.perf_counter()
            row = _audit_one(
                evaluator,
                sample,
                source,
                checkpoint,
                checkpoint_metadata,
                seed,
                args,
                config_hash,
            )
            row["runtime_seconds"] = time.perf_counter() - started
            rows.append(row)
            persist("in_progress")
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"counterfactual audit produced {len(rows)} of {expected_rows} expected rows"
        )
    persist("complete")
    print(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--evaluator-sha256")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case-json", action="append")
    source.add_argument("--root")
    source.add_argument(
        "--official-cases",
        help="all or comma-separated visible official validation case ids",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=1)
    parser.add_argument("--heldout-max-layouts-per-file", type=int, default=1)
    parser.add_argument("--min-blocks", type=int, default=1)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--score-aware", action="store_true")
    parser.add_argument("--seed", action="append", required=True)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=0)
    parser.add_argument("--projection-steps", type=int, default=4)
    parser.add_argument("--direction-beam", type=int, default=1)
    parser.add_argument("--component-bdp", action="store_true")
    parser.add_argument("--component-beam", type=int, default=4)
    parser.add_argument("--component-limit", type=int, default=24)
    parser.add_argument("--component-uncertain-pairs", type=int, default=8)
    parser.add_argument("--component-sweeps", type=int, default=2)
    parser.add_argument("--component-reset-limit", type=int, default=2)
    parser.add_argument("--topology-seeds", type=int, default=0)
    parser.add_argument("--constraint-seeds", type=int, default=0)
    parser.add_argument("--treemap-seeds", type=int, default=0)
    parser.add_argument("--btree-seeds", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--collective-steps", type=int, default=0)
    parser.add_argument("--tail-topk", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.root and args.limit <= 0:
        raise ValueError("--root requires a positive --limit")
    if args.constraint_seeds and not args.topology_seeds:
        raise ValueError("--constraint-seeds requires --topology-seeds")
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    for name in (
        "population",
        "projection_steps",
        "direction_beam",
        "component_beam",
        "component_limit",
        "component_uncertain_pairs",
        "component_sweeps",
        "component_reset_limit",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "dynamics_steps",
        "topology_seeds",
        "constraint_seeds",
        "treemap_seeds",
        "btree_seeds",
        "flow_steps",
        "collective_steps",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def _load_case_sources(
    args: argparse.Namespace,
    evaluator: Any,
) -> list[tuple[DataSample, dict[str, Any]]]:
    if args.case_json:
        cases = []
        for value in args.case_json:
            path = Path(value)
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("case fixture entries must be objects")
                sample = sample_from_fixture(item)
                source = _source_from_fixture(item)
                cases.append((sample, source))
        return cases
    if args.official_cases:
        return _official_case_sources(
            evaluator,
            Path(args.data_path),
            args.official_cases,
        )
    heldout, _provenance = _collect_heldout(
        args.root,
        exclude_ids=set(),
        exclude_provenance={"source": "ranker-counterfactual-audit"},
        heldout_limit=args.limit,
        heldout_seed=args.heldout_seed,
        heldout_max_layouts_per_file=args.heldout_max_layouts_per_file,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        score_aware=args.score_aware,
    )
    return heldout


def _official_case_sources(
    evaluator_module: Any,
    data_path: Path,
    case_spec: str,
) -> list[tuple[DataSample, dict[str, Any]]]:
    contest = evaluator_module.ContestEvaluator(str(data_path), verbose=False)
    contest._load_dataset()
    case_ids = (
        list(range(len(contest.dataset)))
        if case_spec == "all"
        else [int(value) for value in case_spec.split(",") if value]
    )
    if not case_ids:
        raise ValueError("--official-cases selected no cases")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate official case IDs")
    if min(case_ids) < 0 or max(case_ids) >= len(contest.dataset):
        raise ValueError("official case ID is outside the visible dataset")

    cases = []
    for case_id in case_ids:
        item = contest.dataset[case_id]
        inputs, labels = item["input"], item["label"]
        area, b2b, p2b, pins, constraints = inputs
        block_count = int((area != -1).sum().item())
        baseline, target_positions = contest._extract_baseline(
            case_id,
            labels,
            b2b,
            p2b,
            pins,
            block_count,
        )
        if target_positions is None:
            raise ValueError(f"official case {case_id} is missing target geometry")
        optimizer_targets = _optimizer_targets(
            constraints[:block_count],
            target_positions[:block_count],
        )
        case = from_official(
            block_count,
            area[:block_count],
            b2b,
            p2b,
            pins,
            constraints[:block_count],
            optimizer_targets,
        )
        sample = DataSample(
            f"official_visible:{case_id}",
            case,
            extract_labels(
                case,
                target_positions[:block_count],
                normalized=False,
                baseline_area=baseline["area_baseline"],
                baseline_hpwl=baseline["hpwl_baseline"],
            ),
        )
        source = {
            "block_count": block_count,
            "area_targets": area[:block_count],
            "b2b_connectivity": b2b,
            "p2b_connectivity": p2b,
            "pins_pos": pins,
            "constraints": constraints[:block_count],
            "target_positions": optimizer_targets,
        }
        cases.append((sample, source))
    return cases


def _optimizer_targets(
    constraints: torch.Tensor,
    target_positions: Any,
) -> torch.Tensor:
    targets = torch.as_tensor(target_positions, dtype=torch.float32)
    output = torch.full((len(constraints), 4), -1.0, dtype=torch.float32)
    columns = constraints.shape[1] if constraints.ndim > 1 else 0
    for index in range(len(constraints)):
        is_fixed = columns > 0 and bool(constraints[index, 0] != 0)
        is_preplaced = columns > 1 and bool(constraints[index, 1] != 0)
        if is_preplaced:
            output[index] = targets[index]
        elif is_fixed:
            output[index, 2:] = targets[index, 2:]
    return output


def _source_from_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    if "source" in payload:
        source = dict(payload["source"])
    else:
        source = {
            "block_count": payload.get("block_count"),
            "area_targets": payload.get("area_targets"),
            "b2b_connectivity": payload.get("b2b_connectivity", []),
            "p2b_connectivity": payload.get("p2b_connectivity", []),
            "pins_pos": payload.get("pins_pos", []),
            "constraints": payload.get("constraints", []),
            "target_positions": payload.get("target_positions"),
        }
    if source.get("target_positions") is None:
        source["target_positions"] = _target_positions_from_case(source)
    return source


def _target_positions_from_case(source: dict[str, Any]) -> list[list[float]]:
    case = from_official(
        int(source["block_count"]),
        source["area_targets"],
        source.get("b2b_connectivity", []),
        source.get("p2b_connectivity", []),
        source.get("pins_pos", []),
        source.get("constraints", []),
        source.get("target_positions"),
    )
    target = case.target.detach().cpu()
    valid = case.target_valid_mask.detach().cpu()
    rows = []
    for index in range(case.n):
        rows.append(target[index].tolist() if bool(valid[index]) else [-1.0] * 4)
    return rows


def _parse_seeds(values: list[str]) -> list[int]:
    return [int(value) for value in values]


def _audit_one(
    evaluator: Any,
    sample: DataSample,
    source: dict[str, Any],
    checkpoint: Path,
    checkpoint_metadata: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    config_hash: str,
) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    config = _learned_config(args, seed)
    case = sample.case.to(device=device, dtype=torch.float32)
    analysis = analyze_case_with_checkpoint(case, checkpoint, config)
    if not analysis.result.used_checkpoint:
        raise RuntimeError(f"{sample.sample_id}/{seed}: checkpoint fallback")
    selected = select_official_from_analysis(
        source,
        case,
        analysis,
        config=config,
        device=device,
    )
    snapshot = analysis.analytic.incumbent_snapshot
    shadow = tuple(snapshot.get("ranker_shadow_top4", ()))
    shadow_available = bool(shadow)
    metrics = _evaluate_positions(evaluator, sample, source, selected)
    if not metrics["hard_feasible"]:
        raise RuntimeError(f"{sample.sample_id}/{seed}: returned placement is infeasible")
    counterfactual = snapshot.get("ranker_selection_counterfactual")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "case_id": sample.sample_id,
        "seed": seed,
        "config_hash": config_hash,
        "checkpoint_state_hash": str(checkpoint_metadata["state_hash"]),
        "exact_source": snapshot.get("exact_source"),
        "selected_sha256": _placement_sha256(selected),
        "selected_positions": selected,
        "hard_feasible": metrics["hard_feasible"],
        "selected_metrics": metrics,
        "ranker_shadow_available": shadow_available,
        "ranker_shadow_eligible_count": int(
            snapshot.get("ranker_shadow_eligible_count", 0)
        ),
        "ranker_shadow_skipped_reason": snapshot.get("ranker_shadow_skipped_reason"),
        "ranker_shadow_failure_reason": snapshot.get("ranker_shadow_failure_reason"),
        "ranker_shadow_empty_reason": snapshot.get("ranker_shadow_empty_reason"),
        "shadow_top4": shadow,
        "ranker_selection_counterfactual": counterfactual,
        "ranker_selection_evaluated_top4": tuple(
            snapshot.get("ranker_selection_evaluated_top4", ())
        ),
    }


def _learned_config(args: argparse.Namespace, seed: int) -> LearnedConfig:
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
        collective_steps=args.collective_steps,
        tail_topk=args.tail_topk,
        seed=seed,
        topology_seeds=args.topology_seeds,
        constraint_seeds=args.constraint_seeds,
        treemap_seeds=args.treemap_seeds,
        btree_seeds=args.btree_seeds,
        ranker_selection_experiment=True,
    )


def _evaluate_positions(
    evaluator: Any,
    sample: DataSample,
    source: dict[str, Any],
    positions: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    baseline = {
        "hpwl_baseline": float(sample.labels.baseline_hpwl),
        "area_baseline": float(sample.labels.baseline_area),
    }
    metrics = evaluator.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        baseline,
        source["constraints"],
        source["b2b_connectivity"],
        source["p2b_connectivity"],
        source["pins_pos"],
        source["area_targets"],
        source["target_positions"],
        median_runtime=1.0,
    )
    return {
        "hard_feasible": bool(metrics.is_feasible),
        "cost": float(metrics.cost),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "total_soft_violations": int(metrics.total_soft_violations),
        "max_possible_violations": int(metrics.max_possible_violations),
    }


def _config_payload(
    args: argparse.Namespace,
    checkpoint: Path,
    evaluator_sha256: str,
    seeds: list[int],
    case_ids: list[str],
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "evaluator_sha256": evaluator_sha256,
        "case_ids": case_ids,
        "seeds": seeds,
        "population": args.population,
        "dynamics_steps": args.dynamics_steps,
        "projection_steps": args.projection_steps,
        "direction_beam": args.direction_beam,
        "component_bdp": args.component_bdp,
        "component_beam": args.component_beam,
        "component_limit": args.component_limit,
        "component_uncertain_pairs": args.component_uncertain_pairs,
        "component_sweeps": args.component_sweeps,
        "component_reset_limit": args.component_reset_limit,
        "topology_seeds": args.topology_seeds,
        "constraint_seeds": args.constraint_seeds,
        "treemap_seeds": args.treemap_seeds,
        "btree_seeds": args.btree_seeds,
        "flow_steps": args.flow_steps,
        "collective_steps": args.collective_steps,
        "tail_topk": args.tail_topk,
        "device": args.device,
        "audit_script_sha256": file_sha256(Path(__file__)),
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _resume_key(case_id: str, seed: int, config_hash: str) -> str:
    return f"{config_hash}:{case_id}:{seed}"


def _load_previous(path: Path, config_hash: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("resume file schema mismatch")
    if payload.get("config_hash") != config_hash:
        raise ValueError("resume config hash mismatch")
    rows = payload.get("cases", [])
    previous = {}
    for row in rows:
        key = _resume_key(str(row["case_id"]), int(row["seed"]), config_hash)
        if key in previous:
            raise ValueError("resume file contains duplicate case+seed rows")
        previous[key] = row
    return previous


def _provenance(
    command: list[str],
    checkpoint: Path,
    checkpoint_metadata: dict[str, Any],
    evaluator_path: Path,
) -> dict[str, Any]:
    return {
        "command": ["scripts/audit_hcfp_ranker_counterfactual.py", *command],
        "git": _git_provenance(),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "file_sha256": file_sha256(checkpoint),
            "state_hash": str(checkpoint_metadata["state_hash"]),
            "normalization": checkpoint_metadata["normalization"],
            "capabilities": checkpoint_metadata.get("capabilities", {}),
            "trained_heads": checkpoint_metadata.get("trained_heads", []),
        },
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": file_sha256(evaluator_path),
            "commit": OFFICIAL_FLOORSET_V10.commit,
        },
        "gpu": _gpu_provenance(),
        "env": _env_provenance(),
        "python": {
            "version": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }


def _git_provenance() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    try:
        status = run(["status", "--porcelain"])
        return {
            "commit": run(["rev-parse", "HEAD"]),
            "branch": run(["branch", "--show-current"]),
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False}


def _gpu_provenance() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    index = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(index),
        "device_capability": torch.cuda.get_device_capability(index),
        "torch_cuda": torch.version.cuda,
    }


def _env_provenance() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("HCFP_") or key in {"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"}
    }


def _summary(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    accept = [
        row
        for row in rows
        if isinstance(row.get("ranker_selection_counterfactual"), dict)
        and row["ranker_selection_counterfactual"].get("would_accept") is True
    ]
    zero_eligible_shadow = [
        row for row in rows if _row_has_zero_eligible_shadow_coverage_failure(row)
    ]
    missing_shadow = [
        row
        for row in rows
        if not _row_has_ranker_shadow(row)
        and not _row_has_zero_eligible_shadow_coverage_failure(row)
    ]
    all_hard_feasible = all(bool(row["hard_feasible"]) for row in rows)
    coverage_complete = len(rows) == expected_rows
    return {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "coverage_complete": coverage_complete,
        "would_accept": len(accept),
        "all_hard_feasible": all_hard_feasible,
        "ranker_shadow_available": len(rows) - len(missing_shadow) - len(zero_eligible_shadow),
        "ranker_shadow_missing": len(missing_shadow),
        "ranker_shadow_zero_eligible": len(zero_eligible_shadow),
        "all_ranker_shadows_available": not missing_shadow and not zero_eligible_shadow,
        "counterfactual_audit_gate_passed": (
            coverage_complete
            and all_hard_feasible
            and not missing_shadow
            and not zero_eligible_shadow
        ),
        "missing_ranker_shadow_rows": [
            {"case_id": row["case_id"], "seed": row["seed"]}
            for row in missing_shadow
        ],
        "zero_eligible_ranker_shadow_rows": [
            {"case_id": row["case_id"], "seed": row["seed"]}
            for row in zero_eligible_shadow
        ],
        "unique_selected_hashes": len({row["selected_sha256"] for row in rows}),
        "output_hashes_by_case": {
            case_id: sorted(
                {row["selected_sha256"] for row in rows if row["case_id"] == case_id}
            )
            for case_id in sorted({str(row["case_id"]) for row in rows})
        },
    }


def _row_has_ranker_shadow(row: dict[str, Any]) -> bool:
    if "ranker_shadow_available" in row:
        return bool(row["ranker_shadow_available"])
    return bool(row.get("shadow_top4"))


def _row_has_zero_eligible_shadow_coverage_failure(row: dict[str, Any]) -> bool:
    return (
        int(row.get("ranker_shadow_eligible_count", -1)) == 0
        and row.get("ranker_shadow_empty_reason") == "no_exact_eligible_candidates"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
