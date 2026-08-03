#!/usr/bin/env python3
"""Resume-safe Q6 full-case evaluation aggregation for HCFP reports."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from statistics import fmean
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.benchmark import percentile, weighted_score  # noqa: E402
from hcfp.cap_margin import attribute_record  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402


AGGREGATOR_SHA256 = file_sha256(Path(__file__))


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", type=_named_path, required=True)
    parser.add_argument("--ranker-eval", type=Path)
    parser.add_argument("--checkpoint", action="append", type=_named_path, default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    _reject_duplicate_names(args.benchmark, "benchmark")
    _reject_duplicate_names(args.checkpoint, "checkpoint")
    device = _resolve_device(args.device)
    output = Path(args.output)
    previous = _load_previous(output) if args.resume and output.exists() else None
    benchmarks = {}
    for name, path in args.benchmark:
        benchmarks[name] = _benchmark_entry(
            name,
            path,
            previous=_previous_benchmark(previous, name, path),
        )
    ranker = _ranker_entry(args.ranker_eval) if args.ranker_eval is not None else None
    checkpoints = {
        name: _checkpoint_probe(name, path, device=device)
        for name, path in args.checkpoint
    }
    report = {
        "schema_version": 1,
        "training_performed": False,
        "official_validation_training_guard": (
            "This orchestrator is eval-only: it loads benchmark/eval reports and checkpoint bytes; "
            "it never writes training shards or invokes training scripts."
        ),
        "provenance": _provenance(
            argv=sys.argv[1:] if argv is None else argv,
            device=device,
            benchmark_names=sorted(benchmarks),
            checkpoint_names=sorted(checkpoints),
            ranker_eval=args.ranker_eval,
            resume=bool(args.resume),
        ),
        "inputs": {
            "benchmark_names": sorted(benchmarks),
            "ranker_eval": str(args.ranker_eval) if args.ranker_eval is not None else None,
            "checkpoint_names": sorted(checkpoints),
            "resume": bool(args.resume),
        },
        "benchmarks": benchmarks,
        "ranker_eval": ranker,
        "checkpoints": checkpoints,
        "summary": _overall_summary(benchmarks, ranker, checkpoints),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output, report)
    print(output)
    return 0


def _reject_duplicate_names(values: list[tuple[str, Path]], noun: str) -> None:
    names = [name for name, _path in values]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {noun} names: {duplicates}")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available")
    return device


def _load_previous(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("resume file schema mismatch")
    return payload


def _previous_benchmark(previous: dict[str, Any] | None, name: str, path: Path) -> dict[str, Any] | None:
    if previous is None:
        return None
    if previous.get("provenance", {}).get("aggregator_sha256") != AGGREGATOR_SHA256:
        return None
    entry = previous.get("benchmarks", {}).get(name)
    if not isinstance(entry, dict):
        return None
    if entry.get("path") == str(path) and entry.get("sha256") == file_sha256(path):
        return entry
    return None


def _benchmark_entry(name: str, path: Path, *, previous: dict[str, Any] | None) -> dict[str, Any]:
    if previous is not None:
        item = dict(previous)
        item["resume_status"] = "reused"
        return item
    payload = json.loads(path.read_text(encoding="utf-8"))
    lanes = _extract_lanes(payload)
    lane_entries = {
        lane: _lane_summary(rows)
        for lane, rows in sorted(lanes.items())
    }
    baseline = payload.get("baseline")
    return {
        "name": name,
        "path": str(path),
        "sha256": file_sha256(path),
        "resume_status": "computed",
        "source_schema": _source_schema(payload),
        "baseline": baseline,
        "case_count": _case_count(lanes),
        "lanes": lane_entries,
        "comparisons": payload.get("comparisons", {}),
        "strict_gates": _strict_gates(lanes, payload.get("comparisons", {}), baseline),
        "promotion_decisions": payload.get("promotion_decisions", {}),
        "source_provenance": payload.get("provenance", {}),
        "limitations": _benchmark_limitations(lanes),
    }


def _extract_lanes(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    lanes = payload.get("lanes")
    if isinstance(lanes, dict):
        return {str(name): _ordered_rows(rows) for name, rows in lanes.items()}
    rows = payload.get("test_results", payload.get("results"))
    if isinstance(rows, list):
        return {"result": _ordered_rows(rows)}
    raise ValueError("benchmark input must contain lanes, test_results, or results")


def _ordered_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark lane rows must be a non-empty list")
    required = {
        "test_id",
        "block_count",
        "hpwl_gap",
        "area_gap",
        "violations_relative",
        "runtime_seconds",
        "cost",
    }
    out = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("benchmark row must be an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"benchmark row is missing {sorted(missing)}")
        item = dict(row)
        if "hard_feasible" not in item:
            if "is_feasible" not in item:
                raise ValueError("benchmark row is missing hard_feasible/is_feasible")
            if type(item["is_feasible"]) is not bool:
                raise ValueError("benchmark row feasibility must be boolean")
            item["hard_feasible"] = item["is_feasible"]
        elif type(item["hard_feasible"]) is not bool:
            raise ValueError("benchmark row feasibility must be boolean")
        elif "is_feasible" in item and type(item["is_feasible"]) is not bool:
            raise ValueError("benchmark row feasibility must be boolean")
        elif "is_feasible" in item and item["hard_feasible"] != item["is_feasible"]:
            raise ValueError("benchmark row feasibility fields disagree")
        item["is_feasible"] = item.get("is_feasible", item["hard_feasible"])
        for field in ("hpwl_gap", "area_gap", "violations_relative", "runtime_seconds", "cost"):
            try:
                value = float(item[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"benchmark row {field} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"benchmark row {field} must be finite")
        out.append(item)
    ordered = sorted(out, key=lambda row: int(row["test_id"]))
    if len({int(row["test_id"]) for row in ordered}) != len(ordered):
        raise ValueError("benchmark lane contains duplicate test_id")
    return ordered


def _source_schema(payload: dict[str, Any]) -> str:
    if "lanes" in payload:
        return "benchmark_report"
    if "test_results" in payload:
        return "official_results"
    if "results" in payload:
        return "results"
    return "unknown"


def _case_count(lanes: dict[str, list[dict[str, Any]]]) -> int:
    counts = {len(rows) for rows in lanes.values()}
    if len(counts) != 1:
        raise ValueError("benchmark lanes must have matching case counts")
    return next(iter(counts))


def _lane_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attributed = [
        attribute_record(row, default_runtime_factor=1.0)
        for row in rows
    ]
    runtimes = sorted(float(row["runtime_seconds"]) for row in rows)
    hard = sum(bool(row["hard_feasible"]) for row in rows)
    cap_cross = sum(bool(row["hard_feasible"]) and float(attr["cap_margin"]) > 0.0 for row, attr in zip(rows, attributed))
    large = [row for row in rows if 106 <= int(row["block_count"]) <= 120]
    large_attr = [
        attr for row, attr in zip(rows, attributed) if 106 <= int(row["block_count"]) <= 120
    ]
    return {
        "cases": len(rows),
        "hard_feasible": hard,
        "hard_feasibility_rate": hard / len(rows),
        "weighted_capped_cost": weighted_score(rows),
        "mean_log_uncapped_j": fmean(float(attr["log_uncapped_cost"]) for attr in attributed),
        "weighted_log_uncapped_j": _weighted_mean(
            [(int(row["block_count"]), float(attr["log_uncapped_cost"])) for row, attr in zip(rows, attributed)]
        ),
        "cap_cross_count": cap_cross,
        "cap_cross_rate": cap_cross / len(rows),
        "capped_or_infeasible_count": len(rows) - cap_cross,
        "runtime_p50": percentile(runtimes, 0.50),
        "runtime_p95": percentile(runtimes, 0.95),
        "runtime_max": max(runtimes),
        "subset_106_120": {
            "cases": len(large),
            "hard_feasible": sum(bool(row["hard_feasible"]) for row in large),
            "weighted_capped_cost": weighted_score(large),
            "weighted_log_uncapped_j": _weighted_mean(
                [(int(row["block_count"]), float(attr["log_uncapped_cost"])) for row, attr in zip(large, large_attr)]
            ) if large else None,
            "cap_cross_count": sum(
                bool(row["hard_feasible"]) and float(attr["cap_margin"]) > 0.0
                for row, attr in zip(large, large_attr)
            ),
            "hard_feasibility_rate": (
                sum(bool(row["hard_feasible"]) for row in large) / len(large)
                if large
                else None
            ),
        },
    }


def _strict_gates(
    lanes: dict[str, list[dict[str, Any]]],
    comparisons: Any,
    baseline: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(baseline, str) or baseline not in lanes:
        return {}
    if not isinstance(comparisons, dict):
        comparisons = {}
    baseline_large = [row for row in lanes[baseline] if 106 <= int(row["block_count"]) <= 120]
    out = {}
    for lane, rows in sorted(lanes.items()):
        if lane == baseline:
            continue
        comparison = comparisons.get(lane, {})
        if not isinstance(comparison, dict):
            comparison = {}
        hard_met = all(bool(row["hard_feasible"]) for row in rows)
        regressed = _regressed_cases(lanes[baseline], rows)
        reported_regressed = comparison.get("regressed_cases")
        if reported_regressed is not None and int(reported_regressed) != regressed:
            raise ValueError("benchmark comparison regression count disagrees with lane rows")
        regression_met = regressed == 0
        candidate_large = [row for row in rows if 106 <= int(row["block_count"]) <= 120]
        large_met: bool | None
        large_delta: float | None
        if not baseline_large or not candidate_large:
            large_met = None
            large_delta = None
        else:
            large_delta = weighted_score(candidate_large) - weighted_score(baseline_large)
            large_met = large_delta <= 1.0e-9
        passed = hard_met and regression_met and large_met is True
        out[lane] = {
            "passed": passed,
            "hard_feasibility_100_met": hard_met,
            "regressed_cases_zero_met": regression_met,
            "regressed_cases": regressed,
            "large_subset_nonregression_met": large_met,
            "large_subset_weighted_cost_delta": large_delta,
            "large_subset_cases": len(candidate_large),
            "baseline_large_subset_cases": len(baseline_large),
            "source": "benchmark_comparisons_plus_recomputed_large_subset",
            "reason": _strict_gate_reason(hard_met, regression_met, large_met),
        }
    return out


def _regressed_cases(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> int:
    if [int(row["test_id"]) for row in baseline] != [int(row["test_id"]) for row in candidate]:
        raise ValueError("benchmark lanes must have matching test ids for strict gate")
    return sum(
        float(current["cost"]) > float(base["cost"]) + 1.0e-9
        for base, current in zip(baseline, candidate)
    )


def _strict_gate_reason(
    hard_met: bool,
    regression_met: bool,
    large_met: bool | None,
) -> str:
    if hard_met and regression_met and large_met is True:
        return "pass"
    if large_met is None:
        return "no_106_120_subset_to_prove_large_nonregression"
    failed = []
    if not hard_met:
        failed.append("hard_feasibility")
    if not regression_met:
        failed.append("per_case_regression")
    if large_met is False:
        failed.append("large_subset_regression")
    return ",".join(failed)


def _weighted_mean(values: list[tuple[int, float]]) -> float:
    if not values:
        return 0.0
    weights = [math.exp(blocks / 12.0) for blocks, _ in values]
    return sum(value * weight for (_, value), weight in zip(values, weights)) / sum(weights)


def _benchmark_limitations(lanes: dict[str, list[dict[str, Any]]]) -> list[str]:
    limitations = []
    if any("runtime_factor" not in row for rows in lanes.values() for row in rows):
        limitations.append("Rows do not expose official cross-submission runtime_factor; local J uses runtime_factor=1.")
    if any(
        not {"boundary_violations", "grouping_violations", "mib_violations", "max_possible_violations"} <= row.keys()
        for rows in lanes.values()
        for row in rows
    ):
        limitations.append("Rows expose only violations_relative; boundary/group/MIB attribution is unavailable.")
    return limitations


def _ranker_entry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("ranker eval schema_version must be 2")
    cases = _ranker_cases(payload)
    if not cases:
        raise ValueError("ranker eval contains no cases")
    top1 = sum(bool(case.get("top1_exact_best")) for case in cases)
    top4 = sum(bool(case.get("top4_oracle_recall")) for case in cases)
    false = sum(bool(case.get("false_promotion")) for case in cases)
    regrets = [float(case.get("score_regret", 0.0)) for case in cases]
    rank_regrets = [float(case.get("rank_regret", 0.0)) for case in cases]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "cases": len(cases),
        "oracle_at_1": top1,
        "oracle_at_4": top4,
        "oracle_at_1_rate": top1 / len(cases),
        "oracle_at_4_rate": top4 / len(cases),
        "false_promotion": false,
        "mean_selected_score_regret": fmean(regrets),
        "mean_selected_rank_regret": fmean(rank_regrets),
        "p95_selected_rank_regret": _nearest_percentile(sorted(rank_regrets), 0.95),
        "by_stage": _ranker_by_stage(cases),
    }


def _ranker_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("ranker eval is missing results")
    cases = []
    for split in results.values():
        if not isinstance(split, dict):
            continue
        for checkpoint in split.values():
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("cases"), list):
                cases.extend(checkpoint["cases"])
    return cases


def _ranker_by_stage(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for stage in sorted({str(case.get("candidate_stage", "unknown")) for case in cases}):
        rows = [case for case in cases if str(case.get("candidate_stage", "unknown")) == stage]
        out[stage] = {
            "cases": len(rows),
            "oracle_at_1": sum(bool(row.get("top1_exact_best")) for row in rows),
            "oracle_at_4": sum(bool(row.get("top4_oracle_recall")) for row in rows),
            "false_promotion": sum(bool(row.get("false_promotion")) for row in rows),
            "oracle_at_1_rate": (
                sum(bool(row.get("top1_exact_best")) for row in rows) / len(rows)
            ),
            "oracle_at_4_rate": (
                sum(bool(row.get("top4_oracle_recall")) for row in rows) / len(rows)
            ),
            "mean_selected_rank_regret": fmean(float(row.get("rank_regret", 0.0)) for row in rows),
        }
    return out


def _nearest_percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[min(len(sorted_values) - 1, rank - 1)]


def _checkpoint_probe(name: str, path: Path, *, device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    model, metadata = load_checkpoint(
        path,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    load_seconds = time.perf_counter() - load_start
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats()
    transfer_start = time.perf_counter()
    model = model.to(device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_start
    total_seconds = time.perf_counter() - total_start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    is_a100 = gpu_name is not None and "A100" in gpu_name.upper()
    return {
        "name": name,
        "path": str(path),
        "sha256": file_sha256(path),
        "state_hash": metadata["state_hash"],
        "parent_state_hash": metadata.get("parent_state_hash"),
        "device": str(device),
        "cpu_file_load_seconds": load_seconds,
        "device_transfer_seconds": transfer_seconds,
        "cold_load_total_seconds": total_seconds,
        "peak_cuda_memory_bytes": peak,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "a100_profile_available": is_a100,
        "a100_profile_gap": (
            None if is_a100 else "A100 metrics require running this script on an A100 host."
        ),
    }


def _overall_summary(
    benchmarks: dict[str, dict[str, Any]],
    ranker: dict[str, Any] | None,
    checkpoints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lane_count = sum(len(entry["lanes"]) for entry in benchmarks.values())
    cases = sum(int(entry["case_count"]) for entry in benchmarks.values())
    strict_gates = [
        gate
        for entry in benchmarks.values()
        for gate in entry.get("strict_gates", {}).values()
    ]
    execution_seeds = {
        seed
        for entry in benchmarks.values()
        if (seed := _benchmark_execution_seed(entry)) is not None
    }
    shadow_blockers = []
    if len(benchmarks) < 3:
        shadow_blockers.append("fewer_than_three_benchmark_reports")
    if len(execution_seeds) < 3:
        shadow_blockers.append("fewer_than_three_distinct_execution_seeds")
    if not strict_gates:
        shadow_blockers.append("no_candidate_strict_gates")
    elif not all(bool(gate.get("passed")) for gate in strict_gates):
        shadow_blockers.append("benchmark_strict_gate_failure")
    shadow_gate = {
        "passed": not shadow_blockers,
        "blockers": shadow_blockers,
        "benchmark_reports": len(benchmarks),
        "distinct_execution_seeds": sorted(execution_seeds),
        "candidate_strict_gates": len(strict_gates),
    }

    ranker_blockers = []
    initial_stage = None if ranker is None else ranker.get("by_stage", {}).get("initial")
    if ranker is None:
        ranker_blockers.append("ranker_eval_missing")
    elif not isinstance(initial_stage, dict):
        ranker_blockers.append("initial_stage_missing")
    else:
        if int(initial_stage["cases"]) < 16:
            ranker_blockers.append("initial_stage_fewer_than_16_cases")
        if float(initial_stage["oracle_at_1_rate"]) < 12.0 / 16.0:
            ranker_blockers.append("initial_oracle_at_1_rate_below_12_of_16")
        if float(initial_stage["oracle_at_4_rate"]) < 15.0 / 16.0:
            ranker_blockers.append("initial_oracle_at_4_rate_below_15_of_16")
        if int(initial_stage["false_promotion"]) != 0:
            ranker_blockers.append("initial_false_promotion_nonzero")
    ranker_gate = {
        "passed": not ranker_blockers,
        "blockers": ranker_blockers,
        "stage": "initial",
        "observed": initial_stage,
        "required": {
            "minimum_cases": 16,
            "oracle_at_1_rate": 12.0 / 16.0,
            "oracle_at_4_rate": 15.0 / 16.0,
            "false_promotion": 0,
        },
    }

    a100_profiles = [
        name
        for name, checkpoint in checkpoints.items()
        if bool(checkpoint.get("a100_profile_available"))
    ]
    freeze_blockers = []
    if not shadow_gate["passed"]:
        freeze_blockers.append("shadow_validation_gate_failed")
    if not ranker_gate["passed"]:
        freeze_blockers.append("ranker_quality_gate_failed")
    if not a100_profiles:
        freeze_blockers.append("a100_profile_missing")

    summary = {
        "benchmark_reports": len(benchmarks),
        "benchmark_case_evaluations": cases,
        "lane_evaluations": lane_count,
        "ranker_eval_present": ranker is not None,
        "q6_shadow_validation_gate": shadow_gate,
        "ranker_quality_gate": ranker_gate,
        "submission_freeze_gate": {
            "passed": not freeze_blockers,
            "blockers": freeze_blockers,
            "a100_profile_checkpoints": a100_profiles,
            "active_ranker_selection_proven": False,
            "active_ranker_selection_gap": (
                "Current Q5 checkpoint and runtime remain counterfactual-only."
            ),
        },
    }
    if ranker is not None:
        summary["ranker_oracle_at_1"] = ranker["oracle_at_1"]
        summary["ranker_oracle_at_4"] = ranker["oracle_at_4"]
    return summary


def _benchmark_execution_seed(entry: dict[str, Any]) -> int | None:
    provenance = entry.get("source_provenance", {})
    if not isinstance(provenance, dict):
        return None
    search = provenance.get("search_config", {})
    if not isinstance(search, dict) or search.get("execution_seed") is None:
        return None
    try:
        return int(search["execution_seed"])
    except (TypeError, ValueError):
        return None


def _provenance(
    *,
    argv: list[str],
    device: torch.device,
    benchmark_names: list[str],
    checkpoint_names: list[str],
    ranker_eval: Path | None,
    resume: bool,
) -> dict[str, Any]:
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "git_status_porcelain": _git(["status", "--porcelain"]),
        "command": " ".join([sys.executable, str(Path(__file__)), *argv]),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "benchmark_names": benchmark_names,
        "checkpoint_names": checkpoint_names,
        "ranker_eval": str(ranker_eval) if ranker_eval is not None else None,
        "resume": resume,
        "aggregator_sha256": AGGREGATOR_SHA256,
        "hcfp_environment": _hcfp_environment(),
    }


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hcfp_environment() -> dict[str, str]:
    relevant = (
        "TOPOLOGY",
        "CONSTRAINT",
        "RANKER",
        "CHECKPOINT",
        "FLOW",
        "COLLECTIVE",
        "TAIL",
        "BDP",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("HCFP_") and any(token in key for token in relevant)
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
