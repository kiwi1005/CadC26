#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.eval.official import evaluate_positions
from puzzleplace.feedback import load_policy_checkpoint
from puzzleplace.models import TypedActionPolicy
from puzzleplace.repair.finalizer import finalize_layout
from puzzleplace.rollout import semantic_rollout
from puzzleplace.train import load_validation_cases

RESEARCH_DIR = ROOT / "artifacts" / "research"
MODEL_DIR = ROOT / "artifacts" / "models"
GENERALIZATION_JSON = RESEARCH_DIR / "generalization_followup_smallcheckpoints.json"
OUTPUT_MD = RESEARCH_DIR / "best_of_k_oracle_reranking.md"
OUTPUT_JSON = RESEARCH_DIR / "best_of_k_oracle_reranking.json"

CASE_LIMIT = 20
UNTRAINED_SEEDS = tuple(range(11))
TRAINED_SPECS = (
    ("bc", 0, MODEL_DIR / "small_overfit_bc_seed0.pt"),
    ("bc", 1, MODEL_DIR / "small_overfit_bc_seed1.pt"),
    ("awbc", 0, MODEL_DIR / "small_overfit_awbc_seed0.pt"),
    ("awbc", 1, MODEL_DIR / "small_overfit_awbc_seed1.pt"),
)
COMMANDS_USED: list[str] = []


def _run_command(args: list[str]) -> None:
    COMMANDS_USED.append(" ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def _ensure_prerequisites() -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    missing_models = [str(path.relative_to(ROOT)) for _, _, path in TRAINED_SPECS if not path.exists()]
    if missing_models:
        _run_command([sys.executable, str(ROOT / "scripts" / "run_small_overfit_matrix.py")])

    if not GENERALIZATION_JSON.exists():
        _run_command([sys.executable, str(ROOT / "scripts" / "run_generalization_followup.py")])


def _load_policy(kind: str, seed: int | None, checkpoint: Path | None):
    if kind == "heuristic":
        return None
    if kind == "untrained":
        assert seed is not None
        torch.manual_seed(seed)
        return TypedActionPolicy(hidden_dim=32)
    if checkpoint is None:
        raise ValueError(f"missing checkpoint for {kind} seed {seed}")
    return load_policy_checkpoint(checkpoint)


def _histogram(actions) -> dict[str, int]:
    counts = Counter(str(action.primitive) for action in actions)
    return dict(sorted(counts.items()))


def _candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"kind": "heuristic", "seed": None, "checkpoint": None}]
    specs.extend({"kind": "untrained", "seed": seed, "checkpoint": None} for seed in UNTRAINED_SEEDS)
    specs.extend(
        {"kind": kind, "seed": seed, "checkpoint": checkpoint}
        for kind, seed, checkpoint in TRAINED_SPECS
        if checkpoint.exists()
    )
    return specs


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _evaluate_case(case, specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    candidates: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        policy = _load_policy(spec["kind"], spec["seed"], spec["checkpoint"])
        started = time.time()
        semantic = semantic_rollout(case, policy)
        repair = finalize_layout(case, semantic.proposed_positions)
        runtime_seconds = max(time.time() - started, 1e-6)
        observed = evaluate_positions(case, repair.positions, runtime=runtime_seconds)
        standardized = evaluate_positions(case, repair.positions, runtime=1.0)
        candidates.append(
            {
                "candidate_index": index,
                "case_id": str(case.case_id),
                "source_kind": spec["kind"],
                "seed": spec["seed"],
                "checkpoint": (
                    str(spec["checkpoint"].relative_to(ROOT)) if spec["checkpoint"] is not None else None
                ),
                "semantic_completed": semantic.semantic_completed,
                "semantic_placed_fraction": semantic.semantic_placed_fraction,
                "fallback_fraction": semantic.fallback_fraction,
                "action_histogram": _histogram(semantic.actions),
                "hard_feasible": repair.report.hard_feasible_after,
                "repair_displacement": repair.report.mean_displacement,
                "max_displacement": repair.report.max_displacement,
                "moved_block_count": repair.report.moved_block_count,
                "changed_block_fraction": repair.report.moved_block_count / max(case.block_count, 1),
                "shelf_fallback_count": repair.report.shelf_fallback_count,
                "fallback_used": bool(
                    semantic.fallback_fraction > 0.0 or repair.report.shelf_fallback_count > 0
                ),
                "runtime_seconds": runtime_seconds,
                "official_cost": float(observed["official"]["cost"]),
                "official_cost_runtime1": float(standardized["official"]["cost"]),
                "hpwl_gap": float(observed["official"]["hpwl_gap"]),
                "hpwl_b2b": float(observed["official"]["hpwl_b2b"]),
                "hpwl_p2b": float(observed["official"]["hpwl_p2b"]),
                "hpwl_total": float(observed["official"]["hpwl_total"]),
                "area_gap": float(observed["official"]["area_gap"]),
                "bbox_area": float(observed["official"]["bbox_area"]),
                "bbox_area_baseline": float(observed["official"]["bbox_area_baseline"]),
                "violations_relative": float(observed["official"]["violations_relative"]),
                "total_soft_violations": int(observed["official"]["total_soft_violations"]),
                "boundary_violations": int(observed["official"]["boundary_violations"]),
                "grouping_violations": int(observed["official"]["grouping_violations"]),
                "mib_violations": int(observed["official"]["mib_violations"]),
            }
        )

    hpwl_scores = [float(candidate["hpwl_gap"]) + float(candidate["area_gap"]) for candidate in candidates]
    hpwl_norm = _min_max_normalize([float(candidate["hpwl_gap"]) for candidate in candidates])
    area_norm = _min_max_normalize([float(candidate["area_gap"]) for candidate in candidates])
    viol_norm = _min_max_normalize([float(candidate["violations_relative"]) for candidate in candidates])
    displacement_scores = [float(candidate["repair_displacement"]) for candidate in candidates]
    soft_scores = [float(candidate["violations_relative"]) for candidate in candidates]
    combined_scores = [
        0.5 * (hpwl_norm[idx] + area_norm[idx]) + viol_norm[idx] for idx in range(len(candidates))
    ]

    selectors = {
        "oracle_best": min(range(len(candidates)), key=lambda idx: float(candidates[idx]["official_cost"])),
        "displacement_best": min(range(len(candidates)), key=lambda idx: displacement_scores[idx]),
        "hpwl_bbox_proxy_best": min(range(len(candidates)), key=lambda idx: hpwl_scores[idx]),
        "soft_violation_best": min(range(len(candidates)), key=lambda idx: soft_scores[idx]),
        "combined_proxy_best": min(range(len(candidates)), key=lambda idx: combined_scores[idx]),
    }
    for idx, candidate in enumerate(candidates):
        candidate["proxy_hpwl_bbox"] = hpwl_scores[idx]
        candidate["proxy_combined"] = combined_scores[idx]

    summary = {
        "best_of_k_cost": min(float(candidate["official_cost"]) for candidate in candidates),
        "mean_of_k_cost": _mean([float(candidate["official_cost"]) for candidate in candidates]),
        "median_of_k_cost": statistics.median(
            [float(candidate["official_cost"]) for candidate in candidates]
        ),
        "std_of_k_cost": _std([float(candidate["official_cost"]) for candidate in candidates]),
        "best_candidate_source": str(candidates[selectors["oracle_best"]]["source_kind"]),
        "best_candidate_seed": candidates[selectors["oracle_best"]]["seed"],
    }
    return candidates, {"selectors": selectors, "summary": summary}


def _load_baselines() -> dict[str, Any]:
    payload = json.loads(GENERALIZATION_JSON.read_text())
    rows = payload["rows"]
    best_untrained = min(
        (row for row in rows if row["variant"] == "untrained"),
        key=lambda row: float(row["avg_official_cost"]),
    )
    best_trained = min(
        (row for row in rows if row["variant"] in {"bc", "awbc"}),
        key=lambda row: float(row["avg_official_cost"]),
    )
    return {
        "best_untrained": {
            "variant": best_untrained["variant"],
            "seed": best_untrained["seed"],
            "avg_official_cost": float(best_untrained["avg_official_cost"]),
            "cases": {str(item["case_id"]): float(item["official_cost"]) for item in best_untrained["cases"]},
        },
        "best_trained": {
            "variant": best_trained["variant"],
            "seed": best_trained["seed"],
            "avg_official_cost": float(best_trained["avg_official_cost"]),
            "cases": {str(item["case_id"]): float(item["official_cost"]) for item in best_trained["cases"]},
        },
    }


def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": candidate["source_kind"],
        "seed": candidate["seed"],
        "official_cost": candidate["official_cost"],
        "official_cost_runtime1": candidate["official_cost_runtime1"],
        "hpwl_gap": candidate["hpwl_gap"],
        "area_gap": candidate["area_gap"],
        "violations_relative": candidate["violations_relative"],
        "repair_displacement": candidate["repair_displacement"],
        "fallback_used": candidate["fallback_used"],
    }


def _driver(candidate: dict[str, Any]) -> str:
    if bool(candidate["fallback_used"]):
        return "finalizer_fallback"
    ranked = [
        ("hpwl", float(candidate["hpwl_gap"])),
        ("bbox", float(candidate["area_gap"])),
        ("soft", float(candidate["violations_relative"])),
    ]
    return max(ranked, key=lambda item: item[1])[0]


def main() -> None:
    _ensure_prerequisites()
    specs = _candidate_specs()
    if len(specs) < 16:
        raise ValueError(f"expected at least 16 candidates, found {len(specs)}")

    cases = load_validation_cases(case_limit=CASE_LIMIT)
    baselines = _load_baselines()

    per_case_candidates: dict[str, list[dict[str, Any]]] = {}
    per_case_best_by_rule: dict[str, dict[str, Any]] = {}
    per_case_rollup: list[dict[str, Any]] = []
    rule_costs: dict[str, list[float]] = {
        "oracle_best": [],
        "displacement_best": [],
        "hpwl_bbox_proxy_best": [],
        "soft_violation_best": [],
        "combined_proxy_best": [],
    }
    rule_costs_runtime1 = {rule: [] for rule in rule_costs}
    source_wins = {rule: Counter() for rule in rule_costs}

    for case in cases:
        case_id = str(case.case_id)
        candidates, selection = _evaluate_case(case, specs)
        selectors = selection["selectors"]
        per_case_candidates[case_id] = candidates
        per_case_best_by_rule[case_id] = {
            rule: _candidate_brief(candidates[idx]) for rule, idx in selectors.items()
        }

        for rule, idx in selectors.items():
            rule_costs[rule].append(float(candidates[idx]["official_cost"]))
            rule_costs_runtime1[rule].append(float(candidates[idx]["official_cost_runtime1"]))
            source_wins[rule][str(candidates[idx]["source_kind"])] += 1

        oracle = candidates[selectors["oracle_best"]]
        untrained_baseline = float(baselines["best_untrained"]["cases"][case_id])
        trained_baseline = float(baselines["best_trained"]["cases"][case_id])
        per_case_rollup.append(
            {
                "case_id": case_id,
                "best_of_k_cost": selection["summary"]["best_of_k_cost"],
                "mean_of_k_cost": selection["summary"]["mean_of_k_cost"],
                "median_of_k_cost": selection["summary"]["median_of_k_cost"],
                "std_of_k_cost": selection["summary"]["std_of_k_cost"],
                "best_candidate_source": selection["summary"]["best_candidate_source"],
                "best_candidate_seed": selection["summary"]["best_candidate_seed"],
                "gap_to_best_untrained_baseline": float(oracle["official_cost"]) - untrained_baseline,
                "gap_to_best_trained_baseline": float(oracle["official_cost"]) - trained_baseline,
                "remaining_bad_case_driver": _driver(oracle),
            }
        )

    aggregate_by_rule: dict[str, dict[str, Any]] = {}
    for rule, costs in rule_costs.items():
        untrained_deltas = [
            costs[idx] - float(baselines["best_untrained"]["cases"][row["case_id"]])
            for idx, row in enumerate(per_case_rollup)
        ]
        trained_deltas = [
            costs[idx] - float(baselines["best_trained"]["cases"][row["case_id"]])
            for idx, row in enumerate(per_case_rollup)
        ]
        improved_untrained = [delta for delta in untrained_deltas if delta < 0.0]
        degraded_untrained = [delta for delta in untrained_deltas if delta >= 0.0]
        aggregate_by_rule[rule] = {
            "mean_cost": _mean(costs),
            "mean_cost_runtime1": _mean(rule_costs_runtime1[rule]),
            "median_cost": statistics.median(costs),
            "std_cost": _std(costs),
            "win_rate_vs_best_untrained": sum(delta < 0.0 for delta in untrained_deltas)
            / max(len(untrained_deltas), 1),
            "win_rate_vs_best_trained": sum(delta < 0.0 for delta in trained_deltas)
            / max(len(trained_deltas), 1),
            "number_of_cases_where_rule_improves_vs_best_untrained": sum(
                delta < 0.0 for delta in untrained_deltas
            ),
            "average_improvement_on_improved_cases_vs_best_untrained": (
                abs(_mean(improved_untrained)) if improved_untrained else 0.0
            ),
            "average_degradation_on_non_improved_cases_vs_best_untrained": (
                _mean(degraded_untrained) if degraded_untrained else 0.0
            ),
            "winner_source_breakdown": dict(sorted(source_wins[rule].items())),
        }

    oracle_summary = aggregate_by_rule["oracle_best"]
    displacement_summary = aggregate_by_rule["displacement_best"]
    hpwl_proxy_summary = aggregate_by_rule["hpwl_bbox_proxy_best"]
    bad_cases = [row for row in per_case_rollup if row["gap_to_best_untrained_baseline"] >= 0.0]
    improved_cases = [row for row in per_case_rollup if row["gap_to_best_untrained_baseline"] < 0.0]

    if oracle_summary["mean_cost"] < float(baselines["best_untrained"]["avg_official_cost"]):
        primary_recommendation = (
            "Oracle best-of-K beats the best untrained baseline, so the next step should be a "
            "post-finalizer cost-aware reranker / scorer rather than more displacement-centric tuning."
        )
    else:
        primary_recommendation = (
            "Oracle best-of-K does not beat the best untrained baseline on mean official cost, so the "
            "candidate/finalizer family still looks ceiling-limited and should be redesigned before training a scorer."
        )

    if hpwl_proxy_summary["mean_cost"] <= oracle_summary["mean_cost"] + 0.5:
        proxy_takeaway = (
            "The HPWL+bbox proxy stays close to oracle, so a cheap quality proxy looks viable inside the rollout loop."
        )
    else:
        proxy_takeaway = (
            "The HPWL+bbox proxy leaves noticeable gap to oracle, so a learned reranker is more justified than a hand proxy alone."
        )

    if displacement_summary["win_rate_vs_best_untrained"] < hpwl_proxy_summary["win_rate_vs_best_untrained"]:
        displacement_takeaway = (
            "Displacement remains a poor scorer; keep it as a secondary regularizer, not the primary selection objective."
        )
    else:
        displacement_takeaway = (
            "Displacement is not completely dominated here, but it still lags oracle and should not be treated as sufficient by itself."
        )

    result = {
        "experiment": {
            "case_limit": CASE_LIMIT,
            "candidate_count": len(specs),
            "k_values_run": [len(specs)],
            "did_not_run_k32_reason": "Bounded run kept K=16-equivalent because measured per-candidate runtime is ~1.7s on this machine.",
            "candidate_specs": [
                {
                    "kind": spec["kind"],
                    "seed": spec["seed"],
                    "checkpoint": (
                        str(spec["checkpoint"].relative_to(ROOT))
                        if spec["checkpoint"] is not None
                        else None
                    ),
                }
                for spec in specs
            ],
        },
        "baseline_rows": baselines,
        "per_case_candidates": per_case_candidates,
        "per_case_best_by_rule": per_case_best_by_rule,
        "per_case_rollup": per_case_rollup,
        "aggregate_by_rule": aggregate_by_rule,
        "oracle_vs_baseline": {
            "best_untrained_avg_official_cost": baselines["best_untrained"]["avg_official_cost"],
            "best_trained_avg_official_cost": baselines["best_trained"]["avg_official_cost"],
            "oracle_best_avg_official_cost": oracle_summary["mean_cost"],
            "oracle_best_avg_official_cost_runtime1": oracle_summary["mean_cost_runtime1"],
            "delta_oracle_minus_best_untrained": oracle_summary["mean_cost"]
            - float(baselines["best_untrained"]["avg_official_cost"]),
            "delta_oracle_minus_best_trained": oracle_summary["mean_cost"]
            - float(baselines["best_trained"]["avg_official_cost"]),
            "oracle_improved_cases": [row["case_id"] for row in improved_cases],
            "oracle_non_improved_cases": [row["case_id"] for row in bad_cases],
        },
        "final_recommendation": {
            "diagnosis": primary_recommendation,
            "proxy_takeaway": proxy_takeaway,
            "displacement_takeaway": displacement_takeaway,
            "remaining_bad_case_drivers": dict(
                sorted(Counter(row["remaining_bad_case_driver"] for row in bad_cases).items())
            ),
            "top_k_diversity_dominant_source": dict(
                sorted(aggregate_by_rule["oracle_best"]["winner_source_breakdown"].items())
            ),
            "commands_used": COMMANDS_USED + [f"{sys.executable} {Path(__file__).relative_to(ROOT)}"],
        },
        "commands_used": COMMANDS_USED + [f"{sys.executable} {Path(__file__).relative_to(ROOT)}"],
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Best-of-K Oracle / Reranking Ceiling",
        "",
        "## Executive Summary",
        f"- Candidate count (`K`): `{len(specs)}` (`heuristic + 11 untrained random initializations + {len(specs) - 12} trained checkpoints`).",
        f"- Best untrained baseline from `generalization_followup_smallcheckpoints.json`: `{baselines['best_untrained']['variant']} seed {baselines['best_untrained']['seed']}` with mean official cost `{baselines['best_untrained']['avg_official_cost']:.3f}`.",
        f"- Best trained baseline: `{baselines['best_trained']['variant']} seed {baselines['best_trained']['seed']}` with mean official cost `{baselines['best_trained']['avg_official_cost']:.3f}`.",
        f"- Oracle best mean official cost: `{oracle_summary['mean_cost']:.3f}` (runtime-standardized mean `{oracle_summary['mean_cost_runtime1']:.3f}`).",
        f"- {primary_recommendation}",
        f"- {proxy_takeaway}",
        f"- {displacement_takeaway}",
        "",
        "## Per-case Oracle Table",
        "",
        "| Case | best-of-K | mean-of-K | std | best source | seed | Δ vs best untrained | Δ vs best trained | Driver |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in per_case_rollup:
        lines.append(
            f"| {row['case_id']} | {row['best_of_k_cost']:.3f} | {row['mean_of_k_cost']:.3f} | "
            f"{row['std_of_k_cost']:.3f} | {row['best_candidate_source']} | "
            f"{row['best_candidate_seed'] if row['best_candidate_seed'] is not None else '-'} | "
            f"{row['gap_to_best_untrained_baseline']:.3f} | {row['gap_to_best_trained_baseline']:.3f} | "
            f"{row['remaining_bad_case_driver']} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Oracle Table",
            "",
            "| Rule | Mean cost | Mean cost (runtime=1) | Median | Std | Win-rate vs best untrained | Win-rate vs best trained | Winner sources |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rule, summary in aggregate_by_rule.items():
        lines.append(
            f"| {rule} | {summary['mean_cost']:.3f} | {summary['mean_cost_runtime1']:.3f} | "
            f"{summary['median_cost']:.3f} | {summary['std_cost']:.3f} | "
            f"{summary['win_rate_vs_best_untrained']:.3f} | {summary['win_rate_vs_best_trained']:.3f} | "
            f"`{summary['winner_source_breakdown']}` |"
        )

    lines.extend(
        [
            "",
            "## Scorer Comparison",
            f"- `oracle_best` mean cost delta vs best untrained: `{result['oracle_vs_baseline']['delta_oracle_minus_best_untrained']:.3f}`.",
            f"- `oracle_best` mean cost delta vs best trained: `{result['oracle_vs_baseline']['delta_oracle_minus_best_trained']:.3f}`.",
            f"- Oracle improved cases vs best untrained: `{result['oracle_vs_baseline']['oracle_improved_cases']}`.",
            f"- Oracle non-improved cases vs best untrained: `{result['oracle_vs_baseline']['oracle_non_improved_cases']}`.",
            f"- Remaining bad-case drivers: `{result['final_recommendation']['remaining_bad_case_drivers']}`.",
            "",
            "## Decision Recommendation",
            f"- {primary_recommendation}",
            f"- {proxy_takeaway}",
            f"- {displacement_takeaway}",
            "- Immediate next code files to change if reranking proceeds: `src/puzzleplace/rollout/semantic.py` (candidate scoring hook), `src/puzzleplace/optimizer/contest.py` (selection/evaluation path), and a new scorer dataset/training script alongside `scripts/run_generalization_followup.py`.",
            "- Experiments to avoid for now: using repair displacement alone as the selection metric, or expanding validation breadth before the scorer/finalizer direction is chosen.",
            "",
            "## Commands Used",
        ]
    )
    lines.extend([f"- `{command}`" for command in result["commands_used"]])
    OUTPUT_MD.write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "cases": len(cases),
                "candidate_count": len(specs),
                "oracle_mean_cost": oracle_summary["mean_cost"],
                "baseline_mean_cost": baselines["best_untrained"]["avg_official_cost"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
