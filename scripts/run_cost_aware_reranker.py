#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
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
from puzzleplace.repair import finalize_layout
from puzzleplace.rollout import semantic_rollout
from puzzleplace.train import load_validation_cases

RESEARCH_DIR = ROOT / "artifacts" / "research"
MODEL_DIR = ROOT / "artifacts" / "models"
GENERALIZATION_JSON = RESEARCH_DIR / "generalization_followup_smallcheckpoints.json"

DEFAULT_SCORERS = (
    "hpwl_bbox_proxy",
    "hpwl_bbox_soft_proxy",
    "hpwl_bbox_soft_repair_proxy",
    "displacement_proxy",
    "oracle_official_cost",
)
OBJECTIVE_PROXY_SCORERS = {
    "hpwl_bbox_proxy",
    "hpwl_bbox_soft_proxy",
    "hpwl_bbox_soft_repair_proxy",
}

CandidateSpec = dict[str, Any]
CandidateRow = dict[str, Any]


def _average(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _candidate_id(kind: str, seed: int | None) -> str:
    if kind == "heuristic":
        return "heuristic"
    if seed is None:
        return kind
    return f"{kind}_seed{seed}"


def build_candidate_specs(k: int) -> list[CandidateSpec]:
    """Return the reproducible candidate source portfolio.

    The K=16 shape intentionally matches the existing Step5 v0 artifact:
    heuristic + untrained seeds 0..10 + bc/awbc seeds 0..1.
    For larger K, keep the four trained checkpoints and fill the extra slots
    with additional untrained seeds. For small smoke K, use heuristic plus
    untrained seeds only to avoid expensive checkpoint prerequisites.
    """

    if k < 1:
        raise ValueError("k must be >= 1")

    trained_specs: list[CandidateSpec] = [
        {
            "candidate_id": _candidate_id(kind, seed),
            "kind": kind,
            "seed": seed,
            "checkpoint": str(
                (MODEL_DIR / f"small_overfit_{kind}_seed{seed}.pt").relative_to(ROOT)
            ),
        }
        for kind in ("bc", "awbc")
        for seed in (0, 1)
    ]

    specs: list[CandidateSpec] = [
        {"candidate_id": "heuristic", "kind": "heuristic", "seed": None, "checkpoint": None}
    ]
    trained_count = len(trained_specs) if k >= 16 else 0
    untrained_count = max(k - len(specs) - trained_count, 0)
    specs.extend(
        {
            "candidate_id": _candidate_id("untrained", seed),
            "kind": "untrained",
            "seed": seed,
            "checkpoint": None,
        }
        for seed in range(untrained_count)
    )
    specs.extend(trained_specs[: max(k - len(specs), 0)])
    return specs[:k]


def _run_prerequisite(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def ensure_prerequisites(specs: Iterable[CandidateSpec]) -> list[str]:
    commands: list[str] = []
    missing_checkpoints = []
    for spec in specs:
        checkpoint = spec.get("checkpoint")
        if checkpoint is None:
            continue
        checkpoint_path = ROOT / str(checkpoint)
        if not checkpoint_path.exists():
            missing_checkpoints.append(checkpoint_path)
    if missing_checkpoints:
        command = [sys.executable, str(ROOT / "scripts" / "run_small_overfit_matrix.py")]
        commands.append(" ".join(command))
        _run_prerequisite(command)

    if not GENERALIZATION_JSON.exists():
        command = [sys.executable, str(ROOT / "scripts" / "run_generalization_followup.py")]
        commands.append(" ".join(command))
        _run_prerequisite(command)
    return commands


def _load_policy(spec: CandidateSpec):
    kind = str(spec["kind"])
    seed = spec.get("seed")
    if kind == "heuristic":
        return None
    if kind == "untrained":
        if seed is None:
            raise ValueError("untrained candidate requires seed")
        torch.manual_seed(int(seed))
        return TypedActionPolicy(hidden_dim=32)
    checkpoint = spec.get("checkpoint")
    if checkpoint is None:
        raise ValueError(f"{kind} candidate requires checkpoint")
    return load_policy_checkpoint(ROOT / str(checkpoint))


def _action_histogram(actions: Sequence[Any]) -> dict[str, int]:
    counts = Counter(str(action.primitive) for action in actions)
    return dict(sorted(counts.items()))


def _bbox_features(positions: Sequence[tuple[float, float, float, float]]) -> dict[str, float]:
    if not positions:
        return {
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_area": 0.0,
            "total_block_area": 0.0,
            "whitespace_ratio": 0.0,
        }
    x0 = min(float(x) for x, _y, _w, _h in positions)
    y0 = min(float(y) for _x, y, _w, _h in positions)
    x1 = max(float(x + w) for x, _y, w, _h in positions)
    y1 = max(float(y + h) for _x, y, _w, h in positions)
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    bbox_area = width * height
    total_block_area = sum(max(float(w), 0.0) * max(float(h), 0.0) for _x, _y, w, h in positions)
    whitespace = (bbox_area - total_block_area) / bbox_area if bbox_area > 0 else 0.0
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": bbox_area,
        "total_block_area": total_block_area,
        "whitespace_ratio": whitespace,
    }


def evaluate_candidate(case: Any, spec: CandidateSpec) -> CandidateRow:
    policy = _load_policy(spec)
    started = time.time()
    semantic = semantic_rollout(case, policy)
    repair = finalize_layout(case, semantic.proposed_positions)
    runtime_seconds = max(time.time() - started, 1e-6)
    observed = evaluate_positions(case, repair.positions, runtime=runtime_seconds)
    runtime1 = evaluate_positions(case, repair.positions, runtime=1.0)
    official = observed["official"]
    official_runtime1 = runtime1["official"]
    bbox = _bbox_features(repair.positions)
    proxy_features = {
        **bbox,
        "proxy_hpwl_b2b": float(official["hpwl_b2b"]),
        "proxy_hpwl_p2b": float(official["hpwl_p2b"]),
        "proxy_hpwl_total": float(official["hpwl_total"]),
        "proxy_boundary_violations": float(official["boundary_violations"]),
        "proxy_grouping_violations": float(official["grouping_violations"]),
        "proxy_mib_violations": float(official["mib_violations"]),
        "proxy_total_soft_violations": float(official["total_soft_violations"]),
        "changed_block_fraction": repair.report.moved_block_count / max(case.block_count, 1),
        "shelf_fallback_count": float(repair.report.shelf_fallback_count),
        "mean_displacement": float(repair.report.mean_displacement),
        "semantic_fallback_fraction": float(semantic.fallback_fraction),
        "hard_feasible_after": bool(repair.report.hard_feasible_after),
    }
    analysis_metrics = {
        "official_cost": float(official["cost"]),
        "official_cost_runtime1": float(official_runtime1["cost"]),
        "hpwl_gap": float(official["hpwl_gap"]),
        "hpwl_b2b": float(official["hpwl_b2b"]),
        "hpwl_p2b": float(official["hpwl_p2b"]),
        "hpwl_total": float(official["hpwl_total"]),
        "area_gap": float(official["area_gap"]),
        "bbox_area": float(official["bbox_area"]),
        "violations_relative": float(official["violations_relative"]),
        "runtime_factor": official.get("runtime_factor"),
        "is_feasible": bool(official["is_feasible"]),
        "total_soft_violations": int(official["total_soft_violations"]),
        "boundary_violations": int(official["boundary_violations"]),
        "grouping_violations": int(official["grouping_violations"]),
        "mib_violations": int(official["mib_violations"]),
        "runtime_seconds": runtime_seconds,
        "semantic_completed": bool(semantic.semantic_completed),
        "semantic_placed_fraction": float(semantic.semantic_placed_fraction),
        "semantic_fallback_fraction": float(semantic.fallback_fraction),
        "action_count": len(semantic.actions),
        "action_histogram": _action_histogram(semantic.actions),
    }
    return {
        "candidate_id": spec["candidate_id"],
        "kind": spec["kind"],
        "seed": spec.get("seed"),
        "checkpoint": spec.get("checkpoint"),
        "proxy_features": proxy_features,
        "analysis_metrics": analysis_metrics,
    }


def _feature_values(rows: Sequence[CandidateRow], key: str) -> list[float]:
    return [float(row["proxy_features"][key]) for row in rows]


def score_candidate_rows(rows: Sequence[CandidateRow], scorer_name: str) -> list[float]:
    if scorer_name == "oracle_official_cost":
        return [float(row["analysis_metrics"]["official_cost"]) for row in rows]
    if scorer_name == "displacement_proxy":
        return _normalize(_feature_values(rows, "mean_displacement"))
    if scorer_name not in OBJECTIVE_PROXY_SCORERS:
        raise ValueError(f"unknown scorer: {scorer_name}")

    hpwl = _normalize(_feature_values(rows, "proxy_hpwl_total"))
    bbox = _normalize(_feature_values(rows, "bbox_area"))
    scores = [h + b for h, b in zip(hpwl, bbox, strict=True)]
    if scorer_name in {"hpwl_bbox_soft_proxy", "hpwl_bbox_soft_repair_proxy"}:
        soft = _normalize(_feature_values(rows, "proxy_total_soft_violations"))
        scores = [score + 0.25 * value for score, value in zip(scores, soft, strict=True)]
    if scorer_name == "hpwl_bbox_soft_repair_proxy":
        changed = _normalize(_feature_values(rows, "changed_block_fraction"))
        shelf = _normalize(_feature_values(rows, "shelf_fallback_count"))
        semantic_fallback = _normalize(_feature_values(rows, "semantic_fallback_fraction"))
        scores = [
            score + 0.05 * changed_i + 0.05 * shelf_i + 0.05 * fallback_i
            for score, changed_i, shelf_i, fallback_i in zip(
                scores, changed, shelf, semantic_fallback, strict=True
            )
        ]
    return scores


def _rank_indices(values: Sequence[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: (float(values[idx]), idx))


def summarize_scorer_for_case(
    *,
    case_id: str,
    rows: Sequence[CandidateRow],
    scorer_name: str,
    best_untrained_cost: float,
    best_trained_cost: float,
    top_m_values: Sequence[int],
) -> tuple[dict[str, Any], dict[int, bool], list[str]]:
    scores = score_candidate_rows(rows, scorer_name)
    scored_order = _rank_indices(scores)
    official_costs = [float(row["analysis_metrics"]["official_cost"]) for row in rows]
    official_order = _rank_indices(official_costs)
    selected_idx = scored_order[0]
    oracle_idx = official_order[0]
    selected = rows[selected_idx]
    metrics = selected["analysis_metrics"]
    proxy = selected["proxy_features"]
    oracle_candidate_id = str(rows[oracle_idx]["candidate_id"])
    rank_of_selected = official_order.index(selected_idx) + 1
    recall = {
        int(m): oracle_idx in scored_order[: min(int(m), len(scored_order))]
        for m in top_m_values
    }
    row = {
        "case_id": case_id,
        "scorer_name": scorer_name,
        "selected_candidate_id": selected["candidate_id"],
        "selected_index": selected_idx,
        "selected_score": float(scores[selected_idx]),
        "official_cost": float(metrics["official_cost"]),
        "oracle_official_cost": float(official_costs[oracle_idx]),
        "oracle_candidate_id": oracle_candidate_id,
        "oracle_rank_of_selected": rank_of_selected,
        "oracle_rank_by_scorer": scored_order.index(oracle_idx) + 1,
        "regret_to_oracle": float(metrics["official_cost"]) - float(official_costs[oracle_idx]),
        "best_untrained_cost": best_untrained_cost,
        "best_trained_cost": best_trained_cost,
        "is_feasible": bool(metrics["is_feasible"]),
        "hpwl_gap": float(metrics["hpwl_gap"]),
        "area_gap": float(metrics["area_gap"]),
        "violations_relative": float(metrics["violations_relative"]),
        "runtime_factor": metrics.get("runtime_factor"),
        "changed_block_fraction": float(proxy["changed_block_fraction"]),
        "shelf_fallback_count": float(proxy["shelf_fallback_count"]),
        "semantic_fallback_fraction": float(proxy["semantic_fallback_fraction"]),
        "source_kind": selected["kind"],
        "seed": selected.get("seed"),
    }
    top_order_ids = [str(rows[idx]["candidate_id"]) for idx in scored_order]
    return row, recall, top_order_ids


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
            "cases": {
                str(item["case_id"]): float(item["official_cost"])
                for item in best_untrained["cases"]
            },
        },
        "best_trained": {
            "variant": best_trained["variant"],
            "seed": best_trained["seed"],
            "avg_official_cost": float(best_trained["avg_official_cost"]),
            "cases": {
                str(item["case_id"]): float(item["official_cost"])
                for item in best_trained["cases"]
            },
        },
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    costs = [float(row["official_cost"]) for row in rows]
    regrets = [float(row["regret_to_oracle"]) for row in rows]
    untrained_deltas = [
        float(row["official_cost"]) - float(row["best_untrained_cost"]) for row in rows
    ]
    trained_deltas = [
        float(row["official_cost"]) - float(row["best_trained_cost"]) for row in rows
    ]
    runtime_factors = [row.get("runtime_factor") for row in rows]
    numeric_runtime = [float(value) for value in runtime_factors if value is not None]
    return {
        "mean_official_cost": _average(costs),
        "median_official_cost": statistics.median(costs) if costs else 0.0,
        "std_official_cost": _std(costs),
        "mean_regret_to_oracle": _average(regrets),
        "median_regret_to_oracle": statistics.median(regrets) if regrets else 0.0,
        "oracle_hit_count": sum(math.isclose(regret, 0.0, abs_tol=1e-9) for regret in regrets),
        "win_rate_vs_best_untrained": sum(delta < 0.0 for delta in untrained_deltas)
        / max(len(untrained_deltas), 1),
        "win_rate_vs_best_trained": sum(delta < 0.0 for delta in trained_deltas)
        / max(len(trained_deltas), 1),
        "feasibility_rate": _average([1.0 if row["is_feasible"] else 0.0 for row in rows]),
        "mean_hpwl_gap": _average([float(row["hpwl_gap"]) for row in rows]),
        "mean_area_gap": _average([float(row["area_gap"]) for row in rows]),
        "mean_violations_relative": _average([float(row["violations_relative"]) for row in rows]),
        "mean_changed_block_fraction": _average(
            [float(row["changed_block_fraction"]) for row in rows]
        ),
        "mean_shelf_fallback_count": _average(
            [float(row["shelf_fallback_count"]) for row in rows]
        ),
        "mean_semantic_fallback_fraction": _average(
            [float(row["semantic_fallback_fraction"]) for row in rows]
        ),
        "mean_runtime_factor": _average(numeric_runtime) if numeric_runtime else None,
    }


def _select_cases(case_ids: Sequence[int]) -> list[Any]:
    if not case_ids:
        raise ValueError("at least one case id is required")
    loaded = load_validation_cases(case_limit=max(case_ids) + 1)
    by_index = {idx: case for idx, case in enumerate(loaded)}
    missing = [case_id for case_id in case_ids if case_id not in by_index]
    if missing:
        raise ValueError(f"requested validation case ids were not loaded: {missing}")
    return [by_index[case_id] for case_id in case_ids]


def run_experiment(
    *,
    case_ids: Sequence[int],
    k: int,
    scorers: Sequence[str],
    top_m_values: Sequence[int],
    ensure: bool = True,
) -> dict[str, Any]:
    started = time.time()
    specs = build_candidate_specs(k)
    prerequisite_commands = ensure_prerequisites(specs) if ensure else []
    baselines = _load_baselines()
    cases = _select_cases(case_ids)

    candidate_features: dict[str, list[CandidateRow]] = {}
    selected_candidates: dict[str, list[dict[str, Any]]] = {scorer: [] for scorer in scorers}
    per_case_by_scorer: dict[str, list[dict[str, Any]]] = {scorer: [] for scorer in scorers}
    top_m_recall_counts: dict[str, dict[int, int]] = {
        scorer: {int(m): 0 for m in top_m_values} for scorer in scorers
    }
    top_order_by_scorer: dict[str, dict[str, list[str]]] = {scorer: {} for scorer in scorers}

    for case in cases:
        case_id = str(case.case_id)
        rows = [evaluate_candidate(case, spec) for spec in specs]
        candidate_features[case_id] = rows
        best_untrained_cost = float(baselines["best_untrained"]["cases"][case_id])
        best_trained_cost = float(baselines["best_trained"]["cases"][case_id])
        for scorer in scorers:
            summary, recall, top_order = summarize_scorer_for_case(
                case_id=case_id,
                rows=rows,
                scorer_name=scorer,
                best_untrained_cost=best_untrained_cost,
                best_trained_cost=best_trained_cost,
                top_m_values=top_m_values,
            )
            per_case_by_scorer[scorer].append(summary)
            selected_candidates[scorer].append(
                {
                    "case_id": case_id,
                    "candidate_id": summary["selected_candidate_id"],
                    "kind": summary["source_kind"],
                    "seed": summary["seed"],
                    "checkpoint": rows[int(summary["selected_index"])].get("checkpoint"),
                }
            )
            for m, hit in recall.items():
                top_m_recall_counts[scorer][m] += int(hit)
            top_order_by_scorer[scorer][case_id] = top_order

    aggregate_by_scorer = {
        scorer: _aggregate_rows(per_case_by_scorer[scorer]) for scorer in scorers
    }
    top_m_recall_by_scorer = {
        scorer: {
            str(m): top_m_recall_counts[scorer][m] / max(len(cases), 1)
            for m in sorted(top_m_recall_counts[scorer])
        }
        for scorer in scorers
    }

    best_proxy_name = min(
        (scorer for scorer in scorers if scorer != "oracle_official_cost"),
        key=lambda scorer: float(aggregate_by_scorer[scorer]["mean_official_cost"]),
    )
    oracle_mean = float(aggregate_by_scorer["oracle_official_cost"]["mean_official_cost"])
    best_proxy_mean = float(aggregate_by_scorer[best_proxy_name]["mean_official_cost"])
    result = {
        "experiment": {
            "case_ids": list(case_ids),
            "k_requested": k,
            "candidate_count": len(specs),
            "scorers": list(scorers),
            "top_m_values": list(top_m_values),
            "elapsed_seconds": time.time() - started,
        },
        "candidate_pool_spec": {
            "source_specs": specs,
            "selection_policy": (
                "K=16 matches Step5 v0: heuristic + untrained seeds 0..10 + bc/awbc seeds 0..1. "
                "K>=16 keeps trained checkpoints and fills extra slots with untrained seeds; "
                "K<16 is smoke-only."
            ),
        },
        "baseline_rows": baselines,
        "aggregate_by_scorer": aggregate_by_scorer,
        "top_m_recall_by_scorer": top_m_recall_by_scorer,
        "per_case_by_scorer": per_case_by_scorer,
        "selected_candidates": selected_candidates,
        "top_order_by_scorer": top_order_by_scorer,
        "candidate_features": candidate_features,
        "oracle_comparison": {
            "best_untrained_avg_official_cost": baselines["best_untrained"]["avg_official_cost"],
            "best_trained_avg_official_cost": baselines["best_trained"]["avg_official_cost"],
            "oracle_mean_official_cost": oracle_mean,
            "best_proxy_name": best_proxy_name,
            "best_proxy_mean_official_cost": best_proxy_mean,
            "best_proxy_regret_to_oracle": best_proxy_mean - oracle_mean,
        },
        "commands_used": prerequisite_commands,
        "missing_metrics": ["runtime_factor"],
        "final_recommendation": _recommendation(
            aggregate_by_scorer=aggregate_by_scorer,
            top_m_recall_by_scorer=top_m_recall_by_scorer,
            best_proxy_name=best_proxy_name,
        ),
    }
    return result


def _recommendation(
    *,
    aggregate_by_scorer: dict[str, dict[str, Any]],
    top_m_recall_by_scorer: dict[str, dict[str, float]],
    best_proxy_name: str,
) -> dict[str, Any]:
    best_proxy = aggregate_by_scorer[best_proxy_name]
    best_untrained_win_rate = float(best_proxy["win_rate_vs_best_untrained"])
    top4 = top_m_recall_by_scorer.get(best_proxy_name, {}).get("4", 0.0)
    if top4 >= 0.75:
        next_step = "train_or_test_two_stage_pairwise_reranker_on_proxy_top_m"
    else:
        next_step = "improve_proxy_features_or_candidate_portfolio_before_learned_reranker"
    if best_untrained_win_rate > 0.5:
        recommendation = (
            f"{best_proxy_name} beats the best untrained baseline on most cases; keep the "
            "objective-aware selector path and use top-M recall to decide whether to add "
            "a learned reranker."
        )
    else:
        recommendation = (
            f"{best_proxy_name} is not stable enough against best untrained; improve "
            "proxy features "
            "or candidate diversity before integrating it as the default selector."
        )
    return {
        "best_proxy_name": best_proxy_name,
        "recommendation": recommendation,
        "top_m_gate_next_step": next_step,
        "top4_oracle_recall": top4,
    }


def write_outputs(payload: dict[str, Any], output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    md_path = output.with_suffix(".md")
    md_path.write_text(render_markdown(payload))
    return output, md_path


def render_markdown(payload: dict[str, Any]) -> str:
    experiment = payload["experiment"]
    aggregate = payload["aggregate_by_scorer"]
    oracle = payload["oracle_comparison"]
    top_m = payload["top_m_recall_by_scorer"]
    best_proxy = oracle["best_proxy_name"]
    lines = [
        "# Cost-Aware Reranker / Oracle-Recall Audit",
        "",
        "## Executive Summary",
        f"- Cases: `{experiment['case_ids']}`",
        "- Requested K: "
        f"`{experiment['k_requested']}`; resolved candidate count: "
        f"`{experiment['candidate_count']}`.",
        "- Best untrained baseline mean official cost: "
        f"`{oracle['best_untrained_avg_official_cost']:.3f}`.",
        "- Best trained baseline mean official cost: "
        f"`{oracle['best_trained_avg_official_cost']:.3f}`.",
        f"- Oracle mean official cost: `{oracle['oracle_mean_official_cost']:.3f}`.",
        f"- Best proxy scorer: `{best_proxy}` with mean official cost "
        f"`{oracle['best_proxy_mean_official_cost']:.3f}` and regret "
        f"`{oracle['best_proxy_regret_to_oracle']:.3f}`.",
        f"- Recommendation: {payload['final_recommendation']['recommendation']}",
        "",
        "## Scorer Comparison Table",
        "",
        "| Scorer | Mean cost | Median | Std | Mean regret | Oracle hits | "
        "Win vs best untrained | Win vs best trained | Feasible rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scorer, summary in aggregate.items():
        lines.append(
            f"| {scorer} | {summary['mean_official_cost']:.3f} | "
            f"{summary['median_official_cost']:.3f} | {summary['std_official_cost']:.3f} | "
            f"{summary['mean_regret_to_oracle']:.3f} | {summary['oracle_hit_count']} | "
            f"{summary['win_rate_vs_best_untrained']:.3f} | "
            f"{summary['win_rate_vs_best_trained']:.3f} | {summary['feasibility_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top-M Oracle Recall",
            "",
            "This is the gate for whether a two-stage learned reranker is worth trying: "
            "if oracle candidates are usually inside proxy top-M, a learned stage can "
            "focus on local reordering; otherwise proxy features or candidate diversity "
            "need work first.",
            "",
            "| Scorer | " + " | ".join(f"top-{m}" for m in experiment["top_m_values"]) + " |",
            "| --- | " + " | ".join("---:" for _ in experiment["top_m_values"]) + " |",
        ]
    )
    for scorer, recalls in top_m.items():
        lines.append(
            f"| {scorer} | "
            + " | ".join(f"{float(recalls[str(m)]):.3f}" for m in experiment["top_m_values"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-case Table for Best Proxy",
            "",
            "| Case | Selected | Cost | Oracle | Regret | Oracle rank by scorer | "
            "Official rank of selected |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["per_case_by_scorer"][best_proxy]:
        lines.append(
            f"| {row['case_id']} | {row['selected_candidate_id']} | "
            f"{row['official_cost']:.3f} | {row['oracle_official_cost']:.3f} | "
            f"{row['regret_to_oracle']:.3f} | {row['oracle_rank_by_scorer']} | "
            f"{row['oracle_rank_of_selected']} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Pool",
            "",
            "| Candidate ID | Kind | Seed | Checkpoint |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for spec in payload["candidate_pool_spec"]["source_specs"]:
        lines.append(
            f"| {spec['candidate_id']} | {spec['kind']} | "
            f"{spec['seed'] if spec['seed'] is not None else '-'} | "
            f"{spec['checkpoint'] if spec['checkpoint'] is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Commands Used",
        ]
    )
    for command in payload.get("commands_used", []):
        lines.append(f"- `{command}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate objective-aware cost proxy selectors over finalized candidate pools and "
            "report selected-vs-oracle plus top-M oracle recall."
        )
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=list(range(20)))
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--scorers", nargs="*", choices=DEFAULT_SCORERS, default=list(DEFAULT_SCORERS)
    )
    parser.add_argument("--top-m", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH_DIR / "cost_aware_reranker_v0.json",
        help="JSON output path; markdown is written next to it with .md suffix.",
    )
    parser.add_argument(
        "--no-ensure-prerequisites",
        action="store_true",
        help="Do not auto-run prerequisite training/generalization scripts if inputs are missing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_experiment(
        case_ids=args.case_ids,
        k=args.k,
        scorers=args.scorers,
        top_m_values=args.top_m,
        ensure=not args.no_ensure_prerequisites,
    )
    output_path = ROOT / args.output if not args.output.is_absolute() else args.output
    json_path, md_path = write_outputs(payload, output_path)
    print(
        json.dumps(
            {
                "json": str(
                    json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path
                ),
                "markdown": str(
                    md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path
                ),
                "best_proxy": payload["oracle_comparison"]["best_proxy_name"],
                "best_proxy_mean": payload["oracle_comparison"]["best_proxy_mean_official_cost"],
                "oracle_mean": payload["oracle_comparison"]["oracle_mean_official_cost"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
