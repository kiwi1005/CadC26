#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

_COST_RUNNER = Path(__file__).resolve().with_name("run_cost_aware_reranker.py")
_SPEC = importlib.util.spec_from_file_location("run_cost_aware_reranker", _COST_RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_COST_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COST_MODULE)
score_candidate_rows = _COST_MODULE.score_candidate_rows

RESEARCH_DIR = ROOT / "artifacts" / "research"
DEFAULT_INPUT = RESEARCH_DIR / "cost_aware_reranker_v0.json"
DEFAULT_OUTPUT = RESEARCH_DIR / "two_stage_reranker_loco_v0.json"

FEATURE_KEYS = (
    ("proxy_features", "proxy_hpwl_total"),
    ("proxy_features", "bbox_area"),
    ("proxy_features", "proxy_total_soft_violations"),
    ("proxy_features", "changed_block_fraction"),
    ("proxy_features", "shelf_fallback_count"),
    ("proxy_features", "mean_displacement"),
    ("proxy_features", "semantic_fallback_fraction"),
    ("proxy_features", "whitespace_ratio"),
)

CandidateRow = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainedLinearRanker:
    weights: torch.Tensor
    bias: torch.Tensor
    feature_names: list[str]

    def scores(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weights + self.bias


def _cost(row: CandidateRow) -> float:
    return float(row["analysis_metrics"]["official_cost"])


def _feature_name(key: tuple[str, str]) -> str:
    return f"{key[0]}.{key[1]}"


def _raw_feature_matrix(rows: Sequence[CandidateRow]) -> torch.Tensor:
    values: list[list[float]] = []
    for row in rows:
        values.append([float(row[group][name]) for group, name in FEATURE_KEYS])
    if not values:
        return torch.empty((0, len(FEATURE_KEYS)), dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


def normalized_feature_matrix(rows: Sequence[CandidateRow]) -> torch.Tensor:
    raw = _raw_feature_matrix(rows)
    if raw.numel() == 0:
        return raw
    lo = raw.min(dim=0).values
    hi = raw.max(dim=0).values
    denom = torch.where(torch.isclose(hi, lo), torch.ones_like(hi), hi - lo)
    return (raw - lo) / denom


def proxy_order(rows: Sequence[CandidateRow], scorer_name: str) -> list[int]:
    scores = score_candidate_rows(rows, scorer_name)
    return sorted(range(len(rows)), key=lambda idx: (float(scores[idx]), idx))


def train_pairwise_ranker(
    train_cases: dict[str, list[CandidateRow]],
    *,
    scorer_name: str,
    top_m: int,
    epochs: int = 600,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
) -> TrainedLinearRanker:
    feature_chunks: list[torch.Tensor] = []
    pair_left: list[int] = []
    pair_right: list[int] = []
    offset = 0
    for rows in train_cases.values():
        order = proxy_order(rows, scorer_name)[: min(top_m, len(rows))]
        features = normalized_feature_matrix(rows)[order]
        costs = [_cost(rows[idx]) for idx in order]
        feature_chunks.append(features)
        for i, left_cost in enumerate(costs):
            for j, right_cost in enumerate(costs):
                if left_cost + 1e-9 < right_cost:
                    pair_left.append(offset + i)
                    pair_right.append(offset + j)
        offset += len(order)

    if not feature_chunks:
        raise ValueError("cannot train two-stage reranker without training candidates")
    all_features = torch.cat(feature_chunks, dim=0)
    feature_dim = all_features.shape[1]
    weights = torch.zeros(feature_dim, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    if not pair_left:
        return TrainedLinearRanker(
            weights.detach(),
            bias.detach(),
            [_feature_name(key) for key in FEATURE_KEYS],
        )

    left_idx = torch.tensor(pair_left, dtype=torch.long)
    right_idx = torch.tensor(pair_right, dtype=torch.long)
    optimizer = torch.optim.Adam([weights, bias], lr=lr, weight_decay=weight_decay)
    for _epoch in range(epochs):
        optimizer.zero_grad()
        scores = all_features @ weights + bias
        # Lower score should mean lower official cost. For a better left item,
        # penalize score_left >= score_right.
        loss = F.softplus(scores[left_idx] - scores[right_idx]).mean()
        loss.backward()
        optimizer.step()

    return TrainedLinearRanker(
        weights.detach(),
        bias.detach(),
        [_feature_name(key) for key in FEATURE_KEYS],
    )


def _select_by_ranker(
    rows: Sequence[CandidateRow],
    *,
    scorer_name: str,
    top_m: int,
    ranker: TrainedLinearRanker,
) -> tuple[int, list[int], list[float]]:
    shortlist = proxy_order(rows, scorer_name)[: min(top_m, len(rows))]
    features = normalized_feature_matrix(rows)[shortlist]
    learned_scores = ranker.scores(features).tolist()
    selected_local = min(range(len(shortlist)), key=lambda idx: (learned_scores[idx], idx))
    return shortlist[selected_local], shortlist, [float(score) for score in learned_scores]


def _oracle_index(rows: Sequence[CandidateRow]) -> int:
    return min(range(len(rows)), key=lambda idx: (_cost(rows[idx]), idx))


def evaluate_loco(
    candidate_features: dict[str, list[CandidateRow]],
    *,
    scorer_name: str,
    top_m: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    final_weights: list[list[float]] = []
    for heldout_case, rows in candidate_features.items():
        train_cases = {
            case_id: case_rows
            for case_id, case_rows in candidate_features.items()
            if case_id != heldout_case
        }
        ranker = train_pairwise_ranker(
            train_cases,
            scorer_name=scorer_name,
            top_m=top_m,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        final_weights.append([float(value) for value in ranker.weights.tolist()])
        selected_idx, shortlist, learned_scores = _select_by_ranker(
            rows,
            scorer_name=scorer_name,
            top_m=top_m,
            ranker=ranker,
        )
        proxy_idx = proxy_order(rows, scorer_name)[0]
        oracle_idx = _oracle_index(rows)
        row = rows[selected_idx]
        proxy_row = rows[proxy_idx]
        oracle_row = rows[oracle_idx]
        per_case.append(
            {
                "case_id": heldout_case,
                "selected_candidate_id": row["candidate_id"],
                "selected_cost": _cost(row),
                "proxy_candidate_id": proxy_row["candidate_id"],
                "proxy_cost": _cost(proxy_row),
                "oracle_candidate_id": oracle_row["candidate_id"],
                "oracle_cost": _cost(oracle_row),
                "regret_to_oracle": _cost(row) - _cost(oracle_row),
                "delta_vs_proxy": _cost(row) - _cost(proxy_row),
                "oracle_in_top_m": oracle_idx in shortlist,
                "shortlist": [rows[idx]["candidate_id"] for idx in shortlist],
                "learned_scores": learned_scores,
            }
        )
    return {
        "per_case": per_case,
        "weight_mean": _mean_columns(final_weights),
        "feature_names": [_feature_name(key) for key in FEATURE_KEYS],
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _mean_columns(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        return []
    width = len(rows[0])
    return [_mean([float(row[idx]) for row in rows]) for idx in range(width)]


def _aggregate(per_case: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected_costs = [float(row["selected_cost"]) for row in per_case]
    proxy_costs = [float(row["proxy_cost"]) for row in per_case]
    oracle_costs = [float(row["oracle_cost"]) for row in per_case]
    regrets = [float(row["regret_to_oracle"]) for row in per_case]
    proxy_deltas = [float(row["delta_vs_proxy"]) for row in per_case]
    return {
        "mean_selected_cost": _mean(selected_costs),
        "median_selected_cost": sorted(selected_costs)[len(selected_costs) // 2]
        if selected_costs
        else 0.0,
        "std_selected_cost": _std(selected_costs),
        "mean_proxy_cost": _mean(proxy_costs),
        "mean_oracle_cost": _mean(oracle_costs),
        "mean_regret_to_oracle": _mean(regrets),
        "mean_delta_vs_proxy": _mean(proxy_deltas),
        "oracle_hit_count": sum(math.isclose(value, 0.0, abs_tol=1e-9) for value in regrets),
        "improved_vs_proxy_count": sum(value < -1e-9 for value in proxy_deltas),
        "worse_than_proxy_count": sum(value > 1e-9 for value in proxy_deltas),
        "oracle_in_top_m_rate": _mean([1.0 if row["oracle_in_top_m"] else 0.0 for row in per_case]),
    }


def run(
    *,
    input_path: Path,
    scorer_name: str,
    top_m: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    candidate_features = source["candidate_features"]
    result = evaluate_loco(
        candidate_features,
        scorer_name=scorer_name,
        top_m=top_m,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    aggregate = _aggregate(result["per_case"])
    if aggregate["mean_delta_vs_proxy"] < 0:
        recommendation = "two_stage_loco_improves_proxy_continue_to_guarded_top_m_integration"
    else:
        recommendation = (
            "two_stage_loco_does_not_improve_proxy_keep_proxy_default_and_iterate_features"
        )
    return {
        "experiment": {
            "source_artifact": str(
                input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path
            ),
            "scorer_name": scorer_name,
            "top_m": top_m,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "case_count": len(candidate_features),
        },
        "aggregate": aggregate,
        "feature_names": result["feature_names"],
        "mean_loco_weights": result["weight_mean"],
        "per_case": result["per_case"],
        "recommendation": recommendation,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    experiment = payload["experiment"]
    aggregate = payload["aggregate"]
    lines = [
        "# Two-Stage Pairwise Reranker LOCO Audit",
        "",
        "## Executive Summary",
        f"- Source artifact: `{experiment['source_artifact']}`",
        f"- Stage-1 scorer: `{experiment['scorer_name']}`",
        f"- Top-M shortlist: `{experiment['top_m']}`",
        f"- Case count: `{experiment['case_count']}`",
        f"- Proxy mean cost: `{aggregate['mean_proxy_cost']:.3f}`",
        f"- Two-stage LOCO mean cost: `{aggregate['mean_selected_cost']:.3f}`",
        f"- Oracle mean cost: `{aggregate['mean_oracle_cost']:.3f}`",
        f"- Mean delta vs proxy: `{aggregate['mean_delta_vs_proxy']:.3f}`",
        f"- Oracle hit count: `{aggregate['oracle_hit_count']}`",
        f"- Recommendation: `{payload['recommendation']}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in aggregate.items():
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.3f} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Per-case LOCO Selection",
            "",
            "| Case | Two-stage | Cost | Proxy | Proxy cost | Oracle | Oracle cost | "
            "Δ proxy | Oracle in top-M |",
            "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["per_case"]:
        lines.append(
            f"| {row['case_id']} | {row['selected_candidate_id']} | "
            f"{row['selected_cost']:.3f} | {row['proxy_candidate_id']} | "
            f"{row['proxy_cost']:.3f} | {row['oracle_candidate_id']} | "
            f"{row['oracle_cost']:.3f} | {row['delta_vs_proxy']:.3f} | "
            f"{row['oracle_in_top_m']} |"
        )
    lines.extend(
        [
            "",
            "## Mean LOCO Weights",
            "",
            "| Feature | Mean weight |",
            "| --- | ---: |",
        ]
    )
    for feature, weight in zip(
        payload["feature_names"], payload["mean_loco_weights"], strict=True
    ):
        lines.append(f"| `{feature}` | {float(weight):.4f} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a top-M two-stage pairwise reranker with LOCO validation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scorer-name", default="hpwl_bbox_soft_repair_proxy")
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = ROOT / args.input if not args.input.is_absolute() else args.input
    output_path = ROOT / args.output if not args.output.is_absolute() else args.output
    payload = run(
        input_path=input_path,
        scorer_name=args.scorer_name,
        top_m=args.top_m,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_markdown(payload))
    print(
        json.dumps(
            {
                "json": str(output_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
                "mean_selected_cost": payload["aggregate"]["mean_selected_cost"],
                "mean_proxy_cost": payload["aggregate"]["mean_proxy_cost"],
                "recommendation": payload["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
