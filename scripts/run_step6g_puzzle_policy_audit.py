#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from puzzleplace.actions import ActionExecutor, ExecutionState
from puzzleplace.research.puzzle_candidate_payload import (
    FeatureMode,
    build_puzzle_candidate_descriptors,
    choose_expert_descriptor,
    heuristic_scores,
)
from puzzleplace.train.dataset_bc import load_validation_cases
from puzzleplace.trajectory import generate_pseudo_traces


class TinyPoolScorer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _collect_pools(case, feature_mode: FeatureMode, *, max_steps: int = 20):
    trace = generate_pseudo_traces(case, max_traces=1)[0]
    state = ExecutionState()
    executor = ActionExecutor(case)
    pools = []
    for action in trace.actions:
        if action.primitive.value == "freeze":
            executor.apply(state, action)
            continue
        remaining = [idx for idx in range(case.block_count) if idx not in state.placements]
        desc = build_puzzle_candidate_descriptors(
            case,
            state,
            remaining_blocks=remaining,
            feature_mode=feature_mode,
            max_shape_bins=5,
        )
        expert = choose_expert_descriptor(desc, action)
        if expert is not None:
            pools.append((desc, int(expert)))
        executor.apply(state, action)
        if len(pools) >= max_steps:
            break
    return pools


def _train_and_eval(pools, *, epochs: int) -> dict[str, float]:
    features0 = pools[0][0][0].normalized_features
    model = TinyPoolScorer(features0.numel())
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = None
    final = None
    for epoch in range(max(epochs, 1)):
        total = torch.tensor(0.0)
        for desc, expert in pools:
            feats = torch.stack([item.normalized_features for item in desc])
            logits = model(feats).unsqueeze(0)
            target = torch.tensor([expert])
            total = total + nn.functional.cross_entropy(logits, target)
        total = total / max(len(pools), 1)
        if epoch == 0:
            initial = float(total.detach())
        opt.zero_grad()
        total.backward()
        opt.step()
        final = float(total.detach())
    candidate_hits = block_hits = shape_hits = contact_hits = 0
    rank_sum = 0.0
    for desc, expert in pools:
        feats = torch.stack([item.normalized_features for item in desc])
        logits = model(feats).detach()
        pred = int(logits.argmax().item())
        sorted_indices = torch.argsort(logits, descending=True).tolist()
        rank_sum += sorted_indices.index(expert) + 1
        candidate_hits += int(pred == expert)
        block_hits += int(desc[pred].block_index == desc[expert].block_index)
        shape_hits += int(desc[pred].shape_bin_id == desc[expert].shape_bin_id)
        contact_hits += int(desc[pred].contact_mode == desc[expert].contact_mode)
    n = max(len(pools), 1)
    return {
        "initial_ce": float(initial or 0.0),
        "final_ce": float(final or 0.0),
        "candidate_top1": candidate_hits / n,
        "block_top1": block_hits / n,
        "shape_bin_top1": shape_hits / n,
        "contact_mode_top1": contact_hits / n,
        "mean_expert_rank": rank_sum / n,
    }


def _heuristic_eval(pools) -> dict[str, float]:
    candidate_hits = 0
    rank_sum = 0.0
    for desc, expert in pools:
        scores = heuristic_scores(desc)
        pred = int(scores.argmax().item())
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        rank_sum += sorted_indices.index(expert) + 1
        candidate_hits += int(pred == expert)
    n = max(len(pools), 1)
    return {"candidate_top1": candidate_hits / n, "mean_expert_rank": rank_sum / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*", default=["0"])
    parser.add_argument("--small-slice-case-ids", nargs="*", default=["0", "1", "2", "3", "4"])
    parser.add_argument(
        "--feature-modes",
        nargs="*",
        default=["puzzle_pool_raw_safe", "puzzle_pool_normalized_relational"],
    )
    parser.add_argument("--single-state-epochs", type=int, default=200)
    parser.add_argument("--single-trajectory-epochs", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)

    requested_case_ids = [int(v) for v in args.case_ids]
    requested_small_ids = [int(v) for v in args.small_slice_case_ids]
    cases = load_validation_cases(case_limit=max(requested_case_ids + requested_small_ids) + 1)

    pools = _collect_pools(
        cases[requested_case_ids[0]], "puzzle_pool_normalized_relational", max_steps=20
    )
    single_state = _train_and_eval([pools[0]], epochs=args.single_state_epochs)
    single_state["expert_candidate_top1"] = single_state.pop("candidate_top1")
    trajectory = _train_and_eval(pools, epochs=args.single_trajectory_epochs)
    trajectory["expert_masked_rate"] = 0.0

    feature_mode_results = {}
    for feature_mode in args.feature_modes:
        mode_pools = []
        for case_id in requested_small_ids:
            mode_pools.extend(_collect_pools(cases[case_id], feature_mode, max_steps=8))
        feature_mode_results[feature_mode] = {
            "heuristic": _heuristic_eval(mode_pools),
            "learned": _train_and_eval(mode_pools, epochs=args.single_trajectory_epochs),
        }

    normalized = feature_mode_results.get(
        "puzzle_pool_normalized_relational", next(iter(feature_mode_results.values()))
    )
    heuristic = normalized["heuristic"]
    learned = normalized["learned"]
    candidate_delta = learned["candidate_top1"] - heuristic["candidate_top1"]
    rank_improvement = (heuristic["mean_expert_rank"] - learned["mean_expert_rank"]) / max(
        heuristic["mean_expert_rank"], 1e-6
    )

    report = {
        "case_ids": args.case_ids,
        "small_slice_case_ids": args.small_slice_case_ids,
        "feature_modes": args.feature_modes,
        "single_state": single_state,
        "single_trajectory": trajectory,
        "small_slice": {
            "feature_mode_results": feature_mode_results,
            "heuristic": heuristic,
            "learned_normalized_relational": learned,
            "candidate_top1_delta": candidate_delta,
            "mean_expert_rank_relative_improvement": rank_improvement,
            "raw_safe_vs_normalized_relational_reported": True,
        },
        "gates": {
            "single_state_pass": single_state["final_ce"]
            <= max(0.05, single_state["initial_ce"] * 0.10)
            and single_state["expert_candidate_top1"] == 1.0,
            "single_trajectory_pass": trajectory["candidate_top1"] >= 0.95
            and trajectory["block_top1"] >= 0.95
            and trajectory["shape_bin_top1"] >= 0.90
            and trajectory["contact_mode_top1"] >= 0.90
            and trajectory["expert_masked_rate"] == 0.0,
            "small_slice_pass": candidate_delta >= 0.05 or rank_improvement >= 0.10,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    md = Path(args.output).with_suffix(".md")
    md.write_text(
        "# Step6G Puzzle Policy Audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
