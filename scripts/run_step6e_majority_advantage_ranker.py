#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import ActionExecutor, ExecutionState, canonical_action_key, generate_candidate_actions  # noqa: E402
from puzzleplace.actions.schema import ActionPrimitive  # noqa: E402
from puzzleplace.models import CandidateRelationalActionQRanker  # noqa: E402
from puzzleplace.roles import label_case_roles  # noqa: E402
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases  # noqa: E402

from scripts.run_step6_hierarchical_rollout_control_audit import (  # noqa: E402
    RolloutJob,
    _forced_progress_action,
    _score_hierarchical_action,
    _seed_first_action,
    _train_hierarchical_policy,
)
from scripts.run_step6c_hierarchical_action_q_audit import (  # noqa: E402
    _build_pool_features,
    _normalize_pools_for_ranker,
)
from scripts.run_step6e_rollout_label_stability import _continue_after_action  # noqa: E402
from scripts.run_step6c_hierarchical_quality_alignment_audit import _quality_after_action  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step6E experiment: train a pool-local pairwise ranker from "
            "majority cross-continuation advantage labels instead of hard rollout top1 labels."
        )
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=[1, 4, 6])
    parser.add_argument("--policy-seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument(
        "--continuation-policies",
        nargs="*",
        choices=["policy_greedy", "immediate_oracle", "policy_topk_sample"],
        default=["policy_greedy", "immediate_oracle", "policy_topk_sample"],
    )
    parser.add_argument("--sample-topk", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--policy-epochs", type=int, default=80)
    parser.add_argument("--ranker-epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ranker-lr", type=float, default=1e-2)
    parser.add_argument("--primitive-set-weight", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument("--encoder-kind", choices=["graph", "relation_aware", "typed_constraint_graph", "typed_constraint_graph_no_anchor", "typed_constraint_graph_no_boundary", "typed_constraint_graph_no_groups"], default="relation_aware")
    parser.add_argument(
        "--feature-mode",
        choices=["relational_state_pool_no_raw_logits", "state_pool_no_raw_logits"],
        default="relational_state_pool_no_raw_logits",
    )
    parser.add_argument("--feature-normalization", choices=["global", "per_case"], default="per_case")
    parser.add_argument(
        "--ranker-input",
        choices=["candidate_features", "graph_action_embeddings", "joint_typed_graph_action", "action_delta_features"],
        default="candidate_features",
        help="Use existing candidate feature rows or direct policy encoder block/target/graph action embeddings.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-candidates-per-step", type=int, default=8)
    parser.add_argument("--continuation-horizon", type=int, default=4)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6e_majority_advantage_ranker_micro.json",
    )
    return parser.parse_args()




def _clone_state_for_delta(state: ExecutionState) -> ExecutionState:
    return ExecutionState(
        placements=dict(state.placements),
        frozen_blocks=set(state.frozen_blocks),
        proposed_positions=dict(state.proposed_positions),
        shape_assigned=set(state.shape_assigned),
        semantic_placed=set(state.semantic_placed),
        physically_placed=set(state.physically_placed),
        step=state.step,
        history=list(state.history),
        last_rollout_mode=state.last_rollout_mode,
    )


def _group_stats(case, placed: set[int], block_index: int, column) -> tuple[float, float, float]:
    value = float(case.constraints[block_index, column].item())
    if value <= 0:
        return (0.0, 0.0, 0.0)
    members = [idx for idx in range(case.block_count) if float(case.constraints[idx, column].item()) == value]
    before = sum(1 for idx in members if idx in placed) / max(len(members), 1)
    after_placed = set(placed)
    after_placed.add(block_index)
    after = sum(1 for idx in members if idx in after_placed) / max(len(members), 1)
    return (1.0, before, after - before)


def _action_delta_feature_rows(case, state: ExecutionState, selected) -> list[list[float]]:
    from puzzleplace.data import ConstraintColumns
    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    scale = max(total_area ** 0.5, 1e-6)
    rows = []
    placed_before = set(state.semantic_placed)
    base_fraction = len(placed_before) / max(case.block_count, 1)
    for policy_score, candidate, components in selected:
        trial = _clone_state_for_delta(state)
        ActionExecutor(case).apply(trial, candidate)
        x, y, w, h = trial.proposed_positions.get(int(candidate.block_index), (0.0, 0.0, 0.0, 0.0))
        quality = _quality_after_action(case, state, candidate)["quality"]
        primitive = torch.zeros(len(ActionPrimitive), dtype=torch.float32)
        primitive[list(ActionPrimitive).index(candidate.primitive)] = 1.0
        block_idx = int(candidate.block_index)
        target_idx = None if candidate.target_index is None else int(candidate.target_index)
        cluster_present, cluster_before, cluster_delta = _group_stats(case, placed_before, block_idx, ConstraintColumns.CLUSTER)
        mib_present, mib_before, mib_delta = _group_stats(case, placed_before, block_idx, ConstraintColumns.MIB)
        boundary_present, boundary_before, boundary_delta = _group_stats(case, placed_before, block_idx, ConstraintColumns.BOUNDARY)
        target_same_cluster = 0.0
        target_same_mib = 0.0
        target_same_boundary = 0.0
        if target_idx is not None and 0 <= target_idx < case.block_count:
            target_same_cluster = float(case.constraints[block_idx, ConstraintColumns.CLUSTER].item() > 0 and case.constraints[block_idx, ConstraintColumns.CLUSTER].item() == case.constraints[target_idx, ConstraintColumns.CLUSTER].item())
            target_same_mib = float(case.constraints[block_idx, ConstraintColumns.MIB].item() > 0 and case.constraints[block_idx, ConstraintColumns.MIB].item() == case.constraints[target_idx, ConstraintColumns.MIB].item())
            target_same_boundary = float(case.constraints[block_idx, ConstraintColumns.BOUNDARY].item() > 0 and case.constraints[block_idx, ConstraintColumns.BOUNDARY].item() == case.constraints[target_idx, ConstraintColumns.BOUNDARY].item())
        row = [
            float(policy_score),
            float(components.get("block_logit") or 0.0),
            float(components.get("primitive_logit") or 0.0),
            float(components.get("target_logit") or 0.0),
            float(components.get("heuristic_score") or 0.0),
            float(block_idx) / max(case.block_count - 1, 1),
            -1.0 if target_idx is None else float(target_idx) / max(case.block_count - 1, 1),
            0.0 if target_idx is None else 1.0,
            float(x) / scale,
            float(y) / scale,
            float(w) / scale,
            float(h) / scale,
            float(w) * float(h) / total_area,
            base_fraction,
            len(trial.semantic_placed) / max(case.block_count, 1) - base_fraction,
            float(case.constraints[block_idx, ConstraintColumns.FIXED].item()),
            float(case.constraints[block_idx, ConstraintColumns.PREPLACED].item()),
            cluster_present, cluster_before, cluster_delta,
            mib_present, mib_before, mib_delta,
            boundary_present, boundary_before, boundary_delta,
            target_same_cluster, target_same_mib, target_same_boundary,
            float(quality["HPWLgap"]),
            float(quality["Areagap_bbox"]),
            float(quality["Violationsrelative"]),
        ]
        row.extend(primitive.tolist())
        rows.append(row)
    return rows

def _graph_action_feature_rows(case, policy, role_evidence, state: ExecutionState, selected) -> list[list[float]]:
    with torch.no_grad():
        encoded = policy.encoder(
            case,
            role_evidence=role_evidence,
            placements=state.proposed_positions,
            state_step=state.step,
        )
    primitive_count = len(__import__("puzzleplace.actions.schema", fromlist=["ActionPrimitive"]).ActionPrimitive)
    rows = []
    zero_block = torch.zeros_like(encoded.graph_embedding)
    for _score, candidate, _components in selected:
        block_emb = encoded.block_embeddings[int(candidate.block_index)]
        target_emb = (
            encoded.block_embeddings[int(candidate.target_index)]
            if candidate.target_index is not None and 0 <= int(candidate.target_index) < case.block_count
            else zero_block
        )
        primitive_one_hot = torch.zeros(primitive_count, dtype=torch.float32)
        primitive_one_hot[list(__import__("puzzleplace.actions.schema", fromlist=["ActionPrimitive"]).ActionPrimitive).index(candidate.primitive)] = 1.0
        row = torch.cat(
            [
                block_emb,
                target_emb,
                encoded.graph_embedding,
                primitive_one_hot,
                torch.tensor([0.0 if candidate.target_index is None else 1.0], dtype=torch.float32),
            ]
        )
        rows.append(row.tolist())
    return rows

def _collect_case_seed(args_dict: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    case_id = int(args_dict["case_id"])
    seed = int(args_dict["seed"])
    max_steps = int(args_dict["max_steps"])
    max_candidates = int(args_dict["max_candidates_per_step"])
    policies = list(args_dict["continuation_policies"])
    job = RolloutJob(
        case_id=case_id,
        seed=seed,
        hidden_dim=int(args_dict["hidden_dim"]),
        epochs=int(args_dict["policy_epochs"]),
        lr=float(args_dict["lr"]),
        primitive_set_weight=float(args_dict["primitive_set_weight"]),
        block_weight=float(args_dict["block_weight"]),
        encoder_kind=str(args_dict["encoder_kind"]),
    )
    cases = load_validation_cases(case_limit=case_id + 1)
    case = cases[case_id]
    dataset = build_bc_dataset_from_cases([case], max_traces_per_case=1)
    policy, training_summary, _primitive_sets = _train_hierarchical_policy(dataset, job)
    role_evidence = label_case_roles(case)
    state = ExecutionState(last_rollout_mode="semantic")
    executor = ActionExecutor(case)
    seed_action = _seed_first_action(
        case,
        [idx for idx in range(case.block_count) if idx not in state.semantic_placed],
    )
    executor.apply(state, seed_action)

    pools = []
    no_progress = 0
    step = 0
    while len(state.semantic_placed) < case.block_count and state.step < case.block_count * 4 and step < max_steps:
        remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
        candidates = generate_candidate_actions(case, state, remaining_blocks=remaining, mode="semantic", max_per_primitive=8)
        if not candidates:
            chosen = _forced_progress_action(case, state, remaining[0])
            executor.apply(state, chosen)
            step += 1
            continue
        scored = []
        for candidate in candidates:
            policy_score, components = _score_hierarchical_action(case, policy, role_evidence, state, candidate)
            scored.append((float(policy_score), candidate, components))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:max_candidates]
        ranker_input = str(args_dict.get("ranker_input", "candidate_features"))
        if ranker_input == "graph_action_embeddings":
            feature_rows = _graph_action_feature_rows(case, policy, role_evidence, state, selected)
        elif ranker_input == "action_delta_features":
            feature_rows = _action_delta_feature_rows(case, state, selected)
        else:
            feature_rows = _build_pool_features(case, state, selected, feature_mode=str(args_dict["feature_mode"]))
        candidate_rows = []
        for policy_rank, (_score, candidate, _components) in enumerate(selected, start=1):
            per_policy = {}
            for continuation_policy in policies:
                quality, meta = _continue_after_action(
                    case,
                    policy,
                    role_evidence,
                    state,
                    candidate,
                    policy_name=continuation_policy,
                    horizon=int(args_dict["continuation_horizon"]),
                    max_candidates=max_candidates,
                    sample_topk=int(args_dict["sample_topk"]),
                    seed=seed,
                    pool_step=step,
                )
                per_policy[continuation_policy] = {"quality": quality, "meta": meta}
            candidate_rows.append(
                {
                    "policy_rank": policy_rank,
                    "action_key": canonical_action_key(candidate),
                    "primitive": candidate.primitive.value,
                    "block_index": int(candidate.block_index),
                    "target_index": None if candidate.target_index is None else int(candidate.target_index),
                    "source": candidate.metadata.get("source"),
                    "intent_type": candidate.metadata.get("intent_type"),
                    "rollout_return_by_policy": per_policy,
                }
            )
        if str(args_dict.get("ranker_input", "candidate_features")) == "joint_typed_graph_action":
            block_features, _role_ids = build_relation_aware_block_features(
                case,
                role_evidence=role_evidence,
                placements=state.proposed_positions,
                state_step=state.step,
            )
            for row, (_score, candidate, _components) in zip(candidate_rows, selected, strict=True):
                row["primitive_id"] = list(_E9ActionPrimitive).index(candidate.primitive)
            graph_block_features = block_features.tolist()
            typed_edges = _typed_edges_for_pool(case, state)
        else:
            graph_block_features = None
            typed_edges = None
        quality_values = _mean_normalized_quality(candidate_rows, policies)
        oracle_index = min(range(len(quality_values)), key=lambda idx: quality_values[idx])
        pools.append(
            {
                "case_id": str(case.case_id),
                "case_index": case_id,
                "policy_seed": seed,
                "step": step,
                "feature_rows": feature_rows,
                "quality_values": quality_values,
                "oracle_index": int(oracle_index),
                "candidate_rows": candidate_rows,
                "graph_block_features": graph_block_features,
                "typed_edges": typed_edges,
            }
        )
        chosen = scored[0][1]
        if no_progress >= 2 and chosen.block_index in state.semantic_placed:
            chosen = _forced_progress_action(case, state, remaining[0])
        before = len(state.semantic_placed)
        executor.apply(state, chosen)
        no_progress = 0 if len(state.semantic_placed) > before else no_progress + 1
        step += 1
    return {
        "case_id": str(case.case_id),
        "case_index": case_id,
        "policy_seed": seed,
        "job": asdict(job),
        "training_summary": training_summary,
        "pools": pools,
    }


def _mean_normalized_quality(rows: list[dict[str, Any]], policies: list[str]) -> list[float]:
    values_by_policy = {
        policy: [float(row["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"]) for row in rows]
        for policy in policies
    }
    result = []
    for idx in range(len(rows)):
        parts = []
        for policy in policies:
            values = values_by_policy[policy]
            lo = min(values)
            hi = max(values)
            parts.append((values[idx] - lo) / max(hi - lo, 1e-9))
        result.append(sum(parts) / max(len(parts), 1))
    return result


def _pair_targets(pool: dict[str, Any], policies: list[str]) -> torch.Tensor:
    rows = pool["candidate_rows"]
    n = len(rows)
    targets = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wins = 0
            for policy in policies:
                qi = float(rows[i]["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"])
                qj = float(rows[j]["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"])
                wins += int(qi < qj)
            targets[i, j] = wins / max(len(policies), 1)
    return targets


def _train_pairwise_ranker(pools: list[dict[str, Any]], args: argparse.Namespace):
    torch.manual_seed(int(args.policy_seeds[0] if args.policy_seeds else 0))
    policies = [str(policy) for policy in args.continuation_policies]
    if str(getattr(args, "ranker_input", "candidate_features")) == "joint_typed_graph_action":
        ranker = _JointTypedGraphActionRanker(
            block_feature_dim=len(pools[0]["graph_block_features"][0]),
            hidden_dim=64,
            primitive_count=len(_E9ActionPrimitive),
        )
        optimizer = torch.optim.Adam(ranker.parameters(), lr=float(args.ranker_lr))
        for _epoch in range(int(args.ranker_epochs)):
            for pool in pools:
                targets = _pair_targets(pool, policies)
                mask = ~torch.eye(targets.shape[0], dtype=torch.bool)
                logits = ranker.pair_logits(pool)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return ranker, {"mean": [], "std": [], "kind": "joint_typed_graph_action"}

    feature_dim = len(pools[0]["feature_rows"][0])
    ranker = CandidateRelationalActionQRanker(feature_dim=feature_dim, hidden_dim=64, num_heads=4)
    all_features = torch.tensor([f for pool in pools for f in pool["feature_rows"]], dtype=torch.float32)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0).clamp_min(1e-6)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=float(args.ranker_lr))
    for _epoch in range(int(args.ranker_epochs)):
        for pool in pools:
            features = torch.tensor(pool["feature_rows"], dtype=torch.float32)
            features = (features - mean) / std
            targets = _pair_targets(pool, policies)
            mask = ~torch.eye(targets.shape[0], dtype=torch.bool)
            logits = ranker.pair_logits(features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return ranker, {"mean": mean.tolist(), "std": std.tolist(), "kind": "feature_pairwise"}


def _pairwise_scores(ranker: CandidateRelationalActionQRanker, features: torch.Tensor) -> torch.Tensor:
    logits = ranker.pair_logits(features)
    mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    probs = torch.sigmoid(logits).masked_fill(~mask, 0.0)
    return probs.sum(dim=1) / max(logits.shape[0] - 1, 1)


def _evaluate(ranker, pools: list[dict[str, Any]], stats: dict[str, list[float]]) -> dict[str, Any]:
    is_joint = stats.get("kind") == "joint_typed_graph_action"
    mean = torch.tensor(stats.get("mean", []), dtype=torch.float32)
    std = torch.tensor(stats.get("std", []), dtype=torch.float32)
    ranks = []
    regrets = []
    top1 = 0
    rows = []
    for pool in pools:
        if is_joint:
            scores = ranker.score_candidates(pool).detach()
        else:
            features = torch.tensor(pool["feature_rows"], dtype=torch.float32)
            features = (features - mean) / std
            scores = _pairwise_scores(ranker, features).detach()
        pred = int(scores.argmax().item())
        order = sorted(range(len(pool["quality_values"])), key=lambda idx: float(pool["quality_values"][idx]))
        rank = order.index(pred) + 1
        regret = float(pool["quality_values"][pred]) - float(pool["quality_values"][order[0]])
        ranks.append(rank)
        regrets.append(regret)
        top1 += int(rank == 1)
        rows.append(
            {
                "case_id": pool["case_id"],
                "case_index": int(pool["case_index"]),
                "policy_seed": int(pool["policy_seed"]),
                "step": int(pool["step"]),
                "predicted_index": pred,
                "oracle_index": int(order[0]),
                "selected_quality_rank": rank,
                "selected_quality_regret": regret,
                "predicted_action_key": pool["candidate_rows"][pred]["action_key"],
                "oracle_action_key": pool["candidate_rows"][order[0]]["action_key"],
            }
        )
    return {
        "pool_count": len(pools),
        "mean_selected_quality_rank": sum(ranks) / max(len(ranks), 1),
        "mean_selected_quality_regret": sum(regrets) / max(len(regrets), 1),
        "oracle_top1_selected_fraction": top1 / max(len(pools), 1),
        "rows": rows,
    }




def _run_loco_split_job(job: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    heldout = int(job["heldout"])
    pools = job["pools"]
    args = argparse.Namespace(**job["args"])
    train = [pool for pool in pools if int(pool["case_index"]) != heldout]
    eval_pools = [pool for pool in pools if int(pool["case_index"]) == heldout]
    ranker, stats = _train_pairwise_ranker(train, args)
    train_eval = _evaluate(ranker, train, stats)
    evaluation = _evaluate(ranker, eval_pools, stats)
    gate = {
        "mean_selected_quality_rank_lt_4": evaluation["mean_selected_quality_rank"] < 4.0,
        "oracle_top1_selected_fraction_gt_0_30": evaluation["oracle_top1_selected_fraction"] > 0.30,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    return {
        "heldout_case_index": heldout,
        "train_pool_count": len(train),
        "eval_pool_count": len(eval_pools),
        "train_evaluation": train_eval,
        "evaluation": evaluation,
        "heldout_gate": gate,
    }


def _run_loco_splits(pools: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    case_ids = sorted({int(pool["case_index"]) for pool in pools})
    if len(case_ids) < 2:
        return None
    args_dict = vars(args).copy()
    jobs = [{"heldout": heldout, "pools": pools, "args": args_dict} for heldout in case_ids]
    max_workers = max(1, min(int(getattr(args, "workers", 1)), len(jobs)))
    if max_workers == 1:
        splits = [_run_loco_split_job(job) for job in jobs]
    else:
        splits = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(_run_loco_split_job, job) for job in jobs]
            for future in as_completed(futures):
                splits.append(future.result())
        splits.sort(key=lambda item: int(item["heldout_case_index"]))
    mean_rank = sum(split["evaluation"]["mean_selected_quality_rank"] for split in splits) / max(len(splits), 1)
    mean_regret = sum(split["evaluation"]["mean_selected_quality_regret"] for split in splits) / max(len(splits), 1)
    mean_top1 = sum(split["evaluation"]["oracle_top1_selected_fraction"] for split in splits) / max(len(splits), 1)
    gate = {
        "mean_selected_quality_rank_lt_4": mean_rank < 4.0,
        "oracle_top1_selected_fraction_gt_0_30": mean_top1 > 0.30,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    return {
        "split_count": len(splits),
        "parallel_workers_used": max_workers,
        "mean_selected_quality_rank": mean_rank,
        "mean_selected_quality_regret": mean_regret,
        "oracle_top1_selected_fraction": mean_top1,
        "gate": gate,
        "splits": splits,
    }

def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    ev = payload["evaluation"]
    gate = payload["micro_gate"]
    lines = [
        "# Step6E Majority Advantage Ranker",
        "",
        "- purpose: `train from majority cross-continuation pairwise advantage labels`",
        f"- cases: `{payload['case_ids']}`",
        f"- policy seeds: `{payload['policy_seeds']}`",
        f"- continuation policies: `{payload['continuation_policies']}`",
        f"- feature mode: `{payload['feature_mode']}`",
        f"- ranker input: `{payload['ranker_input']}`",
        f"- feature normalization: `{payload['feature_normalization']}`",
        f"- pool count: `{payload['pool_count']}`",
        f"- mean selected quality rank: `{ev['mean_selected_quality_rank']:.4f}`",
        f"- mean selected quality regret: `{ev['mean_selected_quality_regret']:.4f}`",
        f"- oracle top1 selected fraction: `{ev['oracle_top1_selected_fraction']:.4f}`",
        f"- micro gate pass: `{gate['pass']}`",
        "",
        "## Leave-One-Case-Out",
        "",
        *( [
            f"- mean selected quality rank: `{payload['leave_one_case_out']['mean_selected_quality_rank']:.4f}`",
            f"- mean selected quality regret: `{payload['leave_one_case_out']['mean_selected_quality_regret']:.4f}`",
            f"- oracle top1 selected fraction: `{payload['leave_one_case_out']['oracle_top1_selected_fraction']:.4f}`",
            f"- LOCO gate pass: `{payload['leave_one_case_out']['gate']['pass']}`",
        ] if payload.get('leave_one_case_out') is not None else ["- skipped: fewer than two cases"] ),
        "",
        "## Gate",
        "",
        f"- mean rank < 4: `{gate['mean_selected_quality_rank_lt_4']}`",
        f"- top1 > 0.30: `{gate['oracle_top1_selected_fraction_gt_0_30']}`",
        "",
        "Interpretation: this is a method probe for robust advantage learning. Passing micro-overfit would justify widening to LOCO; failing would push the next branch toward a true typed constraint graph/state encoder.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    jobs = [
        {
            "case_id": int(case_id),
            "seed": int(seed),
            "hidden_dim": int(args.hidden_dim),
            "policy_epochs": int(args.policy_epochs),
            "lr": float(args.lr),
            "primitive_set_weight": float(args.primitive_set_weight),
            "block_weight": float(args.block_weight),
            "encoder_kind": str(args.encoder_kind),
            "feature_mode": str(args.feature_mode),
            "ranker_input": str(args.ranker_input),
            "max_steps": int(args.max_steps),
            "max_candidates_per_step": int(args.max_candidates_per_step),
            "continuation_horizon": int(args.continuation_horizon),
            "continuation_policies": [str(p) for p in args.continuation_policies],
            "sample_topk": int(args.sample_topk),
        }
        for case_id in args.case_ids
        for seed in args.policy_seeds
    ]
    max_workers = max(1, min(int(args.workers), len(jobs)))
    collections = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_collect_case_seed, job) for job in jobs]
        for future in as_completed(futures):
            collections.append(future.result())
    pools = [pool for collection in collections for pool in collection["pools"]]
    pools = _normalize_pools_for_ranker(pools, mode=str(args.feature_normalization))
    ranker, stats = _train_pairwise_ranker(pools, args)
    evaluation = _evaluate(ranker, pools, stats)
    gate = {
        "mean_selected_quality_rank_lt_4": evaluation["mean_selected_quality_rank"] < 4.0,
        "oracle_top1_selected_fraction_gt_0_30": evaluation["oracle_top1_selected_fraction"] > 0.30,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    payload = {
        "status": "complete",
        "purpose": "Step6E majority cross-continuation advantage ranker micro-overfit",
        "case_ids": [int(v) for v in args.case_ids],
        "policy_seeds": [int(v) for v in args.policy_seeds],
        "continuation_policies": [str(p) for p in args.continuation_policies],
        "feature_mode": str(args.feature_mode),
        "ranker_input": str(args.ranker_input),
        "feature_normalization": str(args.feature_normalization),
        "max_steps": int(args.max_steps),
        "max_candidates_per_step": int(args.max_candidates_per_step),
        "continuation_horizon": int(args.continuation_horizon),
        "collection_count": len(collections),
        "pool_count": len(pools),
        "workers_requested": max_workers,
        "feature_stats": stats,
        "evaluation": evaluation,
        "micro_gate": gate,
        "leave_one_case_out": _run_loco_splits(pools, args),
        "collections": sorted(collections, key=lambda item: (int(item["case_index"]), int(item["policy_seed"]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "case_ids": payload["case_ids"],
                "policy_seeds": payload["policy_seeds"],
                "pool_count": payload["pool_count"],
                "ranker_input": payload["ranker_input"],
                "mean_selected_quality_rank": evaluation["mean_selected_quality_rank"],
                "mean_selected_quality_regret": evaluation["mean_selected_quality_regret"],
                "oracle_top1_selected_fraction": evaluation["oracle_top1_selected_fraction"],
                "micro_gate_pass": gate["pass"],
                "loco_mean_selected_quality_rank": None if payload["leave_one_case_out"] is None else payload["leave_one_case_out"]["mean_selected_quality_rank"],
                "loco_oracle_top1_selected_fraction": None if payload["leave_one_case_out"] is None else payload["leave_one_case_out"]["oracle_top1_selected_fraction"],
                "loco_gate_pass": None if payload["leave_one_case_out"] is None else payload["leave_one_case_out"]["gate"]["pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )


# --- E9 joint trainable typed-graph/action ranker helpers ---
# Kept inside the research script to avoid changing production policy APIs.
from puzzleplace.models.encoders import build_relation_aware_block_features  # noqa: E402
from puzzleplace.actions.schema import ActionPrimitive as _E9ActionPrimitive  # noqa: E402


class _JointTypedGraphActionRanker(torch.nn.Module):
    def __init__(self, block_feature_dim: int, hidden_dim: int = 64, primitive_count: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.primitive_count = primitive_count or len(_E9ActionPrimitive)
        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(block_feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.relation_embedding = torch.nn.Embedding(6, hidden_dim)
        self.message_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = torch.nn.LayerNorm(hidden_dim)
        self.action_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4 + self.primitive_count + 1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.pair_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def encode_actions(self, pool: dict[str, Any]) -> torch.Tensor:
        block_features = torch.tensor(pool["graph_block_features"], dtype=torch.float32)
        nodes = self.node_proj(block_features)
        if pool.get("typed_edges"):
            agg = torch.zeros_like(nodes)
            for src, dst, relation_id, weight in pool["typed_edges"]:
                i, j, rel = int(src), int(dst), int(relation_id)
                if not (0 <= i < nodes.shape[0] and 0 <= j < nodes.shape[0]):
                    continue
                relation = self.relation_embedding(torch.tensor(rel, dtype=torch.long, device=nodes.device))
                w = torch.tensor([float(weight)], dtype=torch.float32, device=nodes.device)
                agg[j] += self.message_proj(torch.cat([nodes[i], nodes[j], relation, w]))
            nodes = self.node_norm(nodes + agg)
        graph = nodes.mean(dim=0)
        zero = torch.zeros_like(graph)
        action_vectors = []
        for row in pool["candidate_rows"]:
            block_index = int(row["block_index"])
            target_index = row.get("target_index")
            block = nodes[block_index]
            target = nodes[int(target_index)] if target_index is not None else zero
            primitive = torch.zeros(self.primitive_count, dtype=torch.float32, device=nodes.device)
            primitive[int(row["primitive_id"])] = 1.0
            has_target = torch.tensor([0.0 if target_index is None else 1.0], dtype=torch.float32, device=nodes.device)
            action_vectors.append(
                self.action_proj(torch.cat([block, target, graph, (block - target).abs(), primitive, has_target]))
            )
        return torch.stack(action_vectors, dim=0)

    def pair_logits(self, pool: dict[str, Any]) -> torch.Tensor:
        encoded = self.encode_actions(pool)
        lhs = encoded.unsqueeze(1).expand(-1, encoded.shape[0], -1)
        rhs = encoded.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        pair = torch.cat([lhs, rhs, lhs - rhs, (lhs - rhs).abs()], dim=-1)
        return self.pair_head(pair).squeeze(-1)

    def score_candidates(self, pool: dict[str, Any]) -> torch.Tensor:
        logits = self.pair_logits(pool)
        mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        return torch.sigmoid(logits).masked_fill(~mask, 0.0).sum(dim=1) / max(logits.shape[0] - 1, 1)


def _typed_edges_for_pool(case, state: ExecutionState) -> list[list[float]]:
    edges: list[list[float]] = []
    if case.b2b_edges.numel() > 0:
        denom = max(float(case.b2b_edges[:, 2].abs().max().item()), 1.0)
        for src, dst, weight in case.b2b_edges.tolist():
            i, j = int(src), int(dst)
            if 0 <= i < case.block_count and 0 <= j < case.block_count:
                w = float(weight) / denom
                edges.append([i, j, 0, w])
                edges.append([j, i, 0, w])
    if case.p2b_edges.numel() > 0:
        denom = max(float(case.p2b_edges[:, 2].abs().max().item()), 1.0)
        for _pin, block, weight in case.p2b_edges.tolist():
            j = int(block)
            if 0 <= j < case.block_count:
                edges.append([j, j, 1, float(weight) / denom])
    from puzzleplace.data import ConstraintColumns
    groups: dict[tuple[int, float], list[int]] = {}
    anchors: list[int] = []
    for idx in range(case.block_count):
        cluster = float(case.constraints[idx, ConstraintColumns.CLUSTER].item())
        mib = float(case.constraints[idx, ConstraintColumns.MIB].item())
        boundary = float(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if cluster > 0:
            groups.setdefault((2, cluster), []).append(idx)
        if mib > 0:
            groups.setdefault((3, mib), []).append(idx)
        if boundary > 0:
            groups.setdefault((4, boundary), []).append(idx)
        if bool(case.constraints[idx, ConstraintColumns.FIXED].item()) or bool(case.constraints[idx, ConstraintColumns.PREPLACED].item()):
            anchors.append(idx)
    for (relation_id, _value), members in groups.items():
        if len(members) < 2:
            continue
        denom = max(len(members) - 1, 1)
        for src in members:
            for dst in members:
                if src != dst:
                    edges.append([src, dst, relation_id, 1.0 / denom])
    if anchors:
        denom = max(len(anchors), 1)
        for src in anchors:
            for dst in range(case.block_count):
                if src != dst:
                    edges.append([src, dst, 5, 1.0 / denom])
    return edges


if __name__ == "__main__":
    main()
