#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import (  # noqa: E402
    ActionExecutor,
    ActionPrimitive,
    ExecutionState,
    TypedAction,
    actions_match,
    generate_candidate_actions,
)
from puzzleplace.data import FloorSetCase  # noqa: E402
from puzzleplace.eval import evaluate_positions  # noqa: E402
from puzzleplace.models import (  # noqa: E402
    DENIED_TRANSITION_PAYLOAD_FIELDS,
    HierarchicalSetPolicy,  # noqa: E402
    SharedEncoderTransitionComparator,
    build_transition_payload,
)
from puzzleplace.repair.finalizer import finalize_layout  # noqa: E402
from puzzleplace.roles import label_case_roles  # noqa: E402
from puzzleplace.rollout.semantic import (  # noqa: E402
    _forced_progress_action,
    _seed_first_action,
    _semantic_heuristic_score,
)
from puzzleplace.train import (  # noqa: E402
    BCStepRecord,
    build_bc_dataset_from_cases,
    load_validation_cases,
)
from puzzleplace.train.dataset_bc import action_to_targets  # noqa: E402


@dataclass(frozen=True, slots=True)
class RolloutJob:
    case_id: int
    seed: int
    hidden_dim: int
    epochs: int
    lr: float
    primitive_set_weight: float
    block_weight: float
    encoder_kind: str = "typed_constraint_graph"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step6F research-sidecar: shared-encoder pre/post transition comparator "
            "trained on majority cross-continuation pairwise advantage labels."
        )
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=[0, 1, 2, 3, 4])
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
    parser.add_argument(
        "--encoder-kind",
        choices=[
            "graph",
            "relation_aware",
            "typed_constraint_graph",
            "typed_constraint_graph_no_anchor",
            "typed_constraint_graph_no_boundary",
            "typed_constraint_graph_no_groups",
        ],
        default="typed_constraint_graph",
        help="Policy encoder used only to collect candidate pools; not a model input shortcut.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-candidates-per-step", type=int, default=8)
    parser.add_argument("--continuation-horizon", type=int, default=4)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--eval-mode",
        choices=["micro", "loco", "micro-loco"],
        default="micro-loco",
        help="micro always trains/evaluates on all pools; loco adds leave-one-case-out splits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6f_transition_comparator.json",
    )
    return parser.parse_args()


def _thread_limit() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _canonical_action_key(action: TypedAction) -> str:
    parts = [
        action.primitive.value,
        str(int(action.block_index)),
        "none" if action.target_index is None else str(int(action.target_index)),
        "none" if action.boundary_code is None else str(int(action.boundary_code)),
    ]
    for value in (action.x, action.y, action.w, action.h, action.dx, action.dy):
        parts.append("none" if value is None else f"{float(value):.6g}")
    return "|".join(parts)


def _clone_state(state: ExecutionState) -> ExecutionState:
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


def _target_blind_case(case: FloorSetCase) -> FloorSetCase:
    """Return a case view for policy/candidate/payload paths with no targets."""

    return replace(case, target_positions=None)


def _state_from_record(record: BCStepRecord) -> ExecutionState:
    placed = dict(record.placements)
    return ExecutionState(
        placements=placed,
        proposed_positions=placed,
        shape_assigned=set(placed),
        semantic_placed=set(placed),
        physically_placed=set(placed),
        step=len(placed),
        last_rollout_mode="semantic",
    )


def _acceptable_primitive_set(record: BCStepRecord) -> list[int]:
    state = _state_from_record(record)
    remaining = [idx for idx in range(record.case.block_count) if idx not in state.placements]
    candidates = generate_candidate_actions(
        record.case,
        state,
        remaining_blocks=remaining,
        mode="semantic",
    )
    target_block_candidates = [
        candidate for candidate in candidates if candidate.block_index == record.action.block_index
    ]
    semantic_matches = [
        candidate
        for candidate in candidates
        if actions_match(candidate, record.action, mode="semantic")
    ]
    target_primitive = list(ActionPrimitive).index(record.action.primitive)
    return sorted(
        {
            list(ActionPrimitive).index(candidate.primitive)
            for candidate in (semantic_matches or target_block_candidates)
        }
        or {target_primitive}
    )


def _set_cross_entropy(logits: torch.Tensor, target_ids: list[int]) -> torch.Tensor:
    ids = sorted({int(idx) for idx in target_ids if 0 <= int(idx) < int(logits.shape[0])})
    if not ids:
        raise ValueError("set loss requires at least one valid target id")
    log_probs = torch.nn.functional.log_softmax(logits, dim=0)
    return -torch.logsumexp(log_probs[torch.tensor(ids, dtype=torch.long)], dim=0)


def _train_hierarchical_policy(
    dataset: list[BCStepRecord],
    job: RolloutJob,
) -> tuple[HierarchicalSetPolicy, dict[str, Any], list[list[int]]]:
    torch.manual_seed(job.seed)
    policy = HierarchicalSetPolicy(hidden_dim=job.hidden_dim, encoder_kind=job.encoder_kind)
    primitive_sets = [_acceptable_primitive_set(record) for record in dataset]
    optimizer = torch.optim.Adam(policy.parameters(), lr=job.lr)
    initial_loss = 0.0
    final_loss = 0.0
    for epoch in range(job.epochs):
        epoch_loss = 0.0
        for step, record in enumerate(dataset):
            optimizer.zero_grad(set_to_none=True)
            output = policy(
                record.case,
                role_evidence=record.role_evidence,
                placements=record.placements,
                state_step=len(record.placements),
            )
            targets = action_to_targets(record.action)
            block_index = int(targets["block_index"])
            block_loss = torch.nn.functional.cross_entropy(
                output.block_logits.unsqueeze(0), torch.tensor([block_index])
            )
            primitive_loss = _set_cross_entropy(
                output.primitive_logits_by_block[block_index], primitive_sets[step]
            )
            loss = job.block_weight * block_loss + job.primitive_set_weight * primitive_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        if epoch == 0:
            initial_loss = epoch_loss / max(len(dataset), 1)
        final_loss = epoch_loss / max(len(dataset), 1)
    return (
        policy,
        {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "dataset_size": len(dataset),
            "epochs": job.epochs,
            "hidden_dim": job.hidden_dim,
            "seed": job.seed,
            "primitive_set_weight": job.primitive_set_weight,
            "block_weight": job.block_weight,
            "encoder_kind": job.encoder_kind,
        },
        primitive_sets,
    )


def _score_hierarchical_action(
    case: FloorSetCase,
    policy: HierarchicalSetPolicy | None,
    role_evidence,
    state: ExecutionState,
    action: TypedAction,
) -> tuple[float, dict[str, Any]]:
    heuristic = _semantic_heuristic_score(action)
    if policy is None:
        return heuristic, {"components_used": ["semantic_heuristic"]}
    output = policy(
        case,
        role_evidence=role_evidence,
        placements=state.placements,
        state_step=state.step,
    )
    primitive_id = list(ActionPrimitive).index(action.primitive)
    block_logit = float(output.block_logits[action.block_index].item())
    primitive_logit = float(
        output.primitive_logits_by_block[action.block_index, primitive_id].item()
    )
    target_logit = 0.0
    if action.target_index is not None:
        target_logit = float(output.target_logits[action.block_index, action.target_index].item())
    return block_logit + primitive_logit + target_logit + heuristic, {
        "components_used": ["block", "primitive_by_block", "target", "semantic_heuristic"],
    }


def _quality_from_positions(
    case: FloorSetCase,
    positions: dict[int, tuple[float, float, float, float]],
) -> dict[str, Any]:
    repair = finalize_layout(case, positions)
    evaluation = evaluate_positions(case, repair.positions, runtime=1.0, median_runtime=1.0)
    official = evaluation["official"]
    return {
        "quality_cost_runtime1": float(official["cost"]),
        "HPWLgap": float(official["hpwl_gap"]),
        "Areagap_bbox": float(official["area_gap"]),
        "Violationsrelative": float(official["violations_relative"]),
        "feasible": bool(official["is_feasible"]),
        "repair": {
            "hard_feasible_after": bool(repair.report.hard_feasible_after),
            "mean_displacement": float(repair.report.mean_displacement),
            "moved_block_count": int(repair.report.moved_block_count),
            "shelf_fallback_count": int(repair.report.shelf_fallback_count),
        },
    }


def _quality_after_action(
    eval_case: FloorSetCase,
    blind_case: FloorSetCase,
    state: ExecutionState,
    action,
) -> dict[str, Any]:
    trial = _clone_state(state)
    ActionExecutor(blind_case).apply(trial, action)
    return _quality_from_positions(eval_case, trial.proposed_positions)


def _stable_random(seed: int, *parts: object) -> random.Random:
    joined = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _choose_continuation_action(
    eval_case: FloorSetCase,
    blind_case: FloorSetCase,
    policy,
    role_evidence,
    state: ExecutionState,
    *,
    policy_name: str,
    max_candidates: int,
    sample_topk: int,
    seed: int,
    pool_step: int,
    candidate_key: str,
    continuation_step: int,
):
    remaining = [idx for idx in range(blind_case.block_count) if idx not in state.semantic_placed]
    candidates = generate_candidate_actions(
        blind_case,
        state,
        remaining_blocks=remaining,
        mode="semantic",
        max_per_primitive=8,
    )
    if not candidates:
        return _forced_progress_action(blind_case, state, remaining[0]), True

    if policy_name == "immediate_oracle":
        limited = candidates[: max(max_candidates * 2, max_candidates)]
        chosen = min(
            limited,
            key=lambda action: float(
                _quality_after_action(eval_case, blind_case, state, action)["quality_cost_runtime1"]
            ),
        )
        return chosen, False

    scored = []
    for action in candidates[: max(max_candidates * 4, max_candidates)]:
        policy_score, _components = _score_hierarchical_action(
            blind_case, policy, role_evidence, state, action
        )
        scored.append((float(policy_score), action))
    scored.sort(key=lambda item: item[0], reverse=True)
    if policy_name == "policy_greedy":
        return scored[0][1], False
    if policy_name == "policy_topk_sample":
        rng = _stable_random(seed, candidate_key, pool_step, continuation_step)
        choices = scored[: max(1, min(sample_topk, len(scored)))]
        return rng.choice(choices)[1], False
    raise ValueError(f"unknown continuation policy: {policy_name}")


def _continue_after_action(
    eval_case: FloorSetCase,
    blind_case: FloorSetCase,
    policy,
    role_evidence,
    state: ExecutionState,
    action,
    *,
    policy_name: str,
    horizon: int,
    max_candidates: int,
    sample_topk: int,
    seed: int,
    pool_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_key = _canonical_action_key(action)
    trial = _clone_state(state)
    executor = ActionExecutor(blind_case)
    executor.apply(trial, action)
    actions_taken = 1
    forced_count = 0
    no_progress = 0
    while (
        len(trial.semantic_placed) < blind_case.block_count
        and trial.step < blind_case.block_count * 4
        and actions_taken < horizon + 1
    ):
        remaining = [
            idx for idx in range(blind_case.block_count) if idx not in trial.semantic_placed
        ]
        if not remaining:
            break
        chosen, forced = _choose_continuation_action(
            eval_case,
            blind_case,
            policy,
            role_evidence,
            trial,
            policy_name=policy_name,
            max_candidates=max_candidates,
            sample_topk=sample_topk,
            seed=seed,
            pool_step=pool_step,
            candidate_key=candidate_key,
            continuation_step=actions_taken,
        )
        if no_progress >= 2 and chosen.block_index in trial.semantic_placed:
            chosen = _forced_progress_action(blind_case, trial, remaining[0])
            forced = True
        before = len(trial.semantic_placed)
        executor.apply(trial, chosen)
        forced_count += int(forced)
        no_progress = 0 if len(trial.semantic_placed) > before else no_progress + 1
        actions_taken += 1
    return _quality_from_positions(eval_case, trial.proposed_positions), {
        "continuation_policy": policy_name,
        "continuation_actions_taken": actions_taken - 1,
        "forced_continuation_count": forced_count,
        "completed": len(trial.semantic_placed) >= blind_case.block_count,
        "semantic_placed_fraction": len(trial.semantic_placed) / max(blind_case.block_count, 1),
    }


def _mean_normalized_quality(rows: list[dict[str, Any]], policies: list[str]) -> list[float]:
    values_by_policy = {
        policy: [
            float(row["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"])
            for row in rows
        ]
        for policy in policies
    }
    result = []
    for idx in range(len(rows)):
        normalized = []
        for policy in policies:
            values = values_by_policy[policy]
            lo = min(values)
            hi = max(values)
            normalized.append((values[idx] - lo) / max(hi - lo, 1e-9))
        result.append(sum(normalized) / max(len(normalized), 1))
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
                qi = float(
                    rows[i]["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"]
                )
                qj = float(
                    rows[j]["rollout_return_by_policy"][policy]["quality"]["quality_cost_runtime1"]
                )
                wins += int(qi < qj)
            targets[i, j] = wins / max(len(policies), 1)
    return targets


def _collect_case_seed(args_dict: dict[str, Any]) -> dict[str, Any]:
    _thread_limit()
    case_id = int(args_dict["case_id"])
    seed = int(args_dict["seed"])
    max_steps = int(args_dict["max_steps"])
    max_candidates = int(args_dict["max_candidates_per_step"])
    policies = [str(policy) for policy in args_dict["continuation_policies"]]

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
    eval_case = cases[case_id]
    blind_case = _target_blind_case(eval_case)
    dataset = build_bc_dataset_from_cases([eval_case], max_traces_per_case=1)
    policy, training_summary, _primitive_sets = _train_hierarchical_policy(dataset, job)
    role_evidence = label_case_roles(blind_case)
    state = ExecutionState(last_rollout_mode="semantic")
    executor = ActionExecutor(blind_case)
    seed_action = _seed_first_action(
        blind_case,
        [idx for idx in range(blind_case.block_count) if idx not in state.semantic_placed],
    )
    executor.apply(state, seed_action)

    pools = []
    no_progress = 0
    step = 0
    while (
        len(state.semantic_placed) < blind_case.block_count
        and state.step < blind_case.block_count * 4
        and step < max_steps
    ):
        remaining = [
            idx for idx in range(blind_case.block_count) if idx not in state.semantic_placed
        ]
        candidates = generate_candidate_actions(
            blind_case,
            state,
            remaining_blocks=remaining,
            mode="semantic",
            max_per_primitive=8,
        )
        if not candidates:
            chosen = _forced_progress_action(blind_case, state, remaining[0])
            executor.apply(state, chosen)
            step += 1
            continue

        scored = []
        for candidate in candidates:
            policy_score, _components = _score_hierarchical_action(
                blind_case, policy, role_evidence, state, candidate
            )
            scored.append((float(policy_score), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:max_candidates]

        candidate_rows = []
        for policy_rank, (_score, candidate) in enumerate(selected, start=1):
            per_policy = {}
            for continuation_policy in policies:
                quality, meta = _continue_after_action(
                    eval_case,
                    blind_case,
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
                    "action_key": _canonical_action_key(candidate),
                    "primitive": candidate.primitive.value,
                    "block_index": int(candidate.block_index),
                    "target_index": None
                    if candidate.target_index is None
                    else int(candidate.target_index),
                    "source": candidate.metadata.get("source"),
                    "intent_type": candidate.metadata.get("intent_type"),
                    "rollout_return_by_policy": per_policy,
                }
            )

        quality_values = _mean_normalized_quality(candidate_rows, policies)
        targets = _pair_targets({"candidate_rows": candidate_rows}, policies)
        transition_payloads = []
        for idx, (_score, candidate) in enumerate(selected):
            transition_payloads.append(
                build_transition_payload(
                    blind_case,
                    state,
                    candidate,
                    role_evidence=role_evidence,
                    pairwise_majority_target=targets[idx].tolist(),
                ).to_mapping()
            )
        oracle_index = min(range(len(quality_values)), key=lambda idx: float(quality_values[idx]))
        pools.append(
            {
                "case_id": str(eval_case.case_id),
                "case_index": case_id,
                "policy_seed": seed,
                "step": step,
                "quality_values": quality_values,
                "oracle_index": int(oracle_index),
                "candidate_rows": candidate_rows,
                "transition_payloads": transition_payloads,
                "payload_contract": {
                    "ranker_input": "transition_comparator",
                    "model_input_keys": list(SharedEncoderTransitionComparator.model_input_keys()),
                    "target_positions_in_payload": False,
                    "candidate_features_in_payload": False,
                    "manual_delta_rows_in_payload": False,
                },
            }
        )

        chosen = scored[0][1]
        if no_progress >= 2 and chosen.block_index in state.semantic_placed:
            chosen = _forced_progress_action(blind_case, state, remaining[0])
        before = len(state.semantic_placed)
        executor.apply(state, chosen)
        no_progress = 0 if len(state.semantic_placed) > before else no_progress + 1
        step += 1

    return {
        "case_id": str(eval_case.case_id),
        "case_index": case_id,
        "policy_seed": seed,
        "job": asdict(job),
        "training_summary": training_summary,
        "target_positions_used_by_model_payload": False,
        "pools": pools,
    }


def _train_transition_ranker(pools: list[dict[str, Any]], args: argparse.Namespace):
    torch.manual_seed(int(args.policy_seeds[0] if args.policy_seeds else 0))
    policies = [str(policy) for policy in args.continuation_policies]
    first_payload = pools[0]["transition_payloads"][0]
    block_feature_dim = len(first_payload["pre_block_features"][0])
    ranker = SharedEncoderTransitionComparator(
        block_feature_dim=block_feature_dim,
        action_token_dim=len(first_payload["action_token"]),
        hidden_dim=int(args.hidden_dim),
    )
    optimizer = torch.optim.Adam(ranker.parameters(), lr=float(args.ranker_lr))
    for _epoch in range(int(args.ranker_epochs)):
        for pool in pools:
            targets = _pair_targets(pool, policies)
            mask = ~torch.eye(targets.shape[0], dtype=torch.bool)
            logits = ranker.pair_logits(pool["transition_payloads"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return ranker, {
        "kind": "transition_comparator",
        "block_feature_dim": block_feature_dim,
        "action_token_dim": len(first_payload["action_token"]),
        "hidden_dim": int(args.hidden_dim),
        "shared_encoder_module": "SharedEncoderTransitionComparator.encoder",
        "model_input_keys": list(SharedEncoderTransitionComparator.model_input_keys()),
        "denied_payload_fields": sorted(DENIED_TRANSITION_PAYLOAD_FIELDS),
    }


def _evaluate(ranker, pools: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = []
    regrets = []
    top1 = 0
    rows = []
    for pool in pools:
        scores = ranker.score_candidates(pool["transition_payloads"]).detach()
        pred = int(scores.argmax().item())
        order = sorted(
            range(len(pool["quality_values"])), key=lambda idx: float(pool["quality_values"][idx])
        )
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
    _thread_limit()
    heldout = int(job["heldout"])
    pools = job["pools"]
    args = argparse.Namespace(**job["args"])
    train = [pool for pool in pools if int(pool["case_index"]) != heldout]
    eval_pools = [pool for pool in pools if int(pool["case_index"]) == heldout]
    ranker, _stats = _train_transition_ranker(train, args)
    train_eval = _evaluate(ranker, train)
    evaluation = _evaluate(ranker, eval_pools)
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


def _run_loco_splits(
    pools: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any] | None:
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
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(_run_loco_split_job, job) for job in jobs]
            for future in as_completed(futures):
                splits.append(future.result())
        splits.sort(key=lambda item: int(item["heldout_case_index"]))
    mean_rank = sum(split["evaluation"]["mean_selected_quality_rank"] for split in splits) / max(
        len(splits), 1
    )
    mean_regret = sum(
        split["evaluation"]["mean_selected_quality_regret"] for split in splits
    ) / max(len(splits), 1)
    mean_top1 = sum(split["evaluation"]["oracle_top1_selected_fraction"] for split in splits) / max(
        len(splits), 1
    )
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
    payload_guardrails = {
        "target_positions": payload["target_positions_used_by_model_payload"],
        "candidate_features": payload["candidate_features_used_by_model_payload"],
        "manual_delta_rows": payload["manual_delta_rows_used_by_model_payload"],
    }
    loco = payload.get("leave_one_case_out")
    loco_lines = (
        [
            f"- mean selected quality rank: `{loco['mean_selected_quality_rank']:.4f}`",
            f"- mean selected quality regret: `{loco['mean_selected_quality_regret']:.4f}`",
            f"- oracle top1 selected fraction: `{loco['oracle_top1_selected_fraction']:.4f}`",
            f"- LOCO gate pass: `{loco['gate']['pass']}`",
        ]
        if loco is not None
        else ["- skipped: eval mode did not request LOCO or fewer than two cases"]
    )
    lines = [
        "# Step6F Shared Pre/Post Transition Comparator",
        "",
        "- purpose: `shared encoder over pre/post typed graphs with action token`",
        f"- cases: `{payload['case_ids']}`",
        f"- policy seeds: `{payload['policy_seeds']}`",
        f"- continuation policies: `{payload['continuation_policies']}`",
        f"- encoder kind for collection policy: `{payload['encoder_kind']}`",
        f"- ranker input: `{payload['ranker_input']}`",
        f"- pool count: `{payload['pool_count']}`",
        f"- model input keys: `{payload['model_input_keys']}`",
        f"- payload guardrails: `{payload_guardrails}`",
        f"- micro mean selected quality rank: `{ev['mean_selected_quality_rank']:.4f}`",
        f"- micro mean selected quality regret: `{ev['mean_selected_quality_regret']:.4f}`",
        f"- micro oracle top1 selected fraction: `{ev['oracle_top1_selected_fraction']:.4f}`",
        f"- micro gate pass: `{gate['pass']}`",
        "",
        "## Leave-One-Case-Out",
        "",
        *loco_lines,
        "",
        "## Guardrail",
        "",
        "- payloads contain only pre/post typed graph tensors, action token, and pairwise target",
        "- model scoring ignores the pairwise target and consumes only the legal model input keys",
        "- target-position, candidate-feature, raw-logit, heuristic-score, case-id, and",
        "  manual-delta fields are denied by validator",
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
    collections.sort(key=lambda item: (int(item["case_index"]), int(item["policy_seed"])))
    pools = [pool for collection in collections for pool in collection["pools"]]
    if not pools:
        raise RuntimeError("no transition pools collected; cannot train Step6F comparator")

    ranker, stats = _train_transition_ranker(pools, args)
    evaluation = _evaluate(ranker, pools)
    gate = {
        "mean_selected_quality_rank_lt_4": evaluation["mean_selected_quality_rank"] < 4.0,
        "oracle_top1_selected_fraction_gt_0_30": evaluation["oracle_top1_selected_fraction"] > 0.30,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    should_run_loco = str(args.eval_mode) in {"loco", "micro-loco"}
    payload = {
        "status": "complete",
        "purpose": "Step6F shared pre/post transition comparator",
        "case_ids": [int(v) for v in args.case_ids],
        "policy_seeds": [int(v) for v in args.policy_seeds],
        "continuation_policies": [str(p) for p in args.continuation_policies],
        "encoder_kind": str(args.encoder_kind),
        "ranker_input": "transition_comparator",
        "eval_mode": str(args.eval_mode),
        "max_steps": int(args.max_steps),
        "max_candidates_per_step": int(args.max_candidates_per_step),
        "continuation_horizon": int(args.continuation_horizon),
        "collection_count": len(collections),
        "pool_count": len(pools),
        "workers_requested": max_workers,
        "model_stats": stats,
        "model_input_keys": list(SharedEncoderTransitionComparator.model_input_keys()),
        "target_positions_used_by_model_payload": False,
        "candidate_features_used_by_model_payload": False,
        "manual_delta_rows_used_by_model_payload": False,
        "denied_payload_fields": sorted(DENIED_TRANSITION_PAYLOAD_FIELDS),
        "evaluation": evaluation,
        "micro_gate": gate,
        "leave_one_case_out": _run_loco_splits(pools, args) if should_run_loco else None,
        "collections": collections,
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
                "loco_mean_selected_quality_rank": None
                if payload["leave_one_case_out"] is None
                else payload["leave_one_case_out"]["mean_selected_quality_rank"],
                "loco_oracle_top1_selected_fraction": None
                if payload["leave_one_case_out"] is None
                else payload["leave_one_case_out"]["oracle_top1_selected_fraction"],
                "loco_gate_pass": None
                if payload["leave_one_case_out"] is None
                else payload["leave_one_case_out"]["gate"]["pass"],
                "target_positions_used_by_model_payload": False,
                "candidate_features_used_by_model_payload": False,
                "manual_delta_rows_used_by_model_payload": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
