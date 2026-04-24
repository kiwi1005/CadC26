#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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

from puzzleplace.actions import (  # noqa: E402
    ActionExecutor,
    ExecutionState,
    canonical_action_key,
    generate_candidate_actions,
)
from puzzleplace.eval import evaluate_positions  # noqa: E402
from puzzleplace.repair.finalizer import finalize_layout  # noqa: E402
from puzzleplace.roles import label_case_roles  # noqa: E402
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases  # noqa: E402

from scripts.run_step6_hierarchical_rollout_control_audit import (  # noqa: E402
    RolloutJob,
    _forced_progress_action,
    _score_hierarchical_action,
    _seed_first_action,
    _train_hierarchical_policy,
)
from scripts.run_step6c_hierarchical_quality_alignment_audit import _quality_after_action  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step6E diagnostic: measure whether rollout-return labels are stable "
            "across materially different continuation policies."
        )
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=[1, 4, 6])
    parser.add_argument("--policy-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--continuation-policies",
        nargs="*",
        choices=["policy_greedy", "immediate_oracle", "policy_topk_sample"],
        default=["policy_greedy", "immediate_oracle", "policy_topk_sample"],
    )
    parser.add_argument("--sample-topk", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--policy-epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--primitive-set-weight", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument(
        "--encoder-kind",
        choices=["graph", "relation_aware", "typed_constraint_graph", "typed_constraint_graph_no_anchor", "typed_constraint_graph_no_boundary", "typed_constraint_graph_no_groups"],
        default="relation_aware",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-candidates-per-step", type=int, default=16)
    parser.add_argument("--continuation-horizon", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker cap for independent case/seed or LOCO split jobs. Keep at 1 for smoke; use 48 only after the relevant smoke/neutral gate passes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6e_rollout_label_stability.json",
    )
    return parser.parse_args()


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


def _quality_from_positions(case, positions: dict[int, tuple[float, float, float, float]]) -> dict[str, Any]:
    repair = finalize_layout(case, positions)
    evaluation = evaluate_positions(case, repair.positions, runtime=1.0, median_runtime=1.0)
    quality = evaluation["quality"]
    return {
        "quality_cost_runtime1": float(quality["quality_cost_runtime1"]),
        "HPWLgap": float(quality["HPWLgap"]),
        "Areagap_bbox": float(quality["Areagap_bbox"]),
        "Violationsrelative": float(quality["Violationsrelative"]),
        "feasible": bool(quality["feasible"]),
        "repair": {
            "hard_feasible_after": bool(repair.report.hard_feasible_after),
            "mean_displacement": float(repair.report.mean_displacement),
            "moved_block_count": int(repair.report.moved_block_count),
            "shelf_fallback_count": int(repair.report.shelf_fallback_count),
        },
    }


def _stable_random(seed: int, *parts: object) -> random.Random:
    joined = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _choose_continuation_action(
    case,
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
    remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
    candidates = generate_candidate_actions(
        case,
        state,
        remaining_blocks=remaining,
        mode="semantic",
        max_per_primitive=8,
    )
    if not candidates:
        return _forced_progress_action(case, state, remaining[0]), True

    if policy_name == "immediate_oracle":
        limited = candidates[: max(max_candidates * 2, max_candidates)]
        chosen = min(
            limited,
            key=lambda action: float(
                _quality_after_action(case, state, action)["quality"]["quality_cost_runtime1"]
            ),
        )
        return chosen, False

    scored = []
    for action in candidates[: max(max_candidates * 4, max_candidates)]:
        policy_score, _components = _score_hierarchical_action(
            case, policy, role_evidence, state, action
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
    case,
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
    candidate_key = canonical_action_key(action)
    trial = _clone_state(state)
    executor = ActionExecutor(case)
    executor.apply(trial, action)
    actions_taken = 1
    forced_count = 0
    no_progress = 0
    while (
        len(trial.semantic_placed) < case.block_count
        and trial.step < case.block_count * 4
        and actions_taken < horizon + 1
    ):
        remaining = [idx for idx in range(case.block_count) if idx not in trial.semantic_placed]
        if not remaining:
            break
        chosen, forced = _choose_continuation_action(
            case,
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
            chosen = _forced_progress_action(case, trial, remaining[0])
            forced = True
        before = len(trial.semantic_placed)
        executor.apply(trial, chosen)
        forced_count += int(forced)
        no_progress = 0 if len(trial.semantic_placed) > before else no_progress + 1
        actions_taken += 1
    return _quality_from_positions(case, trial.proposed_positions), {
        "continuation_policy": policy_name,
        "continuation_actions_taken": actions_taken - 1,
        "forced_continuation_count": forced_count,
        "completed": len(trial.semantic_placed) >= case.block_count,
        "semantic_placed_fraction": len(trial.semantic_placed) / max(case.block_count, 1),
    }


def _evaluate_pool(
    case,
    policy,
    role_evidence,
    state: ExecutionState,
    selected,
    *,
    policies: list[str],
    horizon: int,
    max_candidates: int,
    sample_topk: int,
    seed: int,
    pool_step: int,
) -> dict[str, Any]:
    rows = []
    for policy_rank, (policy_score, candidate) in enumerate(selected, start=1):
        immediate = _quality_after_action(case, state, candidate)["quality"]
        row = {
            "policy_rank": policy_rank,
            "policy_score": float(policy_score),
            "action_key": canonical_action_key(candidate),
            "primitive": candidate.primitive.value,
            "block_index": int(candidate.block_index),
            "target_index": None if candidate.target_index is None else int(candidate.target_index),
            "source": candidate.metadata.get("source"),
            "intent_type": candidate.metadata.get("intent_type"),
            "immediate": {
                "quality_cost_runtime1": float(immediate["quality_cost_runtime1"]),
                "HPWLgap": float(immediate["HPWLgap"]),
                "Areagap_bbox": float(immediate["Areagap_bbox"]),
                "Violationsrelative": float(immediate["Violationsrelative"]),
                "feasible": bool(immediate["feasible"]),
            },
            "rollout_return_by_policy": {},
        }
        for continuation_policy in policies:
            quality, meta = _continue_after_action(
                case,
                policy,
                role_evidence,
                state,
                candidate,
                policy_name=continuation_policy,
                horizon=horizon,
                max_candidates=max_candidates,
                sample_topk=sample_topk,
                seed=seed,
                pool_step=pool_step,
            )
            row["rollout_return_by_policy"][continuation_policy] = {
                "quality": quality,
                "meta": meta,
            }
        rows.append(row)

    immediate_oracle_index = min(
        range(len(rows)), key=lambda idx: rows[idx]["immediate"]["quality_cost_runtime1"]
    )
    policy_oracle_indexes = {}
    for continuation_policy in policies:
        order = sorted(
            range(len(rows)),
            key=lambda idx: rows[idx]["rollout_return_by_policy"][continuation_policy]["quality"][
                "quality_cost_runtime1"
            ],
        )
        policy_oracle_indexes[continuation_policy] = int(order[0])
        for rank, idx in enumerate(order, start=1):
            rows[idx]["rollout_return_by_policy"][continuation_policy]["quality_rank"] = rank
            rows[idx]["rollout_return_by_policy"][continuation_policy]["quality_regret"] = float(
                rows[idx]["rollout_return_by_policy"][continuation_policy]["quality"][
                    "quality_cost_runtime1"
                ]
                - rows[order[0]]["rollout_return_by_policy"][continuation_policy]["quality"][
                    "quality_cost_runtime1"
                ]
            )

    oracle_keys = {rows[idx]["action_key"] for idx in policy_oracle_indexes.values()}
    immediate_key = rows[immediate_oracle_index]["action_key"]
    pairwise = {}
    for left_idx, left in enumerate(policies):
        for right in policies[left_idx + 1 :]:
            pairwise[f"{left}__{right}"] = bool(
                policy_oracle_indexes[left] == policy_oracle_indexes[right]
            )
    immediate_matches = {
        continuation_policy: bool(immediate_oracle_index == oracle_index)
        for continuation_policy, oracle_index in policy_oracle_indexes.items()
    }
    reference_policy = policies[0]
    reference_oracle = policy_oracle_indexes[reference_policy]
    reference_regret_under_policy = {}
    for continuation_policy in policies:
        reference_regret_under_policy[continuation_policy] = rows[reference_oracle][
            "rollout_return_by_policy"
        ][continuation_policy]["quality_regret"]

    return {
        "candidate_rows": rows,
        "immediate_oracle_index": int(immediate_oracle_index),
        "immediate_oracle_action_key": immediate_key,
        "rollout_oracle_indexes": policy_oracle_indexes,
        "rollout_oracle_action_keys": {
            continuation_policy: rows[idx]["action_key"]
            for continuation_policy, idx in policy_oracle_indexes.items()
        },
        "rollout_unique_oracle_count": len(oracle_keys),
        "all_rollout_policies_agree": len(oracle_keys) == 1,
        "pairwise_rollout_policy_agreement": pairwise,
        "immediate_matches_rollout_policy": immediate_matches,
        "reference_policy": reference_policy,
        "reference_oracle_regret_under_policy": reference_regret_under_policy,
    }


def _run_case_seed(args_dict: dict[str, Any]) -> dict[str, Any]:
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

    rows = []
    no_progress = 0
    step = 0
    while (
        len(state.semantic_placed) < case.block_count
        and state.step < case.block_count * 4
        and step < max_steps
    ):
        remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
        candidates = generate_candidate_actions(
            case,
            state,
            remaining_blocks=remaining,
            mode="semantic",
            max_per_primitive=8,
        )
        if not candidates:
            chosen = _forced_progress_action(case, state, remaining[0])
            rows.append({"step": step, "candidate_count": 0, "forced_progress": True})
        else:
            scored = [
                (_score_hierarchical_action(case, policy, role_evidence, state, candidate)[0], candidate)
                for candidate in candidates
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = scored[:max_candidates]
            pool_eval = _evaluate_pool(
                case,
                policy,
                role_evidence,
                state,
                selected,
                policies=policies,
                horizon=int(args_dict["continuation_horizon"]),
                max_candidates=max_candidates,
                sample_topk=int(args_dict["sample_topk"]),
                seed=seed,
                pool_step=step,
            )
            chosen = scored[0][1]
            if no_progress >= 2 and chosen.block_index in state.semantic_placed:
                chosen = _forced_progress_action(case, state, remaining[0])
            rows.append(
                {
                    "step": step,
                    "candidate_count": len(candidates),
                    "evaluated_candidate_count": len(selected),
                    "chosen_action_key": canonical_action_key(chosen),
                    **pool_eval,
                }
            )
        before = len(state.semantic_placed)
        executor.apply(state, chosen)
        no_progress = 0 if len(state.semantic_placed) > before else no_progress + 1
        step += 1

    evaluated = [row for row in rows if row.get("evaluated_candidate_count", 0) > 0]
    pairwise_keys = [
        f"{left}__{right}"
        for left_idx, left in enumerate(policies)
        for right in policies[left_idx + 1 :]
    ]
    return {
        "case_id": str(case.case_id),
        "case_index": case_id,
        "policy_seed": seed,
        "job": asdict(job),
        "training_summary": training_summary,
        "steps_audited": len(rows),
        "evaluated_pool_count": len(evaluated),
        "all_rollout_policies_agree_fraction": sum(
            1 for row in evaluated if row["all_rollout_policies_agree"]
        )
        / max(len(evaluated), 1),
        "mean_rollout_unique_oracle_count": sum(
            int(row["rollout_unique_oracle_count"]) for row in evaluated
        )
        / max(len(evaluated), 1),
        "pairwise_rollout_policy_agreement_fraction": {
            key: sum(
                1 for row in evaluated if row["pairwise_rollout_policy_agreement"][key]
            )
            / max(len(evaluated), 1)
            for key in pairwise_keys
        },
        "immediate_matches_rollout_policy_fraction": {
            continuation_policy: sum(
                1 for row in evaluated if row["immediate_matches_rollout_policy"][continuation_policy]
            )
            / max(len(evaluated), 1)
            for continuation_policy in policies
        },
        "mean_reference_oracle_regret_under_policy": {
            continuation_policy: sum(
                float(row["reference_oracle_regret_under_policy"][continuation_policy])
                for row in evaluated
            )
            / max(len(evaluated), 1)
            for continuation_policy in policies
        },
        "rows": rows,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _aggregate(results: list[dict[str, Any]], workers: int, args: argparse.Namespace) -> dict[str, Any]:
    evaluated = [result for result in results if result["evaluated_pool_count"] > 0]
    policies = [str(policy) for policy in args.continuation_policies]
    pairwise_keys = [
        f"{left}__{right}"
        for left_idx, left in enumerate(policies)
        for right in policies[left_idx + 1 :]
    ]
    case_groups: dict[int, list[dict[str, Any]]] = {}
    for result in evaluated:
        case_groups.setdefault(int(result["case_index"]), []).append(result)
    per_case = []
    for case_index, case_results in sorted(case_groups.items()):
        per_case.append(
            {
                "case_index": case_index,
                "case_id": case_results[0]["case_id"],
                "collection_count": len(case_results),
                "evaluated_pool_count": sum(r["evaluated_pool_count"] for r in case_results),
                "all_rollout_policies_agree_fraction": _mean(
                    [r["all_rollout_policies_agree_fraction"] for r in case_results]
                ),
                "mean_rollout_unique_oracle_count": _mean(
                    [r["mean_rollout_unique_oracle_count"] for r in case_results]
                ),
                "pairwise_rollout_policy_agreement_fraction": {
                    key: _mean(
                        [r["pairwise_rollout_policy_agreement_fraction"][key] for r in case_results]
                    )
                    for key in pairwise_keys
                },
                "immediate_matches_rollout_policy_fraction": {
                    policy: _mean(
                        [r["immediate_matches_rollout_policy_fraction"][policy] for r in case_results]
                    )
                    for policy in policies
                },
            }
        )
    stability_gate = {
        "all_policy_agreement_ge_0_60": _mean(
            [r["all_rollout_policies_agree_fraction"] for r in evaluated]
        )
        >= 0.60,
        "mean_unique_oracle_count_le_1_5": _mean(
            [r["mean_rollout_unique_oracle_count"] for r in evaluated]
        )
        <= 1.5,
        "policy_greedy_immediate_oracle_agreement_ge_0_50": _mean(
            [
                r["pairwise_rollout_policy_agreement_fraction"].get(
                    "policy_greedy__immediate_oracle", 0.0
                )
                for r in evaluated
            ]
        )
        >= 0.50,
    }
    stability_gate["rollout_label_stable_enough"] = all(stability_gate.values())
    return {
        "status": "complete",
        "purpose": "Step6E rollout label stability diagnostic",
        "case_ids": [int(v) for v in args.case_ids],
        "policy_seeds": [int(v) for v in args.policy_seeds],
        "continuation_policies": policies,
        "collection_count": len(results),
        "case_count": len(case_groups),
        "workers_requested": workers,
        "max_steps": int(args.max_steps),
        "max_candidates_per_step": int(args.max_candidates_per_step),
        "continuation_horizon": int(args.continuation_horizon),
        "sample_topk": int(args.sample_topk),
        "all_rollout_policies_agree_fraction": _mean(
            [r["all_rollout_policies_agree_fraction"] for r in evaluated]
        ),
        "mean_rollout_unique_oracle_count": _mean(
            [r["mean_rollout_unique_oracle_count"] for r in evaluated]
        ),
        "pairwise_rollout_policy_agreement_fraction": {
            key: _mean([r["pairwise_rollout_policy_agreement_fraction"][key] for r in evaluated])
            for key in pairwise_keys
        },
        "immediate_matches_rollout_policy_fraction": {
            policy: _mean([r["immediate_matches_rollout_policy_fraction"][policy] for r in evaluated])
            for policy in policies
        },
        "mean_reference_oracle_regret_under_policy": {
            policy: _mean([r["mean_reference_oracle_regret_under_policy"][policy] for r in evaluated])
            for policy in policies
        },
        "stability_gate": stability_gate,
        "per_case": per_case,
        "results": sorted(results, key=lambda item: (int(item["case_index"]), int(item["policy_seed"]))),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload["per_case"]:
        rows.append(
            "| {case} | {collections} | {pools} | {agree:.4f} | {unique:.4f} | {imm:.4f} |".format(
                case=item["case_id"],
                collections=item["collection_count"],
                pools=item["evaluated_pool_count"],
                agree=item["all_rollout_policies_agree_fraction"],
                unique=item["mean_rollout_unique_oracle_count"],
                imm=item["immediate_matches_rollout_policy_fraction"].get("policy_greedy", math.nan),
            )
        )
    lines = [
        "# Step6E Rollout Label Stability Diagnostic",
        "",
        "- purpose: `test whether rollout-return oracle labels survive different continuation policies`",
        f"- case ids: `{payload['case_ids']}`",
        f"- policy seeds: `{payload['policy_seeds']}`",
        f"- continuation policies: `{payload['continuation_policies']}`",
        f"- collection count: `{payload['collection_count']}`",
        f"- continuation horizon: `{payload['continuation_horizon']}`",
        f"- all rollout policies agree fraction: `{payload['all_rollout_policies_agree_fraction']:.4f}`",
        f"- mean rollout unique oracle count: `{payload['mean_rollout_unique_oracle_count']:.4f}`",
        f"- rollout label stable enough: `{payload['stability_gate']['rollout_label_stable_enough']}`",
        "",
        "## Pairwise rollout-policy oracle agreement",
        "",
        *[
            f"- {key}: `{value:.4f}`"
            for key, value in payload["pairwise_rollout_policy_agreement_fraction"].items()
        ],
        "",
        "## Immediate-oracle match rate",
        "",
        *[
            f"- immediate vs {key}: `{value:.4f}`"
            for key, value in payload["immediate_matches_rollout_policy_fraction"].items()
        ],
        "",
        "## Per Case",
        "",
        "| Case | collections | pools | all-policy agree | unique rollout oracles | immediate vs policy_greedy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Interpretation: if continuation-policy agreement is low, rollout-return labels are not a clean replacement target. The next research move should denoise labels or learn from robust cross-continuation advantages, not tune ranker weights.",
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
            "continuation_policies": [str(policy) for policy in args.continuation_policies],
            "sample_topk": int(args.sample_topk),
        }
        for case_id in args.case_ids
        for seed in args.policy_seeds
    ]
    max_workers = max(1, min(int(args.workers), len(jobs)))
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_case_seed, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    payload = _aggregate(results, max_workers, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "case_ids": payload["case_ids"],
                "policy_seeds": payload["policy_seeds"],
                "continuation_policies": payload["continuation_policies"],
                "collection_count": payload["collection_count"],
                "all_rollout_policies_agree_fraction": payload[
                    "all_rollout_policies_agree_fraction"
                ],
                "mean_rollout_unique_oracle_count": payload["mean_rollout_unique_oracle_count"],
                "pairwise_rollout_policy_agreement_fraction": payload[
                    "pairwise_rollout_policy_agreement_fraction"
                ],
                "rollout_label_stable_enough": payload["stability_gate"][
                    "rollout_label_stable_enough"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
