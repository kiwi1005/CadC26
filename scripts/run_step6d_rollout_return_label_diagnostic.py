#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step6D diagnostic: compare immediate quality oracle against "
            "bounded rollout-return oracle on zero-top1 held-out cases."
        )
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=[1, 4, 6])
    parser.add_argument("--policy-seeds", nargs="*", type=int, default=[0, 1, 2])
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
    parser.add_argument(
        "--continuation-horizon",
        type=int,
        default=8,
        help="Maximum greedy continuation actions after the candidate action.",
    )
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "research"
        / "step6d_rollout_return_label_diagnostic.json",
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


def _choose_policy_action(case, policy, role_evidence, state: ExecutionState, max_candidates: int):
    remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
    candidates = generate_candidate_actions(
        case,
        state,
        remaining_blocks=remaining,
        mode="semantic",
        max_per_primitive=8,
    )
    if not candidates:
        return _forced_progress_action(case, state, remaining[0]), [], True
    scored = [
        (_score_hierarchical_action(case, policy, role_evidence, state, candidate)[0], candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[:max_candidates], False


def _continue_after_action(
    case,
    policy,
    role_evidence,
    state: ExecutionState,
    action,
    *,
    horizon: int,
    max_candidates: int,
) -> tuple[ExecutionState, dict[str, Any]]:
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
        chosen, _scored, forced = _choose_policy_action(
            case,
            policy,
            role_evidence,
            trial,
            max_candidates,
        )
        if no_progress >= 2 and chosen.block_index in trial.semantic_placed:
            chosen = _forced_progress_action(case, trial, remaining[0])
            forced = True
        before = len(trial.semantic_placed)
        executor.apply(trial, chosen)
        forced_count += int(forced)
        no_progress = 0 if len(trial.semantic_placed) > before else no_progress + 1
        actions_taken += 1
    return trial, {
        "continuation_actions_taken": actions_taken - 1,
        "forced_continuation_count": forced_count,
        "completed": len(trial.semantic_placed) >= case.block_count,
        "semantic_placed_fraction": len(trial.semantic_placed) / max(case.block_count, 1),
    }


def _evaluate_pool(case, policy, role_evidence, state: ExecutionState, selected, *, horizon: int, max_candidates: int) -> dict[str, Any]:
    candidate_rows = []
    for policy_rank, (policy_score, candidate) in enumerate(selected, start=1):
        immediate_state = _clone_state(state)
        ActionExecutor(case).apply(immediate_state, candidate)
        immediate = _quality_from_positions(case, immediate_state.proposed_positions)
        rollout_state, rollout_meta = _continue_after_action(
            case,
            policy,
            role_evidence,
            state,
            candidate,
            horizon=horizon,
            max_candidates=max_candidates,
        )
        rollout = _quality_from_positions(case, rollout_state.proposed_positions)
        candidate_rows.append(
            {
                "policy_rank": policy_rank,
                "policy_score": float(policy_score),
                "action_key": canonical_action_key(candidate),
                "primitive": candidate.primitive.value,
                "block_index": int(candidate.block_index),
                "target_index": None if candidate.target_index is None else int(candidate.target_index),
                "source": candidate.metadata.get("source"),
                "intent_type": candidate.metadata.get("intent_type"),
                "immediate": immediate,
                "rollout_return": rollout,
                "rollout_meta": rollout_meta,
            }
        )
    immediate_order = sorted(
        range(len(candidate_rows)),
        key=lambda idx: candidate_rows[idx]["immediate"]["quality_cost_runtime1"],
    )
    rollout_order = sorted(
        range(len(candidate_rows)),
        key=lambda idx: candidate_rows[idx]["rollout_return"]["quality_cost_runtime1"],
    )
    for rank, idx in enumerate(immediate_order, start=1):
        candidate_rows[idx]["immediate_rank"] = rank
    for rank, idx in enumerate(rollout_order, start=1):
        candidate_rows[idx]["rollout_return_rank"] = rank
    immediate_best = immediate_order[0]
    rollout_best = rollout_order[0]
    immediate_best_rollout_rank = candidate_rows[immediate_best]["rollout_return_rank"]
    rollout_best_immediate_rank = candidate_rows[rollout_best]["immediate_rank"]
    return {
        "candidate_rows": candidate_rows,
        "immediate_oracle_index": int(immediate_best),
        "rollout_return_oracle_index": int(rollout_best),
        "oracle_agreement": immediate_best == rollout_best,
        "immediate_best_rollout_rank": int(immediate_best_rollout_rank),
        "rollout_best_immediate_rank": int(rollout_best_immediate_rank),
        "immediate_best_rollout_regret": float(
            candidate_rows[immediate_best]["rollout_return"]["quality_cost_runtime1"]
            - candidate_rows[rollout_best]["rollout_return"]["quality_cost_runtime1"]
        ),
        "rollout_best_immediate_regret": float(
            candidate_rows[rollout_best]["immediate"]["quality_cost_runtime1"]
            - candidate_rows[immediate_best]["immediate"]["quality_cost_runtime1"]
        ),
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
    horizon = int(args_dict["continuation_horizon"])
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
            rows.append(
                {
                    "step": step,
                    "candidate_count": 0,
                    "evaluated_candidate_count": 0,
                    "forced_progress": True,
                }
            )
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
                horizon=horizon,
                max_candidates=max_candidates,
            )
            chosen = scored[0][1]
            if no_progress >= 2 and chosen.block_index in state.semantic_placed:
                chosen = _forced_progress_action(case, state, remaining[0])
            top_rows = sorted(
                pool_eval["candidate_rows"],
                key=lambda row: row["rollout_return_rank"],
            )[:5]
            rows.append(
                {
                    "step": step,
                    "candidate_count": len(candidates),
                    "evaluated_candidate_count": len(selected),
                    "chosen_action_key": canonical_action_key(chosen),
                    "oracle_agreement": pool_eval["oracle_agreement"],
                    "immediate_best_rollout_rank": pool_eval["immediate_best_rollout_rank"],
                    "rollout_best_immediate_rank": pool_eval["rollout_best_immediate_rank"],
                    "immediate_best_rollout_regret": pool_eval["immediate_best_rollout_regret"],
                    "rollout_best_immediate_regret": pool_eval["rollout_best_immediate_regret"],
                    "immediate_oracle_action": pool_eval["candidate_rows"][pool_eval["immediate_oracle_index"]],
                    "rollout_return_oracle_action": pool_eval["candidate_rows"][pool_eval["rollout_return_oracle_index"]],
                    "top_rollout_return_rows": top_rows,
                }
            )
        before = len(state.semantic_placed)
        executor.apply(state, chosen)
        no_progress = 0 if len(state.semantic_placed) > before else no_progress + 1
        step += 1

    evaluated = [row for row in rows if row.get("evaluated_candidate_count", 0) > 0]
    agreement_count = sum(1 for row in evaluated if row["oracle_agreement"])
    mismatch_rows = [row for row in evaluated if not row["oracle_agreement"]]
    return {
        "case_id": str(case.case_id),
        "case_index": case_id,
        "policy_seed": seed,
        "job": asdict(job),
        "training_summary": training_summary,
        "steps_audited": len(rows),
        "evaluated_pool_count": len(evaluated),
        "oracle_agreement_fraction": agreement_count / max(len(evaluated), 1),
        "oracle_mismatch_fraction": len(mismatch_rows) / max(len(evaluated), 1),
        "mean_immediate_best_rollout_rank": sum(
            int(row["immediate_best_rollout_rank"]) for row in evaluated
        )
        / max(len(evaluated), 1),
        "mean_immediate_best_rollout_regret": sum(
            float(row["immediate_best_rollout_regret"]) for row in evaluated
        )
        / max(len(evaluated), 1),
        "mean_rollout_best_immediate_rank": sum(
            int(row["rollout_best_immediate_rank"]) for row in evaluated
        )
        / max(len(evaluated), 1),
        "mean_rollout_best_immediate_regret": sum(
            float(row["rollout_best_immediate_regret"]) for row in evaluated
        )
        / max(len(evaluated), 1),
        "rows": rows,
    }


def _aggregate(results: list[dict[str, Any]], workers: int, args: argparse.Namespace) -> dict[str, Any]:
    evaluated = [result for result in results if result["evaluated_pool_count"] > 0]
    case_groups: dict[int, list[dict[str, Any]]] = {}
    for result in evaluated:
        case_groups.setdefault(int(result["case_index"]), []).append(result)
    per_case = []
    for case_index, case_results in sorted(case_groups.items()):
        denom = max(len(case_results), 1)
        per_case.append(
            {
                "case_index": case_index,
                "case_id": case_results[0]["case_id"],
                "collection_count": len(case_results),
                "evaluated_pool_count": sum(r["evaluated_pool_count"] for r in case_results),
                "oracle_agreement_fraction": sum(r["oracle_agreement_fraction"] for r in case_results) / denom,
                "oracle_mismatch_fraction": sum(r["oracle_mismatch_fraction"] for r in case_results) / denom,
                "mean_immediate_best_rollout_rank": sum(
                    r["mean_immediate_best_rollout_rank"] for r in case_results
                )
                / denom,
                "mean_immediate_best_rollout_regret": sum(
                    r["mean_immediate_best_rollout_regret"] for r in case_results
                )
                / denom,
            }
        )
    denom = max(len(evaluated), 1)
    mismatch = sum(result["oracle_mismatch_fraction"] for result in evaluated) / denom
    agreement = sum(result["oracle_agreement_fraction"] for result in evaluated) / denom
    rollout_rank = sum(result["mean_immediate_best_rollout_rank"] for result in evaluated) / denom
    rollout_regret = sum(result["mean_immediate_best_rollout_regret"] for result in evaluated) / denom
    horizon_gate = {
        "oracle_mismatch_fraction_lt_0_25": mismatch < 0.25,
        "mean_immediate_best_rollout_rank_le_2": rollout_rank <= 2.0,
        "mean_immediate_best_rollout_regret_lt_0_25": rollout_regret < 0.25,
    }
    horizon_gate["immediate_label_ok"] = all(bool(value) for value in horizon_gate.values())
    return {
        "status": "complete",
        "purpose": "Step6D rollout-return label horizon diagnostic",
        "case_ids": [int(v) for v in args.case_ids],
        "policy_seeds": [int(v) for v in args.policy_seeds],
        "collection_count": len(results),
        "case_count": len(case_groups),
        "workers_requested": workers,
        "max_steps": int(args.max_steps),
        "max_candidates_per_step": int(args.max_candidates_per_step),
        "continuation_horizon": int(args.continuation_horizon),
        "oracle_agreement_fraction": agreement,
        "oracle_mismatch_fraction": mismatch,
        "mean_immediate_best_rollout_rank": rollout_rank,
        "mean_immediate_best_rollout_regret": rollout_regret,
        "horizon_gate": horizon_gate,
        "per_case": per_case,
        "results": sorted(results, key=lambda item: (int(item["case_index"]), int(item["policy_seed"]))),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        "| {case} | {collections} | {pools} | {agree:.4f} | {mismatch:.4f} | {rank:.4f} | {regret:.4f} |".format(
            case=item["case_id"],
            collections=item["collection_count"],
            pools=item["evaluated_pool_count"],
            agree=item["oracle_agreement_fraction"],
            mismatch=item["oracle_mismatch_fraction"],
            rank=item["mean_immediate_best_rollout_rank"],
            regret=item["mean_immediate_best_rollout_regret"],
        )
        for item in payload["per_case"]
    ]
    gate = payload["horizon_gate"]
    lines = [
        "# Step6D Rollout-Return Label Diagnostic",
        "",
        "- purpose: `compare immediate action-quality labels against bounded rollout-return labels`",
        f"- case ids: `{payload['case_ids']}`",
        f"- policy seeds: `{payload['policy_seeds']}`",
        f"- collections: `{payload['collection_count']}`",
        f"- continuation horizon: `{payload['continuation_horizon']}`",
        f"- max steps: `{payload['max_steps']}`",
        f"- max candidates per step: `{payload['max_candidates_per_step']}`",
        f"- oracle agreement fraction: `{payload['oracle_agreement_fraction']:.4f}`",
        f"- oracle mismatch fraction: `{payload['oracle_mismatch_fraction']:.4f}`",
        f"- mean immediate-best rollout rank: `{payload['mean_immediate_best_rollout_rank']:.4f}`",
        f"- mean immediate-best rollout regret: `{payload['mean_immediate_best_rollout_regret']:.4f}`",
        "",
        "## Horizon Gate",
        "",
        f"- oracle mismatch fraction < 0.25: `{gate['oracle_mismatch_fraction_lt_0_25']}`",
        f"- mean immediate-best rollout rank <= 2: `{gate['mean_immediate_best_rollout_rank_le_2']}`",
        f"- mean immediate-best rollout regret < 0.25: `{gate['mean_immediate_best_rollout_regret_lt_0_25']}`",
        f"- immediate label ok: `{gate['immediate_label_ok']}`",
        "",
        "## Per Case",
        "",
        "| Case | collections | pools | agreement | mismatch | immediate-best rollout rank | immediate-best rollout regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Interpretation: if this gate fails, one-step immediate `quality_after_action` labels are not reliable enough for the zero-top1 held-out cases, and Step6D should prioritize rollout-return/action-value labels before widening the ranker or re-enabling pairwise auxiliary loss.",
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
                "collection_count": payload["collection_count"],
                "oracle_agreement_fraction": payload["oracle_agreement_fraction"],
                "oracle_mismatch_fraction": payload["oracle_mismatch_fraction"],
                "mean_immediate_best_rollout_rank": payload[
                    "mean_immediate_best_rollout_rank"
                ],
                "mean_immediate_best_rollout_regret": payload[
                    "mean_immediate_best_rollout_regret"
                ],
                "immediate_label_ok": payload["horizon_gate"]["immediate_label_ok"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
