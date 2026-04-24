#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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
    canonical_action_key,
    generate_candidate_actions,
)
from puzzleplace.models import HierarchicalSetPolicy  # noqa: E402
from puzzleplace.roles import label_case_roles  # noqa: E402
from puzzleplace.rollout.semantic import (  # noqa: E402
    _forced_progress_action,
    _score_action,
    _seed_first_action,
    _semantic_heuristic_score,
)
from puzzleplace.train import BCStepRecord, build_bc_dataset_from_cases, load_validation_cases  # noqa: E402
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
    encoder_kind: str = "graph"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step6C hierarchical rollout-control audit without contest integration."
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--primitive-set-weight", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument(
        "--encoder-kind",
        choices=["graph", "relation_aware", "typed_constraint_graph", "typed_constraint_graph_no_anchor", "typed_constraint_graph_no_boundary", "typed_constraint_graph_no_groups"],
        default="graph",
        help="Policy state encoder used by this audit sidecar.",
    )
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6c_hierarchical_rollout_control_audit.json",
    )
    return parser.parse_args()


def _state_from_record(record: BCStepRecord) -> ExecutionState:
    placed = dict(record.placements)
    return ExecutionState(
        placements=placed,
        proposed_positions=placed,
        shape_assigned=set(placed),
        semantic_placed=set(placed),
        physically_placed=set(placed),
        step=record.state_step,
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
        candidate for candidate in candidates if actions_match(candidate, record.action, mode="semantic")
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


def _masked_block_logits(record: BCStepRecord, logits: torch.Tensor) -> torch.Tensor:
    if not record.legal_block_mask:
        return logits
    mask = torch.tensor(record.legal_block_mask, dtype=torch.bool, device=logits.device)
    if mask.ndim == 1 and mask.numel() == logits.shape[0] and mask.any():
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return logits


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
                state_step=record.state_step,
            )
            targets = action_to_targets(record.action)
            block_index = int(targets["block_index"])
            block_logits = _masked_block_logits(record, output.block_logits)
            block_loss = torch.nn.functional.cross_entropy(
                block_logits.unsqueeze(0), torch.tensor([block_index])
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
    case,
    policy: HierarchicalSetPolicy | None,
    role_evidence,
    state: ExecutionState,
    action: TypedAction,
) -> tuple[float, dict[str, Any]]:
    heuristic = _semantic_heuristic_score(action)
    if policy is None:
        return heuristic, {
            "block_logit": None,
            "primitive_logit": None,
            "target_logit": None,
            "heuristic_score": heuristic,
            "components_used": ["semantic_heuristic"],
        }
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
    components = ["block", "primitive_by_block", "semantic_heuristic"]
    if action.target_index is not None:
        target_logit = float(output.target_logits[action.block_index, action.target_index].item())
        components.append("target")
    score = block_logit + primitive_logit + target_logit + heuristic
    return score, {
        "block_logit": block_logit,
        "primitive_logit": primitive_logit,
        "target_logit": target_logit if action.target_index is not None else None,
        "heuristic_score": heuristic,
        "components_used": components,
    }


def _trace_distance(left: list[TypedAction], right: list[TypedAction]) -> int:
    shared = min(len(left), len(right))
    mismatches = sum(
        1
        for idx in range(shared)
        if canonical_action_key(left[idx]) != canonical_action_key(right[idx])
    )
    return mismatches + abs(len(left) - len(right))


def _rollout_with_hierarchical_policy(
    case,
    policy: HierarchicalSetPolicy | None,
    *,
    label: str,
    teacher_records: list[BCStepRecord],
    primitive_sets: list[list[int]],
) -> dict[str, Any]:
    role_evidence = label_case_roles(case)
    state = ExecutionState(last_rollout_mode="semantic")
    executor = ActionExecutor(case)
    max_steps = case.block_count * 4
    seed = _seed_first_action(
        case,
        [idx for idx in range(case.block_count) if idx not in state.semantic_placed],
    )
    executor.apply(state, seed)
    no_progress = 0
    forced_actions = 0
    model_selected_steps = 0
    events: list[dict[str, Any]] = [
        {
            "event": "seed",
            "run_id": label,
            "chosen_action": {
                "primitive": seed.primitive.value,
                "block_index": seed.block_index,
                "source": seed.metadata.get("source"),
                "intent_type": seed.metadata.get("intent_type"),
            },
        }
    ]
    step = 0
    while len(state.semantic_placed) < case.block_count and state.step < max_steps:
        remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
        before = len(state.semantic_placed)
        candidates = generate_candidate_actions(
            case,
            state,
            remaining_blocks=remaining,
            mode="semantic",
            max_per_primitive=8,
        )
        forced_progress = False
        forced_reason = None
        candidate_breakdowns: list[dict[str, Any]] = []
        if not candidates:
            chosen = _forced_progress_action(case, state, remaining[0])
            forced_progress = True
            forced_reason = "no_candidates"
            forced_actions += 1
        else:
            scored = []
            for candidate in candidates:
                score, components = _score_hierarchical_action(
                    case, policy, role_evidence, state, candidate
                )
                scored.append((score, candidate, components))
            _chosen_score, chosen, _chosen_components = max(scored, key=lambda item: item[0])
            if no_progress >= 2 and chosen.block_index in state.semantic_placed:
                chosen = _forced_progress_action(case, state, remaining[0])
                forced_progress = True
                forced_reason = "no_progress_counter"
                forced_actions += 1
            else:
                model_selected_steps += int(policy is not None)
                for score, candidate, components in sorted(
                    scored, key=lambda item: item[0], reverse=True
                )[:5]:
                    candidate_breakdowns.append(
                        {
                            "score": float(score),
                            "chosen": canonical_action_key(candidate)
                            == canonical_action_key(chosen),
                            "primitive": candidate.primitive.value,
                            "block_index": candidate.block_index,
                            "target_index": candidate.target_index,
                            "source": candidate.metadata.get("source"),
                            "intent_type": candidate.metadata.get("intent_type"),
                            **components,
                        }
                    )
        executor.apply(state, chosen)
        after = len(state.semantic_placed)
        progress = after > before
        no_progress = 0 if progress else no_progress + 1

        teacher_row: dict[str, Any] | None = None
        if step < len(teacher_records):
            teacher = teacher_records[step]
            primitive_id = list(ActionPrimitive).index(chosen.primitive)
            primitive_set = primitive_sets[step] if step < len(primitive_sets) else []
            teacher_row = {
                "teacher_block": teacher.action.block_index,
                "chosen_block_matches_teacher": chosen.block_index == teacher.action.block_index,
                "chosen_primitive_in_teacher_set": primitive_id in primitive_set,
                "teacher_primitive_set_size": len(primitive_set),
            }
        events.append(
            {
                "event": "step",
                "run_id": label,
                "step": step,
                "candidate_count": len(candidates),
                "remaining_blocks": remaining,
                "chosen_action": {
                    "primitive": chosen.primitive.value,
                    "block_index": chosen.block_index,
                    "target_index": chosen.target_index,
                    "source": chosen.metadata.get("source"),
                    "intent_type": chosen.metadata.get("intent_type"),
                },
                "semantic_before": before,
                "semantic_after": after,
                "progress_made": progress,
                "forced_progress": forced_progress,
                "forced_reason": forced_reason,
                "teacher_comparison": teacher_row,
                "candidate_breakdowns": candidate_breakdowns,
            }
        )
        step += 1
    return {
        "label": label,
        "events": events,
        "actions": list(state.history),
        "trace": [canonical_action_key(action) for action in state.history],
        "semantic_completed": len(state.semantic_placed) == case.block_count,
        "semantic_placed_fraction": len(state.semantic_placed) / max(case.block_count, 1),
        "fallback_fraction": forced_actions / max(len(state.history), 1),
        "model_selected_fraction": model_selected_steps / max(len(state.history) - 1, 1),
        "stopped_reason": "completed"
        if len(state.semantic_placed) == case.block_count
        else "max_steps",
    }


def _run_job(job: RolloutJob) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    cases = load_validation_cases(case_limit=job.case_id + 1)
    case = cases[job.case_id]
    dataset = build_bc_dataset_from_cases([case], max_traces_per_case=1)
    trained_policy, training_summary, primitive_sets = _train_hierarchical_policy(dataset, job)
    torch.manual_seed(job.seed)
    untrained_policy = HierarchicalSetPolicy(hidden_dim=job.hidden_dim)
    trained = _rollout_with_hierarchical_policy(
        case,
        trained_policy,
        label="trained_hierarchical",
        teacher_records=dataset,
        primitive_sets=primitive_sets,
    )
    untrained = _rollout_with_hierarchical_policy(
        case,
        untrained_policy,
        label="untrained_hierarchical",
        teacher_records=dataset,
        primitive_sets=primitive_sets,
    )
    heuristic = _rollout_with_hierarchical_policy(
        case,
        None,
        label="heuristic_only",
        teacher_records=dataset,
        primitive_sets=primitive_sets,
    )
    trace_distance = _trace_distance(trained["actions"], untrained["actions"])
    trained_logits = trained_policy(
        case,
        role_evidence=label_case_roles(case),
        placements={},
        state_step=0,
    )
    untrained_logits = untrained_policy(
        case,
        role_evidence=label_case_roles(case),
        placements={},
        state_step=0,
    )
    logit_delta = float(
        (trained_logits.block_logits - untrained_logits.block_logits).norm().item()
        + (
            trained_logits.primitive_logits_by_block
            - untrained_logits.primitive_logits_by_block
        )
        .norm()
        .item()
    )
    return {
        "case_id": str(case.case_id),
        "job": asdict(job),
        "training_summary": training_summary,
        "logits_changed": logit_delta > 1e-6,
        "logit_delta_norm": logit_delta,
        "trace_distance_trained_vs_untrained": trace_distance,
        "trained": {key: value for key, value in trained.items() if key != "actions"},
        "untrained": {key: value for key, value in untrained.items() if key != "actions"},
        "heuristic": {key: value for key, value in heuristic.items() if key != "actions"},
    }


def _aggregate(results: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    trace_changed = sum(
        1 for result in results if int(result["trace_distance_trained_vs_untrained"]) > 0
    )
    completed = sum(1 for result in results if bool(result["trained"]["semantic_completed"]))
    denom = max(len(results), 1)
    return {
        "status": "complete",
        "purpose": "Step6C hierarchical rollout-control audit",
        "workers_requested": workers,
        "case_count": len(results),
        "trace_distance_nonzero_cases": trace_changed,
        "semantic_completed_cases": completed,
        "gate_pass": trace_changed >= min(3, len(results)),
        "mean_trained_model_selected_fraction": sum(
            float(result["trained"]["model_selected_fraction"]) for result in results
        )
        / denom,
        "mean_trained_semantic_placed_fraction": sum(
            float(result["trained"]["semantic_placed_fraction"]) for result in results
        )
        / denom,
        "mean_trained_fallback_fraction": sum(
            float(result["trained"]["fallback_fraction"]) for result in results
        )
        / denom,
        "results": sorted(results, key=lambda item: item["case_id"]),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for result in payload["results"]:
        rows.append(
            "| {case} | {dist} | {done} | {placed:.4f} | {model:.4f} | {fallback:.4f} |".format(
                case=result["case_id"],
                dist=result["trace_distance_trained_vs_untrained"],
                done=result["trained"]["semantic_completed"],
                placed=result["trained"]["semantic_placed_fraction"],
                model=result["trained"]["model_selected_fraction"],
                fallback=result["trained"]["fallback_fraction"],
            )
        )
    lines = [
        "# Step6C Hierarchical Rollout-Control Audit",
        "",
        "- purpose: `prove hierarchical trained logits affect semantic rollout choice`",
        f"- cases: `{payload['case_count']}`",
        f"- workers requested: `{payload['workers_requested']}`",
        f"- trace distance nonzero cases: `{payload['trace_distance_nonzero_cases']}`",
        f"- semantic completed cases: `{payload['semantic_completed_cases']}`",
        f"- gate pass: `{payload['gate_pass']}`",
        f"- mean trained model-selected fraction: `{payload['mean_trained_model_selected_fraction']:.4f}`",
        f"- mean trained semantic placed fraction: `{payload['mean_trained_semantic_placed_fraction']:.4f}`",
        f"- mean trained fallback fraction: `{payload['mean_trained_fallback_fraction']:.4f}`",
        "",
        "| Case | trace distance | trained completed | trained placed frac | model-selected frac | fallback frac |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
        *rows,
        "",
        "Interpretation: this is a sidecar rollout-control audit only. It does not "
        "modify contest runtime, repair/finalizer behavior, reranker settings, or "
        "proxy weights.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    jobs = [
        RolloutJob(
            case_id=int(case_id),
            seed=int(args.seed),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            lr=float(args.lr),
            primitive_set_weight=float(args.primitive_set_weight),
            block_weight=float(args.block_weight),
            encoder_kind=str(args.encoder_kind),
        )
        for case_id in args.case_ids
    ]
    max_workers = max(1, min(int(args.workers), len(jobs)))
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    payload = _aggregate(results, max_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "case_count": payload["case_count"],
                "trace_distance_nonzero_cases": payload["trace_distance_nonzero_cases"],
                "gate_pass": payload["gate_pass"],
                "mean_trained_model_selected_fraction": payload[
                    "mean_trained_model_selected_fraction"
                ],
                "mean_trained_semantic_placed_fraction": payload[
                    "mean_trained_semantic_placed_fraction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
