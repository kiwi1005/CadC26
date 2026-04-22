#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.feedback import (
    build_advantage_dataset_from_cases,
    run_advantage_weighted_bc,
    save_policy_checkpoint,
)
from puzzleplace.models import TypedActionPolicy
from puzzleplace.repair.finalizer import finalize_layout
from puzzleplace.rollout import semantic_rollout
from puzzleplace.train import (
    build_bc_dataset_from_cases,
    compute_bc_loss,
    load_training_cases,
    load_validation_cases,
    measure_candidate_recall,
    run_bc_overfit,
)
from puzzleplace.train.dataset_bc import action_to_targets


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return seeds or [0, 1]


def _average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _evaluate_loss(policy: TypedActionPolicy, dataset) -> float:
    total = 0.0
    for record in dataset:
        total += float(compute_bc_loss(policy, record).total.item())
    return total / max(len(dataset), 1)


def _evaluate_accuracy(policy: TypedActionPolicy, dataset) -> tuple[float, float]:
    primitive_hits = 0
    block_hits = 0
    for record in dataset:
        output = policy(
            record.case,
            role_evidence=record.role_evidence,
            placements=record.placements,
        )
        targets = action_to_targets(record.action)
        primitive_hits += int(int(output.primitive_logits.argmax().item()) == int(targets["primitive_id"]))
        block_hits += int(int(output.block_logits.argmax().item()) == int(targets["block_index"]))
    denom = max(len(dataset), 1)
    return primitive_hits / denom, block_hits / denom


def _evaluate_rollout_variant(name: str, policy, cases) -> dict[str, object]:
    started_at = time.time()
    details = []
    for case in cases:
        semantic = semantic_rollout(case, policy)
        repair = finalize_layout(case, semantic.proposed_positions)
        details.append(
            {
                "case_id": str(case.case_id),
                "semantic_completed": semantic.semantic_completed,
                "semantic_placed_fraction": semantic.semantic_placed_fraction,
                "fallback_fraction": semantic.fallback_fraction,
                "overlap_pairs": semantic.violation_profile.overlap_pairs,
                "overlap_area": semantic.violation_profile.total_overlap_area,
                "repair_hard_feasible_after": repair.report.hard_feasible_after,
                "repair_overlap_pairs_after": repair.report.overlap_pairs_after,
                "repair_shelf_fallback_count": repair.report.shelf_fallback_count,
            }
        )
    return {
        "name": name,
        "runtime_seconds": time.time() - started_at,
        "semantic_rollout_completion": _average(
            [1.0 if row["semantic_completed"] else 0.0 for row in details]
        ),
        "avg_semantic_placed_fraction": _average(
            [float(row["semantic_placed_fraction"]) for row in details]
        ),
        "avg_fallback_fraction": _average([float(row["fallback_fraction"]) for row in details]),
        "avg_overlap_pairs": _average([float(row["overlap_pairs"]) for row in details]),
        "repair_success_rate": _average(
            [1.0 if row["repair_hard_feasible_after"] else 0.0 for row in details]
        ),
        "details": details,
    }


def _untrained_metrics(policy: TypedActionPolicy, train_dataset, val_dataset) -> dict[str, float]:
    train_primitive_accuracy, train_block_accuracy = _evaluate_accuracy(policy, train_dataset)
    val_primitive_accuracy, val_block_accuracy = _evaluate_accuracy(policy, val_dataset)
    return {
        "train_loss": _evaluate_loss(policy, train_dataset),
        "val_loss": _evaluate_loss(policy, val_dataset),
        "train_primitive_accuracy": train_primitive_accuracy,
        "train_block_accuracy": train_block_accuracy,
        "val_primitive_accuracy": val_primitive_accuracy,
        "val_block_accuracy": val_block_accuracy,
    }


def main() -> None:
    train_case_count = int(os.environ.get("SMALL_OVERFIT_TRAIN_CASES", "16"))
    val_case_count = int(os.environ.get("SMALL_OVERFIT_VAL_CASES", "8"))
    max_traces = int(os.environ.get("SMALL_OVERFIT_MAX_TRACES", "1"))
    hidden_dim = int(os.environ.get("SMALL_OVERFIT_HIDDEN_DIM", "32"))
    lr = float(os.environ.get("SMALL_OVERFIT_LR", "1e-3"))
    epochs = int(os.environ.get("SMALL_OVERFIT_EPOCHS", "5"))
    seeds = _parse_seeds(os.environ.get("SMALL_OVERFIT_SEEDS", "0,1"))
    candidate_mode = "semantic"

    research_dir = ROOT / "artifacts" / "research"
    model_dir = ROOT / "artifacts" / "models"
    research_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_cases = load_training_cases(case_limit=train_case_count, batch_size=1)
    val_cases = load_validation_cases(case_limit=val_case_count)
    train_dataset = build_bc_dataset_from_cases(train_cases, max_traces_per_case=max_traces)
    val_dataset = build_bc_dataset_from_cases(val_cases, max_traces_per_case=max_traces)
    awbc_train_dataset = build_advantage_dataset_from_cases(
        train_cases,
        max_traces_per_case=max_traces,
        candidate_mode=candidate_mode,  # type: ignore[arg-type]
    )
    shared = {
        "train_case_ids": [str(case.case_id) for case in train_cases],
        "val_case_ids": [str(case.case_id) for case in val_cases],
        "train_candidate_recall_miss_rate": measure_candidate_recall(
            train_cases,
            max_traces_per_case=max_traces,
            candidate_mode=candidate_mode,  # type: ignore[arg-type]
        ).miss_rate,
        "val_candidate_recall_miss_rate": measure_candidate_recall(
            val_cases,
            max_traces_per_case=max_traces,
            candidate_mode=candidate_mode,  # type: ignore[arg-type]
        ).miss_rate,
    }

    heuristic_rollout = _evaluate_rollout_variant("heuristic", None, val_cases)
    bc_runs = []
    awbc_runs = []

    for seed in seeds:
        torch.manual_seed(seed)
        untrained_policy = TypedActionPolicy(hidden_dim=hidden_dim)
        untrained = _untrained_metrics(untrained_policy, train_dataset, val_dataset)
        untrained_rollout = _evaluate_rollout_variant("untrained", untrained_policy, val_cases)

        bc_policy, bc_summary = run_bc_overfit(
            train_dataset,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        bc_checkpoint = model_dir / f"small_overfit_bc_seed{seed}.pt"
        save_policy_checkpoint(
            bc_policy,
            bc_checkpoint,
            metadata={
                "variant": "bc",
                "seed": seed,
                "train_case_count": train_case_count,
                "val_case_count": val_case_count,
                "max_traces_per_case": max_traces,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "epochs": epochs,
                "candidate_mode": candidate_mode,
            },
        )
        bc_train_primitive_accuracy, bc_train_block_accuracy = _evaluate_accuracy(
            bc_policy,
            train_dataset,
        )
        bc_val_primitive_accuracy, bc_val_block_accuracy = _evaluate_accuracy(bc_policy, val_dataset)
        bc_runs.append(
            {
                "seed": seed,
                "untrained": untrained,
                "trained": {
                    "initial_loss": bc_summary.initial_loss,
                    "final_loss": bc_summary.final_loss,
                    "train_loss": _evaluate_loss(bc_policy, train_dataset),
                    "val_loss": _evaluate_loss(bc_policy, val_dataset),
                    "train_primitive_accuracy": bc_train_primitive_accuracy,
                    "train_block_accuracy": bc_train_block_accuracy,
                    "val_primitive_accuracy": bc_val_primitive_accuracy,
                    "val_block_accuracy": bc_val_block_accuracy,
                    "checkpoint_path": str(bc_checkpoint.relative_to(ROOT)),
                },
                "rollout_variants": {
                    "heuristic": heuristic_rollout,
                    "untrained": untrained_rollout,
                    "bc_trained": _evaluate_rollout_variant("bc_trained", bc_policy, val_cases),
                },
            }
        )

        awbc_policy, awbc_summary = run_advantage_weighted_bc(
            awbc_train_dataset,
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            seed=seed,
        )
        awbc_checkpoint = model_dir / f"small_overfit_awbc_seed{seed}.pt"
        save_policy_checkpoint(
            awbc_policy,
            awbc_checkpoint,
            metadata={
                "variant": "awbc",
                "seed": seed,
                "train_case_count": train_case_count,
                "val_case_count": val_case_count,
                "max_traces_per_case": max_traces,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "epochs": epochs,
                "candidate_mode": candidate_mode,
            },
        )
        awbc_train_primitive_accuracy, awbc_train_block_accuracy = _evaluate_accuracy(
            awbc_policy,
            train_dataset,
        )
        awbc_val_primitive_accuracy, awbc_val_block_accuracy = _evaluate_accuracy(
            awbc_policy,
            val_dataset,
        )
        awbc_runs.append(
            {
                "seed": seed,
                "untrained": untrained,
                "trained": {
                    "initial_loss": awbc_summary.initial_loss,
                    "final_loss": awbc_summary.final_loss,
                    "train_loss": _evaluate_loss(awbc_policy, train_dataset),
                    "val_loss": _evaluate_loss(awbc_policy, val_dataset),
                    "train_primitive_accuracy": awbc_train_primitive_accuracy,
                    "train_block_accuracy": awbc_train_block_accuracy,
                    "val_primitive_accuracy": awbc_val_primitive_accuracy,
                    "val_block_accuracy": awbc_val_block_accuracy,
                    "mean_advantage": awbc_summary.mean_advantage,
                    "mean_weight": awbc_summary.mean_weight,
                    "checkpoint_path": str(awbc_checkpoint.relative_to(ROOT)),
                },
                "rollout_variants": {
                    "heuristic": heuristic_rollout,
                    "untrained": untrained_rollout,
                    "awbc_trained": _evaluate_rollout_variant("awbc_trained", awbc_policy, val_cases),
                },
            }
        )

    bc_payload = {
        "study": "small_overfit",
        "variant": "bc",
        "train_case_count": train_case_count,
        "val_case_count": val_case_count,
        "max_traces_per_case": max_traces,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "epochs": epochs,
        "candidate_mode": candidate_mode,
        "shared": shared,
        "runs": bc_runs,
    }
    awbc_payload = {
        "study": "small_overfit",
        "variant": "awbc",
        "train_case_count": train_case_count,
        "val_case_count": val_case_count,
        "max_traces_per_case": max_traces,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "epochs": epochs,
        "candidate_mode": candidate_mode,
        "shared": shared,
        "runs": awbc_runs,
    }

    (research_dir / "small_overfit_bc.json").write_text(json.dumps(bc_payload, indent=2))
    (research_dir / "small_overfit_awbc.json").write_text(json.dumps(awbc_payload, indent=2))

    summary_lines = [
        "# Small Overfit Summary",
        "",
        f"- train cases: `{train_case_count}`",
        f"- val cases: `{val_case_count}`",
        f"- max traces per case: `{max_traces}`",
        f"- epochs: `{epochs}`",
        f"- seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- train candidate miss rate: `{shared['train_candidate_recall_miss_rate']:.4f}`",
        f"- val candidate miss rate: `{shared['val_candidate_recall_miss_rate']:.4f}`",
        "- candidate mode: `semantic`",
        "",
        "| Seed | Variant | Train loss | Val loss | Train block acc | Val block acc | Repair success |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in bc_runs:
        seed = run["seed"]
        summary_lines.append(
            f"| {seed} | untrained | {run['untrained']['train_loss']:.2f} | {run['untrained']['val_loss']:.2f} | "
            f"{run['untrained']['train_block_accuracy']:.4f} | {run['untrained']['val_block_accuracy']:.4f} | "
            f"{run['rollout_variants']['untrained']['repair_success_rate']:.3f} |"
        )
        summary_lines.append(
            f"| {seed} | bc | {run['trained']['train_loss']:.2f} | {run['trained']['val_loss']:.2f} | "
            f"{run['trained']['train_block_accuracy']:.4f} | {run['trained']['val_block_accuracy']:.4f} | "
            f"{run['rollout_variants']['bc_trained']['repair_success_rate']:.3f} |"
        )
    for run in awbc_runs:
        seed = run["seed"]
        summary_lines.append(
            f"| {seed} | awbc | {run['trained']['train_loss']:.2f} | {run['trained']['val_loss']:.2f} | "
            f"{run['trained']['train_block_accuracy']:.4f} | {run['trained']['val_block_accuracy']:.4f} | "
            f"{run['rollout_variants']['awbc_trained']['repair_success_rate']:.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "Checkpoint paths:",
            *(f"- BC seed {run['seed']}: `{run['trained']['checkpoint_path']}`" for run in bc_runs),
            *(f"- AWBC seed {run['seed']}: `{run['trained']['checkpoint_path']}`" for run in awbc_runs),
            "",
            f"- heuristic repair success: `{heuristic_rollout['repair_success_rate']:.3f}`",
            "> Proposal/survival gates remain pending until these numbers are interpreted against the approved thresholds.",
        ]
    )
    (research_dir / "small_overfit_summary.md").write_text("\n".join(summary_lines))
    print(json.dumps({"bc_runs": len(bc_runs), "awbc_runs": len(awbc_runs)}, indent=2))


if __name__ == "__main__":
    main()
