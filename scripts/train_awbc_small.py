#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
from puzzleplace.train import load_training_cases, load_validation_cases


def main() -> None:
    case_limit = int(os.environ.get("AWBC_CASE_LIMIT", "5"))
    max_traces = int(os.environ.get("AWBC_MAX_TRACES", "2"))
    hidden_dim = int(os.environ.get("AWBC_HIDDEN_DIM", "64"))
    lr = float(os.environ.get("AWBC_LR", "1e-3"))
    epochs = int(os.environ.get("AWBC_EPOCHS", "10"))
    seed = int(os.environ.get("AWBC_SEED", "0"))
    candidate_mode = os.environ.get("AWBC_CANDIDATE_MODE", "semantic")
    split = os.environ.get("AWBC_SPLIT", "validation")
    batch_size = int(os.environ.get("AWBC_BATCH_SIZE", "1"))

    if split == "training":
        cases = load_training_cases(case_limit=case_limit, batch_size=batch_size)
    else:
        cases = load_validation_cases(case_limit=case_limit)
    dataset = build_advantage_dataset_from_cases(
        cases,
        max_traces_per_case=max_traces,
        candidate_mode=candidate_mode,  # type: ignore[arg-type]
    )
    policy, summary = run_advantage_weighted_bc(
        dataset,
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        seed=seed,
    )

    checkpoint_path = ROOT / "artifacts" / "models" / "agent11_awbc_policy.pt"
    save_policy_checkpoint(
        policy,
        checkpoint_path,
        metadata={
            "case_limit": case_limit,
            "split": split,
            "batch_size": batch_size,
            "max_traces_per_case": max_traces,
            "epochs": epochs,
            "seed": seed,
            "candidate_mode": candidate_mode,
        },
    )

    payload = {
        "case_limit": case_limit,
        "split": split,
        "batch_size": batch_size,
        "max_traces_per_case": max_traces,
        "dataset_size": summary.dataset_size,
        "epochs": summary.epochs,
        "initial_loss": summary.initial_loss,
        "final_loss": summary.final_loss,
        "primitive_accuracy": summary.primitive_accuracy,
        "block_accuracy": summary.block_accuracy,
        "mean_advantage": summary.mean_advantage,
        "mean_weight": summary.mean_weight,
        "candidate_mode": candidate_mode,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }
    report_dir = ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "agent11_awbc_validation0_4.json").write_text(json.dumps(payload, indent=2))
    (report_dir / "agent11_awbc_summary.md").write_text(
        "\n".join(
            [
                "# Agent 11 Summary",
                "",
                f"- dataset size: `{summary.dataset_size}`",
                f"- epochs: `{summary.epochs}`",
                f"- weighted loss: `{summary.initial_loss:.4f}` -> `{summary.final_loss:.4f}`",
                f"- primitive accuracy: `{summary.primitive_accuracy:.4f}`",
                f"- block accuracy: `{summary.block_accuracy:.4f}`",
                f"- mean verifier advantage: `{summary.mean_advantage:.4f}`",
                f"- mean weight: `{summary.mean_weight:.4f}`",
                f"- checkpoint: `{checkpoint_path.relative_to(ROOT)}`",
            ]
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
