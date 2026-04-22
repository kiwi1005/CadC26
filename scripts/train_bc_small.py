#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.feedback import save_policy_checkpoint
from puzzleplace.train import (
    build_bc_dataset_from_cases,
    load_training_cases,
    load_validation_cases,
    measure_candidate_recall,
    run_bc_overfit,
)


def main() -> None:
    config_path = Path(os.environ.get("BC_CONFIG", ROOT / "configs" / "bc_small.yaml"))
    config = yaml.safe_load(config_path.read_text())
    candidate_mode = os.environ.get("BC_CANDIDATE_MODE", "semantic")
    split = os.environ.get("BC_SPLIT", "validation")
    batch_size = int(os.environ.get("BC_BATCH_SIZE", "1"))
    checkpoint_path = Path(
        os.environ.get("BC_CHECKPOINT_PATH", ROOT / "artifacts" / "models" / "agent8_bc_policy.pt")
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    case_limit = int(config["case_limit"])
    if split == "training":
        cases = load_training_cases(case_limit=case_limit, batch_size=batch_size)
    else:
        cases = load_validation_cases(case_limit=case_limit)
    dataset = build_bc_dataset_from_cases(
        cases,
        max_traces_per_case=int(config["max_traces_per_case"]),
    )
    recall = measure_candidate_recall(
        cases,
        max_traces_per_case=int(config["max_traces_per_case"]),
        candidate_mode=candidate_mode,  # type: ignore[arg-type]
    )
    policy, summary = run_bc_overfit(
        dataset,
        hidden_dim=int(config["hidden_dim"]),
        lr=float(config["lr"]),
        epochs=int(config["epochs"]),
        seed=int(config["seed"]),
    )
    save_policy_checkpoint(
        policy,
        checkpoint_path,
        metadata={
            "split": split,
            "batch_size": batch_size,
            "case_limit": case_limit,
            "max_traces_per_case": int(config["max_traces_per_case"]),
            "epochs": int(config["epochs"]),
            "seed": int(config["seed"]),
            "candidate_mode": candidate_mode,
        },
    )
    payload = {
        "split": split,
        "batch_size": batch_size,
        "case_limit": case_limit,
        "max_traces_per_case": int(config["max_traces_per_case"]),
        "dataset_size": summary.dataset_size,
        "epochs": summary.epochs,
        "initial_loss": summary.initial_loss,
        "final_loss": summary.final_loss,
        "primitive_accuracy": summary.primitive_accuracy,
        "block_accuracy": summary.block_accuracy,
        "candidate_mode": candidate_mode,
        "candidate_miss_rate": recall.miss_rate,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
