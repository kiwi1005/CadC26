#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases, run_bc_overfit


def main() -> None:
    config_path = Path(os.environ.get("BC_CONFIG", ROOT / "configs" / "bc_small.yaml"))
    config = yaml.safe_load(config_path.read_text())
    cases = load_validation_cases(case_limit=int(config["case_limit"]))
    dataset = build_bc_dataset_from_cases(cases, max_traces_per_case=int(config["max_traces_per_case"]))
    _policy, summary = run_bc_overfit(
        dataset,
        hidden_dim=int(config["hidden_dim"]),
        lr=float(config["lr"]),
        epochs=int(config["epochs"]),
        seed=int(config["seed"]),
    )
    payload = {
        "case_limit": int(config["case_limit"]),
        "max_traces_per_case": int(config["max_traces_per_case"]),
        "dataset_size": summary.dataset_size,
        "epochs": summary.epochs,
        "initial_loss": summary.initial_loss,
        "final_loss": summary.final_loss,
        "primitive_accuracy": summary.primitive_accuracy,
        "block_accuracy": summary.block_accuracy,
    }
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
