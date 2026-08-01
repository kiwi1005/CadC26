#!/usr/bin/env python3
"""Train HCFP structure/initializer/flow heads from auditable tar shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint  # noqa: E402
from hcfp.data import file_sha256, iter_shard, read_shard_manifest  # noqa: E402
from hcfp.model import HCFPModel, ModelConfig  # noqa: E402
from hcfp.training import TRAINING_STAGES, train_steps  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="input .tar shards")
    parser.add_argument("-o", "--output", required=True, help="output checkpoint")
    parser.add_argument("--stage", choices=TRAINING_STAGES, default="all")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    manifests = [read_shard_manifest(path) for path in args.shards]
    for manifest in manifests:
        provenance = manifest.get("provenance", {})
        if provenance.get("split") != "train":
            raise ValueError("training accepts only shards with provenance split=train")
        source = str(provenance.get("source", "")).lower()
        if not source or any(token in source for token in ("validation", "visible", "test")):
            raise ValueError("training shard source is missing or validation/test-like")
        if not provenance.get("denylist_sha256"):
            raise ValueError("training shard provenance is missing a validation denylist checksum")
    sample_count = sum(len(manifest.get("samples", [])) for manifest in manifests)

    def training_samples():
        for path in args.shards:
            for sample in iter_shard(path):
                if sample.sample_id.lower().startswith(("validation-", "val-", "official/")):
                    raise ValueError(f"official validation-like sample ID is forbidden: {sample.sample_id}")
                yield sample

    model = HCFPModel(ModelConfig(hidden_dim=args.hidden_dim, encoder_layers=args.encoder_layers)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = train_steps(
        model,
        training_samples,
        optimizer,
        steps=args.steps,
        population=args.population,
        stage=args.stage,
        seed=args.seed,
    )
    checkpoint_hash = save_checkpoint(model, args.output, RUNTIME_NORMALIZATION)
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(args.output)),
        "checkpoint_hash": checkpoint_hash,
        "stage": args.stage,
        "steps": args.steps,
        "population": args.population,
        "device": str(device),
        "sample_count": sample_count,
        "shards": [{"path": str(path), "sha256": file_sha256(path)} for path in args.shards],
        "first_loss": history[0],
        "last_loss": history[-1],
    }
    report_path = Path(f"{args.output}.training.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
