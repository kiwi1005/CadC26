#!/usr/bin/env python3
"""Train HCFP structure/initializer/flow heads from auditable tar shards."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint  # noqa: E402
from hcfp.data import file_sha256, iter_shard, read_shard_manifest  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite  # noqa: E402
from hcfp.model import HCFPModel, ModelConfig  # noqa: E402
from hcfp.training import ExponentialMovingAverage, TRAINING_STAGES, train_steps  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", help="input .tar shards")
    parser.add_argument("--floorset-lite-root", help="direct official training root; avoids copied shards")
    parser.add_argument("--sample-limit", type=int, help="bounded direct-stream subset for smoke/ablation")
    parser.add_argument("-o", "--output", required=True, help="output checkpoint")
    parser.add_argument("--stage", choices=TRAINING_STAGES, default="all")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--init-checkpoint", help="warm-start model weights from a runtime checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if bool(args.shards) == bool(args.floorset_lite_root):
        raise ValueError("provide either shards or --floorset-lite-root")
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
    sample_count = sum(len(manifest.get("samples", [])) for manifest in manifests) if manifests else args.sample_limit

    def training_samples():
        if args.floorset_lite_root:
            yield from iter_floorset_lite(args.floorset_lite_root, limit=args.sample_limit)
            return
        for path in args.shards:
            for sample in iter_shard(path):
                if sample.sample_id.lower().startswith(("validation-", "val-", "official/")):
                    raise ValueError(f"official validation-like sample ID is forbidden: {sample.sample_id}")
                yield sample

    compute_dtype = "bfloat16" if args.amp == "bf16" else "float32"
    if args.init_checkpoint:
        loaded, _ = load_checkpoint(
            args.init_checkpoint,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        config = replace(loaded.config, compute_dtype=compute_dtype)
        model = HCFPModel(config)
        model.load_state_dict(loaded.state_dict(), strict=True)
    else:
        model = HCFPModel(
            ModelConfig(
                hidden_dim=args.hidden_dim,
                encoder_layers=args.encoder_layers,
                compute_dtype=compute_dtype,
            )
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(model, args.ema_decay) if args.ema_decay > 0.0 else None

    def checkpoint_step(step: int, _report) -> None:
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(model, f"{args.output}.step-{step:08d}.pt", RUNTIME_NORMALIZATION)

    history = train_steps(
        model,
        training_samples,
        optimizer,
        steps=args.steps,
        population=args.population,
        stage=args.stage,
        seed=args.seed,
        ema=ema,
        on_step=checkpoint_step,
    )
    if ema is not None:
        ema.copy_to(model)
    checkpoint_hash = save_checkpoint(model, args.output, RUNTIME_NORMALIZATION)
    report = {
        "schema_version": 1,
        "checkpoint": str(Path(args.output)),
        "checkpoint_hash": checkpoint_hash,
        "stage": args.stage,
        "steps": args.steps,
        "population": args.population,
        "device": str(device),
        "compute_dtype": compute_dtype,
        "ema_decay": args.ema_decay if ema is not None else None,
        "init_checkpoint": args.init_checkpoint,
        "sample_count": sample_count,
        "shards": [{"path": str(path), "sha256": file_sha256(path)} for path in args.shards],
        "floorset_lite_root": args.floorset_lite_root,
        "first_loss": history[0],
        "last_loss": history[-1],
    }
    report_path = Path(f"{args.output}.training.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
