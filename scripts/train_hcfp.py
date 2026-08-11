#!/usr/bin/env python3
"""Train HCFP structure/initializer/flow heads from auditable tar shards."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
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
    command_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", help="input .tar shards")
    parser.add_argument("--floorset-lite-root", help="direct official training root; avoids copied shards")
    parser.add_argument("--sample-limit", type=int, help="bounded direct-stream subset for smoke/ablation")
    parser.add_argument("--sampling", choices=("uniform", "score-aware"), default="score-aware")
    parser.add_argument("-o", "--output", required=True, help="output checkpoint")
    parser.add_argument("--stage", choices=TRAINING_STAGES, default="all")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument(
        "--absolute-initializer",
        action="store_true",
        default=None,
        help="predict normalized centers directly instead of shelf residuals",
    )
    parser.add_argument("--center-bound", type=float)
    parser.add_argument("--aspect-residual-bound", type=float)
    parser.add_argument(
        "--topology",
        action="store_true",
        default=None,
        help="opt in to typed membership messages and the dual-permutation head",
    )
    parser.add_argument(
        "--constraints",
        action="store_true",
        default=None,
        help="opt in to learned Q2 constraint heads and supervision",
    )
    parser.add_argument(
        "--collective",
        action="store_true",
        default=None,
        help="opt in to Q3 dynamic pair messages and collective rollout supervision",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--no-ema-warmup",
        action="store_true",
        help="use the target EMA decay from the first update",
    )
    parser.add_argument("--init-checkpoint", help="warm-start model weights from a runtime checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(command_args)

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if bool(args.shards) == bool(args.floorset_lite_root):
        raise ValueError("provide either shards or --floorset-lite-root")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit must be positive")
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

    consumed_sample_ids: list[str] = []

    def training_samples():
        if args.floorset_lite_root:
            stream = iter_floorset_lite(
                args.floorset_lite_root,
                limit=args.sample_limit,
                seed=args.seed,
                score_aware=args.sampling == "score-aware",
            )
        else:
            stream = (
                sample
                for path in args.shards
                for sample in iter_shard(path)
            )
        for sample in stream:
            if sample.sample_id.lower().startswith(("validation-", "val-", "official/")):
                raise ValueError(
                    f"official validation-like sample ID is forbidden: {sample.sample_id}"
                )
            consumed_sample_ids.append(sample.sample_id)
            yield sample

    compute_dtype = "bfloat16" if args.amp == "bf16" else "float32"
    source_metadata = None
    if args.init_checkpoint:
        loaded, source_metadata = load_checkpoint(
            args.init_checkpoint,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        topology_enabled = (
            loaded.config.topology_enabled
            if args.topology is None
            else args.topology
        )
        config = replace(
            loaded.config,
            compute_dtype=compute_dtype,
            residual_bound=(
                loaded.config.residual_bound
                if args.center_bound is None
                else args.center_bound
            ),
            aspect_residual_bound=(
                loaded.config.aspect_residual_bound
                if args.aspect_residual_bound is None
                else args.aspect_residual_bound
            ),
            initializer_absolute=(
                loaded.config.initializer_absolute
                if args.absolute_initializer is None
                else args.absolute_initializer
            ),
            topology_enabled=topology_enabled,
            constraint_enabled=(
                loaded.config.constraint_enabled
                if args.constraints is None
                else args.constraints
            ),
            collective_enabled=(
                loaded.config.collective_enabled
                if args.collective is None
                else args.collective
            ),
        )
        model = HCFPModel(config)
        incompatible = model.load_state_dict(loaded.state_dict(), strict=False)
        allowed_missing_prefixes = (
            "topology.",
            "encoder.group_message.",
            "encoder.mib_message.",
            "constraints.",
            "collective.",
        )
        invalid_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith(allowed_missing_prefixes)
        ]
        invalid_unexpected = list(incompatible.unexpected_keys)
        if invalid_missing or invalid_unexpected:
            raise ValueError(
                f"init checkpoint state mismatch: missing={invalid_missing} unexpected={invalid_unexpected}"
            )
    else:
        model = HCFPModel(
            ModelConfig(
                hidden_dim=args.hidden_dim,
                encoder_layers=args.encoder_layers,
                residual_bound=(0.10 if args.center_bound is None else args.center_bound),
                aspect_residual_bound=(
                    0.25
                    if args.aspect_residual_bound is None
                    else args.aspect_residual_bound
                ),
                initializer_absolute=bool(args.absolute_initializer),
                compute_dtype=compute_dtype,
                topology_enabled=bool(args.topology),
                constraint_enabled=bool(args.constraints),
                collective_enabled=bool(args.collective),
            )
        )
    if args.stage == "collective" and not model.config.collective_enabled:
        raise ValueError("--stage collective requires --collective or a collective checkpoint")
    checkpoint_metadata = _training_checkpoint_metadata(
        args.stage,
        model.config,
        source_metadata,
    )
    model = model.to(device)
    trainable_parameter_names = _select_trainable_parameters(model, args.stage)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    ema = (
        ExponentialMovingAverage(model, args.ema_decay, warmup=not args.no_ema_warmup)
        if args.ema_decay > 0.0
        else None
    )

    def checkpoint_step(step: int, _report) -> None:
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(
                model,
                f"{args.output}.step-{step:08d}.pt",
                RUNTIME_NORMALIZATION,
                metadata=checkpoint_metadata,
            )

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
    if len(consumed_sample_ids) != args.steps:
        raise RuntimeError(
            f"training consumed {len(consumed_sample_ids)} samples for {args.steps} steps"
        )
    if ema is not None:
        ema.copy_to(model)
    checkpoint_hash = save_checkpoint(
        model,
        args.output,
        RUNTIME_NORMALIZATION,
        metadata=checkpoint_metadata,
    )
    unique_sample_ids = sorted(set(consumed_sample_ids))
    direct_stream = None
    if args.floorset_lite_root:
        direct_stream = {
            "root": str(Path(args.floorset_lite_root).resolve()),
            "sampling": args.sampling,
            "seed": args.seed,
            "source_limit": args.sample_limit,
            "max_layouts_per_file": None,
            "consumed_count": len(consumed_sample_ids),
            "ordered_sample_id_count": len(consumed_sample_ids),
            "ordered_sample_id_sha256": _sample_id_hash(consumed_sample_ids),
            "unique_sample_id_count": len(unique_sample_ids),
            "unique_sample_id_sha256": _sample_id_hash(unique_sample_ids),
            "checkpoint_hash": checkpoint_hash,
        }
    parent_training_report = _parent_training_report(
        args.init_checkpoint,
        source_metadata,
    )
    report = {
        "schema_version": 3,
        "command": ["scripts/train_hcfp.py", *command_args],
        "checkpoint": str(Path(args.output).resolve()),
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_metadata": checkpoint_metadata,
        "model_config": asdict(model.config),
        "stage": args.stage,
        "steps": args.steps,
        "population": args.population,
        "seed": args.seed,
        "device": str(device),
        "compute_dtype": compute_dtype,
        "topology_enabled": model.config.topology_enabled,
        "constraint_enabled": model.config.constraint_enabled,
        "collective_enabled": model.config.collective_enabled,
        "trainable_parameter_names": trainable_parameter_names,
        "ema_target_decay": ema.target_decay if ema is not None else None,
        "ema_warmup_enabled": ema.warmup if ema is not None else False,
        "ema_update_count": ema.update_count if ema is not None else 0,
        "ema_final_effective_decay": ema.effective_decay if ema is not None else None,
        "init_checkpoint": args.init_checkpoint,
        "sample_count": sample_count,
        "sampling": args.sampling,
        "direct_floorset_lite_stream": direct_stream,
        "parent_training_report": parent_training_report,
        "shards": [{"path": str(path), "sha256": file_sha256(path)} for path in args.shards],
        "floorset_lite_root": args.floorset_lite_root,
        "first_loss": history[0],
        "last_loss": history[-1],
    }
    report_path = Path(f"{args.output}.training.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sample_id_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


def _parent_training_report(
    init_checkpoint: str | None,
    source_metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    if not init_checkpoint or source_metadata is None:
        return None
    path = Path(f"{init_checkpoint}.training.json").resolve()
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "checkpoint_hash": source_metadata.get("state_hash"),
    }


def _training_checkpoint_metadata(
    stage: str,
    config: ModelConfig,
    source: dict[str, object] | None,
) -> dict[str, object]:
    trained_heads = set(source.get("trained_heads", [])) if source is not None else set()
    if stage in {"structure", "all"}:
        trained_heads.add("encoder")
    if stage in {"structure", "all"}:
        trained_heads.add("structure")
        if config.topology_enabled:
            trained_heads.add("topology")
        if config.constraint_enabled:
            trained_heads.add("constraints")
    if stage in {"initializer", "all"}:
        trained_heads.add("initializer")
    if stage in {"flow", "all"}:
        trained_heads.add("flow")
    if stage in {"collective", "all"} and config.collective_enabled:
        trained_heads.add("collective")
    capabilities = (
        dict(source.get("capabilities", {})) if source is not None else {"flow": False}
    )
    if stage in {"flow", "all"}:
        capabilities["flow"] = True
    if stage in {"collective", "all"} and config.collective_enabled:
        capabilities["collective"] = True
    objective = (
        "collective_rollout_v1"
        if stage == "collective"
        else (
            "supervised_collective_v1"
            if stage == "all" and config.collective_enabled
            else "supervised_loss_v1"
        )
    )
    return {
        "capabilities": capabilities,
        "trained_heads": sorted(trained_heads),
        "training_objective_version": objective,
        "parent_state_hash": source.get("state_hash") if source is not None else None,
    }


def _select_trainable_parameters(model: HCFPModel, stage: str) -> list[str]:
    prefixes = {
        "structure": ("encoder.", "structure.", "topology.", "constraints."),
        "initializer": ("initializer.",),
        "flow": ("flow.",),
        "collective": ("collective.",),
    }
    names = []
    for name, parameter in model.named_parameters():
        trainable = stage == "all" or name.startswith(prefixes[stage])
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise ValueError(f"stage {stage!r} has no trainable parameters")
    return names


if __name__ == "__main__":
    raise SystemExit(main())
