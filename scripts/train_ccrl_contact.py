#!/usr/bin/env python3
"""Train and exactly evaluate the fixed Contact-only Gate D model."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.repair.actions import action_sha256  # noqa: E402
from hcfp.repair.decoders.contact import decode_contact_action  # noqa: E402
from hcfp.repair.losses import contact_action_loss  # noqa: E402
from hcfp.repair.model import (  # noqa: E402
    ContactRepairModel,
    RepairModelConfig,
    topk_contact_actions,
)
from hcfp.repair.replay import (  # noqa: E402
    repair_generation_loads,
    repair_replay_loads,
)


_KINDS = ("C0", "C1", "C2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-replay", required=True)
    parser.add_argument("--heldout-replay", required=True)
    parser.add_argument("--train-generation")
    parser.add_argument("--heldout-generation")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=_positive, default=50_000)
    parser.add_argument("--learning-rate", type=_positive_float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing training directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    device = _device(args.device)
    torch.manual_seed(args.seed)
    train = _load_replay(Path(args.train_replay), "train")
    heldout = _load_replay(Path(args.heldout_replay), "heldout")
    manifest = _load_manifest(Path(args.source_manifest))
    train_generation_path = _generation_path(args.train_generation, args.train_replay)
    heldout_generation_path = _generation_path(
        args.heldout_generation, args.heldout_replay
    )
    train_generation = _load_generation(train_generation_path, "train")
    heldout_generation = _load_generation(heldout_generation_path, "heldout")
    train_coverage = _validate_cache(train, train_generation, manifest, "train")
    heldout_coverage = _validate_cache(heldout, heldout_generation, manifest, "heldout")
    verifiers = _load_verifiers(
        args.floorset_lite_root,
        set(record.source_id for record in heldout),
        seed=int(manifest["config"]["seed"]),
        max_layouts_per_file=int(manifest["config"]["max_layouts_per_file"]),
    )

    config = RepairModelConfig()
    model = ContactRepairModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    started = time.perf_counter()
    total_loss = 0.0
    model.train()
    for step in range(args.steps):
        if step and step % len(order) == 0:
            random.Random(args.seed + step).shuffle(order)
        record = train[order[step % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        report = contact_action_loss(
            model(record.state, record.obligation), record.action
        )
        report.total.backward()
        optimizer.step()
        total_loss += float(report.total.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    model.eval()
    evaluation_started = time.perf_counter()
    evaluation = _evaluate(model, heldout, verifiers)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    checkpoint = output_dir / "contact_gate_d.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_kind": "ccrl-contact-gate-d-v1",
            "config": asdict(config),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "source_manifest_sha256": file_sha256(args.source_manifest),
            "train_replay_sha256": file_sha256(args.train_replay),
            "heldout_replay_sha256": file_sha256(args.heldout_replay),
            "train_generation_sha256": file_sha256(train_generation_path),
            "heldout_generation_sha256": file_sha256(heldout_generation_path),
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        checkpoint,
    )
    report = {
        "schema_version": 1,
        "purpose": "P11.4 Gate D Contact-only held-out generalization",
        "model": {"kind": "fixed-debug", **asdict(config)},
        "training": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": str(device),
            "train_rows": len(train),
            "mean_factorized_nll": total_loss / args.steps,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
        },
        "provenance": {
            "source_manifest": {
                "path": str(Path(args.source_manifest).resolve()),
                "sha256": file_sha256(args.source_manifest),
                "train_selection_sha256": manifest["selected"]["train"][
                    "selection_sha256"
                ],
                "heldout_selection_sha256": manifest["selected"]["heldout"][
                    "selection_sha256"
                ],
            },
            "train_replay": {
                "path": str(Path(args.train_replay).resolve()),
                "sha256": file_sha256(args.train_replay),
            },
            "heldout_replay": {
                "path": str(Path(args.heldout_replay).resolve()),
                "sha256": file_sha256(args.heldout_replay),
            },
            "train_generation": {
                "path": str(train_generation_path.resolve()),
                "sha256": file_sha256(train_generation_path),
            },
            "heldout_generation": {
                "path": str(heldout_generation_path.resolve()),
                "sha256": file_sha256(heldout_generation_path),
            },
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": file_sha256(checkpoint),
            },
        },
        "coverage": {"train": train_coverage, "heldout": heldout_coverage},
        "heldout": evaluation,
        "gate": _gate(evaluation, train_coverage, heldout_coverage),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_replay(path: Path, split: str):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [repair_replay_loads(line) for line in stream]
    if not records or any(record.source_split != split for record in records):
        raise ValueError(f"{split} replay is empty or has a split mismatch")
    return records


def _load_generation(path: Path, split: str):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [repair_generation_loads(line) for line in stream]
    if not records or any(record.source_split != split for record in records):
        raise ValueError(f"{split} generation is empty or has a split mismatch")
    return records


def _generation_path(value: str | None, replay: str) -> Path:
    if value is not None:
        return Path(value)
    replay_path = Path(replay)
    suffix = ".replay.jsonl.gz"
    if not replay_path.name.endswith(suffix):
        raise ValueError(
            "pass --*-generation when replay does not end in .replay.jsonl.gz"
        )
    return replay_path.with_name(
        replay_path.name.replace(suffix, ".generation.jsonl.gz")
    )


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("integrity", {}).get("artifact_sha256")
    canonical = json.loads(json.dumps(payload))
    canonical["integrity"]["artifact_sha256"] = None
    actual = hashlib.sha256(
        (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("source manifest canonical SHA-256 mismatch")
    if payload.get("overlap", {}).get("disjoint") is not True:
        raise ValueError("source manifest split overlap")
    for split in ("train", "heldout"):
        selected = [
            str(item["source_id"]) for item in payload["selected"][split]["records"]
        ]
        if len(selected) != len(set(selected)):
            raise ValueError(f"duplicate {split} source IDs in source manifest")
        if (
            _sha256_lines(set(selected))
            != payload["selected"][split]["selection_sha256"]
        ):
            raise ValueError(f"{split} selection SHA-256 mismatch")
    return payload


def _validate_cache(replay, generation, manifest: dict, split: str) -> dict:
    selected = {
        str(item["source_id"]) for item in manifest["selected"][split]["records"]
    }
    generation_keys = {
        (record.source_id, record.corruption_kind) for record in generation
    }
    expected_keys = {(source_id, kind) for source_id in selected for kind in _KINDS}
    generation_sources = {record.source_id for record in generation}
    split_version = str(manifest["config"]["split_version"])
    if any(record.split_version != split_version for record in generation + replay):
        raise ValueError(f"{split} cache split version does not match source manifest")
    if len(generation_keys) != len(generation):
        raise ValueError(f"{split} generation has duplicate source/kind rows")
    if not generation_sources <= selected or not generation_keys <= expected_keys:
        raise ValueError(f"{split} generation has rows outside frozen manifest")
    cached_keys = {
        (record.source_id, (record.state.corruption_kind or "").upper())
        for record in replay
    }
    if len(cached_keys) != len(replay):
        raise ValueError(f"{split} replay has duplicate source/kind rows")
    generated = {
        (record.source_id, record.corruption_kind): record
        for record in generation
        if record.corruption_generated
    }
    if cached_keys != set(generated):
        raise ValueError(f"{split} replay rows do not match generated corruptions")
    for record in replay:
        generated_record = generated[
            (record.source_id, (record.state.corruption_kind or "").upper())
        ]
        if action_sha256(record.action) != action_sha256(
            generated_record.inverse_action
        ):
            raise ValueError(f"{split} replay inverse action disagrees with generation")
    by_kind = {}
    for kind in _KINDS:
        rows = [record for record in generation if record.corruption_kind == kind]
        generated_rows = [record for record in rows if record.corruption_generated]
        by_kind[kind] = {
            "requested": len(rows),
            "generated": len(generated_rows),
            "failed": len(rows) - len(generated_rows),
            "inverse_decoder_generation_rate": len(generated_rows) / len(rows),
            "failure_categories": dict(
                sorted(
                    Counter(
                        record.generation_failure_reason
                        for record in rows
                        if not record.corruption_generated
                    ).items()
                )
            ),
        }
    return {
        "manifest_source_count": len(selected),
        "generation_source_count": len(generation_sources),
        "generation_source_id_sha256": _sha256_lines(generation_sources),
        "generation_complete_for_manifest": generation_keys == expected_keys,
        "replay_row_count": len(replay),
        "replay_source_count": len({record.source_id for record in replay}),
        "by_kind": by_kind,
    }


def _load_verifiers(
    root: str, source_ids: set[str], *, seed: int, max_layouts_per_file: int
):
    found = {}
    for sample, source in iter_floorset_lite_with_source(
        root,
        limit=None,
        seed=seed,
        max_layouts_per_file=max_layouts_per_file,
    ):
        if sample.sample_id in source_ids:
            found[sample.sample_id] = source
            if len(found) == len(source_ids):
                break
    missing = sorted(source_ids - found.keys())
    if missing:
        raise RuntimeError(f"missing official verifier sources: {missing[:8]}")
    return found


@torch.inference_mode()
def _evaluate(model, records, verifiers: dict) -> dict:
    totals = Counter()
    retained = []
    per_kind: dict[str, Counter] = {}
    for record in records:
        kind = (record.state.corruption_kind or "unknown").upper()
        metrics = per_kind.setdefault(kind, Counter())
        output = model(record.state, record.obligation)
        actions = topk_contact_actions(output, record.obligation, k=4)
        teacher = action_sha256(record.action)
        top1 = bool(actions) and action_sha256(actions[0]) == teacher
        top4 = any(action_sha256(action) == teacher for action in actions)
        decoded = [
            result
            for action in actions
            if (
                result := decode_contact_action(
                    record.state.case,
                    record.decoder_placement,
                    action,
                    verify_case=verifiers[record.source_id],
                )
            ).succeeded
        ]
        best = min(
            decoded,
            key=lambda result: (result.debt_after, action_sha256(result.action)),
            default=None,
        )
        teacher_gain = record.outcome.debt_before - record.outcome.debt_after
        gain = 0 if best is None else record.outcome.debt_before - best.debt_after
        recovery = gain / max(teacher_gain, 1)
        retained.append(recovery)
        for bucket in (totals, metrics):
            bucket["count"] += 1
            bucket["top1_inverse"] += int(top1)
            bucket["top4_inverse"] += int(top4)
            bucket["top4_exact"] += int(best is not None)
            bucket["full_teacher_gain"] += int(gain >= teacher_gain)
            bucket["recovery_sum"] += recovery
    return {
        "record_count": totals["count"],
        "source_count": len({record.source_id for record in records}),
        "source_id_sha256": _sha256_lines({record.source_id for record in records}),
        "top1_inverse_action_recall": totals["top1_inverse"] / totals["count"],
        "top4_inverse_action_recall": totals["top4_inverse"] / totals["count"],
        "decoded_top4_exact_success_rate": totals["top4_exact"] / totals["count"],
        "grouping_recovery_vs_inverse_mean": sum(retained) / len(retained),
        "full_inverse_gain_recovery_rate": totals["full_teacher_gain"]
        / totals["count"],
        "by_kind": {
            kind: {
                "count": bucket["count"],
                "top1_inverse_action_recall": bucket["top1_inverse"] / bucket["count"],
                "top4_inverse_action_recall": bucket["top4_inverse"] / bucket["count"],
                "decoded_top4_exact_success_rate": bucket["top4_exact"]
                / bucket["count"],
                "grouping_recovery_vs_inverse_mean": bucket["recovery_sum"]
                / bucket["count"],
                "full_inverse_gain_recovery_rate": bucket["full_teacher_gain"]
                / bucket["count"],
            }
            for kind, bucket in sorted(per_kind.items())
        },
    }


def _gate(evaluation: dict, train_coverage: dict, heldout_coverage: dict) -> dict:
    thresholds = {
        "top4_inverse_action_recall": 0.80,
        "decoded_top4_exact_success_rate": 0.99,
        "grouping_recovery_vs_inverse_mean": 0.90,
    }
    train_complete = (
        train_coverage["manifest_source_count"] == 2_000
        and train_coverage["generation_complete_for_manifest"]
    )
    heldout_complete = (
        heldout_coverage["manifest_source_count"] == 512
        and heldout_coverage["generation_complete_for_manifest"]
    )
    checks = {}
    checks.update(
        {name: evaluation[name] >= threshold for name, threshold in thresholds.items()}
    )
    complete = train_complete and heldout_complete
    return {
        "thresholds": thresholds,
        "checks": {
            "complete_2000_train_source_generation": train_complete,
            "complete_512_heldout_source_generation": heldout_complete,
            **checks,
        },
        "decision": (
            "NOT_EVALUATED_PARTIAL_CACHE"
            if not complete
            else "KEEP"
            if all(checks.values())
            else "REJECT"
        ),
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sha256_lines(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
