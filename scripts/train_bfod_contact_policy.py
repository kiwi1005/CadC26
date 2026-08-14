#!/usr/bin/env python3
"""Train the bounded BFOD contact-patch ranker from official training layouts.

The ranker only chooses which four deterministic local patches reach the exact
scorer.  It never changes the production solver or manufactures placements.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_hcfp import _load_evaluator  # noqa: E402
from experiment_bfod_v1 import _admitted  # noqa: E402
from experiment_p10_case70 import _measure  # noqa: E402
from hcfp.contact_patch import dense_contact_patch_candidates  # noqa: E402
from hcfp.contact_policy import (  # noqa: E402
    CONTACT_FEATURE_VERSION,
    ContactPolicy,
    ContactPolicyConfig,
    contact_candidate_features,
    file_sha256,
    save_contact_policy,
)
from hcfp.fallback import safe_shelf  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.verify import verify  # noqa: E402


@dataclass(frozen=True)
class ConstraintSignature:
    block_count: int
    group_count: int
    mib_count: int
    mib_sizes: tuple[int, ...]
    boundary_blocks: int
    boundary_side_bits: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--training-root", default="artifacts/floorset-v10")
    parser.add_argument("--signature-case", type=int, default=70)
    parser.add_argument("--train-states", type=_positive, default=32)
    parser.add_argument("--heldout-states", type=_positive, default=8)
    parser.add_argument("--source-limit", type=_positive, default=3000)
    parser.add_argument("--layouts-per-file", type=_positive, default=16)
    parser.add_argument("--max-candidates", type=_positive, default=16)
    parser.add_argument("--patch-sizes", default="4,8,12,16")
    parser.add_argument("--steps", type=_positive, default=3000)
    parser.add_argument("--checkpoint-every", type=_positive, default=500)
    parser.add_argument("--learning-rate", type=_positive_float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=_positive, default=32)
    parser.add_argument("--seed", type=int, default=5170)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--case70-runtime-ceiling", type=_positive_float, default=30.0)
    parser.add_argument(
        "--output-dir", default="artifacts/experiments/bfod_contact_policy_v1"
    )
    args = parser.parse_args(argv)
    if args.steps > 3000:
        parser.error("--steps is capped at 3000 for this learned-expert gate")
    if args.steps % args.checkpoint_every:
        parser.error("--steps must be divisible by --checkpoint-every")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signature, signature_input = _signature_from_input(
        Path(args.data_path), args.signature_case
    )
    _assert_training_root(Path(args.training_root))
    evaluator = _load_evaluator(Path(args.data_path))
    teachers, collection = _collect_teachers(
        evaluator,
        Path(args.training_root),
        signature,
        args,
    )
    teacher_path = output_dir / "teacher_actions.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": "bfod_contact_teacher_v1",
            "signature": asdict(signature),
            "feature_version": CONTACT_FEATURE_VERSION,
            "collection": collection,
            **teachers,
        },
        teacher_path,
    )
    _dump(
        output_dir / "teacher_summary.json",
        {
            "signature": asdict(signature),
            "signature_input": str(signature_input),
            "signature_input_sha256": file_sha256(signature_input),
            "feature_version": CONTACT_FEATURE_VERSION,
            "teacher_actions": {
                split: _teacher_brief(states) for split, states in teachers.items()
            },
            "collection": collection,
            "teacher_actions_sha256": file_sha256(teacher_path),
        },
    )

    device = _device(args.device)
    torch.manual_seed(args.seed)
    policy = ContactPolicy(ContactPolicyConfig(hidden_dim=args.hidden_dim)).to(device)
    train_features = torch.cat([state["features"] for state in teachers["train"]])
    policy.set_normalization(train_features.mean(0), train_features.std(0).clamp_min(1.0e-6))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    baseline_case70 = _run_case70(
        Path(args.data_path),
        output_dir / "case70_baseline",
        args.case70_runtime_ceiling,
    )
    best_case70 = float(baseline_case70["uncapped_cost"])
    best_checkpoint: str | None = None
    stale = 0
    checkpoints: list[dict[str, Any]] = []
    loss_total = 0.0
    started = time.perf_counter()
    order = _training_order(len(teachers["train"]), args.seed)

    for step in range(1, args.steps + 1):
        state = teachers["train"][order[(step - 1) % len(order)]]
        policy.train()
        logits = policy(state["features"].to(device)).unsqueeze(0)
        target = torch.tensor([state["teacher_index"]], dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach().cpu())

        if step % args.checkpoint_every:
            continue
        checkpoint_path = output_dir / "checkpoints" / f"contact_policy_step_{step:04d}.pt"
        metadata = {
            "method": "bfod_contact_policy_v1",
            "feature_version": CONTACT_FEATURE_VERSION,
            "step": step,
            "seed": args.seed,
            "teacher_actions_sha256": file_sha256(teacher_path),
            "signature": asdict(signature),
            "train_state_ids_sha256": _id_hash(teachers["train"]),
            "heldout_state_ids_sha256": _id_hash(teachers["heldout"]),
        }
        checkpoint_sha = save_contact_policy(policy, checkpoint_path, metadata=metadata)
        heldout = _evaluate_heldout(policy, teachers["heldout"])
        case70 = _run_case70(
            Path(args.data_path),
            output_dir / f"case70_step_{step:04d}",
            args.case70_runtime_ceiling,
            checkpoint_path,
        )
        improved = float(case70["uncapped_cost"]) < best_case70 - 1.0e-10
        stale = 0 if improved else stale + 1
        if improved:
            best_case70 = float(case70["uncapped_cost"])
            best_checkpoint = str(checkpoint_path)
        checkpoint = {
            "step": step,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "train_cross_entropy": loss_total / args.checkpoint_every,
            "heldout": heldout,
            "case70": case70,
            "case70_improved": improved,
            "consecutive_non_improving_case70": stale,
        }
        checkpoints.append(checkpoint)
        _dump(output_dir / "progress.json", {"checkpoints": checkpoints})
        loss_total = 0.0
        if stale >= 2:
            break

    summary = {
        "method": "BFOD bounded learned contact policy v1",
        "scope": "experiment-only candidate ranker; production solve path unchanged",
        "signature": asdict(signature),
        "signature_input": str(signature_input),
        "signature_input_sha256": file_sha256(signature_input),
        "training_source": {
            "root": str(Path(args.training_root).resolve()),
            "guard": "official floorset_lite stream only; visible validation/test paths rejected",
            "case70_training_labels": False,
        },
        "teacher_actions": {
            "path": str(teacher_path),
            "sha256": file_sha256(teacher_path),
            **{split: _teacher_brief(states) for split, states in teachers.items()},
        },
        "config": _config_dict(args, device),
        "case70_baseline": baseline_case70,
        "checkpoints": checkpoints,
        "best_case70_uncapped_cost": best_case70,
        "best_checkpoint": best_checkpoint,
        "decision": "KEEP" if best_checkpoint else "REJECT",
        "stop_reason": (
            "two consecutive checkpoints did not improve Case70"
            if stale >= 2
            else "maximum training steps reached"
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    _dump(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    print(output_dir / "summary.json")
    return 0


def _signature_from_input(
    data_path: Path, case_id: int
) -> tuple[ConstraintSignature, Path]:
    """Read only the runtime-visible validation input tensor, never its label."""

    input_path = data_path / "LiteTensorDataTest" / f"config_{case_id + 21}" / "litedata_1.pth"
    if not input_path.is_file():
        raise FileNotFoundError(f"official input tensor not found: {input_path}")
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    rows = torch.as_tensor(payload[0][0], dtype=torch.long)
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError("official signature input must have [area,fixed,preplaced,MIB,group,boundary]")
    constraints = rows[:, 1:6]
    group_sizes = _group_sizes(constraints[:, 3])
    mib_sizes = _group_sizes(constraints[:, 2])
    boundary = constraints[:, 4]
    return (
        ConstraintSignature(
            block_count=int(rows.shape[0]),
            group_count=len(group_sizes),
            mib_count=len(mib_sizes),
            mib_sizes=mib_sizes,
            boundary_blocks=int((boundary != 0).sum()),
            boundary_side_bits=sum(int(value).bit_count() for value in boundary.tolist()),
        ),
        input_path,
    )


def _collect_teachers(
    evaluator: Any,
    root: Path,
    signature: ConstraintSignature,
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    targets = {"train": args.train_states, "heldout": args.heldout_states}
    stats = {
        "examined": 0,
        "signature_mismatch": 0,
        "quota_already_full": 0,
        "shelf_infeasible": 0,
        "no_contact_candidate": 0,
        "no_admissible_teacher": 0,
    }
    for sample, source in iter_floorset_lite_with_source(
        root,
        limit=args.source_limit,
        seed=args.seed,
        max_layouts_per_file=args.layouts_per_file,
        min_blocks=signature.block_count,
        max_blocks=signature.block_count,
    ):
        stats["examined"] += 1
        if not _signature_matches(sample.case, signature):
            stats["signature_mismatch"] += 1
            continue
        split = "heldout" if _heldout(sample.sample_id) else "train"
        if len(buckets[split]) >= targets[split]:
            stats["quota_already_full"] += 1
            if all(len(buckets[name]) >= targets[name] for name in targets):
                break
            continue
        try:
            positions = safe_shelf(source)
        except (RuntimeError, ValueError):
            stats["shelf_infeasible"] += 1
            continue
        if not verify(source, positions).feasible:
            stats["shelf_infeasible"] += 1
            continue
        candidates = dense_contact_patch_candidates(
            sample.case,
            positions,
            verify_case=source,
            patch_sizes=_parse_positive_ints(args.patch_sizes),
            max_candidates=args.max_candidates,
        )
        if not candidates:
            stats["no_contact_candidate"] += 1
            continue
        metric_args = (
            {
                "area_baseline": float(sample.labels.baseline_area),
                "hpwl_baseline": float(sample.labels.baseline_hpwl),
            },
            source["constraints"],
            source["b2b_connectivity"],
            source["p2b_connectivity"],
            source["pins_pos"],
            source["area_targets"],
            source["target_positions"],
        )
        before = _measure(evaluator, source, metric_args, positions)
        metrics = [
            _measure(evaluator, source, metric_args, candidate.placement)
            for candidate in candidates
        ]
        admitted = [
            index
            for index, metric in enumerate(metrics)
            if _admitted(before, metric, "boundary_or_group")
        ]
        if not admitted:
            stats["no_admissible_teacher"] += 1
            continue
        teacher_index = min(
            admitted,
            key=lambda index: _metric_key(metrics[index], index),
        )
        buckets[split].append(
            {
                "sample_id": sample.sample_id,
                "features": torch.stack(
                    [
                        contact_candidate_features(
                            sample.case, source, positions, candidate
                        )
                        for candidate in candidates
                    ]
                ),
                "teacher_index": teacher_index,
                "candidate_uncapped_cost": torch.tensor(
                    [metric["uncapped_cost"] for metric in metrics], dtype=torch.float64
                ),
                "candidate_admitted": torch.tensor(admitted, dtype=torch.long),
                "baseline_uncapped_cost": float(before["uncapped_cost"]),
                "teacher_uncapped_cost": float(metrics[teacher_index]["uncapped_cost"]),
                "teacher": {
                    "grouping_before": candidates[teacher_index].grouping_before,
                    "grouping_after": candidates[teacher_index].grouping_after,
                    "bridge_member": candidates[teacher_index].bridge_member,
                    "anchor_member": candidates[teacher_index].anchor_member,
                    "side": candidates[teacher_index].side,
                },
            }
        )
        if all(len(buckets[name]) >= targets[name] for name in targets):
            break
    if any(len(buckets[name]) < targets[name] for name in targets):
        collected = {name: len(states) for name, states in buckets.items()}
        raise RuntimeError(
            "not enough official training teacher states: "
            f"wanted={targets}, got={collected}, "
            f"stats={stats}"
        )
    return buckets, {"targets": targets, "collected": {name: len(states) for name, states in buckets.items()}, "stats": stats}


def _evaluate_heldout(policy: ContactPolicy, states: list[dict[str, Any]]) -> dict[str, Any]:
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    top1 = top4 = 0
    oracle_gain = selected_gain = 0.0
    with torch.inference_mode():
        for state in states:
            ranking = torch.argsort(
                policy(state["features"].to(device)), descending=True, stable=True
            ).tolist()
            teacher = int(state["teacher_index"])
            top1 += int(ranking[0] == teacher)
            top4 += int(teacher in ranking[:4])
            selected = _selected_cost(state, ranking[:4])
            baseline = float(state["baseline_uncapped_cost"])
            oracle_gain += baseline - float(state["teacher_uncapped_cost"])
            selected_gain += baseline - selected
    count = len(states)
    if was_training:
        policy.train()
    return {
        "states": count,
        "top1_teacher_recall": top1 / count,
        "top4_teacher_recall": top4 / count,
        "mean_oracle_uncapped_gain": oracle_gain / count,
        "mean_top4_uncapped_gain": selected_gain / count,
    }


def _selected_cost(state: dict[str, Any], selected: list[int]) -> float:
    baseline = float(state["baseline_uncapped_cost"])
    admitted = set(int(index) for index in state["candidate_admitted"].tolist())
    scores = state["candidate_uncapped_cost"]
    return min(
        [baseline]
        + [float(scores[index]) for index in selected if index in admitted]
    )


def _run_case70(
    data_path: Path,
    output_dir: Path,
    runtime_ceiling: float,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_bfod_v1.py"),
        "--cases",
        "70",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--runtime-ceiling",
        str(runtime_ceiling),
    ]
    if checkpoint is not None:
        command.extend(("--contact-policy", str(checkpoint)))
    subprocess.run(command, cwd=ROOT, check=True)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    result_path = Path(summary["cases"][0]["artifact"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "cost": result["winner"]["metrics"]["cost"],
        "uncapped_cost": result["winner"]["metrics"]["uncapped_cost"],
        "boundary_violations": result["winner"]["metrics"]["boundary_violations"],
        "grouping_violations": result["winner"]["metrics"]["grouping_violations"],
        "mib_violations": result["winner"]["metrics"]["mib_violations"],
        "hard_feasible": result["winner"]["metrics"]["hard_feasible"],
        "runtime_seconds": result["runtime_seconds"],
        "artifact": str(result_path),
    }


def _signature_matches(case: Any, signature: ConstraintSignature) -> bool:
    groups = _membership_sizes(case.group_membership)
    mibs = _membership_sizes(case.mib_membership)
    boundary_blocks = int(torch.as_tensor(case.boundary_bits, dtype=torch.bool).any(1).sum())
    return (
        int(case.n) == signature.block_count
        and len(groups) == signature.group_count
        and len(mibs) == signature.mib_count
        and mibs == signature.mib_sizes
        and max(groups, default=0) >= max(signature.mib_sizes, default=0)
        and abs(boundary_blocks - signature.boundary_blocks) <= 10
    )


def _metric_key(metric: dict[str, Any], index: int) -> tuple[Any, ...]:
    return (
        metric["uncapped_cost"],
        metric["total_soft_violations"],
        metric["bbox_area"],
        metric["hpwl_total"],
        index,
    )


def _membership_sizes(membership: Any) -> tuple[int, ...]:
    values = torch.as_tensor(membership, dtype=torch.bool)
    return tuple(sorted(int(value) for value in values.sum(1).tolist())) if values.numel() else ()


def _group_sizes(ids: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        int((ids == group_id).sum())
        for group_id in torch.unique(ids[ids > 0], sorted=True).tolist()
    )


def _heldout(sample_id: str) -> bool:
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 5 == 0


def _training_order(count: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(count, generator=generator).tolist()


def _id_hash(states: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(state["sample_id"] for state in states).encode()).hexdigest()


def _teacher_brief(states: list[dict[str, Any]]) -> dict[str, Any]:
    return {"states": len(states), "sample_ids_sha256": _id_hash(states)}


def _assert_training_root(root: Path) -> None:
    text = str(root.resolve()).lower()
    if any(token in text for token in ("litetensordatatest", "validation", "visible")):
        raise ValueError("visible validation/test paths are forbidden for teacher generation")


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part.strip())
    if not values or any(part <= 0 for part in values):
        raise ValueError("patch sizes must be positive integers")
    return values


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _config_dict(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {**vars(args), "resolved_device": str(device)}


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BFOD learned contact policy v1",
        "",
        "| Variant | Cost | Uncapped | B | G | M | Feasible | Runtime |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    baseline = summary["case70_baseline"]
    lines.append(_report_row("baseline", baseline))
    for checkpoint in summary["checkpoints"]:
        lines.append(_report_row(f"step {checkpoint['step']}", checkpoint["case70"]))
    lines.extend(
        (
            "",
            f"Decision: **{summary['decision']}**",
            f"Stop: {summary['stop_reason']}",
            "",
            "Teacher actions came only from the official `floorset_lite` training stream; "
            "Case70 supplied an input-only signature and checkpoint evaluation, never a teacher label.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_row(name: str, metric: dict[str, Any]) -> str:
    return (
        f"| {name} | {metric['cost']:.6f} | {metric['uncapped_cost']:.6f} | "
        f"{metric['boundary_violations']} | {metric['grouping_violations']} | "
        f"{metric['mib_violations']} | {metric['hard_feasible']} | "
        f"{metric['runtime_seconds']:.3f}s |"
    )


if __name__ == "__main__":
    raise SystemExit(main())
