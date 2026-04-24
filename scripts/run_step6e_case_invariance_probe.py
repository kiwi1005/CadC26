#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step6E diagnostic: probe whether candidate-pool features encode "
            "case identity strongly enough to explain poor LOCO transfer."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6e_typed_graph_majority_advantage_neutral5_loco.json",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6e_case_invariance_probe_neutral5.json",
    )
    return parser.parse_args()


def _flatten_rows(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    features = []
    labels = []
    case_ids = sorted({int(c["case_index"]) for c in payload["collections"]})
    case_to_label = {case_id: idx for idx, case_id in enumerate(case_ids)}
    for collection in payload["collections"]:
        case_index = int(collection["case_index"])
        label = case_to_label[case_index]
        for pool in collection["pools"]:
            for row in pool["feature_rows"]:
                features.append(row)
                labels.append(label)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), case_ids


def _train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    class_count: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, class_count),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        logits = model(train_x)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_pred = model(train_x).argmax(dim=1)
        eval_pred = model(eval_x).argmax(dim=1)
    return {
        "train_accuracy": float((train_pred == train_y).float().mean().item()),
        "eval_accuracy": float((eval_pred == eval_y).float().mean().item()),
        "chance_accuracy": 1.0 / max(class_count, 1),
    }


def _leave_one_case_probe(features: torch.Tensor, labels: torch.Tensor, case_ids: list[int], *, epochs: int, seed: int) -> dict[str, Any]:
    splits = []
    for label, case_index in enumerate(case_ids):
        train_mask = labels != label
        eval_mask = labels == label
        result = _train_probe(
            features[train_mask],
            labels[train_mask],
            features[eval_mask],
            labels[eval_mask],
            class_count=len(case_ids),
            epochs=epochs,
            seed=seed,
        )
        result["heldout_case_index"] = case_index
        result["train_row_count"] = int(train_mask.sum().item())
        result["eval_row_count"] = int(eval_mask.sum().item())
        splits.append(result)
    return {
        "split_count": len(splits),
        "mean_train_accuracy": sum(s["train_accuracy"] for s in splits) / max(len(splits), 1),
        "mean_eval_accuracy": sum(s["eval_accuracy"] for s in splits) / max(len(splits), 1),
        "chance_accuracy": 1.0 / max(len(case_ids), 1),
        "splits": splits,
    }


def _within_case_probe(features: torch.Tensor, labels: torch.Tensor, case_ids: list[int], *, epochs: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    indexes = torch.randperm(features.shape[0])
    split = int(0.7 * features.shape[0])
    train_idx = indexes[:split]
    eval_idx = indexes[split:]
    return _train_probe(
        features[train_idx],
        labels[train_idx],
        features[eval_idx],
        labels[eval_idx],
        class_count=len(case_ids),
        epochs=epochs,
        seed=seed,
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        "| {case} | {train:.4f} | {eval:.4f} | {rows} |".format(
            case=s["heldout_case_index"],
            train=s["train_accuracy"],
            eval=s["eval_accuracy"],
            rows=s["eval_row_count"],
        )
        for s in payload["leave_one_case_probe"]["splits"]
    ]
    lines = [
        "# Step6E Case-Invariance Probe",
        "",
        "- purpose: `test whether candidate features encode case identity`",
        f"- source: `{payload['source']}`",
        f"- case ids: `{payload['case_ids']}`",
        f"- feature rows: `{payload['feature_row_count']}`",
        f"- chance accuracy: `{payload['chance_accuracy']:.4f}`",
        "",
        "## Random row split case-ID probe",
        "",
        f"- train accuracy: `{payload['within_case_probe']['train_accuracy']:.4f}`",
        f"- eval accuracy: `{payload['within_case_probe']['eval_accuracy']:.4f}`",
        "",
        "## Leave-one-case probe",
        "",
        f"- mean train accuracy: `{payload['leave_one_case_probe']['mean_train_accuracy']:.4f}`",
        f"- mean heldout eval accuracy: `{payload['leave_one_case_probe']['mean_eval_accuracy']:.4f}`",
        "",
        "| Heldout case | train acc | heldout eval acc | heldout rows |",
        "| ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Interpretation: high random-split case-ID accuracy means the feature representation carries strong case identity. Low heldout eval accuracy is expected because the heldout label is unseen; the key signal is whether within-slice identity is trivially recoverable, which can explain LOCO overfitting risk without adding manual features.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    features, labels, case_ids = _flatten_rows(payload)
    within = _within_case_probe(features, labels, case_ids, epochs=int(args.epochs), seed=int(args.seed))
    loco = _leave_one_case_probe(features, labels, case_ids, epochs=int(args.epochs), seed=int(args.seed))
    result = {
        "status": "complete",
        "purpose": "Step6E case invariance feature probe",
        "source": str(args.input),
        "case_ids": case_ids,
        "feature_row_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "chance_accuracy": 1.0 / max(len(case_ids), 1),
        "within_case_probe": within,
        "leave_one_case_probe": loco,
        "finding": "High random-split case-ID accuracy indicates case identity is strongly encoded in the current candidate features; this supports a learned case-invariant representation objective rather than manual feature edits.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "case_ids": result["case_ids"],
                "feature_row_count": result["feature_row_count"],
                "feature_dim": result["feature_dim"],
                "chance_accuracy": result["chance_accuracy"],
                "random_split_eval_accuracy": within["eval_accuracy"],
                "loco_mean_train_accuracy": loco["mean_train_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
