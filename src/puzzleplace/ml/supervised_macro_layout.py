"""Step7ML-H supervised macro closure layout sidecar.

This module trains small deterministic closure-layout priors from Step7ML-G
FloorSet training labels and evaluates their geometry transfer on Step7P macro
payloads. It deliberately keeps Step7 candidate-quality labels out of the layout
training target.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from puzzleplace.ml.floorset_training_corpus import write_json
from puzzleplace.ml.training_backed_data_mart import as_float, rows, stable_split

FEATURE_NAMES = (
    "rel_w",
    "rel_h",
    "area_fraction",
    "fixed",
    "preplaced",
    "boundary_nonzero",
    "block_count_norm",
    "closure_aspect_log",
    "closure_type_mib",
    "closure_type_cluster",
    "area_rank_norm",
)
STEP7P_FALLBACK_DIRS = (
    Path("artifacts/research"),
    Path("/home/hwchen/PROJ/floorplan-step7p/artifacts/research"),
)
STEP7P_BASELINES = {
    "hard_feasible_non_noop": 11,
    "official_like_improving": 2,
    "quality_gate_pass": 2,
    "overlap_after_repack": 53,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_artifact(filename: str, *, base_dir: Path) -> Path | None:
    candidates = [base_dir / "artifacts/research" / filename]
    candidates.extend(path / filename for path in STEP7P_FALLBACK_DIRS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def bucket_block_count(count: int) -> str:
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    if count <= 8:
        return "6-8"
    return "9+"


def closure_key(example: dict[str, Any]) -> str:
    return f"{example.get('closure_type')}|{bucket_block_count(int(example.get('block_count', 0)))}"


def _closure_blocks(example: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = example.get("block_geometry")
    normalized = example.get("normalized_internal_coordinates")
    if not isinstance(geometry, list) or not isinstance(normalized, list):
        return []
    by_id = {
        int(row["block_id"]): row
        for row in normalized
        if isinstance(row, dict) and row.get("block_id") is not None
    }
    blocks: list[dict[str, Any]] = []
    for row in geometry:
        if not isinstance(row, dict):
            continue
        block_id = int(row.get("block_id", -1))
        target = by_id.get(block_id)
        if target is None:
            continue
        item = dict(row)
        item.update(
            {
                "target_rel_x": as_float(target.get("rel_x")),
                "target_rel_y": as_float(target.get("rel_y")),
                "target_rel_w": as_float(target.get("rel_w")),
                "target_rel_h": as_float(target.get("rel_h")),
            }
        )
        blocks.append(item)
    return blocks


def feature_vector(
    block: dict[str, Any],
    *,
    closure_type: str,
    block_count: int,
    closure_aspect: float | None,
    area_rank: int,
) -> list[float]:
    rel_w = as_float(block.get("target_rel_w", block.get("rel_w", block.get("w", 0.0))))
    rel_h = as_float(block.get("target_rel_h", block.get("rel_h", block.get("h", 0.0))))
    if "target_rel_w" not in block:
        rel_w = as_float(block.get("w"))
    if "target_rel_h" not in block:
        rel_h = as_float(block.get("h"))
    area_fraction = max(rel_w * rel_h, 0.0)
    return [
        rel_w,
        rel_h,
        area_fraction,
        1.0 if block.get("fixed") else 0.0,
        1.0 if block.get("preplaced") else 0.0,
        1.0 if int(block.get("boundary") or 0) > 0 else 0.0,
        min(block_count, 20) / 20.0,
        math.log(max(as_float(closure_aspect, 1.0), 1e-6)),
        1.0 if closure_type == "mib" else 0.0,
        1.0 if closure_type == "cluster" else 0.0,
        area_rank / max(block_count - 1, 1),
    ]


def _ranked_blocks(blocks: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    ranked = sorted(
        blocks,
        key=lambda row: (
            -(as_float(row.get("w")) * as_float(row.get("h"))),
            int(row.get("block_id", 0)),
        ),
    )
    return list(enumerate(ranked))


@dataclass
class LayoutDataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    val_records: list[dict[str, Any]]
    train_records: list[dict[str, Any]]
    template: dict[str, Any]
    split_counts: dict[str, int]


def build_layout_dataset(layout_rows: list[dict[str, Any]]) -> LayoutDataset:
    train_x: list[list[float]] = []
    train_y: list[list[float]] = []
    val_x: list[list[float]] = []
    val_y: list[list[float]] = []
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    template_acc: dict[str, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    split_counts: Counter[str] = Counter()
    for example in layout_rows:
        split = str(example.get("mart_split") or stable_split(str(example.get("sample_id"))))
        split_counts[split] += 1
        closure_type = str(example.get("closure_type"))
        block_count = int(example.get("block_count", 0))
        closure_aspect = example.get("closure_aspect")
        blocks = _closure_blocks(example)
        for area_rank, block in _ranked_blocks(blocks):
            features = feature_vector(
                block,
                closure_type=closure_type,
                block_count=block_count,
                closure_aspect=as_float(closure_aspect, 1.0),
                area_rank=area_rank,
            )
            target = [as_float(block["target_rel_x"]), as_float(block["target_rel_y"])]
            record = {
                "sample_id": example.get("sample_id"),
                "closure_type": closure_type,
                "block_count": block_count,
                "block_count_bucket": bucket_block_count(block_count),
                "features": features,
                "target": target,
                "area_rank": area_rank,
            }
            if split == "train":
                train_x.append(features)
                train_y.append(target)
                train_records.append(record)
                template_acc[closure_key(example)][area_rank].append(target)
            else:
                val_x.append(features)
                val_y.append(target)
                val_records.append(record)
    if not train_x or not val_x:
        raise ValueError("Step7ML-H needs non-empty train and validation splits")
    template: dict[str, Any] = {}
    for key, by_rank in template_acc.items():
        template[key] = {
            str(rank): [
                sum(values[0] for values in targets) / len(targets),
                sum(values[1] for values in targets) / len(targets),
            ]
            for rank, targets in by_rank.items()
        }
    return LayoutDataset(
        x_train=torch.tensor(train_x, dtype=torch.float32),
        y_train=torch.tensor(train_y, dtype=torch.float32),
        x_val=torch.tensor(val_x, dtype=torch.float32),
        y_val=torch.tensor(val_y, dtype=torch.float32),
        val_records=val_records,
        train_records=train_records,
        template=template,
        split_counts=dict(split_counts),
    )


class MacroLayoutMLP(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def _pairwise_order_accuracy(records: list[dict[str, Any]], preds: torch.Tensor) -> float | None:
    by_sample: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
    for record, pred in zip(records, preds.tolist(), strict=False):
        by_sample[str(record.get("sample_id"))].append((record["target"], pred))
    correct = 0
    total = 0
    for pairs in by_sample.values():
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                true_i, pred_i = pairs[i]
                true_j, pred_j = pairs[j]
                for axis in (0, 1):
                    true_sign = true_i[axis] <= true_j[axis]
                    pred_sign = pred_i[axis] <= pred_j[axis]
                    correct += int(true_sign == pred_sign)
                    total += 1
    return correct / total if total else None


def _metrics_by(
    records: list[dict[str, Any]],
    preds: torch.Tensor,
    targets: torch.Tensor,
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[str(record.get(key))].append(idx)
    out: dict[str, Any] = {}
    for name, indices in sorted(groups.items()):
        index_tensor = torch.tensor(indices, dtype=torch.long)
        out[name] = {
            "count": len(indices),
            "coordinate_mae": _mae(preds[index_tensor], targets[index_tensor]),
        }
    return out


def _template_predict(dataset: LayoutDataset) -> torch.Tensor:
    preds: list[list[float]] = []
    global_mean = torch.mean(dataset.y_train, dim=0).tolist()
    for record in dataset.val_records:
        key = f"{record['closure_type']}|{record['block_count_bucket']}"
        by_rank = dataset.template.get(key, {})
        pred = by_rank.get(str(record["area_rank"]), global_mean)
        preds.append([as_float(pred[0]), as_float(pred[1])])
    return torch.tensor(preds, dtype=torch.float32)


def train_macro_layout_model(
    layout_prior_path: Path,
    output_dir: Path,
    model_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    seed: int = 7,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    started = time.perf_counter()
    layout_rows = rows(load_json(layout_prior_path))
    dataset = build_layout_dataset(layout_rows)
    model = MacroLayoutMLP(hidden_dim=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    loader = DataLoader(
        TensorDataset(dataset.x_train, dataset.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        losses: list[float] = []
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        epoch_losses.append(sum(losses) / max(len(losses), 1))
    model.eval()
    with torch.no_grad():
        val_pred = model(dataset.x_val)
        template_pred = _template_predict(dataset)
    val_mae = _mae(val_pred, dataset.y_val)
    template_mae = _mae(template_pred, dataset.y_val)
    report = {
        "schema": "step7ml_h_training_report_v1",
        "layout_prior_example_count_used": len(layout_rows),
        "block_training_row_count": int(dataset.x_train.shape[0]),
        "block_validation_row_count": int(dataset.x_val.shape[0]),
        "train_validation_split_counts": dataset.split_counts,
        "model_type": "mlp_set_block_baseline",
        "baseline_type": "area_rank_template_mean",
        "feature_names": list(FEATURE_NAMES),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "training_losses": epoch_losses,
        "training_time_proxy_ms": (time.perf_counter() - started) * 1000.0,
        "validation_coordinate_mae": val_mae,
        "normalized_coordinate_mae": val_mae,
        "template_validation_coordinate_mae": template_mae,
        "model_improves_template": val_mae < template_mae,
        "pairwise_order_accuracy": _pairwise_order_accuracy(dataset.val_records, val_pred),
        "template_pairwise_order_accuracy": _pairwise_order_accuracy(
            dataset.val_records, template_pred
        ),
        "metrics_by_closure_type": _metrics_by(
            dataset.val_records, val_pred, dataset.y_val, "closure_type"
        ),
        "metrics_by_block_count_bucket": _metrics_by(
            dataset.val_records, val_pred, dataset.y_val, "block_count_bucket"
        ),
    }
    config = {
        "schema": "step7ml_h_training_config_v1",
        "model_type": report["model_type"],
        "feature_names": list(FEATURE_NAMES),
        "target": "normalized internal x/y from FloorSet training fp_sol only",
        "uses_step7_candidate_quality_as_target": False,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "macro_layout_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "template": dataset.template,
        },
        checkpoint_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "step7ml_h_training_config.json",
        {**config, "checkpoint_path": str(checkpoint_path)},
    )
    write_json(output_dir / "step7ml_h_training_report.json", report)
    write_json(
        output_dir / "step7ml_h_validation_report.json",
        {
            "schema": "step7ml_h_validation_report_v1",
            "validation_coordinate_mae": val_mae,
            "template_validation_coordinate_mae": template_mae,
            "model_improves_template": val_mae < template_mae,
            "pairwise_order_accuracy": report["pairwise_order_accuracy"],
            "metrics_by_closure_type": report["metrics_by_closure_type"],
            "metrics_by_block_count_bucket": report["metrics_by_block_count_bucket"],
        },
    )
    write_json(
        output_dir / "step7ml_h_ablation_report.json",
        {
            "schema": "step7ml_h_ablation_report_v1",
            "ablations": [
                {"name": "area_rank_template_mean", "validation_coordinate_mae": template_mae},
                {"name": "mlp_set_block_baseline", "validation_coordinate_mae": val_mae},
            ],
        },
    )
    return {"report": report, "checkpoint_path": str(checkpoint_path)}


def _load_model(checkpoint_path: Path) -> tuple[MacroLayoutMLP, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config", {})
    model = MacroLayoutMLP(hidden_dim=int(config.get("hidden_dim", 64)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _payload_bbox(blocks: list[dict[str, Any]]) -> dict[str, float]:
    x0 = min(as_float(block["x"]) for block in blocks)
    y0 = min(as_float(block["y"]) for block in blocks)
    x1 = max(as_float(block["x"]) + as_float(block["w"]) for block in blocks)
    y1 = max(as_float(block["y"]) + as_float(block["h"]) for block in blocks)
    return {"x": x0, "y": y0, "w": max(x1 - x0, 1.0), "h": max(y1 - y0, 1.0)}


def _overlap_area(a: dict[str, float], b: dict[str, float]) -> float:
    dx = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    dy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return max(0.0, dx) * max(0.0, dy)


def _overlap_count(blocks: list[dict[str, Any]]) -> int:
    count = 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if _overlap_area(blocks[i], blocks[j]) > 1e-9:
                count += 1
    return count


def _predict_payload_layout(model: MacroLayoutMLP, payload: dict[str, Any]) -> dict[str, Any]:
    blocks = [dict(block) for block in payload.get("blocks", []) if isinstance(block, dict)]
    box = _payload_bbox(blocks)
    closure_type = "mib" if any(int(block.get("mib") or 0) > 0 for block in blocks) else "cluster"
    block_count = len(blocks)
    ranked = _ranked_blocks(blocks)
    predicted_blocks: list[dict[str, Any]] = []
    model_moved = 0
    for area_rank, block in ranked:
        item = dict(block)
        item["target_rel_w"] = as_float(block.get("w")) / max(box["w"], 1e-9)
        item["target_rel_h"] = as_float(block.get("h")) / max(box["h"], 1e-9)
        features = feature_vector(
            item,
            closure_type=closure_type,
            block_count=block_count,
            closure_aspect=box["w"] / max(box["h"], 1e-9),
            area_rank=area_rank,
        )
        with torch.no_grad():
            rel_x, rel_y = model(torch.tensor([features], dtype=torch.float32))[0].tolist()
        movable = bool(block.get("movable", not (block.get("fixed") or block.get("preplaced"))))
        if movable:
            new_x = box["x"] + max(0.0, min(rel_x, 1.0)) * max(
                box["w"] - as_float(block["w"]), 0.0
            )
            new_y = box["y"] + max(0.0, min(rel_y, 1.0)) * max(
                box["h"] - as_float(block["h"]), 0.0
            )
            moved_x = abs(new_x - as_float(block["x"])) > 1e-6
            moved_y = abs(new_y - as_float(block["y"])) > 1e-6
            model_moved += int(moved_x or moved_y)
        else:
            new_x = as_float(block["x"])
            new_y = as_float(block["y"])
        predicted_blocks.append(
            {
                **block,
                "x": new_x,
                "y": new_y,
                "w": as_float(block["w"]),
                "h": as_float(block["h"]),
                "model_rel_x": rel_x,
                "model_rel_y": rel_y,
            }
        )
    return {
        "candidate_id": payload.get("candidate_id"),
        "case_id": payload.get("case_id"),
        "source_candidate_id": payload.get("source_candidate_id"),
        "target_region": payload.get("target_region"),
        "variant": payload.get("variant"),
        "block_count": block_count,
        "movable_block_count": payload.get("movable_block_count"),
        "fixed_or_preplaced_in_closure": payload.get("fixed_or_preplaced_in_closure"),
        "prediction_bbox": box,
        "predicted_blocks": predicted_blocks,
        "moved_block_count": model_moved,
        "overlap_pair_count": _overlap_count(predicted_blocks),
    }


def _step7p_baseline_rows(base_dir: Path) -> dict[str, dict[str, Any]]:
    path = find_artifact("step7p_real_repack_candidates.json", base_dir=base_dir)
    if path is None:
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows(load_json(path)):
        for key in ("candidate_id", "step7o_a_variant_candidate_id", "source_candidate_id"):
            if row.get(key):
                indexed[str(row[key])] = row
    return indexed


def evaluate_on_step7p(
    checkpoint_path: Path,
    base_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    model, _payload = _load_model(checkpoint_path)
    payload_path = find_artifact("step7p_geometry_payloads.json", base_dir=base_dir)
    if payload_path is None:
        raise FileNotFoundError(
            "Missing step7p_geometry_payloads.json in current or fallback worktree"
        )
    payloads = rows(load_json(payload_path))
    baseline_by_id = _step7p_baseline_rows(base_dir)
    predictions: list[dict[str, Any]] = []
    for payload_row in payloads:
        pred = _predict_payload_layout(model, payload_row)
        baseline = baseline_by_id.get(str(pred.get("candidate_id")), {})
        overlap = int(pred["overlap_pair_count"])
        non_noop = int(pred["moved_block_count"]) > 0
        hard_feasible_non_noop = overlap == 0 and non_noop
        official_improving = (
            bool(baseline.get("official_like_cost_improving")) and hard_feasible_non_noop
        )
        quality_gate_pass = (
            bool(baseline.get("step7p_selected_for_archive")) and hard_feasible_non_noop
        )
        predictions.append(
            {
                **{k: v for k, v in pred.items() if k != "predicted_blocks"},
                "predicted_blocks_preview": pred["predicted_blocks"][:20],
                "non_original_non_noop": non_noop,
                "hard_feasible_non_noop": hard_feasible_non_noop,
                "overlap_after_model_layout": overlap > 0,
                "official_like_improving": official_improving,
                "quality_gate_pass": quality_gate_pass,
                "route_class": baseline.get("source_actual_locality_class")
                or baseline.get("predicted_locality_class"),
                "dominated_by_original": baseline.get("dominated_by_original"),
                "metric_regressing": as_float(baseline.get("official_like_cost_delta")) > 0
                and not official_improving,
                "failure_attribution": _prediction_failure(
                    overlap, non_noop, hard_feasible_non_noop, baseline
                ),
                "baseline_step7p_status": baseline.get("quality_gate_status"),
                "baseline_hard_feasible_non_noop": baseline.get("hard_feasible_non_noop"),
                "baseline_official_like_improving": baseline.get("official_like_cost_improving"),
            }
        )
    failure_counts = Counter(str(row["failure_attribution"]) for row in predictions)
    route_counts = Counter(str(row.get("route_class")) for row in predictions)
    overlap_count = sum(1 for row in predictions if row["overlap_after_model_layout"])
    hard_feasible = sum(1 for row in predictions if row["hard_feasible_non_noop"])
    official = sum(1 for row in predictions if row["official_like_improving"])
    gate = sum(1 for row in predictions if row["quality_gate_pass"])
    report = {
        "schema": "step7ml_h_step7p_eval_summary_v1",
        "step7p_payload_count": len(payloads),
        "regenerated_candidate_count": len(predictions),
        "non_original_non_noop_count": sum(
            1 for row in predictions if row["non_original_non_noop"]
        ),
        "hard_feasible_non_noop_count": hard_feasible,
        "overlap_after_repack_count": overlap_count,
        "overlap_reduction_vs_step7p": STEP7P_BASELINES["overlap_after_repack"] - overlap_count,
        "official_like_improving_count": official,
        "quality_gate_pass_count": gate,
        "dominated_by_original_count": sum(
            1 for row in predictions if row.get("dominated_by_original")
        ),
        "metric_regressing_count": sum(1 for row in predictions if row.get("metric_regressing")),
        "route_count_by_class": dict(route_counts),
        "projected_model_prediction_survival_count": gate,
        "failure_counts": dict(failure_counts),
        "runtime_proxy_ms": (time.perf_counter() - started) * 1000.0,
        "comparison_baseline": STEP7P_BASELINES,
        "payload_source": str(payload_path),
        "metric_semantics": (
            "Step7P official-like deltas are joined from Step7P rows; model predictions are not "
            "allowed to bypass geometry overlap/fixity screening. Full official "
            "metric rerun remains "
            "future integration work."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "step7ml_h_prediction_candidates.json",
        {"schema": "step7ml_h_prediction_candidates_v1", "rows": predictions},
    )
    write_json(
        output_dir / "step7ml_h_step7p_feasibility_report.json",
        {
            "schema": "step7ml_h_step7p_feasibility_report_v1",
            "summary": report,
            "failure_counts": dict(failure_counts),
        },
    )
    write_json(
        output_dir / "step7ml_h_step7p_metric_report.json",
        {
            "schema": "step7ml_h_step7p_metric_report_v1",
            "summary": report,
            "metric_semantics": report["metric_semantics"],
        },
    )
    write_json(
        output_dir / "step7ml_h_quality_gate_report.json",
        {
            "schema": "step7ml_h_quality_gate_report_v1",
            "quality_gate_pass_count": gate,
            "official_like_improving_count": official,
            "comparison_baseline": STEP7P_BASELINES,
            "decision_inputs": report,
        },
    )
    return report


def _prediction_failure(
    overlap: int,
    non_noop: bool,
    hard_feasible_non_noop: bool,
    baseline: dict[str, Any],
) -> str:
    if not non_noop:
        return "no_op"
    if overlap > 0:
        return "overlap_after_model_layout"
    if not hard_feasible_non_noop:
        return "fixed_preplaced_blocked"
    if as_float(baseline.get("soft_constraint_delta")) > 0:
        return "soft_regression"
    if as_float(baseline.get("hpwl_delta")) > 0:
        return "hpwl_regression"
    if as_float(baseline.get("bbox_area_delta")) > 0:
        return "bbox_regression"
    if baseline.get("predicted_locality_class") == "global":
        return "route_global"
    return "none"


def decide_h(training_report: dict[str, Any], eval_report: dict[str, Any]) -> str:
    if eval_report["overlap_after_repack_count"] < STEP7P_BASELINES["overlap_after_repack"]:
        if eval_report["hard_feasible_non_noop_count"] > STEP7P_BASELINES["hard_feasible_non_noop"]:
            if (
                eval_report["official_like_improving_count"]
                > STEP7P_BASELINES["official_like_improving"]
            ):
                return "promote_supervised_macro_layout_prior"
            return "use_as_repack_initialization_only"
        return "build_geometry_legalization_layer"
    if training_report.get("model_improves_template"):
        return "refine_model_features_or_architecture"
    return "collect_more_training_labels"


def write_decision(
    output_dir: Path,
    training_report: dict[str, Any],
    eval_report: dict[str, Any],
) -> str:
    decision = decide_h(training_report, eval_report)
    text = f"""# Step7ML-H Supervised Macro Closure Layout Training

Decision: `{decision}`

## Training metrics

- layout_prior_example_count_used: {training_report['layout_prior_example_count_used']}
- model_type: {training_report['model_type']}
- training_time_proxy_ms: {training_report['training_time_proxy_ms']:.3f}
- validation_coordinate_mae: {training_report['validation_coordinate_mae']:.6f}
- template_validation_coordinate_mae: {training_report['template_validation_coordinate_mae']:.6f}
- model_improves_template: {training_report['model_improves_template']}
- pairwise_order_accuracy: {training_report['pairwise_order_accuracy']}

## Step7P-style evaluation

- step7p_payload_count: {eval_report['step7p_payload_count']}
- regenerated_candidate_count: {eval_report['regenerated_candidate_count']}
- non_original_non_noop_count: {eval_report['non_original_non_noop_count']}
- hard_feasible_non_noop_count: {eval_report['hard_feasible_non_noop_count']} vs Step7P baseline 11
- overlap_after_repack_count: {eval_report['overlap_after_repack_count']} vs Step7P baseline 53
- overlap_reduction_vs_step7p: {eval_report['overlap_reduction_vs_step7p']}
- official_like_improving_count: {eval_report['official_like_improving_count']} vs Step7P baseline 2
- quality_gate_pass_count: {eval_report['quality_gate_pass_count']} vs Step7P baseline 2
- failure_counts: {eval_report['failure_counts']}

## Interpretation

The supervised layout prior is trained only on FloorSet training `fp_sol` closure
layout labels. Step7 candidate labels are not used as layout targets. Step7P
artifact evaluation screens model layouts for geometry overlap/fixity before
joining Step7P metric/gate provenance, so success is not claimed from training
loss alone.
"""
    (output_dir / "step7ml_h_decision.md").write_text(text, encoding="utf-8")
    return decision
