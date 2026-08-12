#!/usr/bin/env python3
"""Infer the raw FloorSet-Lite tree_sol schema and test gold-size contour decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.floorset_lite import fp_sol_to_xywh  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", default="artifacts/floorset-v10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--min-blocks", type=int, default=21)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--max-layouts-per-file", type=int, default=1)
    args = parser.parse_args(argv)
    if args.limit <= 0 or args.max_layouts_per_file <= 0:
        parser.error("--limit and --max-layouts-per-file must be positive")
    if not 1 <= args.min_blocks <= args.max_blocks:
        parser.error("invalid block range")

    root = Path(args.floorset_lite_root).resolve()
    layout_root = root if root.name == "floorset_lite" else root / "floorset_lite"
    files = sorted(layout_root.glob("worker_*/layouts*"))
    if not files:
        raise FileNotFoundError(f"no FloorSet-Lite files under {layout_root}")
    generator = random.Random(args.seed)
    generator.shuffle(files)
    rows = []
    source_files = set()
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        indices = list(range(len(payload[0])))
        generator.shuffle(indices)
        accepted = 0
        for index in indices:
            block_count = int((payload[0][index][:, 0] != -1).sum().item())
            if not args.min_blocks <= block_count <= args.max_blocks:
                continue
            rows.append(
                _audit_tree(
                    f"{path.parent.name}/{path.name}:{index}",
                    payload[4][index][: block_count - 1],
                    fp_sol_to_xywh(payload[5][index], block_count),
                )
            )
            source_files.add(str(path.relative_to(layout_root)))
            accepted += 1
            if len(rows) >= args.limit or accepted >= args.max_layouts_per_file:
                break
        if len(rows) >= args.limit:
            break
    if len(rows) != args.limit:
        raise RuntimeError(f"collected {len(rows)} samples, expected {args.limit}")

    report = {
        "schema_version": 1,
        "config": vars(args),
        "provenance": {
            "data_root": str(root),
            "source_file_count": len(source_files),
            "source_file_sha256": _sha256_lines(sorted(source_files)),
            "torch_version": torch.__version__,
        },
        "summary": _summary(rows),
        "samples": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _audit_tree(sample_id: str, tree: torch.Tensor, gold: torch.Tensor) -> dict[str, Any]:
    parsed = _parse_parent_child_side(tree, len(gold))
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "block_count": len(gold),
        "row_count": int(tree.shape[0]),
        "schema_valid": parsed is not None,
    }
    if parsed is None:
        return result
    root, left, right = parsed
    decoded = _contour_pack(root, left, right, gold[:, 2:4])
    gold64 = gold.to(dtype=torch.float64)
    decoded_area, decoded_width, decoded_height = _bbox(decoded)
    gold_area, gold_width, gold_height = _bbox(gold64)
    translated_decoded = decoded.clone()
    translated_gold = gold64.clone()
    translated_decoded[:, :2] -= translated_decoded[:, :2].amin(dim=0)
    translated_gold[:, :2] -= translated_gold[:, :2].amin(dim=0)
    edge = _edge_agreement(tree, gold64)
    result.update(
        {
            "root": root,
            "edge_agreement": edge,
            "decoded_overlap_pairs": _overlap_pairs(decoded),
            "decoded_width_relative_error": abs(decoded_width - gold_width) / max(gold_width, 1.0e-12),
            "decoded_height_relative_error": abs(decoded_height - gold_height) / max(gold_height, 1.0e-12),
            "decoded_area_relative_error": abs(decoded_area - gold_area) / max(gold_area, 1.0e-12),
            "decoded_position_exact_fraction": float(
                torch.isclose(
                    translated_decoded[:, :2],
                    translated_gold[:, :2],
                    rtol=0.0,
                    atol=1.0e-5,
                ).all(dim=1).double().mean()
            ),
        }
    )
    return result


def _parse_parent_child_side(
    tree: torch.Tensor,
    block_count: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    values = torch.as_tensor(tree, dtype=torch.float64)
    if values.shape != (block_count - 1, 3) or not bool(torch.isfinite(values).all()):
        return None
    rounded = values.round()
    if not torch.equal(values, rounded):
        return None
    rows = rounded.to(dtype=torch.long)
    if bool((rows[:, :2] < 0).any()) or bool((rows[:, :2] >= block_count).any()):
        return None
    if not bool(((rows[:, 2] == 0) | (rows[:, 2] == 1)).all()):
        return None
    left = [-1] * block_count
    right = [-1] * block_count
    parents = [-1] * block_count
    for parent, child, side in rows.tolist():
        if parent == child or parents[child] != -1:
            return None
        branch = left if side == 0 else right
        if branch[parent] != -1:
            return None
        branch[parent] = child
        parents[child] = parent
    roots = [index for index, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        return None
    root = roots[0]
    visited: set[int] = set()
    active: set[int] = set()

    def visit(node: int) -> bool:
        if node in active:
            return False
        if node in visited:
            return True
        active.add(node)
        for child in (left[node], right[node]):
            if child >= 0 and not visit(child):
                return False
        active.remove(node)
        visited.add(node)
        return True

    if not visit(root) or len(visited) != block_count:
        return None
    return root, tuple(left), tuple(right)


def _contour_pack(
    root: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
    dimensions: torch.Tensor,
) -> torch.Tensor:
    dims = torch.as_tensor(dimensions, dtype=torch.float64)
    boxes = torch.zeros((len(dims), 4), dtype=torch.float64)
    contour: list[tuple[float, float, float]] = []

    def contour_y(x0: float, x1: float) -> float:
        return max(
            (height for start, end, height in contour if x0 < end and x1 > start),
            default=0.0,
        )

    def update(x0: float, x1: float, height: float) -> None:
        retained: list[tuple[float, float, float]] = []
        for start, end, old_height in contour:
            if end <= x0 or start >= x1:
                retained.append((start, end, old_height))
                continue
            if start < x0:
                retained.append((start, x0, old_height))
            if end > x1:
                retained.append((x1, end, old_height))
        retained.append((x0, x1, height))
        retained.sort()
        contour[:] = retained

    def place(node: int, x: float) -> None:
        width, height = (float(value) for value in dims[node])
        y = contour_y(x, x + width)
        boxes[node] = torch.tensor((x, y, width, height), dtype=torch.float64)
        update(x, x + width, y + height)
        if left[node] >= 0:
            place(left[node], x + width)
        if right[node] >= 0:
            place(right[node], x)

    place(root, 0.0)
    return boxes


def _edge_agreement(tree: torch.Tensor, gold: torch.Tensor) -> dict[str, int]:
    counts = {
        "side0_total": 0,
        "side0_child_right": 0,
        "side1_total": 0,
        "side1_same_x": 0,
        "side1_above": 0,
        "side1_below": 0,
    }
    for parent, child, side in torch.as_tensor(tree, dtype=torch.long).tolist():
        p = gold[parent]
        c = gold[child]
        if side == 0:
            counts["side0_total"] += 1
            counts["side0_child_right"] += int(abs(float(c[0] - (p[0] + p[2]))) <= 1.0e-5)
        else:
            counts["side1_total"] += 1
            counts["side1_same_x"] += int(abs(float(c[0] - p[0])) <= 1.0e-5)
            counts["side1_above"] += int(abs(float(c[1] - (p[1] + p[3]))) <= 1.0e-5)
            counts["side1_below"] += int(abs(float(p[1] - (c[1] + c[3]))) <= 1.0e-5)
    return counts


def _overlap_pairs(boxes: torch.Tensor) -> int:
    left = boxes[:, 0]
    bottom = boxes[:, 1]
    right = left + boxes[:, 2]
    top = bottom + boxes[:, 3]
    overlap = (
        (left[:, None] < right[None, :] - 1.0e-9)
        & (right[:, None] > left[None, :] + 1.0e-9)
        & (bottom[:, None] < top[None, :] - 1.0e-9)
        & (top[:, None] > bottom[None, :] + 1.0e-9)
    )
    return int(torch.triu(overlap, diagonal=1).sum())


def _bbox(boxes: torch.Tensor) -> tuple[float, float, float]:
    width = float((boxes[:, 0] + boxes[:, 2]).amax() - boxes[:, 0].amin())
    height = float((boxes[:, 1] + boxes[:, 3]).amax() - boxes[:, 1].amin())
    return width * height, width, height


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["schema_valid"]]
    edge = {
        key: sum(int(row["edge_agreement"][key]) for row in valid)
        for key in (
            "side0_total",
            "side0_child_right",
            "side1_total",
            "side1_same_x",
            "side1_above",
            "side1_below",
        )
    }
    result = {
        "sample_count": len(rows),
        "valid_parent_child_side_count": len(valid),
        "valid_parent_child_side_rate": len(valid) / len(rows),
        "decoded_overlap_free_rate": sum(row["decoded_overlap_pairs"] == 0 for row in valid) / max(len(valid), 1),
        "side0_child_right_rate": edge["side0_child_right"] / max(edge["side0_total"], 1),
        "side1_same_x_rate": edge["side1_same_x"] / max(edge["side1_total"], 1),
        "side1_above_rate": edge["side1_above"] / max(edge["side1_total"], 1),
        "side1_below_rate": edge["side1_below"] / max(edge["side1_total"], 1),
        "median_decoded_area_relative_error": _median(row["decoded_area_relative_error"] for row in valid),
        "p95_decoded_area_relative_error": _percentile(
            [row["decoded_area_relative_error"] for row in valid], 0.95
        ),
        "mean_decoded_position_exact_fraction": sum(
            row["decoded_position_exact_fraction"] for row in valid
        )
        / max(len(valid), 1),
    }
    result["inferred_schema"] = (
        "parent_child_side0_right_side1_above"
        if result["valid_parent_child_side_rate"] >= 0.999
        and result["side0_child_right_rate"] >= 0.99
        and result["side1_same_x_rate"] >= 0.99
        else "unresolved"
    )
    result["decision"] = (
        "KEEP"
        if result["inferred_schema"] != "unresolved"
        and result["decoded_overlap_free_rate"] == 1.0
        else "MODIFY"
    )
    return result


def _median(values: Any) -> float | None:
    ordered = sorted(float(value) for value in values)
    return _percentile(ordered, 0.5) if ordered else None


def _percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = quantile * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
