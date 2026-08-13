#!/usr/bin/env python3
"""Bucket a benchmark by cost and summarize geometry/constraint pressure."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.verify import soft_violation_normalized  # noqa: E402


BANDS = (
    ("elite", -math.inf, 4.5),
    ("strong", 4.5, 5.5),
    ("middle", 5.5, 6.5),
    ("weak", 6.5, 8.0),
    ("critical", 8.0, math.inf),
)


def _membership(ids: torch.Tensor) -> torch.Tensor:
    groups = [int(value) for value in torch.unique(ids).tolist() if int(value) > 0]
    if not groups:
        return torch.zeros((0, int(ids.numel())), dtype=torch.bool)
    return torch.stack([ids == group for group in groups])


def _case_features(row: dict, metadata: dict) -> dict:
    boxes = torch.as_tensor(row["positions"], dtype=torch.float64)
    constraints = torch.as_tensor(metadata["constraints"], dtype=torch.long)
    n = int(row["block_count"])
    group = _membership(constraints[:, 3])
    mib = _membership(constraints[:, 2])
    boundary_codes = constraints[:, 4]
    boundary_bits = torch.stack(
        [(boundary_codes & bit) != 0 for bit in (1, 2, 4, 8)], dim=1
    )
    case = SimpleNamespace(
        group_membership=group,
        mib_membership=mib,
        boundary_bits=boundary_bits,
        normalized=False,
    )
    soft = soft_violation_normalized(case, boxes)
    left = float(boxes[:, 0].min())
    bottom = float(boxes[:, 1].min())
    right = float((boxes[:, 0] + boxes[:, 2]).max())
    top = float((boxes[:, 1] + boxes[:, 3]).max())
    bbox_area = max(0.0, right - left) * max(0.0, top - bottom)
    block_area = float((boxes[:, 2] * boxes[:, 3]).sum())
    aspect = torch.maximum(boxes[:, 2] / boxes[:, 3], boxes[:, 3] / boxes[:, 2])
    b2b_edges = metadata["b2b_connectivity"]
    p2b_edges = metadata["p2b_connectivity"]
    return {
        "test_id": int(row["test_id"]),
        "block_count": n,
        "cost": float(row["cost"]),
        "area_gap": float(row["area_gap"]),
        "hpwl_gap": float(row["hpwl_gap"]),
        "violations_relative": float(row["violations_relative"]),
        "runtime_seconds": float(row["runtime_seconds"]),
        "utilization": block_area / bbox_area if bbox_area else 0.0,
        "bbox_aspect": max(right - left, top - bottom) / max(min(right - left, top - bottom), 1e-12),
        "max_block_aspect": float(aspect.max()),
        "sliver32_ratio": float((aspect > 32.0).double().mean()),
        "boundary_violations": soft.raw_boundary,
        "grouping_violations": soft.raw_grouping,
        "mib_violations": soft.raw_mib,
        "boundary_density": float((boundary_codes != 0).double().mean()),
        "group_density": float((constraints[:, 3] > 0).double().mean()),
        "mib_density": float((constraints[:, 2] > 0).double().mean()),
        "fixed_density": float((constraints[:, 0] > 0).double().mean()),
        "preplaced_density": float((constraints[:, 1] > 0).double().mean()),
        "b2b_density": len(b2b_edges) / max(n * (n - 1), 1),
        "p2b_per_block": len(p2b_edges) / n,
        "pin_count": len(metadata["pins_pos"]),
    }


def _band(cost: float) -> str:
    return next(name for name, low, high in BANDS if low <= cost < high)


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _summarize(rows: list[dict]) -> dict:
    metrics = (
        "cost", "block_count", "violations_relative", "area_gap", "hpwl_gap",
        "utilization", "boundary_violations", "grouping_violations", "mib_violations",
        "preplaced_density", "fixed_density", "b2b_density", "p2b_per_block",
        "max_block_aspect", "sliver32_ratio", "runtime_seconds",
    )
    return {
        "count": len(rows),
        "test_ids": [row["test_id"] for row in rows],
        "mean": {key: _mean(rows, key) for key in metrics},
    }


def _critical_mode(row: dict) -> str:
    if row["area_gap"] <= 0.10:
        return "dense_topology_constraint"
    if row["area_gap"] >= 1.0:
        return "sparse_area_fragmentation"
    return "mixed"


def _markdown(report: dict) -> str:
    lines = [
        "# HCFP-5090 full100 score-bucket diagnosis",
        "",
        "Lower official local cost is better. Bands: elite `<4.5`, strong `4.5–5.5`, "
        "middle `5.5–6.5`, weak `6.5–8.0`, critical `>=8.0`.",
        "",
        "| Band | Cases | Mean cost | N | Vrel | Area gap | HPWL gap | Util | Boundary | Group | P2B/block | B2B density |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, _, _ in BANDS:
        item = report["buckets"][name]
        mean = item["mean"]
        lines.append(
            f"| {name} | {item['count']} | {mean['cost']:.4f} | {mean['block_count']:.1f} | "
            f"{mean['violations_relative']:.3f} | {mean['area_gap']:.3f} | "
            f"{mean['hpwl_gap']:.3f} | {mean['utilization']:.3f} | "
            f"{mean['boundary_violations']:.1f} | {mean['grouping_violations']:.1f} | "
            f"{mean['p2b_per_block']:.2f} | {mean['b2b_density']:.3f} |"
        )
    lines += ["", "## Critical cases", "", "| Case | N | Cost | Mode | Vrel | Area gap | HPWL gap | Util | Boundary | Group |", "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["critical_cases"]:
        lines.append(
            f"| {row['test_id']} | {row['block_count']} | {row['cost']:.4f} | {row['mode']} | "
            f"{row['violations_relative']:.3f} | {row['area_gap']:.3f} | {row['hpwl_gap']:.3f} | "
            f"{row['utilization']:.3f} | {row['boundary_violations']} | {row['grouping_violations']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--lane", default="learned")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    source = json.loads(Path(args.benchmark).read_text())
    rows = [
        _case_features(row, source["case_metadata"][str(row["test_id"])])
        for row in source["lanes"][args.lane]
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_band(row["cost"])].append(row)
    critical = sorted(grouped["critical"], key=lambda row: row["cost"], reverse=True)
    critical_cases = [{**row, "mode": _critical_mode(row)} for row in critical]
    report = {
        "benchmark": args.benchmark,
        "lane": args.lane,
        "buckets": {name: _summarize(grouped[name]) for name, _, _ in BANDS},
        "critical_cases": critical_cases,
        "best_cases": sorted(rows, key=lambda row: row["cost"])[:10],
        "worst_cases": sorted(rows, key=lambda row: row["cost"], reverse=True)[:10],
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_markdown(report))
    print(output_md.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
