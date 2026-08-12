#!/usr/bin/env python3
"""Audit latent-outline recovery on FloorSet-Lite training layouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.floorset_lite import fp_sol_to_xywh, target_positions_from_solution  # noqa: E402
from hcfp.outline_inference import infer_outline_hypotheses  # noqa: E402


def audit_layout(
    sample_id: str,
    area_constraints: torch.Tensor,
    pins_pos: torch.Tensor,
    fp_sol: torch.Tensor,
    *,
    beam: int = 8,
) -> dict[str, Any]:
    """Return compact recovery evidence for one training-only layout."""

    rows = torch.as_tensor(area_constraints)
    block_count = int((rows[:, 0] != -1).sum().item())
    area = rows[:block_count, 0].to(dtype=torch.float64)
    constraints = rows[:block_count, 1:6].to(dtype=torch.long)
    rectangles = fp_sol_to_xywh(fp_sol, block_count).to(dtype=torch.float64)
    targets = target_positions_from_solution(constraints, rectangles).to(dtype=torch.float64)
    pins = torch.as_tensor(pins_pos, dtype=torch.float64)
    pins = pins[pins[:, 0] != -1] if pins.numel() else pins.reshape(0, 2)
    case = SimpleNamespace(
        n=block_count,
        area=area,
        pins=pins,
        target=targets,
        fixed_mask=constraints[:, 0] != 0,
        preplaced_mask=constraints[:, 1] != 0,
    )
    hypotheses = infer_outline_hypotheses(case, max_hypotheses=beam)
    gold = _bbox(rectangles)
    pin_perimeter = _point_bbox(pins)
    preplaced = case.preplaced_mask
    evidence = [
        _hypothesis_metrics(hypothesis, gold, pin_perimeter, rectangles, targets, preplaced)
        for hypothesis in hypotheses
    ]
    top = evidence[0] if evidence else None
    return {
        "sample_id": sample_id,
        "block_count": block_count,
        "pin_count": int(pins.shape[0]),
        "preplaced_density": float(preplaced.to(torch.float64).mean().item()),
        "boundary_density": float((constraints[:, 4] != 0).to(torch.float64).mean().item()),
        "hypotheses": len(evidence),
        "top1": top,
        "oracle": {
            "area_relative_error": _minimum(evidence, "area_relative_error"),
            "width_relative_error": _minimum(evidence, "width_relative_error"),
            "height_relative_error": _minimum(evidence, "height_relative_error"),
            "max_dimension_relative_error": _minimum(
                evidence, "max_dimension_relative_error"
            ),
            "pin_perimeter_side_recovery": _maximum(
                evidence, "pin_perimeter_side_recovery"
            ),
            "pin_side_coverage": _maximum(evidence, "pin_side_coverage"),
            "side_coverage": _maximum(evidence, "side_coverage"),
            "gold_outline_side_recovery": _maximum(
                evidence, "gold_outline_side_recovery"
            ),
            "gold_outside_block_ratio": _minimum(
                evidence, "gold_outside_block_ratio"
            ),
        },
        "all_hypotheses_contain_preplaced": bool(evidence)
        and all(bool(row["contains_preplaced"]) for row in evidence),
    }


def build_report(
    records: Iterable[dict[str, Any]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    rows = list(records)
    sample_hash = hashlib.sha256()
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {
        "block_count": defaultdict(list),
        "pin_count": defaultdict(list),
        "preplaced_density": defaultdict(list),
        "boundary_density": defaultdict(list),
    }
    for row in rows:
        sample_hash.update((str(row["sample_id"]) + "\n").encode())
        buckets["block_count"][_block_bucket(int(row["block_count"]))].append(row)
        buckets["pin_count"][_pin_bucket(int(row["pin_count"]))].append(row)
        buckets["preplaced_density"][
            _density_bucket(float(row["preplaced_density"]))
        ].append(row)
        buckets["boundary_density"][
            _density_bucket(float(row["boundary_density"]))
        ].append(row)
    summary = _summarize(rows)
    requested_cases = int(provenance.get("requested_cases", len(rows)))
    gates = {
        "audited_cases_eq_requested": summary["cases"] == requested_cases,
        "oracle_area_error_median_lt_0_01": (
            summary["oracle_area_relative_error"]["median"] < 0.01
        ),
        "oracle_area_error_p95_lt_0_03": (
            summary["oracle_area_relative_error"]["p95"] < 0.03
        ),
        "gold_outline_side_recovery_ge_0_95": (
            summary["oracle_gold_outline_side_recovery"]["mean"] >= 0.95
        ),
        "preplaced_containment_eq_1": (
            summary["all_hypotheses_contain_preplaced_rate"] == 1.0
        ),
        "nonempty_hypothesis_rate_eq_1": summary["nonempty_hypothesis_rate"] == 1.0,
    }
    gates["passed"] = all(gates.values())
    bucket_summary = {
        dimension: {
            name: _summarize(values) for name, values in sorted(groups.items())
        }
        for dimension, groups in buckets.items()
    }
    stable_summary = {
        "summary": summary,
        "buckets": bucket_summary,
        "gates": gates,
    }
    summary_sha256 = hashlib.sha256(
        json.dumps(stable_summary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "training_only": True,
        "definitions": {
            "gold_outline": "bbox(fp_sol rectangles), used only as an audit label",
            "pin_perimeter": "min/max valid pins_pos coordinates from official input",
            "inference_fields": (
                "area, pins_pos, fixed dimensions, and preplaced rectangles only; movable "
                "fp_sol geometry is never exposed to infer_outline_hypotheses"
            ),
            "side_recovery": "fraction of gold bbox sides recovered within 5% of its maximum extent",
            "oracle_at_k": "best hypothesis per metric; not an input-aware selector result",
        },
        "provenance": {
            **provenance,
            "sample_id_sha256": sample_hash.hexdigest(),
            "summary_sha256": summary_sha256,
        },
        "summary": summary,
        "buckets": bucket_summary,
        "gates": gates,
        "worst_oracle_area_cases": sorted(
            (
                {
                    "sample_id": row["sample_id"],
                    "block_count": row["block_count"],
                    "area_relative_error": row["oracle"]["area_relative_error"],
                    "hypotheses": row["hypotheses"],
                }
                for row in rows
            ),
            key=lambda row: (-_finite_or_inf(row["area_relative_error"]), row["sample_id"]),
        )[:20],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "nonempty_hypothesis_rate": 0.0,
            "all_hypotheses_contain_preplaced_rate": 0.0,
            "oracle_area_relative_error": _stats([]),
            "oracle_width_relative_error": _stats([]),
            "oracle_height_relative_error": _stats([]),
            "oracle_max_dimension_relative_error": _stats([]),
            "oracle_pin_perimeter_side_recovery": _stats([]),
            "oracle_pin_side_coverage": _stats([]),
            "oracle_side_coverage": _stats([]),
            "oracle_gold_outline_side_recovery": _stats([]),
            "oracle_gold_outside_block_ratio": _stats([]),
            "top1_area_relative_error": _stats([]),
            "top1_width_relative_error": _stats([]),
            "top1_height_relative_error": _stats([]),
            "top1_pin_perimeter_side_recovery": _stats([]),
            "top1_pin_side_coverage": _stats([]),
            "top1_side_coverage": _stats([]),
            "top1_gold_outline_side_recovery": _stats([]),
            "top1_gold_outside_block_ratio": _stats([]),
        }
    return {
        "cases": len(rows),
        "nonempty_hypothesis_rate": sum(row["hypotheses"] > 0 for row in rows)
        / len(rows),
        "all_hypotheses_contain_preplaced_rate": sum(
            bool(row["all_hypotheses_contain_preplaced"]) for row in rows
        )
        / len(rows),
        "oracle_area_relative_error": _stats(
            row["oracle"]["area_relative_error"] for row in rows
        ),
        "oracle_width_relative_error": _stats(
            row["oracle"]["width_relative_error"] for row in rows
        ),
        "oracle_height_relative_error": _stats(
            row["oracle"]["height_relative_error"] for row in rows
        ),
        "oracle_max_dimension_relative_error": _stats(
            row["oracle"]["max_dimension_relative_error"] for row in rows
        ),
        "oracle_pin_perimeter_side_recovery": _stats(
            row["oracle"]["pin_perimeter_side_recovery"] for row in rows
        ),
        "oracle_pin_side_coverage": _stats(
            row["oracle"]["pin_side_coverage"] for row in rows
        ),
        "oracle_side_coverage": _stats(
            row["oracle"]["side_coverage"] for row in rows
        ),
        "oracle_gold_outline_side_recovery": _stats(
            row["oracle"]["gold_outline_side_recovery"] for row in rows
        ),
        "oracle_gold_outside_block_ratio": _stats(
            row["oracle"]["gold_outside_block_ratio"] for row in rows
        ),
        "top1_area_relative_error": _stats(
            row["top1"]["area_relative_error"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_width_relative_error": _stats(
            row["top1"]["width_relative_error"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_height_relative_error": _stats(
            row["top1"]["height_relative_error"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_pin_perimeter_side_recovery": _stats(
            row["top1"]["pin_perimeter_side_recovery"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_pin_side_coverage": _stats(
            row["top1"]["pin_side_coverage"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_side_coverage": _stats(
            row["top1"]["side_coverage"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_gold_outline_side_recovery": _stats(
            row["top1"]["gold_outline_side_recovery"]
            for row in rows
            if row["top1"] is not None
        ),
        "top1_gold_outside_block_ratio": _stats(
            row["top1"]["gold_outside_block_ratio"]
            for row in rows
            if row["top1"] is not None
        ),
    }


def _hypothesis_metrics(
    hypothesis: Any,
    gold: tuple[float, float, float, float],
    pin_perimeter: tuple[float, float, float, float] | None,
    rectangles: torch.Tensor,
    targets: torch.Tensor,
    preplaced: torch.Tensor,
) -> dict[str, Any]:
    left, bottom, right, top = (float(value) for value in hypothesis.bounds)
    gold_left, gold_bottom, gold_right, gold_top = gold
    width, height = right - left, top - bottom
    gold_width, gold_height = gold_right - gold_left, gold_top - gold_bottom
    area_error = abs(width * height - gold_width * gold_height) / max(
        gold_width * gold_height, 1.0e-12
    )
    width_error = abs(width - gold_width) / max(gold_width, 1.0e-12)
    height_error = abs(height - gold_height) / max(gold_height, 1.0e-12)
    outside = (
        (rectangles[:, 0] < left - 1.0e-8)
        | (rectangles[:, 1] < bottom - 1.0e-8)
        | (rectangles[:, 0] + rectangles[:, 2] > right + 1.0e-8)
        | (rectangles[:, 1] + rectangles[:, 3] > top + 1.0e-8)
    )
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "source": hypothesis.source,
        "area_relative_error": area_error,
        "width_relative_error": width_error,
        "height_relative_error": height_error,
        "max_dimension_relative_error": max(width_error, height_error),
        "pin_perimeter_side_recovery": _side_recovery(
            (left, bottom, right, top), pin_perimeter, relative_tolerance=1.0e-6
        ),
        "pin_side_coverage": float(hypothesis.pin_side_coverage),
        "side_coverage": float(hypothesis.side_coverage),
        "gold_outline_side_recovery": _side_recovery(
            (left, bottom, right, top), gold, relative_tolerance=0.05
        ),
        "gold_outside_block_ratio": float(outside.to(torch.float64).mean().item()),
        "contains_preplaced": hypothesis.contains_targets(targets, mask=preplaced),
    }


def _side_recovery(
    predicted: tuple[float, float, float, float],
    expected: tuple[float, float, float, float] | None,
    *,
    relative_tolerance: float,
) -> float:
    if expected is None:
        return 0.0
    span = max(expected[2] - expected[0], expected[3] - expected[1], 1.0)
    tolerance = relative_tolerance * span
    return sum(abs(first - second) <= tolerance for first, second in zip(predicted, expected)) / 4.0


def _bbox(rectangles: torch.Tensor) -> tuple[float, float, float, float]:
    return (
        float(rectangles[:, 0].min().item()),
        float(rectangles[:, 1].min().item()),
        float((rectangles[:, 0] + rectangles[:, 2]).max().item()),
        float((rectangles[:, 1] + rectangles[:, 3]).max().item()),
    )


def _point_bbox(points: torch.Tensor) -> tuple[float, float, float, float] | None:
    if not points.numel():
        return None
    return (
        float(points[:, 0].min().item()),
        float(points[:, 1].min().item()),
        float(points[:, 0].max().item()),
        float(points[:, 1].max().item()),
    )


def _minimum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return min(values) if values else None


def _maximum(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def _stats(values: Iterable[float | None]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not ordered:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _finite_or_inf(value: Any) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else math.inf


def _block_bucket(block_count: int) -> str:
    return "106-115" if block_count <= 115 else "116-120"


def _pin_bucket(pin_count: int) -> str:
    if pin_count == 0:
        return "0"
    if pin_count <= 8:
        return "1-8"
    if pin_count <= 32:
        return "9-32"
    return "33+"


def _density_bucket(density: float) -> str:
    if density == 0.0:
        return "0"
    if density <= 0.05:
        return "(0,0.05]"
    if density <= 0.20:
        return "(0.05,0.20]"
    return "(0.20,1]"


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout
    audited_sources = [
        ROOT / "src" / "hcfp" / "outline_inference.py",
        ROOT / "scripts" / "audit_outline_recovery.py",
    ]
    return {
        "git_commit": commit,
        "git_clean": not status.strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in audited_sources
        },
    }


def _layout_files(root: Path) -> list[Path]:
    text = str(root.resolve()).lower()
    if any(token in text for token in ("litetensordatatest", "validation", "visible")):
        raise ValueError("validation/test paths are forbidden for outline recovery audit")
    layout_root = root if root.name == "floorset_lite" else root / "floorset_lite"
    files = sorted(layout_root.glob("worker_*/layouts*"))
    if not files:
        raise FileNotFoundError(f"no FloorSet-Lite training layouts under {layout_root}")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--min-blocks", type=int, default=106)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--beam", type=int, default=8)
    args = parser.parse_args(argv)
    if args.limit <= 0:
        raise ValueError("limit must be positive")

    files = _layout_files(Path(args.floorset_lite_root))
    records: list[dict[str, Any]] = []
    examined_files = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        examined_files += 1
        for index in range(len(payload[0])):
            block_count = int((payload[0][index, :, 0] != -1).sum().item())
            if not args.min_blocks <= block_count <= args.max_blocks:
                continue
            records.append(
                audit_layout(
                    f"{path.parent.name}/{path.name}:{index}",
                    payload[0][index],
                    payload[3][index],
                    payload[5][index],
                    beam=args.beam,
                )
            )
            if len(records) >= args.limit:
                break
        if len(records) >= args.limit:
            break

    report = build_report(
        records,
        provenance={
            **_git_provenance(),
            "data_root": str(Path(args.floorset_lite_root).resolve()),
            "source_files_available": len(files),
            "source_files_examined": examined_files,
            "requested_cases": args.limit,
            "min_blocks": args.min_blocks,
            "max_blocks": args.max_blocks,
            "beam": args.beam,
            "official_hard_target_source": (
                "fixed dimensions and preplaced rectangles reconstructed from training fp_sol; "
                "these fields are exact official runtime inputs"
            ),
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report["gates"], sort_keys=True))
    return 0 if report["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
