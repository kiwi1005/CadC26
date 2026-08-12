#!/usr/bin/env python3
"""Evaluate the case-level baseline head on pinned visible cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.case import from_official  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.reference import OFFICIAL_FLOORSET_V10  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    data_path = Path(args.data_path)
    evaluator = _evaluator(data_path).ContestEvaluator(str(data_path), verbose=False)
    evaluator._load_dataset()
    ids = (
        list(range(len(evaluator.dataset)))
        if args.cases == "all"
        else [int(value) for value in args.cases.split(",") if value.strip()]
    )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, metadata = load_checkpoint(
        args.checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    if not model.config.baseline_enabled or "baseline" not in metadata.get(
        "trained_heads", ()
    ):
        raise ValueError("checkpoint does not declare a trained baseline head")
    model = model.to(device).eval()

    rows = []
    with torch.inference_mode():
        for test_id in ids:
            sample = evaluator.dataset[test_id]
            (area, b2b, p2b, pins, constraints), labels = (
                sample["input"],
                sample["label"],
            )
            n = int((area != -1).sum().item())
            baseline, target = evaluator._extract_baseline(
                test_id, labels, b2b, p2b, pins, n
            )
            targets = torch.full((n, 4), -1.0)
            for index in range(n):
                if constraints[index, 1] != 0:
                    targets[index] = torch.tensor(target[index])
                elif constraints[index, 0] != 0:
                    targets[index, 2:] = torch.tensor(target[index][2:])
            case = from_official(
                n, area, b2b, p2b, pins, constraints, targets, device=device
            )
            output = model(case, population=1)
            pred_area = math.exp(float(output.baseline_log_area)) * case.scale**2
            pred_hpwl = math.exp(float(output.baseline_log_hpwl)) * case.scale
            true_area = float(baseline["area_baseline"])
            true_hpwl = float(baseline["hpwl_baseline"])
            rows.append(
                {
                    "test_id": test_id,
                    "block_count": n,
                    "area_true": true_area,
                    "area_predicted": pred_area,
                    "area_relative_error": pred_area / true_area - 1.0,
                    "hpwl_true": true_hpwl,
                    "hpwl_predicted": pred_hpwl,
                    "hpwl_relative_error": pred_hpwl / true_hpwl - 1.0,
                }
            )

    area_abs = [abs(row["area_relative_error"]) for row in rows]
    hpwl_abs = [abs(row["hpwl_relative_error"]) for row in rows]
    joint_p95 = max(_quantile(area_abs, 0.95), _quantile(hpwl_abs, 0.95))
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_hash": metadata["state_hash"],
        "cases": len(rows),
        "area": _summary(area_abs),
        "hpwl": _summary(hpwl_abs),
        "joint_absolute_relative_error_p95": joint_p95,
        "router_eligible": joint_p95 < 1.0,
        "suggested_joint_margin_p95": joint_p95 if joint_p95 < 1.0 else None,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(output_path)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("cases", "area", "hpwl", "suggested_joint_margin_p95")
            },
            indent=2,
        )
    )
    return 0


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_absolute_relative_error": statistics.median(values),
        "p90_absolute_relative_error": _quantile(values, 0.90),
        "p95_absolute_relative_error": _quantile(values, 0.95),
        "maximum_absolute_relative_error": max(values),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _evaluator(data_path: Path):
    path = data_path / OFFICIAL_FLOORSET_V10.evaluator_path
    spec = importlib.util.spec_from_file_location("hcfp_baseline_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
