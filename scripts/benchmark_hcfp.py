#!/usr/bin/env python3
"""Run or combine official HCFP lanes into an exact comparison report."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.benchmark import build_report  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.reference import OFFICIAL_FLOORSET_V10  # noqa: E402
from hcfp.visualize import render_html  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--optimizer", action="append", metavar="NAME=PATH")
    source.add_argument("--result", action="append", metavar="NAME=JSON")
    parser.add_argument("--baseline", default="fallback")
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--cases", default="all", help="all or comma-separated test ids")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="LANE=PATH",
        help="hash-verified checkpoint for a learned optimizer lane",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--visualize-dir")
    parser.add_argument("--visualize-cases", default="")
    args = parser.parse_args(argv)

    specs = _assignments(args.optimizer or args.result or [])
    checkpoints = _assignments(args.checkpoint or [])
    if checkpoints and not args.optimizer:
        raise ValueError("--checkpoint requires --optimizer mode")
    unknown_checkpoint_lanes = checkpoints.keys() - specs.keys()
    if unknown_checkpoint_lanes:
        raise ValueError(f"checkpoint lanes are missing optimizers: {sorted(unknown_checkpoint_lanes)}")
    if args.optimizer:
        lanes, case_metadata, lane_metadata = _run_optimizers(
            specs,
            Path(args.data_path),
            _case_ids(args.cases),
            args.device,
            checkpoints,
        )
        mode = "optimizer"
    else:
        lanes = {name: _load_rows(path) for name, path in specs.items()}
        case_metadata = {}
        lane_metadata = {}
        mode = "result"

    report = build_report(
        lanes,
        baseline=args.baseline,
        provenance=_provenance(Path(args.data_path), args.device, mode),
        case_metadata=case_metadata,
        lane_metadata=lane_metadata,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.visualize_dir:
        _visualize(report, Path(args.visualize_dir), _case_ids(args.visualize_cases))
    print(output)
    for name, summary in report["lane_summary"].items():
        decision = report["promotion_decisions"].get(name, "BASELINE")
        print(
            f"{name}: feasible={summary['feasible']}/{summary['cases']} "
            f"weighted_cost={summary['weighted_cost']:.6f} decision={decision}"
        )
    return 0 if all(summary["hard_feasibility_rate"] == 1.0 for summary in report["lane_summary"].values()) else 2


def _assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        result[name] = Path(path)
    return result


def _case_ids(value: str) -> list[int] | None:
    if not value or value == "all":
        return None
    return [int(item) for item in value.split(",")]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("test_results", payload.get("results"))
    if not isinstance(rows, list):
        raise ValueError(f"result file {path} does not contain test_results")
    return rows


def _run_optimizers(
    specs: dict[str, Path],
    data_path: Path,
    test_ids: list[int] | None,
    device: str,
    checkpoints: dict[str, Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    evaluator_module = _load_evaluator(data_path)
    lanes = {}
    metadata: dict[str, Any] = {}
    lane_metadata: dict[str, Any] = {}
    with _environment("HCFP_DEVICE", device):
        for name, path in specs.items():
            checkpoint = checkpoints.get(name)
            if checkpoint is not None:
                _, checkpoint_metadata = load_checkpoint(
                    checkpoint,
                    expected_normalization=RUNTIME_NORMALIZATION,
                    map_location="cpu",
                )
                lane_metadata[name] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_hash": checkpoint_metadata["state_hash"],
                    "normalization": checkpoint_metadata["normalization"],
                    "required": True,
                }
            else:
                lane_metadata[name] = {"checkpoint": None, "required": False}
            with _environment("HCFP_CHECKPOINT", str(checkpoint) if checkpoint is not None else None):
                evaluator = evaluator_module.ContestEvaluator(str(data_path), verbose=False)
                result = evaluator.evaluate(str(path), test_ids=test_ids)
            lanes[name] = [asdict(row) for row in result.test_results]
            if not metadata:
                metadata = _case_metadata(evaluator, [int(row.test_id) for row in result.test_results])
    return lanes, metadata, lane_metadata


def _case_metadata(evaluator: Any, test_ids: list[int]) -> dict[str, Any]:
    metadata = {}
    for test_id in test_ids:
        inputs = evaluator.dataset[test_id]["input"]
        area, b2b, p2b, pins, constraints = inputs
        block_count = int((area != -1).sum().item())
        metadata[str(test_id)] = {
            "block_count": block_count,
            "constraints": constraints[:block_count].tolist(),
            "pins_pos": [row for row in pins.tolist() if row != [-1.0, -1.0]],
            "b2b_connectivity": [row for row in b2b.tolist() if row[:2] != [-1.0, -1.0]],
            "p2b_connectivity": [row for row in p2b.tolist() if row[:2] != [-1.0, -1.0]],
        }
    return metadata


def _load_evaluator(data_path: Path):
    path = data_path / OFFICIAL_FLOORSET_V10.evaluator_path
    if not path.is_file():
        raise FileNotFoundError(f"pinned evaluator not found: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != OFFICIAL_FLOORSET_V10.evaluator_sha256:
        raise RuntimeError(f"evaluator SHA256 mismatch: {actual}")
    sys.path.insert(0, str(data_path))
    spec = importlib.util.spec_from_file_location("hcfp_official_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provenance(data_path: Path, device: str, mode: str) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "mode": mode,
        "device": device,
        "data_path": str(data_path),
        "evaluator_commit": OFFICIAL_FLOORSET_V10.commit,
        "evaluator_sha256": OFFICIAL_FLOORSET_V10.evaluator_sha256,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "command": " ".join(sys.argv),
    }


def _visualize(report: dict[str, Any], directory: Path, case_ids: list[int] | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    available = [int(row["test_id"]) for row in next(iter(report["lanes"].values()))]
    selected = case_ids if case_ids is not None else ([available[0], available[-1]] if available else [])
    for test_id in selected:
        items = []
        for name, rows in report["lanes"].items():
            row = next((entry for entry in rows if int(entry["test_id"]) == test_id), None)
            if row is None or row.get("positions") is None:
                continue
            items.append(
                {
                    "title": f"case {test_id} — {name} — cost {float(row['cost']):.6f}",
                    "placements": row["positions"],
                    "case": report.get("case_metadata", {}).get(str(test_id)),
                    "telemetry": {
                        key: row[key]
                        for key in ("cost", "hpwl_gap", "area_gap", "violations_relative")
                    },
                }
            )
        if items:
            (directory / f"case_{test_id}.html").write_text(
                render_html(items, title=f"HCFP case {test_id}"), encoding="utf-8"
            )


@contextmanager
def _environment(name: str, value: str | None):
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


if __name__ == "__main__":
    raise SystemExit(main())
