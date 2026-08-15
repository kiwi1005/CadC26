#!/usr/bin/env python3
"""P8.2: pre-registered held-out evaluation of the frozen group-first loop.

Pre-registration contract (all recorded in registration.json BEFORE any
scoring):

- training root is the official floorset_lite tree; visible validation/test
  roots are rejected (guard tokens: litetensordatatest / validation /
  visible / test);
- training-root inventory: sorted relative paths + sizes + inventory sha256;
- seed, source limit, block range, bucket rule, and the frozen config;
- denylist: constraint signatures of the five design cases 70/89/90/94/97
  (from LiteTensorDataTest/config_{id+21}/litedata_1.pth); any sampled layout
  matching one of them is rejected and recorded;
- source-layout hashes: sha256 of every layout file that actually yielded a
  sampled layout (file identity recovered from sample_id).

Bucket rule (input-only, computed on the deterministic shelf incumbent):
dense utilization >= 0.90 AND residual grouping debt > 0.

Pipeline: `--scan-only` freezes the selected bucket (no QoR scoring), then a
second invocation with `--frozen <path>` runs the QoR loop on the frozen
bucket only. No teacher data is collected.

Gate (predeclared): all winners hard feasible AND (unique wins >= 5% OR
weighted uncapped debt down >= 5%). PASS authorizes a later Contact
Generator Policy task only; FAIL/MODIFY stops before learning.

Usage:
  python3 scripts/experiment_bfod_heldout.py --scan-only \
      --min-blocks 106 --max-blocks 120 --limit 6000 --jobs 24 \
      --output-dir artifacts/experiments/p8_2_groupfirst_contact_heldout
  python3 scripts/experiment_bfod_heldout.py \
      --frozen artifacts/experiments/p8_2_groupfirst_contact_heldout/frozen_bucket.json \
      --jobs 24 --output-dir artifacts/experiments/p8_2_groupfirst_contact_heldout
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "src"), str(ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from benchmark_hcfp import _load_evaluator  # noqa: E402
from experiment_bfod_v1 import (  # noqa: E402
    Config,
    Context,
    _bootstrap_contact,
    _common_loop,
    _measure,
    _metrics_brief,
    _route,
    _state,
    _state_key,
    _state_record,
)
from hcfp.fallback import safe_shelf  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402

DENSE_UTILIZATION = 0.90
FORBIDDEN_ROOT_TOKENS = ("litetensordatatest", "validation", "visible", "test")
DESIGN_CASE_IDS = (70, 89, 90, 94, 97)

_EVALUATOR: Any = None


def _init_worker(data_path: str) -> None:
    global _EVALUATOR
    _EVALUATOR = _load_evaluator(Path(data_path))


# --------------------------------------------------------------------------
# tensor-safe JSON round trip for the frozen bucket
# --------------------------------------------------------------------------


def _encode(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "__torch_tensor__": True,
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__torch_tensor__"):
        return torch.as_tensor(value["data"], dtype=getattr(torch, value["dtype"]))
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_encode(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return _decode(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# pre-registration
# --------------------------------------------------------------------------


def _constraint_signature(
    block_count: int,
    group_sizes: tuple[int, ...],
    mib_sizes: tuple[int, ...],
    boundary_blocks: int,
    boundary_side_bits: int,
) -> dict[str, Any]:
    return {
        "block_count": int(block_count),
        "group_sizes": sorted(int(size) for size in group_sizes),
        "mib_sizes": sorted(int(size) for size in mib_sizes),
        "boundary_blocks": int(boundary_blocks),
        "boundary_side_bits": int(boundary_side_bits),
    }


def _case_signature(case: Any) -> dict[str, Any]:
    groups = torch.as_tensor(case.group_membership, dtype=torch.bool)
    mibs = torch.as_tensor(case.mib_membership, dtype=torch.bool)
    boundary = torch.as_tensor(case.boundary_bits, dtype=torch.bool)
    return _constraint_signature(
        int(case.n),
        tuple(groups.sum(dim=1).tolist()),
        tuple(mibs.sum(dim=1).tolist()),
        int(boundary.any(dim=1).sum()),
        int(boundary.to(torch.long).reshape(-1).sum()),
    )


def _official_design_signatures(data_path: Path) -> dict[str, dict[str, Any]]:
    """Constraint signatures of cases 70/89/90/94/97 from their inputs only."""

    signatures: dict[str, dict[str, Any]] = {}
    for case_id in DESIGN_CASE_IDS:
        input_path = (
            data_path / "LiteTensorDataTest" / f"config_{case_id + 21}" / "litedata_1.pth"
        )
        if not input_path.is_file():
            raise FileNotFoundError(f"design-case input tensor missing: {input_path}")
        payload = torch.load(input_path, map_location="cpu", weights_only=True)
        rows = torch.as_tensor(payload[0][0], dtype=torch.long)
        valid = rows[:, 0] != -1
        constraints = rows[valid]
        signatures[str(case_id)] = _constraint_signature(
            int(valid.sum()),
            tuple(torch.bincount(constraints[:, 4][constraints[:, 4] != 0])[1:].tolist()),
            tuple(torch.bincount(constraints[:, 3][constraints[:, 3] != 0])[1:].tolist()),
            int((constraints[:, 5] != 0).sum()),
            sum(int(value).bit_count() for value in constraints[:, 5].tolist()),
        )
    return signatures


def _inventory_record(layout_root: Path) -> dict[str, Any]:
    files = sorted(layout_root.glob("worker_*/layouts*"))
    rows = [
        {"path": str(path.relative_to(layout_root)), "size": path.stat().st_size}
        for path in files
    ]
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "root": str(layout_root),
        "layout_file_count": len(files),
        "inventory_sha256": digest,
    }


def _hash_files(paths: list[Path]) -> list[dict[str, str]]:
    output = []
    for path in sorted(set(paths)):
        output.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return output


def _design_denylist(data_path: Path) -> dict[str, dict[str, Any]]:
    return _official_design_signatures(data_path)


# --------------------------------------------------------------------------
# scan phase (input-only filtering; no QoR scoring)
# --------------------------------------------------------------------------


def _raw_case_from_sample(sample: Any, source: dict[str, Any]) -> dict[str, Any]:
    case = sample.case
    return {
        "normalized": False,
        "n": int(case.n),
        "area": source["area_targets"],
        "constraints": source["constraints"],
        "target": source["target_positions"],
        "fixed_mask": case.fixed_mask,
        "preplaced_mask": case.preplaced_mask,
        "raw_preplaced_validated": True,
        "boundary_bits": case.boundary_bits,
        "group_membership": case.group_membership,
        "mib_membership": case.mib_membership,
        "b2b_weight": case.b2b_weight,
        "b2b_connectivity": source["b2b_connectivity"],
        "p2b_connectivity": source["p2b_connectivity"],
        "pins_pos": source["pins_pos"],
        "cluster_group_ids": case.cluster_group_ids,
        "mib_group_ids": case.mib_group_ids,
    }


def _metric_args(sample: Any, source: dict[str, Any]) -> tuple[Any, ...]:
    return (
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


def _scan_one(args: tuple[Any, dict[str, Any], dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
    sample, source, denylist = args
    evaluator = _EVALUATOR
    if sample.labels.baseline_area is None or sample.labels.baseline_hpwl is None:
        return None
    if _case_signature(sample.case) in denylist.values():
        return {"sample_id": sample.sample_id, "denied_design_signature": True}
    raw_case = _raw_case_from_sample(sample, source)
    metric_args = _metric_args(sample, source)
    try:
        # safe_shelf must run on the raw official-coordinate source dict;
        # the normalized FloorplanCase yields normalized boxes that the raw
        # evaluator/verifier would reject.
        positions = safe_shelf(source)
    except (RuntimeError, ValueError):
        return None
    metrics = _measure(evaluator, raw_case, metric_args, positions)
    if not metrics["hard_feasible"]:
        return None
    route = _route(raw_case, positions, metrics)
    if route["utilization"] < DENSE_UTILIZATION or metrics["grouping_violations"] <= 0:
        return None
    return {
        "sample_id": sample.sample_id,
        "block_count": int(sample.case.n),
        "utilization": route["utilization"],
        "soft_debt": int(metrics["total_soft_violations"]),
        "baseline_sha256": _placement_sha256(positions),
        "raw_case": raw_case,
        "metric_args": metric_args,
        "positions": positions.tolist(),
        "baseline": _metrics_brief(metrics),
    }


def _placement_sha256(positions: Any) -> str:
    boxes = torch.as_tensor(positions, dtype=torch.float64, device="cpu")
    payload = json.dumps(
        boxes.tolist(), separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------
# QoR phase (frozen bucket only)
# --------------------------------------------------------------------------


def _loop_one(args: tuple[dict[str, Any], Config]) -> dict[str, Any]:
    row, config = args
    evaluator = _EVALUATOR
    raw_case = row["raw_case"]
    metric_args = row["metric_args"]
    positions = torch.as_tensor(row["positions"], dtype=torch.float64)
    from hcfp.case import from_official

    case = from_official(
        raw_case["n"],
        raw_case["area"],
        raw_case["b2b_connectivity"],
        raw_case["p2b_connectivity"],
        raw_case["pins_pos"],
        raw_case["constraints"],
        raw_case["target"],
    )
    context = Context(
        case_id=0,
        evaluator_module=evaluator,
        case=case,
        raw_case=raw_case,
        raw_object=SimpleNamespace(**raw_case),
        metric_args=metric_args,
        visual_case={},
        b2b_edges=[],
        contact_policy=None,
        contact_policy_metadata=None,
        group_first=True,
    )
    started = time.perf_counter()
    full_baseline = _measure(evaluator, raw_case, metric_args, positions)
    if not full_baseline["hard_feasible"]:
        raise RuntimeError(f"frozen baseline not hard feasible: {row['sample_id']}")
    base_state = _state(positions, full_baseline, history=())
    route = {
        "name": "dense_common_loop",
        "utilization": row["utilization"],
        "soft_debt": row["soft_debt"],
        "group_first": True,
    }
    contact_state, _contact_report = _bootstrap_contact(
        context, base_state, route, config
    )
    loop_state, loop_report = _common_loop(
        context,
        contact_state,
        route,
        config,
        runtime_ceiling=config.runtime_ceiling,
    )
    winner = min((base_state, contact_state, loop_state), key=_state_key)
    elapsed = time.perf_counter() - started
    history = winner["history"]
    patch_sizes = [
        len(entry.get("details", {}).get("members", ()))
        for entry in history
        if entry["family"] in {"contact", "joint"}
    ]
    return {
        "sample_id": row["sample_id"],
        "block_count": row["block_count"],
        "utilization": row["utilization"],
        "baseline": _metrics_brief(full_baseline),
        "baseline_sha256": row["baseline_sha256"],
        "winner": _state_record(winner),
        "winner_cost": winner["metrics"]["uncapped_cost"],
        "baseline_cost": full_baseline["uncapped_cost"],
        "delta_grouping": (
            winner["metrics"]["grouping_violations"]
            - full_baseline["grouping_violations"]
        ),
        "delta_hpwl_total": (
            winner["metrics"]["hpwl_total"] - full_baseline["hpwl_total"]
        ),
        "delta_bbox_area": winner["metrics"]["bbox_area"] - full_baseline["bbox_area"],
        "hard_feasible": bool(winner["metrics"]["hard_feasible"]),
        "decodes": int(loop_report["exact_decodes"]),
        "rounds": len(loop_report["rounds"]),
        "accepted_steps": len(history),
        "families": [entry["family"] for entry in history],
        "patch_sizes": patch_sizes,
        "runtime_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# summary + gate
# --------------------------------------------------------------------------


def _summarize(
    rows: list[dict[str, Any]], examined: int, scan_seconds: float, loop_seconds: float
) -> dict[str, Any]:
    feasible = [row for row in rows if row["hard_feasible"]]
    all_feasible = len(feasible) == len(rows) and bool(rows)
    wins = [row for row in rows if row["winner_cost"] < row["baseline_cost"] - 1.0e-10]
    ties = [
        row
        for row in rows
        if abs(row["winner_cost"] - row["baseline_cost"]) <= 1.0e-10
    ]
    losses = [row for row in rows if row["winner_cost"] > row["baseline_cost"] + 1.0e-10]
    baseline_debt = sum(row["baseline_cost"] for row in rows)
    winner_debt = sum(row["winner_cost"] for row in rows)
    grouping_before = sum(row["baseline"]["grouping_violations"] for row in rows)
    grouping_after = sum(row["winner"]["metrics"]["grouping_violations"] for row in rows)
    hpwl_before = sum(row["baseline"]["hpwl_gap"] for row in rows)
    hpwl_after = sum(row["winner"]["metrics"]["hpwl_gap"] for row in rows)
    bbox_before = sum(row["baseline"]["bbox_area"] for row in rows)
    bbox_after = sum(row["winner"]["metrics"]["bbox_area"] for row in rows)
    patch_counter: Counter = Counter()
    family_counter: Counter = Counter()
    for row in rows:
        patch_counter.update(row["patch_sizes"])
        family_counter.update(row["families"])
    win_rate = len(wins) / len(rows) if rows else 0.0
    debt_change = (
        (baseline_debt - winner_debt) / baseline_debt if baseline_debt else 0.0
    )
    decodes = sorted(row["decodes"] for row in rows)
    runtimes = sorted(row["runtime_seconds"] for row in rows)
    gate_pass = all_feasible and (win_rate >= 0.05 or debt_change >= 0.05)
    return {
        "method": "P8.2 held-out dense bucket (frozen group-first v2 loop)",
        "scope": "training-split floorset_lite layouts only; visible/test roots rejected",
        "bucket": {
            "examined": examined,
            "qualifying": len(rows),
            "hard_feasible_winners": len(feasible),
            "hard_infeasible_winners": len(rows) - len(feasible),
        },
        "results": {
            "unique_wins": len(wins),
            "ties": len(ties),
            "losses": len(losses),
            "win_rate": win_rate,
            "weighted_uncapped_debt": {
                "before": baseline_debt,
                "after": winner_debt,
                "relative_change": debt_change,
            },
            "grouping": {"before": grouping_before, "after": grouping_after},
            "hpwl_gap_sum": {"before": hpwl_before, "after": hpwl_after},
            "bbox_area_sum": {"before": bbox_before, "after": bbox_after},
            "mean_delta_grouping": (
                sum(row["delta_grouping"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_delta_hpwl_total": (
                sum(row["delta_hpwl_total"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_delta_bbox_area": (
                sum(row["delta_bbox_area"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "total_accepted_steps": sum(row["accepted_steps"] for row in rows),
            "total_decodes": sum(row["decodes"] for row in rows),
            "decodes_distribution": {
                "min": decodes[0] if decodes else None,
                "median": decodes[len(decodes) // 2] if decodes else None,
                "max": decodes[-1] if decodes else None,
            },
            "runtime_seconds_distribution": {
                "min": round(runtimes[0], 3) if runtimes else None,
                "median": round(runtimes[len(runtimes) // 2], 3) if runtimes else None,
                "max": round(runtimes[-1], 3) if runtimes else None,
            },
            "patch_size_distribution": dict(sorted(patch_counter.items())),
            "family_distribution": dict(sorted(family_counter.items())),
        },
        "gate": {
            "rule": (
                "all winners hard feasible AND "
                "(unique wins >= 5% OR weighted uncapped debt down >= 5%)"
            ),
            "all_winners_hard_feasible": all_feasible,
            "win_rate": win_rate,
            "debt_change": debt_change,
            "pass": gate_pass,
        },
        "runtime_seconds": {"scan": scan_seconds, "loop": loop_seconds},
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    r = summary["results"]
    g = summary["gate"]
    lines = [
        "# P8.2 held-out dense bucket (frozen group-first loop)",
        "",
        f"- examined: {summary['bucket']['examined']}",
        f"- qualifying: {summary['bucket']['qualifying']}",
        f"- hard-feasible winners: {summary['bucket']['hard_feasible_winners']}",
        f"- unique wins: {r['unique_wins']} / {summary['bucket']['qualifying']} "
        f"({g['win_rate'] * 100:.1f}%)",
        f"- ties: {r['ties']}, losses: {r['losses']}",
        f"- weighted uncapped debt: {r['weighted_uncapped_debt']['before']:.4f} -> "
        f"{r['weighted_uncapped_debt']['after']:.4f} "
        f"({g['debt_change'] * 100:+.2f}%)",
        f"- grouping debt: {r['grouping']['before']} -> {r['grouping']['after']} "
        f"({r['grouping']['after'] - r['grouping']['before']:+d})",
        f"- mean Δ grouping per case: {r['mean_delta_grouping']:+.3f}",
        f"- mean Δ hpwl_total: {r['mean_delta_hpwl_total']:+.2e}",
        f"- mean Δ bbox_area: {r['mean_delta_bbox_area']:+.2e}",
        f"- accepted steps: {r['total_accepted_steps']}, "
        f"decodes: {r['total_decodes']} "
        f"(min/med/max {r['decodes_distribution']['min']}/"
        f"{r['decodes_distribution']['median']}/{r['decodes_distribution']['max']})",
        f"- runtime s: min/med/max "
        f"{r['runtime_seconds_distribution']['min']}/"
        f"{r['runtime_seconds_distribution']['median']}/"
        f"{r['runtime_seconds_distribution']['max']}",
        f"- patch sizes: {r['patch_size_distribution']}",
        f"- families: {r['family_distribution']}",
        f"- gate: {'PASS' if g['pass'] else 'FAIL'} "
        f"(all feasible={g['all_winners_hard_feasible']}, "
        f"wins {g['win_rate'] * 100:.1f}% >= 5% or debt {g['debt_change'] * 100:+.2f}% >= 5%)",
        f"- runtime: scan {summary['runtime_seconds']['scan']:.1f}s, "
        f"loop {summary['runtime_seconds']['loop']:.1f}s",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--min-blocks", type=int, default=106)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--layouts-per-file", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--runtime-ceiling", type=float, default=30.0)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument(
        "--frozen",
        help="path to a frozen bucket JSON produced by --scan-only",
    )
    parser.add_argument("--output-dir", default="artifacts/experiments/p8_2_groupfirst_contact_heldout")
    args = parser.parse_args(argv)

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.frozen:
        return _run_qor_phase(args, data_path, output_dir)
    return _run_scan_phase(args, data_path, output_dir)


def _run_scan_phase(
    args: argparse.Namespace, data_path: Path, output_dir: Path
) -> int:
    layout_root = (
        data_path if data_path.name == "floorset_lite" else data_path / "floorset_lite"
    ).resolve()
    forbidden = [
        token
        for token in FORBIDDEN_ROOT_TOKENS
        if token in str(layout_root).lower()
    ]
    if forbidden:
        raise ValueError(
            f"visible validation/test tokens in training root {layout_root}: {forbidden}"
        )
    denylist = _design_denylist(data_path)
    inventory = _inventory_record(layout_root)
    registration = {
        "phase": "scan",
        "command": " ".join(sys.argv),
        "training_root": inventory,
        "seed": args.seed,
        "source_limit": args.limit,
        "layouts_per_file": args.layouts_per_file,
        "block_range": [args.min_blocks, args.max_blocks],
        "bucket_rule": {
            "dense_utilization_min": DENSE_UTILIZATION,
            "grouping_debt_min": 1,
            "incumbent": "deterministic safe_shelf",
        },
        "design_case_denylist": denylist,
        "denylist_guard": (
            "any sampled layout whose constraint signature matches a design "
            "case (70/89/90/94/97) is rejected and recorded"
        ),
        "frozen_config": {
            "contact_only": True,
            "group_first": True,
            "beam_width": 4,
            "max_rounds": 6,
            "top_experts": 2,
            "proposals_per_operator": 4,
            "exact_decode_cap": 96,
            "patch_sizes": [4, 8, 12, 16],
            "runtime_ceiling": args.runtime_ceiling,
        },
    }
    _dump(output_dir / "registration.json", registration)

    started = time.perf_counter()
    pool = Pool(args.jobs, initializer=_init_worker, initargs=(str(data_path),))
    batch: list[tuple[Any, dict[str, Any], dict[str, dict[str, Any]]]] = []
    results: list[dict[str, Any] | None] = []
    touched_files: set[Path] = set()
    collected = 0
    denied = 0
    examined = 0
    for sample, source in iter_floorset_lite_with_source(
        data_path,
        limit=args.limit,
        seed=args.seed,
        max_layouts_per_file=args.layouts_per_file,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
    ):
        examined += 1
        touched_files.add(
            layout_root / Path(sample.sample_id.split(":", 1)[0])
        )
        batch.append((sample, source, denylist))
        if len(batch) >= args.jobs * 4:
            for row in pool.map(_scan_one, batch):
                if row is None:
                    continue
                if row.get("denied_design_signature"):
                    denied += 1
                    continue
                results.append(row)
                collected += 1
            batch = []
            if collected >= args.max_cases:
                break
    if batch:
        for row in pool.map(_scan_one, batch):
            if row is None:
                continue
            if row.get("denied_design_signature"):
                denied += 1
                continue
            results.append(row)
            collected += 1
    pool.close()
    pool.join()
    scan_seconds = time.perf_counter() - started

    source_hashes = _hash_files(list(touched_files))
    frozen = {
        "phase": "frozen",
        "command": " ".join(sys.argv),
        "registration": registration,
        "examined": examined,
        "denied_design_signatures": denied,
        "qualifying": len(results),
        "source_layouts": source_hashes,
        "bucket": results,
    }
    frozen_path = output_dir / "frozen_bucket.json"
    _dump(frozen_path, frozen)
    print(
        f"scan: examined={examined} denied_design={denied} "
        f"qualifying={len(results)} ({scan_seconds:.1f}s) "
        f"frozen -> {frozen_path}"
    )
    return 0


def _run_qor_phase(
    args: argparse.Namespace, data_path: Path, output_dir: Path
) -> int:
    frozen_path = Path(args.frozen)
    frozen = _load_json(frozen_path)
    if frozen.get("phase") != "frozen":
        raise ValueError(f"{frozen_path} is not a frozen bucket artifact")
    rows = frozen["bucket"]
    if not rows:
        raise RuntimeError("frozen bucket is empty; no QoR phase to run")
    config = Config(
        beam_width=4,
        max_rounds=6,
        top_experts=2,
        proposals_per_operator=4,
        exact_decode_cap=96,
        runtime_ceiling=args.runtime_ceiling,
        patch_sizes=(4, 8, 12, 16),
        contact_only=True,
        group_first=True,
    )
    started = time.perf_counter()
    pool = Pool(args.jobs, initializer=_init_worker, initargs=(str(data_path),))
    qor_rows: list[dict[str, Any]] = []
    for index in range(0, len(rows), args.jobs * 4):
        chunk = rows[index : index + args.jobs * 4]
        qor_rows.extend(pool.map(_loop_one, [(row, config) for row in chunk]))
    pool.close()
    pool.join()
    loop_seconds = time.perf_counter() - started

    _dump(output_dir / "cases.json", qor_rows)
    summary = _summarize(
        qor_rows, int(frozen["examined"]), 0.0, loop_seconds
    )
    summary["registration"] = frozen["registration"]
    summary["source_layouts"] = frozen["source_layouts"]
    summary["denied_design_signatures"] = int(frozen["denied_design_signatures"])
    _dump(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
