"""Command-line smoke surfaces for the greenfield HCFP solver."""

from __future__ import annotations

import argparse
import json

from hcfp.analytic import AnalyticConfig, select_device, solve
from hcfp.case import from_official
from hcfp.dynamics import DynamicsConfig
from hcfp.geometry import normalize_xywh
from hcfp.projection import ComponentBDPConfig
from hcfp.runtime import SolveCase
from hcfp.verify import soft_violation_normalized, verify


def _demo(args: argparse.Namespace) -> dict[str, object]:
    runtime_case = SolveCase(
        block_count=6,
        area_targets=[4.0, 9.0, 6.0, 4.0, 8.0, 5.0],
        b2b_connectivity=[[0, 1, 4.0], [1, 2, 2.0], [2, 3, 3.0], [3, 4, 1.0], [4, 5, 2.0]],
        p2b_connectivity=[[0, 0, 1.0]],
        pins_pos=[[0.0, 0.0]],
        constraints=[
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
        ],
        target_positions=[
            [0.0, 0.0, 2.0, 2.0],
            [-1.0, -1.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
    )
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=args.candidates, steps=args.steps),
        projection_iterations=args.projection_steps,
        direction_beam=args.beam,
        component_bdp=ComponentBDPConfig(enabled=args.component_bdp),
    )
    placements = solve(runtime_case, config, device=args.device)
    case = from_official(
        runtime_case.block_count,
        runtime_case.area_targets,
        runtime_case.b2b_connectivity,
        runtime_case.p2b_connectivity,
        runtime_case.pins_pos,
        runtime_case.constraints,
        runtime_case.target_positions,
    )
    normalized = normalize_xywh(case, placements)
    exact = verify(case, normalized)
    soft = soft_violation_normalized(case, normalized)
    return {
        "device": str(select_device(args.device)),
        "block_count": runtime_case.block_count,
        "feasible": exact.feasible,
        "overlap_pairs": len(exact.overlap_pairs),
        "soft_violation_relative": soft.total,
        "placements": placements,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcfp")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the bounded analytic HCFP lane")
    demo.add_argument("--device", default="auto")
    demo.add_argument("--candidates", type=int, default=8)
    demo.add_argument("--steps", type=int, default=12)
    demo.add_argument("--projection-steps", type=int, default=24)
    demo.add_argument("--beam", type=int, default=4)
    demo.add_argument("--component-bdp", action="store_true")
    demo.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _demo(args)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0 if payload["feasible"] else 1
