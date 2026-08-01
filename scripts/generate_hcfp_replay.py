#!/usr/bin/env python3
"""Generate repair-aware ranker replay from the exact HCFP tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import AnalyticConfig, select_device  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite  # noqa: E402
from hcfp.learned import LearnedConfig, analyze_case_with_checkpoint  # noqa: E402
from hcfp.replay import record_from_analysis, write_replay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=4)
    parser.add_argument("--projection-steps", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    analytic = AnalyticConfig(
        dynamics=DynamicsConfig(population=args.population, steps=args.dynamics_steps),
        projection_iterations=args.projection_steps,
        direction_beam=2,
    )
    config = LearnedConfig(analytic=analytic, flow_steps=args.flow_steps)

    def records():
        for sample in iter_floorset_lite(args.floorset_lite_root, limit=args.limit):
            device_sample = sample.case.to(device=device, dtype=None)
            analysis = analyze_case_with_checkpoint(device_sample, args.checkpoint, config)
            if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash is None:
                raise RuntimeError(analysis.result.failure_reason or "checkpoint was not used")
            yield record_from_analysis(
                sample,
                analysis.result.checkpoint_hash,
                analysis.analytic.raw_candidates,
                analysis.analytic.telemetry,
                population=args.population,
            )

    count = write_replay(records(), args.output)
    report = {
        "schema_version": 1,
        "records": count,
        "checkpoint": args.checkpoint,
        "population": args.population,
        "flow_steps": args.flow_steps,
        "device": str(device),
        "output": args.output,
    }
    Path(f"{args.output}.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
