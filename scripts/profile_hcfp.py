#!/usr/bin/env python3
"""Profile HCFP synthetic candidate runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.profile import ProfileConfig, run_profile, write_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=120, choices=(32, 64, 96, 120))
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--projection-steps", type=int, default=8)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--component-bdp", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = run_profile(
        ProfileConfig(
            block_count=args.blocks,
            candidates=args.candidates,
            steps=args.steps,
            repeats=args.repeats,
            warmups=args.warmups,
            projection_iterations=args.projection_steps,
            direction_beam=args.beam,
            component_bdp=args.component_bdp,
            device=args.device,
        )
    )
    write_profile(report, args.output)
    return 0 if report["incumbent"]["feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
