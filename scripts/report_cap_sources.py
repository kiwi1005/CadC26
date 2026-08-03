#!/usr/bin/env python3
"""Report exact official-v10 cap attribution from HCFP JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.cap_margin import build_cap_report, render_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="oracle, benchmark, or minimal case JSON"
    )
    parser.add_argument("--output", required=True, help="destination JSON report")
    parser.add_argument(
        "--markdown",
        help="destination Markdown summary (default: output path with .md suffix)",
    )
    parser.add_argument(
        "--runtime-factor",
        type=float,
        default=1.0,
        help="local attribution default when a row has no runtime_factor",
    )
    parser.add_argument(
        "--lane",
        help="select one lane from a benchmark report instead of reporting every lane",
    )
    args = parser.parse_args(argv)

    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if args.lane:
        lanes = payload.get("lanes") if isinstance(payload, dict) else None
        if not isinstance(lanes, dict) or args.lane not in lanes:
            raise ValueError(f"benchmark input has no lane {args.lane!r}")
        payload = {"lanes": {args.lane: lanes[args.lane]}}
    report = build_cap_report(payload, default_runtime_factor=args.runtime_factor)
    output = Path(args.output)
    markdown = Path(args.markdown) if args.markdown else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(output)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
