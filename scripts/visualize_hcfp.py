#!/usr/bin/env python3
"""Render HCFP placement JSON as SVG or a self-contained HTML page."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.visualize import load_visualization_json, render_html, render_svg  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON file with placements, optional case, telemetry, or candidates")
    parser.add_argument("-o", "--output", required=True, help="output .svg or .html path")
    parser.add_argument("--html", action="store_true", help="force HTML output even for one placement")
    parser.add_argument("--title", default="HCFP floorplan")
    parser.add_argument("--lane", help="select a lane from benchmark JSON")
    parser.add_argument("--case-id", type=int, help="select a test id from benchmark JSON")
    args = parser.parse_args(argv)

    entries = load_visualization_json(args.input, lane=args.lane, test_id=args.case_id)
    output = Path(args.output)
    if args.html or output.suffix.lower() in {".html", ".htm"} or len(entries) > 1:
        text = render_html(entries, title=args.title)
    else:
        entry = entries[0]
        text = render_svg(
            entry["placements"],
            case=entry.get("case"),
            telemetry=entry.get("telemetry"),
            title=str(entry.get("title") or args.title),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
