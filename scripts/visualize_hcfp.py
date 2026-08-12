#!/usr/bin/env python3
"""Render HCFP placement JSON as SVG or a self-contained HTML page."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


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
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="include diagnostic-only overlays such as gold_outline",
    )
    args = parser.parse_args(argv)

    entries = load_visualization_json(args.input, lane=args.lane, test_id=args.case_id)
    output = Path(args.output)
    suffix = output.suffix.lower()
    if suffix == ".png":
        if args.html or len(entries) != 1:
            raise ValueError("PNG output supports exactly one placement entry")
        entry = entries[0]
        text = render_svg(
            entry["placements"],
            case=entry.get("case"),
            telemetry=entry.get("telemetry"),
            title=str(entry.get("title") or args.title),
            diagnostic=args.diagnostic or bool(entry.get("diagnostic", False)),
            **_render_options(entry),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_png(text, output)
        print(output)
        return 0
    if args.html or suffix in {".html", ".htm"} or len(entries) > 1:
        text = render_html(entries, title=args.title, diagnostic=args.diagnostic)
    else:
        entry = entries[0]
        text = render_svg(
            entry["placements"],
            case=entry.get("case"),
            telemetry=entry.get("telemetry"),
            title=str(entry.get("title") or args.title),
            diagnostic=args.diagnostic or bool(entry.get("diagnostic", False)),
            **_render_options(entry),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(output)
    return 0


def _render_options(entry: dict[str, object]) -> dict[str, object]:
    names = (
        "official_candidate_bbox",
        "inferred_latent_outline",
        "model_temporary_outline",
        "pin_perimeter_hypothesis",
        "gold_outline",
        "whitespace",
        "summary_metrics",
        "overlays",
    )
    options = {name: entry[name] for name in names if name in entry}
    return options


def _write_png(svg: str, output: Path) -> None:
    converter = Path("/usr/bin/rsvg-convert")
    if not converter.is_file():
        raise RuntimeError("PNG output requires /usr/bin/rsvg-convert")
    with tempfile.TemporaryDirectory(prefix="hcfp-visualize-") as directory:
        source = Path(directory) / "floorplan.svg"
        source.write_text(svg, encoding="utf-8")
        subprocess.run([str(converter), "-o", str(output), str(source)], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
