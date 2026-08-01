"""Dependency-free SVG/HTML floorplan visualization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import json
import math
from pathlib import Path
from typing import Any


Palette = tuple[str, str]

DEFAULT_SIZE = 760
MARGIN = 42.0
PALETTE: dict[str, Palette] = {
    "soft": ("#e8eaed", "#5f6368"),
    "fixed": ("#d7e7ff", "#1a73e8"),
    "preplaced": ("#fde7c7", "#b06000"),
    "boundary": ("#d9ead3", "#137333"),
    "group": ("#eadcf8", "#7b1fa2"),
    "mib": ("#f8d7da", "#b3261e"),
}


def render_svg(
    placements: Any,
    *,
    case: Any | None = None,
    telemetry: Any | None = None,
    title: str = "HCFP floorplan",
    width: int = DEFAULT_SIZE,
    height: int = DEFAULT_SIZE,
) -> str:
    """Render ``[N,4]`` x/y/w/h placements as deterministic inline SVG."""

    boxes = _boxes(placements)
    if not boxes:
        raise ValueError("placements must contain at least one rectangle")
    constraints = _constraints(case, len(boxes))
    pins = _pins(case)
    bounds = _bounds(boxes, pins)
    sx, sy, scale = _viewport(bounds, width, height)
    bbox = _box_bounds(boxes)
    summary = _summary(boxes, telemetry)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>"
        ".bg{fill:#fff}.bbox{fill:none;stroke:#202124;stroke-width:1.5;stroke-dasharray:6 4}"
        ".block{stroke-width:1.3}.label{font:12px monospace;fill:#202124;text-anchor:middle;dominant-baseline:middle}"
        ".pin{fill:#202124;stroke:#fff;stroke-width:1}.metric{font:13px sans-serif;fill:#202124}"
        ".small{font:11px sans-serif;fill:#5f6368}"
        "</style>",
        '<rect class="bg" x="0" y="0" width="100%" height="100%"/>',
        f'<text class="metric" x="16" y="22">{escape(title)}</text>',
        f'<text class="small" x="16" y="40">{escape(summary)}</text>',
    ]

    bx, by, bw, bh = _screen_rect((bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), sx, sy, scale)
    out.append(f'<rect class="bbox" x="{_fmt(bx)}" y="{_fmt(by)}" width="{_fmt(bw)}" height="{_fmt(bh)}"/>')

    for i, rect in enumerate(boxes):
        kind = _kind(constraints[i])
        fill, stroke = PALETTE[kind]
        x, y, w, h = _screen_rect(rect, sx, sy, scale)
        cx, cy = x + w * 0.5, y + h * 0.5
        out.append(
            f'<rect class="block" data-block="{i}" data-kind="{kind}" x="{_fmt(x)}" y="{_fmt(y)}" '
            f'width="{_fmt(w)}" height="{_fmt(h)}" fill="{fill}" stroke="{stroke}"/>'
        )
        out.append(f'<text class="label" x="{_fmt(cx)}" y="{_fmt(cy)}">B{i}</text>')

    for i, pin in enumerate(pins):
        x, y = _screen_point(pin, sx, sy, scale)
        out.append(f'<circle class="pin" data-pin="{i}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="3"/>')

    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_html(items: Sequence[Mapping[str, Any]], *, title: str = "HCFP comparison") -> str:
    """Render a self-contained HTML page containing one SVG per item."""

    body = []
    for index, item in enumerate(items):
        name = str(item.get("title", f"candidate {index}"))
        body.append("<section>")
        body.append(
            render_svg(
                item["placements"],
                case=item.get("case"),
                telemetry=item.get("telemetry"),
                title=name,
                width=560,
                height=560,
            )
        )
        body.append("</section>")
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>body{font-family:sans-serif;margin:20px;background:#f8f9fa;color:#202124}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}"
        "section{background:white;border:1px solid #dadce0;padding:8px}</style>\n"
        "</head>\n"
        f"<body><h1>{escape(title)}</h1><main>\n{''.join(body)}</main></body>\n"
        "</html>\n"
    )


def load_visualization_json(
    path: str | Path,
    *,
    lane: str | None = None,
    test_id: int | None = None,
) -> list[dict[str, Any]]:
    """Load one or more visualization entries from a JSON payload."""

    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, Mapping) and "lanes" in payload:
        return _benchmark_entries(payload, lane=lane, test_id=test_id)
    if isinstance(payload, list):
        return [_entry(item, i) for i, item in enumerate(payload)]
    if "candidates" in payload:
        case = payload.get("case")
        telemetry = payload.get("telemetry")
        return [
            _entry({"case": case, "telemetry": telemetry, **candidate}, i)
            for i, candidate in enumerate(payload["candidates"])
        ]
    return [_entry(payload, 0)]


def _benchmark_entries(
    payload: Mapping[str, Any],
    *,
    lane: str | None,
    test_id: int | None,
) -> list[dict[str, Any]]:
    lanes = payload["lanes"]
    if not isinstance(lanes, Mapping):
        raise ValueError("benchmark lanes must be a mapping")
    names = [lane] if lane else list(lanes)
    entries = []
    for name in names:
        if name not in lanes:
            raise ValueError(f"benchmark lane {name!r} is missing")
        rows = lanes[name]
        selected = rows if test_id is None else [row for row in rows if int(row["test_id"]) == test_id]
        for row in selected:
            if row.get("positions") is None:
                continue
            entries.append(
                _entry(
                    {
                        "title": f"case {int(row['test_id'])} — {name} — cost {float(row['cost']):.6f}",
                        "placements": row["positions"],
                        "case": payload.get("case_metadata", {}).get(str(int(row["test_id"]))),
                        "telemetry": {
                            key: row[key]
                            for key in ("cost", "hpwl_gap", "area_gap", "violations_relative")
                        },
                    },
                    len(entries),
                )
            )
    if not entries:
        raise ValueError("benchmark selection contains no placements")
    return entries


def _entry(payload: Mapping[str, Any], index: int) -> dict[str, Any]:
    placements = payload.get("placements", payload.get("placement"))
    if placements is None:
        raise ValueError(f"visualization entry {index} is missing placements")
    return {
        "title": payload.get("title", f"candidate {index}"),
        "placements": placements,
        "case": payload.get("case", payload),
        "telemetry": payload.get("telemetry"),
    }


def _boxes(value: Any) -> list[tuple[float, float, float, float]]:
    rows = _tolist(value)
    out: list[tuple[float, float, float, float]] = []
    for row in rows:
        if len(row) != 4:
            raise ValueError("each placement row must contain x, y, width, height")
        x, y, w, h = (float(v) for v in row)
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
            raise ValueError("placements must be finite rectangles with positive dimensions")
        out.append((x, y, w, h))
    return out


def _constraints(case: Any | None, n: int) -> list[tuple[int, int, int, int, int]]:
    rows = _field(case, "constraints", [])
    values = _tolist(rows) if rows is not None else []
    out: list[tuple[int, int, int, int, int]] = []
    for i in range(n):
        row = values[i] if i < len(values) else []
        padded = list(row[:5]) + [0] * max(0, 5 - len(row))
        out.append(tuple(int(float(v)) for v in padded[:5]))
    return out


def _pins(case: Any | None) -> list[tuple[float, float]]:
    rows = _field(case, "pins", _field(case, "pins_pos", []))
    out: list[tuple[float, float]] = []
    for row in _tolist(rows):
        if len(row) >= 2:
            x, y = float(row[0]), float(row[1])
            if math.isfinite(x) and math.isfinite(y) and (x, y) != (-1.0, -1.0):
                out.append((x, y))
    return out


def _kind(row: tuple[int, int, int, int, int]) -> str:
    fixed, preplaced, mib, group, boundary = row
    if preplaced:
        return "preplaced"
    if fixed:
        return "fixed"
    if boundary:
        return "boundary"
    if group > 0:
        return "group"
    if mib > 0:
        return "mib"
    return "soft"


def _summary(boxes: Sequence[tuple[float, float, float, float]], telemetry: Any | None) -> str:
    left, bottom, right, top = _box_bounds(boxes)
    bbox_area = max(0.0, right - left) * max(0.0, top - bottom)
    parts = [f"blocks={len(boxes)}", f"bbox_area={_fmt(bbox_area)}"]
    for key in (
        "cost",
        "hpwl",
        "hpwl_gap",
        "area_gap",
        "soft_violation",
        "violations_relative",
        "projected_overlap",
        "raw_overlap",
    ):
        value = _field(telemetry, key, None)
        if value is not None:
            parts.append(f"{key}={_fmt(_scalar(value))}")
    return "  ".join(parts)


def _bounds(
    boxes: Sequence[tuple[float, float, float, float]],
    pins: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    left, bottom, right, top = _box_bounds(boxes)
    for x, y in pins:
        left, right = min(left, x), max(right, x)
        bottom, top = min(bottom, y), max(top, y)
    dx, dy = max(1.0e-9, right - left), max(1.0e-9, top - bottom)
    pad = 0.05 * max(dx, dy)
    return left - pad, bottom - pad, right + pad, top + pad


def _box_bounds(boxes: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    left = min(rect[0] for rect in boxes)
    bottom = min(rect[1] for rect in boxes)
    right = max(rect[0] + rect[2] for rect in boxes)
    top = max(rect[1] + rect[3] for rect in boxes)
    return left, bottom, right, top


def _viewport(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float]:
    left, bottom, right, top = bounds
    scale = min((width - 2.0 * MARGIN) / max(right - left, 1.0e-9), (height - 2.0 * MARGIN) / max(top - bottom, 1.0e-9))
    return left, top, scale


def _screen_rect(rect: tuple[float, float, float, float], left: float, top: float, scale: float) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    return MARGIN + (x - left) * scale, MARGIN + (top - y - h) * scale, w * scale, h * scale


def _screen_point(point: tuple[float, float], left: float, top: float, scale: float) -> tuple[float, float]:
    x, y = point
    return MARGIN + (x - left) * scale, MARGIN + (top - y) * scale


def _field(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and int(value.numel()) > 1:
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0]
    return float(value)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"
