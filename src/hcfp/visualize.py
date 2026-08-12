"""Dependency-free SVG/HTML floorplan visualization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import json
import math
from pathlib import Path
from typing import Any


Palette = tuple[str, str]
Rect = tuple[float, float, float, float]

DEFAULT_SIZE = 760
MARGIN = 42.0
TOP_MARGIN = 78.0
PALETTE: dict[str, Palette] = {
    "soft": ("#e8eaed", "#5f6368"),
    "fixed": ("#d7e7ff", "#1a73e8"),
    "preplaced": ("#fde7c7", "#b06000"),
    "boundary": ("#d9ead3", "#137333"),
    "group": ("#eadcf8", "#7b1fa2"),
    "mib": ("#f8d7da", "#b3261e"),
}

_OVERLAY_NAMES = (
    "inferred_latent_outline",
    "model_temporary_outline",
    "pin_perimeter_hypothesis",
    "gold_outline",
    "whitespace",
)
_OVERLAY_STYLE: dict[str, tuple[str, str]] = {
    "official_candidate_bbox": ("#202124", "6 4"),
    "inferred_latent_outline": ("#c2185b", "7 4"),
    "model_temporary_outline": ("#ef6c00", "2 4"),
    "pin_perimeter_hypothesis": ("#00838f", "5 3"),
    "gold_outline": ("#2e7d32", "10 3 2 3"),
}


def render_svg(
    placements: Any,
    *,
    case: Any | None = None,
    telemetry: Any | None = None,
    title: str = "HCFP floorplan",
    width: int = DEFAULT_SIZE,
    height: int = DEFAULT_SIZE,
    official_candidate_bbox: Any | None = None,
    inferred_latent_outline: Any | None = None,
    model_temporary_outline: Any | None = None,
    pin_perimeter_hypothesis: Any | None = None,
    gold_outline: Any | None = None,
    whitespace: Any | None = None,
    diagnostic: bool = False,
    summary_metrics: Any | None = None,
    overlays: Mapping[str, Any] | None = None,
) -> str:
    """Render placements and optional, explicitly named diagnostic geometry.

    Rectangles use ``(x, y, width, height)``.  The official candidate bbox is
    always recomputed from ``placements``; the optional argument is accepted so
    JSON payloads can carry the explicit schema field without overriding that
    contract.  ``gold_outline`` is rendered only when ``diagnostic`` is true.
    """

    boxes = _boxes(placements)
    if not boxes:
        raise ValueError("placements must contain at least one rectangle")
    constraints = _constraints(case, len(boxes))
    pins = _pins(case)
    overlay_values = dict(overlays or {})
    for name, value in (
        ("inferred_latent_outline", inferred_latent_outline),
        ("model_temporary_outline", model_temporary_outline),
        ("pin_perimeter_hypothesis", pin_perimeter_hypothesis),
        ("gold_outline", gold_outline),
        ("whitespace", whitespace),
    ):
        if value is not None:
            overlay_values[name] = value
    diagnostic = diagnostic or bool(overlay_values.get("diagnostic", False))
    inferred = _outline_rect(overlay_values.get("inferred_latent_outline"), "inferred_latent_outline")
    model_outline = _outline_rect(
        overlay_values.get("model_temporary_outline"), "model_temporary_outline"
    )
    pin_outline = _outline_rect(overlay_values.get("pin_perimeter_hypothesis"), "pin_perimeter_hypothesis")
    gold = (
        _outline_rect(overlay_values.get("gold_outline"), "gold_outline")
        if diagnostic
        else None
    )
    whitespace_rects = _rectangles(overlay_values.get("whitespace"), "whitespace")
    extra_rects = [rect for rect in (inferred, model_outline, pin_outline, gold) if rect is not None]
    extra_rects.extend(whitespace_rects)
    bounds = _bounds(boxes, pins, extra_rects)
    sx, sy, scale = _viewport(bounds, width, height)
    # This is deliberately derived from the emitted blocks, never from an
    # optional input field: FloorSet has no official fixed canvas.
    bbox = _box_bounds(boxes)
    summary, summary_values = _summary(
        boxes,
        telemetry,
        summary_metrics,
        inferred=inferred,
    )

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>"
        ".bg{fill:#fff}.bbox{fill:none;stroke:#202124;stroke-width:1.5;stroke-dasharray:6 4}"
        ".outline{fill:none;stroke-width:1.7}.inferred_latent_outline{stroke:#c2185b;stroke-dasharray:7 4}"
        ".model_temporary_outline{stroke:#ef6c00;stroke-dasharray:2 4}"
        ".pin_perimeter_hypothesis{stroke:#00838f;stroke-dasharray:5 3}"
        ".gold_outline{stroke:#2e7d32;stroke-width:2;stroke-dasharray:10 3 2 3}"
        ".whitespace{fill:#90caf9;fill-opacity:.32;stroke:#1565c0;stroke-width:1;stroke-dasharray:2 2}"
        ".block{stroke-width:1.3}.outside_inferred_outline{stroke:#d93025!important;stroke-width:2.5}"
        ".label{font:12px monospace;fill:#202124;text-anchor:middle;dominant-baseline:middle}"
        ".pin{fill:#202124;stroke:#fff;stroke-width:1}.metric{font:13px sans-serif;fill:#202124}"
        ".small{font:11px sans-serif;fill:#5f6368}.legend-label{font:11px sans-serif;fill:#202124}"
        "</style>",
        '<rect class="bg" x="0" y="0" width="100%" height="100%"/>',
        f'<text class="metric" x="16" y="22">{escape(title)}</text>',
        _summary_svg(summary, summary_values),
    ]

    out.append(
        _svg_rect(
            "official_candidate_bbox",
            (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
            sx,
            sy,
            scale,
            classes=("bbox", "official_candidate_bbox"),
        )
    )
    if inferred is not None:
        out.append(_svg_rect("inferred_latent_outline", inferred, sx, sy, scale))
    if model_outline is not None:
        out.append(_svg_rect("model_temporary_outline", model_outline, sx, sy, scale))
    if pin_outline is not None:
        out.append(_svg_rect("pin_perimeter_hypothesis", pin_outline, sx, sy, scale))
    if gold is not None:
        out.append(_svg_rect("gold_outline", gold, sx, sy, scale))
    for index, rect in enumerate(whitespace_rects):
        out.append(_svg_rect("whitespace", rect, sx, sy, scale, index=index))

    for i, rect in enumerate(boxes):
        kind = _kind(constraints[i])
        fill, stroke = PALETTE[kind]
        x, y, w, h = _screen_rect(rect, sx, sy, scale)
        cx, cy = x + w * 0.5, y + h * 0.5
        outside = inferred is not None and _outside(rect, inferred)
        classes = "block outside_inferred_outline outside-inferred-outline" if outside else "block"
        outside_attr = ' data-outside-inferred="true"' if outside else ""
        block_stroke = "#d93025" if outside else stroke
        out.append(
            f'<rect class="{classes}" data-block="{i}" data-kind="{kind}"{outside_attr} '
            f'x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" '
            f'fill="{fill}" stroke="{block_stroke}"/>'
        )
        out.append(f'<text class="label" x="{_fmt(cx)}" y="{_fmt(cy)}">B{i}</text>')

    for i, pin in enumerate(pins):
        x, y = _screen_point(pin, sx, sy, scale)
        out.append(f'<circle class="pin" data-pin="{i}" cx="{_fmt(x)}" cy="{_fmt(y)}" r="3"/>')

    out.extend(
        _legend(
            width,
            height,
            inferred is not None,
            model_outline is not None,
            pin_outline is not None,
            gold is not None,
            bool(whitespace_rects),
        )
    )

    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_html(
    items: Sequence[Mapping[str, Any]],
    *,
    title: str = "HCFP comparison",
    diagnostic: bool = False,
) -> str:
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
                diagnostic=diagnostic or bool(item.get("diagnostic", False)),
                **_render_options(item),
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
        shared = {
            key: payload[key]
            for key in (
                *_OVERLAY_NAMES,
                "official_candidate_bbox",
                "summary_metrics",
                "diagnostic",
                "overlays",
            )
            if key in payload
        }
        return [
            _entry({"case": case, "telemetry": telemetry, **shared, **candidate}, i)
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
            optional = {
                key: row[key]
                for key in (
                    *_OVERLAY_NAMES,
                    "official_candidate_bbox",
                    "summary_metrics",
                    "diagnostic",
                    "overlays",
                )
                if key in row
            }
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
                        **optional,
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
    entry = {
        "title": payload.get("title", f"candidate {index}"),
        "placements": placements,
        "case": payload.get("case", payload),
        "telemetry": payload.get("telemetry"),
    }
    for key in (
        *_OVERLAY_NAMES,
        "official_candidate_bbox",
        "summary_metrics",
        "diagnostic",
    ):
        if key in payload:
            entry[key] = payload[key]
    if isinstance(payload.get("overlays"), Mapping):
        entry["overlays"] = dict(payload["overlays"])
    return entry


def _render_options(item: Mapping[str, Any]) -> dict[str, Any]:
    options = {
        key: item[key]
        for key in (*_OVERLAY_NAMES, "official_candidate_bbox", "summary_metrics", "overlays")
        if key in item
    }
    return options


def _outline_rect(value: Any, name: str) -> Rect | None:
    if value is None or value is False:
        return None
    if isinstance(value, Mapping):
        if "rect" in value:
            value = value["rect"]
        elif all(
            key in value for key in ("x_left", "y_bottom", "x_right", "y_top")
        ):
            value = (
                value["x_left"],
                value["y_bottom"],
                float(value["x_right"]) - float(value["x_left"]),
                float(value["y_top"]) - float(value["y_bottom"]),
            )
        elif all(key in value for key in ("x", "y", "width", "height")):
            value = (value["x"], value["y"], value["width"], value["height"])
        elif all(key in value for key in ("left", "bottom", "right", "top")):
            value = (
                value["left"],
                value["bottom"],
                float(value["right"]) - float(value["left"]),
                float(value["top"]) - float(value["bottom"]),
            )
        elif "bounds" in value:
            value = value["bounds"]
        else:
            raise ValueError(f"{name} must describe an x/y/width/height rectangle")
    rows = _tolist(value)
    if len(rows) == 4 and all(not isinstance(row, (list, tuple, Mapping)) for row in rows):
        try:
            rect = tuple(float(row) for row in rows)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric rectangle coordinates") from exc
        if not all(math.isfinite(number) for number in rect) or rect[2] <= 0.0 or rect[3] <= 0.0:
            raise ValueError(f"{name} must be a finite rectangle with positive dimensions")
        return rect  # type: ignore[return-value]
    raise ValueError(f"{name} must describe an x/y/width/height rectangle")


def _rectangles(value: Any, name: str) -> list[Rect]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("leaves", value.get("rectangles", value))
    rows = _tolist(value)
    if len(rows) == 4 and all(not isinstance(row, (list, tuple, Mapping)) for row in rows):
        return [_outline_rect(rows, name)]  # type: ignore[list-item]
    return [_outline_rect(row, name) for row in rows]


def _svg_rect(
    name: str,
    rect: Rect,
    left: float,
    top: float,
    scale: float,
    *,
    classes: Sequence[str] | None = None,
    index: int | None = None,
) -> str:
    x, y, width, height = _screen_rect(rect, left, top, scale)
    class_names = tuple(classes or ("outline", name))
    stroke, dash = _OVERLAY_STYLE.get(name, ("#202124", "6 4"))
    attrs = [
        f'class="{" ".join(class_names)}"',
        f'data-overlay="{name}"',
        f'x="{_fmt(x)}"',
        f'y="{_fmt(y)}"',
        f'width="{_fmt(width)}"',
        f'height="{_fmt(height)}"',
    ]
    if index is not None:
        attrs.append(f'data-whitespace="{index}"')
    if name == "whitespace":
        attrs.extend(('fill="#90caf9"', 'fill-opacity="0.32"', 'stroke="#1565c0"'))
    else:
        attrs.extend(('fill="none"', f'stroke="{stroke}"', f'stroke-dasharray="{dash}"'))
    return f'<rect {" ".join(attrs)}/>'


def _legend(
    width: int,
    height: int,
    inferred: bool,
    model: bool,
    pin: bool,
    gold: bool,
    whitespace: bool,
) -> list[str]:
    entries: list[tuple[str, str, str, bool]] = [("official_candidate_bbox", "#202124", "6 4", False)]
    if inferred:
        entries.extend(
            (("inferred_latent_outline", "#c2185b", "7 4", False), ("outside_inferred_outline", "#d93025", "", False))
        )
    if model:
        entries.append(("model_temporary_outline", "#ef6c00", "2 4", False))
    if pin:
        entries.append(("pin_perimeter_hypothesis", "#00838f", "5 3", False))
    if gold:
        entries.append(("gold_outline", "#2e7d32", "10 3 2 3", False))
    if whitespace:
        entries.append(("whitespace", "#1565c0", "2 2", True))
    row_height = 16
    legend_height = len(entries) * row_height + 8
    y0 = max(48, height - legend_height - 6)
    out = [
        f'<g class="legend" data-legend="true" aria-label="legend"><rect x="10" y="{_fmt(y0 - 4)}" '
        f'width="{_fmt(min(width - 20, 270))}" height="{_fmt(legend_height)}" fill="#fff" fill-opacity="0.9" stroke="#dadce0"/>'
    ]
    for index, (name, stroke, dash, fill) in enumerate(entries):
        y = y0 + index * row_height + 7
        if fill:
            out.append(
                f'<rect x="18" y="{_fmt(y - 5)}" width="14" height="10" fill="#90caf9" '
                f'fill-opacity="0.32" stroke="{stroke}" stroke-dasharray="{dash}"/>'
            )
        else:
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<line x1="18" y1="{_fmt(y)}" x2="32" y2="{_fmt(y)}" stroke="{stroke}" stroke-width="2"{dash_attr}/>' )
        out.append(f'<text class="legend-label" data-legend-overlay="{name}" x="38" y="{_fmt(y + 4)}">{name}</text>')
    out.append("</g>")
    return out


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


def _summary(
    boxes: Sequence[Rect],
    telemetry: Any | None,
    summary_metrics: Any | None,
    *,
    inferred: Rect | None,
) -> tuple[str, dict[str, Any]]:
    left, bottom, right, top = _box_bounds(boxes)
    bbox_area = max(0.0, right - left) * max(0.0, top - bottom)
    parts = [f"blocks={len(boxes)}", f"bbox_area={_fmt(bbox_area)}"]
    values: dict[str, Any] = {"blocks": len(boxes), "bbox_area": bbox_area}
    utilization = _metric(summary_metrics, telemetry, "utilization")
    if utilization is None:
        utilization = sum(rect[2] * rect[3] for rect in boxes) / bbox_area if bbox_area else 0.0
    _add_metric(parts, values, "utilization", utilization)

    overflow = _metric(summary_metrics, telemetry, "outline_overflow_area")
    outside = _metric(summary_metrics, telemetry, "blocks_outside_inferred_outline")
    if inferred is not None:
        overflow = sum(_outside_area(rect, inferred) for rect in boxes) if overflow is None else overflow
        outside = sum(_outside(rect, inferred) for rect in boxes) if outside is None else outside
    if overflow is not None:
        _add_metric(parts, values, "outline_overflow_area", overflow)
    if outside is not None:
        _add_metric(parts, values, "blocks_outside_inferred_outline", outside)

    for key in ("pin_side_coverage", "group_connected_components", "mib_distinct_shape_count"):
        value = _metric(summary_metrics, telemetry, key)
        if value is not None:
            _add_metric(parts, values, key, value)

    boundary_satisfied = _metric(summary_metrics, telemetry, "boundary_satisfied")
    boundary_total = _metric(summary_metrics, telemetry, "boundary_total")
    if boundary_satisfied is not None or boundary_total is not None:
        boundary_satisfied = 0 if boundary_satisfied is None else boundary_satisfied
        boundary_total = boundary_satisfied if boundary_total is None else boundary_total
        rendered = f"{_render_metric(boundary_satisfied)}/{_render_metric(boundary_total)}"
        parts.append(f"boundary_satisfied={rendered}")
        values["boundary_satisfied"] = {"satisfied": _json_metric(boundary_satisfied), "total": _json_metric(boundary_total)}

    displacement = _metric(summary_metrics, telemetry, "raw_to_projected_displacement")
    if displacement is None:
        displacement = _metric(summary_metrics, telemetry, "projection_displacement")
    if displacement is not None:
        _add_metric(parts, values, "raw_to_projected_displacement", displacement)

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
            rendered = _render_metric(value)
            parts.append(f"{key}={rendered}")
            values[key] = _json_metric(value)
    return "  ".join(parts), values


def _metric(summary_metrics: Any | None, telemetry: Any | None, key: str) -> Any:
    for source in (summary_metrics, telemetry):
        if source is not None:
            value = _field(source, key, None)
            if value is not None:
                return value
    return None


def _summary_svg(summary: str, values: Mapping[str, Any]) -> str:
    lines: list[str] = []
    current = ""
    for part in summary.split("  "):
        candidate = part if not current else f"{current}  {part}"
        if current and len(candidate) > 105:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    tspans = "".join(
        f'<tspan x="16" dy="{0 if index == 0 else 14}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    metrics = escape(json.dumps(values, sort_keys=True))
    return f'<text class="small" x="16" y="40" data-summary-metrics="{metrics}">{tspans}</text>'


def _add_metric(parts: list[str], values: dict[str, Any], key: str, value: Any) -> None:
    parts.append(f"{key}={_render_metric(value)}")
    values[key] = _json_metric(value)


def _render_metric(value: Any) -> str:
    try:
        return _fmt(_scalar(value))
    except (TypeError, ValueError):
        return str(value)


def _json_metric(value: Any) -> Any:
    try:
        return _scalar(value)
    except (TypeError, ValueError):
        return str(value)


def _outside_area(rect: Rect, outline: Rect) -> float:
    x, y, width, height = rect
    left, bottom, outline_width, outline_height = outline
    right, top = x + width, y + height
    outline_right, outline_top = left + outline_width, bottom + outline_height
    overlap_width = max(0.0, min(right, outline_right) - max(x, left))
    overlap_height = max(0.0, min(top, outline_top) - max(y, bottom))
    return max(0.0, width * height - overlap_width * overlap_height)


def _outside(rect: Rect, outline: Rect) -> bool:
    x, y, width, height = rect
    left, bottom, outline_width, outline_height = outline
    return x < left or y < bottom or x + width > left + outline_width or y + height > bottom + outline_height


def _bounds(
    boxes: Sequence[Rect],
    pins: Sequence[tuple[float, float]],
    rectangles: Sequence[Rect] = (),
) -> tuple[float, float, float, float]:
    left, bottom, right, top = _box_bounds(boxes)
    for x, y in pins:
        left, right = min(left, x), max(right, x)
        bottom, top = min(bottom, y), max(top, y)
    for x, y, width, height in rectangles:
        left, right = min(left, x), max(right, x + width)
        bottom, top = min(bottom, y), max(top, y + height)
    dx, dy = max(1.0e-9, right - left), max(1.0e-9, top - bottom)
    pad = 0.05 * max(dx, dy)
    return left - pad, bottom - pad, right + pad, top + pad


def _box_bounds(boxes: Sequence[Rect]) -> tuple[float, float, float, float]:
    left = min(rect[0] for rect in boxes)
    bottom = min(rect[1] for rect in boxes)
    right = max(rect[0] + rect[2] for rect in boxes)
    top = max(rect[1] + rect[3] for rect in boxes)
    return left, bottom, right, top


def _viewport(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float]:
    left, bottom, right, top = bounds
    scale = min(
        (width - 2.0 * MARGIN) / max(right - left, 1.0e-9),
        (height - TOP_MARGIN - MARGIN) / max(top - bottom, 1.0e-9),
    )
    return left, top, scale


def _screen_rect(rect: Rect, left: float, top: float, scale: float) -> Rect:
    x, y, w, h = rect
    return MARGIN + (x - left) * scale, TOP_MARGIN + (top - y - h) * scale, w * scale, h * scale


def _screen_point(point: tuple[float, float], left: float, top: float, scale: float) -> tuple[float, float]:
    x, y = point
    return MARGIN + (x - left) * scale, TOP_MARGIN + (top - y) * scale


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
