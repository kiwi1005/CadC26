#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.repair import finalize_layout
from puzzleplace.train.dataset_bc import load_validation_cases

Box = tuple[float, float, float, float]


def _coerce_box(values: Any) -> Box:
    x, y, w, h = values
    return float(x), float(y), float(w), float(h)


def _load_multistart_module() -> ModuleType:
    path = Path(__file__).with_name("run_step6g_multistart_sidecar.py")
    spec = importlib.util.spec_from_file_location("step6g_multistart_sidecar", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bbox(placements: dict[int, Box]) -> Box:
    min_x = min(x for x, _y, _w, _h in placements.values())
    min_y = min(y for _x, y, _w, _h in placements.values())
    max_x = max(x + w for x, _y, w, _h in placements.values())
    max_y = max(y + h for _x, y, _w, h in placements.values())
    return min_x, min_y, max_x - min_x, max_y - min_y


def _constraint_class(case: FloorSetCase, block_index: int) -> str:
    row = case.constraints[block_index]
    if bool(row[ConstraintColumns.FIXED].item()):
        return "fixed"
    if bool(row[ConstraintColumns.PREPLACED].item()):
        return "preplaced"
    if bool(row[ConstraintColumns.MIB].item()):
        return "mib"
    if bool(row[ConstraintColumns.CLUSTER].item()):
        return "cluster"
    if bool(row[ConstraintColumns.BOUNDARY].item()):
        return "boundary"
    return "regular"


def _constraint_color(kind: str) -> str:
    return {
        "fixed": "#c084fc",
        "preplaced": "#94a3b8",
        "mib": "#22c55e",
        "cluster": "#fb7185",
        "boundary": "#38bdf8",
        "regular": "#e5e7eb",
    }[kind]


def _svg_y(y: float, h: float, min_y: float, max_y: float, pad: float) -> float:
    return pad + (max_y - (y + h)) + min(0.0, min_y)


def _center(box: Box) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def render_svg(
    *,
    case: FloorSetCase,
    placements: dict[int, Box],
    title: str,
    metrics: dict[str, Any],
    draw_nets: int = 40,
) -> str:
    min_x, min_y, bbox_w, bbox_h = _bbox(placements)
    max_y = min_y + bbox_h
    pad = max(max(bbox_w, bbox_h) * 0.04, 8.0)
    width = bbox_w + 2 * pad
    height = bbox_h + 2 * pad
    shift_x = pad - min_x

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}" '
            f'height="{height:.2f}" viewBox="0 0 {width:.2f} {height:.2f}">'
        ),
        "<style>",
        ".title{font:700 14px sans-serif}.meta{font:11px monospace;fill:#334155}",
        ".label{font:8px sans-serif;fill:#0f172a}.pin{fill:#16a34a;stroke:#065f46}",
        ".net-b2b{stroke:#ef4444;stroke-opacity:.28;stroke-width:.45;fill:none}",
        ".net-p2b{stroke:#2563eb;stroke-opacity:.35;stroke-width:.45;fill:none}",
        ".block{stroke:#111827;stroke-width:.5}.bbox{fill:none;stroke:#111827;stroke-width:1.0}",
        "</style>",
        f'<rect x="0" y="0" width="{width:.2f}" height="{height:.2f}" fill="#ffffff"/>',
        f'<text class="title" x="{pad:.2f}" y="18">{html.escape(title)}</text>',
    ]
    metric_text = " | ".join(f"{key}={value}" for key, value in metrics.items())
    parts.append(f'<text class="meta" x="{pad:.2f}" y="34">{html.escape(metric_text)}</text>')

    bbox_y = _svg_y(min_y, bbox_h, min_y, max_y, pad)
    parts.append(
        f'<rect class="bbox" x="{pad:.2f}" y="{bbox_y:.2f}" '
        f'width="{bbox_w:.2f}" height="{bbox_h:.2f}"/>'
    )

    if draw_nets > 0:
        weighted_edges = sorted(
            case.b2b_edges.tolist(),
            key=lambda row: abs(float(row[2])) if len(row) > 2 else 1.0,
            reverse=True,
        )[:draw_nets]
        for src, dst, *_rest in weighted_edges:
            i, j = int(src), int(dst)
            if i not in placements or j not in placements:
                continue
            ix, iy = _center(placements[i])
            jx, jy = _center(placements[j])
            parts.append(
                f'<line class="net-b2b" x1="{ix + shift_x:.2f}" '
                f'y1="{_svg_y(iy, 0.0, min_y, max_y, pad):.2f}" '
                f'x2="{jx + shift_x:.2f}" y2="{_svg_y(jy, 0.0, min_y, max_y, pad):.2f}"/>'
            )
        for pin_idx, block_idx, *_rest in case.p2b_edges.tolist()[:draw_nets]:
            pin_i, block_i = int(pin_idx), int(block_idx)
            if pin_i >= len(case.pins_pos) or block_i not in placements:
                continue
            px, py = [float(v) for v in case.pins_pos[pin_i].tolist()]
            bx, by = _center(placements[block_i])
            parts.append(
                f'<line class="net-p2b" x1="{px + shift_x:.2f}" '
                f'y1="{_svg_y(py, 0.0, min_y, max_y, pad):.2f}" '
                f'x2="{bx + shift_x:.2f}" y2="{_svg_y(by, 0.0, min_y, max_y, pad):.2f}"/>'
            )

    for idx, box in sorted(placements.items()):
        x, y, w, h = box
        kind = _constraint_class(case, idx)
        sx = x + shift_x
        sy = _svg_y(y, h, min_y, max_y, pad)
        parts.append(
            f'<rect class="block" x="{sx:.2f}" y="{sy:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" fill="{_constraint_color(kind)}" fill-opacity="0.72">'
            f"<title>block {idx} | {kind} | x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}</title>"
            "</rect>"
        )
        parts.append(
            f'<text class="label" x="{sx + 1.2:.2f}" y="{sy + 8.5:.2f}">{idx}</text>'
        )

    for pin_idx, pin in enumerate(case.pins_pos.tolist()):
        px, py = float(pin[0]), float(pin[1])
        parts.append(
            f'<circle class="pin" cx="{px + shift_x:.2f}" '
            f'cy="{_svg_y(py, 0.0, min_y, max_y, pad):.2f}" r="2.5">'
            f"<title>pin {pin_idx}</title></circle>"
        )

    legend_x = pad
    legend_y = height - pad + 2
    for offset, kind in enumerate(["fixed", "preplaced", "mib", "cluster", "boundary", "regular"]):
        x = legend_x + offset * 74
        parts.append(
            f'<rect x="{x:.2f}" y="{legend_y:.2f}" width="9" height="9" '
            f'fill="{_constraint_color(kind)}" stroke="#111827" stroke-width=".4"/>'
        )
        parts.append(
            f'<text class="meta" x="{x + 12:.2f}" y="{legend_y + 8:.2f}">{kind}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_png(
    *,
    case: FloorSetCase,
    placements: dict[int, Box],
    title: str,
    metrics: dict[str, Any],
    output_path: Path,
    draw_nets: int = 40,
    dpi: int = 180,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Rectangle

    min_x, min_y, bbox_w, bbox_h = _bbox(placements)
    max_x = min_x + bbox_w
    max_y = min_y + bbox_h
    pad = max(max(bbox_w, bbox_h) * 0.04, 8.0)
    fig_w = max(min((bbox_w + 2 * pad) / 24.0, 18.0), 6.0)
    fig_h = max(min((bbox_h + 2 * pad) / 24.0, 18.0), 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_facecolor("#ffffff")

    if draw_nets > 0:
        weighted_edges = sorted(
            case.b2b_edges.tolist(),
            key=lambda row: abs(float(row[2])) if len(row) > 2 else 1.0,
            reverse=True,
        )[:draw_nets]
        for src, dst, *_rest in weighted_edges:
            i, j = int(src), int(dst)
            if i not in placements or j not in placements:
                continue
            ix, iy = _center(placements[i])
            jx, jy = _center(placements[j])
            ax.plot([ix, jx], [iy, jy], color="#ef4444", alpha=0.28, linewidth=0.45)
        for pin_idx, block_idx, *_rest in case.p2b_edges.tolist()[:draw_nets]:
            pin_i, block_i = int(pin_idx), int(block_idx)
            if pin_i >= len(case.pins_pos) or block_i not in placements:
                continue
            px, py = [float(v) for v in case.pins_pos[pin_i].tolist()]
            bx, by = _center(placements[block_i])
            ax.plot([px, bx], [py, by], color="#2563eb", alpha=0.35, linewidth=0.45)

    for idx, box in sorted(placements.items()):
        x, y, w, h = box
        kind = _constraint_class(case, idx)
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=_constraint_color(kind),
                edgecolor="#111827",
                linewidth=0.5,
                alpha=0.72,
            )
        )
        ax.text(x + 0.8, y + min(h * 0.55, 7.5), str(idx), fontsize=5, color="#0f172a")

    for pin in case.pins_pos.tolist():
        px, py = float(pin[0]), float(pin[1])
        ax.add_patch(Circle((px, py), radius=max(pad * 0.06, 1.0), color="#16a34a"))

    metric_text = " | ".join(f"{key}={value}" for key, value in metrics.items())
    ax.set_title(f"{title}\n{metric_text}", fontsize=9)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    legend_handles = [
        Patch(facecolor=_constraint_color(kind), edgecolor="#111827", label=kind, alpha=0.72)
        for kind in ["fixed", "preplaced", "mib", "cluster", "boundary", "regular"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _metrics_for_run(run: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "pre":
        return {
            "bbox": f"{float(run['bbox_area']):.2f}",
            "hpwl": f"{float(run['hpwl_proxy']):.3f}",
            "soft": int(run["soft_boundary_violations"]),
        }
    return {
        "bbox": f"{float(run['post_repair_bbox_area']):.2f}",
        "hpwl": f"{float(run['post_repair_hpwl_proxy']):.3f}",
        "soft": int(run["post_repair_soft_boundary_violations"]),
        "disp": f"{float(run['repair_mean_displacement']):.3f}",
    }


def _write_index(rows: list[dict[str, str]], output_dir: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<section>"
            f"<h2>{html.escape(row['title'])}</h2>"
            f"<p><code>{html.escape(row['metrics'])}</code></p>"
            f"<p><a href=\"{html.escape(row['image'])}\">{html.escape(row['image'])}</a></p>"
            f"<img src=\"{html.escape(row['image'])}\" "
            "style=\"max-width:100%;border:1px solid #ddd\"/>"
            "</section>"
        )
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'>",
            "<title>Step6G layout visualizations</title>",
            "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:32px}</style>",
            "<h1>Step6G layout visualizations</h1>",
            *cards,
        ]
    )
    (output_dir / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Step6G multistart sidecar layouts to PNG/SVG plus an HTML index."
    )
    parser.add_argument("--input", default="artifacts/research/step6g_multistart_sidecar.json")
    parser.add_argument("--output-dir", default="artifacts/research/step6g_visualizations")
    parser.add_argument("--case-ids", nargs="*", type=int)
    parser.add_argument("--stage", choices=["pre", "post", "both"], default="post")
    parser.add_argument("--format", choices=["png", "svg", "both"], default="png")
    parser.add_argument("--draw-nets", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    runs = payload["runs"]
    selected_case_ids = (
        set(args.case_ids) if args.case_ids else {int(run["case_id"]) for run in runs}
    )
    runs = [run for run in runs if int(run["case_id"]) in selected_case_ids]
    if not runs:
        raise SystemExit("no matching case ids in Step6G artifact")

    cases = load_validation_cases(case_limit=max(selected_case_ids) + 1)
    multistart = _load_multistart_module()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    position_dump: dict[str, Any] = {"source": args.input, "layouts": []}
    index_rows: list[dict[str, str]] = []

    for run in runs:
        case_id = int(run["case_id"])
        seed = int(run["best_seed"])
        start = int(run["best_start"])
        case = cases[case_id]
        pre_positions, family_usage = multistart._construct(case, seed, start)
        stages = ["pre", "post"] if args.stage == "both" else [args.stage]
        for stage in stages:
            if stage == "post":
                repair = finalize_layout(case, pre_positions)
                placements = {idx: _coerce_box(box) for idx, box in enumerate(repair.positions)}
            else:
                placements = {idx: _coerce_box(box) for idx, box in pre_positions.items()}
            metrics = _metrics_for_run(run, stage)
            title = f"case {case_id} | {stage}-repair | seed {seed} start {start}"
            stem = f"case{case_id:03d}_seed{seed}_start{start}_{stage}"
            image_filename = f"{stem}.png" if args.format != "svg" else f"{stem}.svg"
            if args.format in {"png", "both"}:
                render_png(
                    case=case,
                    placements=placements,
                    title=title,
                    metrics=metrics,
                    output_path=output_dir / f"{stem}.png",
                    draw_nets=args.draw_nets,
                    dpi=args.dpi,
                )
            if args.format in {"svg", "both"}:
                (output_dir / f"{stem}.svg").write_text(
                    render_svg(
                        case=case,
                        placements=placements,
                        title=title,
                        metrics=metrics,
                        draw_nets=args.draw_nets,
                    )
                )
            position_dump["layouts"].append(
                {
                    "case_id": case_id,
                    "stage": stage,
                    "seed": seed,
                    "start": start,
                    "candidate_family_usage": family_usage,
                    "metrics": metrics,
                    "image": image_filename,
                    "positions": {str(idx): list(box) for idx, box in sorted(placements.items())},
                }
            )
            index_rows.append(
                {
                    "title": title,
                    "image": image_filename,
                    "metrics": ", ".join(f"{key}={value}" for key, value in metrics.items()),
                }
            )

    (output_dir / "positions.json").write_text(json.dumps(position_dump, indent=2))
    _write_index(index_rows, output_dir)
    print(
        json.dumps(
            {
                "input": args.input,
                "output_dir": str(output_dir),
                "rendered": len(index_rows),
                "index": str(output_dir / "index.html"),
                "positions": str(output_dir / "positions.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
