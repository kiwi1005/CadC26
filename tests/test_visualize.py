from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from hcfp.visualize import load_visualization_json, render_html, render_svg


def _payload() -> dict[str, object]:
    return {
        "title": "demo",
        "case": {
            "constraints": [
                [0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 7, 3, 2],
            ],
            "pins_pos": [[0.0, 0.0]],
        },
        "placements": [[0.0, 0.0, 2.0, 2.0], [2.1, 0.0, 3.0, 1.0], [0.0, 2.2, 1.0, 1.5]],
        "telemetry": {"cost": 2.5, "hpwl": 12.5, "soft_violation": 0.25},
    }


def test_render_svg_is_well_formed_and_deterministic() -> None:
    svg = render_svg(**_payload())

    assert svg == render_svg(**_payload())
    ET.fromstring(svg)
    assert "B0" in svg
    assert "data-block=\"2\"" in svg
    assert "data-kind=\"preplaced\"" in svg
    assert "data-kind=\"fixed\"" in svg
    assert "data-pin=\"0\"" in svg
    assert "hpwl=12.5" in svg
    assert "cost=2.5" in svg


def test_render_svg_marks_official_bbox_and_diagnostic_overlays() -> None:
    payload = _payload()
    svg = render_svg(
        **payload,
        inferred_latent_outline=[0.0, 0.0, 2.5, 2.0],
        model_temporary_outline={"left": 0.0, "bottom": 0.0, "right": 5.0, "top": 4.0},
        pin_perimeter_hypothesis=[0.0, 0.0, 5.0, 3.0],
        gold_outline=[0.0, 0.0, 5.0, 4.0],
        whitespace=[[3.0, 1.0, 1.0, 1.0]],
        diagnostic=True,
        summary_metrics={
            "utilization": 0.75,
            "boundary_satisfied": 2,
            "boundary_total": 3,
            "group_connected_components": 1,
            "mib_distinct_shape_count": 2,
            "raw_to_projected_displacement": 0.125,
        },
    )

    ET.fromstring(svg)
    assert 'data-overlay="official_candidate_bbox"' in svg
    for name in (
        "inferred_latent_outline",
        "model_temporary_outline",
        "pin_perimeter_hypothesis",
        "gold_outline",
        "whitespace",
    ):
        assert f'data-overlay="{name}"' in svg
        assert f'data-legend-overlay="{name}"' in svg
    assert "data-outside-inferred=\"true\"" in svg
    assert "outside_inferred_outline" in svg
    assert "utilization=0.75" in svg
    assert "outline_overflow_area=" in svg
    assert "blocks_outside_inferred_outline=" in svg
    assert "boundary_satisfied=2/3" in svg
    assert "group_connected_components=1" in svg
    assert "mib_distinct_shape_count=2" in svg
    assert "raw_to_projected_displacement=0.125" in svg

    without_diagnostic = render_svg(
        **payload,
        gold_outline=[0.0, 0.0, 5.0, 4.0],
    )
    assert 'data-overlay="gold_outline"' not in without_diagnostic


def test_render_svg_accepts_nested_overlay_schema_and_preserves_bbox_contract() -> None:
    svg = render_svg(
        _payload()["placements"],
        case=_payload()["case"],
        official_candidate_bbox=[100.0, 100.0, 1.0, 1.0],
        overlays={
            "inferred_latent_outline": [0.0, 0.0, 4.0, 3.0],
            "whitespace": [[3.0, 1.0, 1.0, 1.0]],
        },
    )

    assert 'class="bbox official_candidate_bbox"' in svg
    assert 'x="100"' not in svg
    assert 'data-overlay="inferred_latent_outline"' in svg
    assert 'data-whitespace="0"' in svg


def test_render_svg_accepts_outline_hypothesis_dict_coordinates() -> None:
    svg = render_svg(
        _payload()["placements"],
        inferred_latent_outline={
            "bounds": [0.0, 0.0, 5.0, 4.0],
            "x_left": 0.0,
            "x_right": 5.0,
            "y_bottom": 0.0,
            "y_top": 4.0,
        },
    )

    assert 'data-overlay="inferred_latent_outline"' in svg
    assert 'blocks_outside_inferred_outline=1' in svg


def test_render_html_embeds_multiple_candidates() -> None:
    html = render_html([_payload(), {**_payload(), "title": "alt"}], title="compare")

    assert html.startswith("<!doctype html>")
    assert html.count("<svg ") == 2
    assert "<title>compare</title>" in html
    assert "alt" in html


def test_json_loader_supports_candidate_list(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"case": _payload()["case"], "candidates": [_payload()]}), encoding="utf-8")

    entries = load_visualization_json(path)

    assert len(entries) == 1
    assert entries[0]["placements"] == _payload()["placements"]
    assert entries[0]["case"] == _payload()["case"]


def test_visualize_cli_writes_svg(tmp_path: Path) -> None:
    src = tmp_path / "placement.json"
    dst = tmp_path / "out.svg"
    src.write_text(json.dumps(_payload()), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/visualize_hcfp.py", str(src), "-o", str(dst)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert str(dst) in result.stdout
    assert dst.read_text(encoding="utf-8").startswith("<svg ")


def test_visualize_cli_selects_benchmark_lane_and_case(tmp_path: Path) -> None:
    src = tmp_path / "benchmark.json"
    dst = tmp_path / "selected.svg"
    row = {
        "test_id": 7,
        "cost": 2.0,
        "hpwl_gap": 0.2,
        "area_gap": 0.1,
        "violations_relative": 0.0,
        "positions": _payload()["placements"],
    }
    src.write_text(json.dumps({"lanes": {"analytic": [row]}}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/visualize_hcfp.py",
            str(src),
            "--lane",
            "analytic",
            "--case-id",
            "7",
            "-o",
            str(dst),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    svg = dst.read_text(encoding="utf-8")
    assert "case 7" in svg
    assert "cost=2" in svg


def test_visualize_cli_writes_single_candidate_png(tmp_path: Path) -> None:
    if not Path("/usr/bin/rsvg-convert").is_file():
        pytest.skip("rsvg-convert is not installed")
    src = tmp_path / "placement.json"
    dst = tmp_path / "out.png"
    src.write_text(
        json.dumps(
            {
                **_payload(),
                "inferred_latent_outline": [0.0, 0.0, 5.0, 4.0],
                "gold_outline": [0.0, 0.0, 5.0, 4.0],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/visualize_hcfp.py",
            str(src),
            "--diagnostic",
            "-o",
            str(dst),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert dst.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
