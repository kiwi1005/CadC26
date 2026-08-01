from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

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
