from __future__ import annotations

import importlib.util
from pathlib import Path

from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case


def _load_visualizer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_step6g_layouts.py"
    spec = importlib.util.spec_from_file_location("visualize_step6g_layouts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step6g_visualizer_renders_svg_with_blocks_pins_and_metrics() -> None:
    module = _load_visualizer()
    case = make_step6g_synthetic_case(block_count=6)
    assert case.target_positions is not None
    placements = {
        idx: tuple(float(v) for v in case.target_positions[idx].tolist())
        for idx in range(case.block_count)
    }

    svg = module.render_svg(
        case=case,
        placements=placements,
        title="synthetic layout",
        metrics={"hpwl": "1.0", "bbox": "2.0"},
        draw_nets=4,
    )

    assert svg.startswith("<?xml")
    assert "<svg" in svg
    assert "synthetic layout" in svg
    assert "hpwl=1.0" in svg
    assert "block 0" in svg
    assert "pin 0" in svg
