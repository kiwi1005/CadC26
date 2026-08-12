from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_hcfp_candidate_funnel import _classify_case  # noqa: E402


def _oracle(value: float):
    return {"uncapped_objective": value}


def _selected(value: float):
    return {"hard_feasible": math.isfinite(value), "uncapped_objective": value}


def test_candidate_funnel_classifies_generation_gap() -> None:
    result = _classify_case(
        {"raw": _oracle(12.0), "post_repair": _oracle(11.0)},
        _selected(11.0),
    )
    assert result["primary"] == "generation"


def test_candidate_funnel_prioritizes_repair_and_selection_gaps() -> None:
    repair = _classify_case(
        {"raw": _oracle(8.0), "post_repair": _oracle(12.0)},
        _selected(12.0),
    )
    selection = _classify_case(
        {"raw": _oracle(8.0), "post_repair": _oracle(7.0)},
        _selected(9.0),
    )
    assert repair["primary"] == "repair"
    assert selection["primary"] == "selection"
    assert selection["selection_regret"] == 2.0
