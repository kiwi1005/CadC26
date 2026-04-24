from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cost_aware_reranker.py"
_SPEC = importlib.util.spec_from_file_location("run_cost_aware_reranker", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_candidate_specs = _MODULE.build_candidate_specs
score_candidate_rows = _MODULE.score_candidate_rows
summarize_scorer_for_case = _MODULE.summarize_scorer_for_case


def _row(
    candidate_id: str,
    *,
    hpwl: float,
    bbox: float,
    soft: float,
    changed: float,
    displacement: float,
    cost: float,
):
    return {
        "candidate_id": candidate_id,
        "kind": "test",
        "seed": None,
        "checkpoint": None,
        "proxy_features": {
            "proxy_hpwl_total": hpwl,
            "bbox_area": bbox,
            "proxy_total_soft_violations": soft,
            "changed_block_fraction": changed,
            "shelf_fallback_count": 0.0,
            "semantic_fallback_fraction": 0.0,
            "mean_displacement": displacement,
        },
        "analysis_metrics": {
            "official_cost": cost,
            "hpwl_gap": hpwl,
            "area_gap": bbox,
            "violations_relative": soft,
            "runtime_factor": None,
            "is_feasible": True,
        },
    }


def test_k16_candidate_specs_match_step5_v0_contract() -> None:
    specs = build_candidate_specs(16)
    assert [spec["candidate_id"] for spec in specs] == [
        "heuristic",
        "untrained_seed0",
        "untrained_seed1",
        "untrained_seed2",
        "untrained_seed3",
        "untrained_seed4",
        "untrained_seed5",
        "untrained_seed6",
        "untrained_seed7",
        "untrained_seed8",
        "untrained_seed9",
        "untrained_seed10",
        "bc_seed0",
        "bc_seed1",
        "awbc_seed0",
        "awbc_seed1",
    ]


def test_proxy_scorer_weights_soft_and_repair_terms() -> None:
    rows = [
        _row("a", hpwl=0.0, bbox=10.0, soft=0.0, changed=0.0, displacement=5.0, cost=5.0),
        _row("b", hpwl=1.0, bbox=20.0, soft=4.0, changed=2.0, displacement=1.0, cost=1.0),
    ]
    assert score_candidate_rows(rows, "hpwl_bbox_proxy") == [0.0, 2.0]
    assert score_candidate_rows(rows, "hpwl_bbox_soft_proxy") == [0.0, 2.25]
    assert score_candidate_rows(rows, "hpwl_bbox_soft_repair_proxy") == [0.0, 2.30]
    assert score_candidate_rows(rows, "displacement_proxy") == [1.0, 0.0]
    assert score_candidate_rows(rows, "oracle_official_cost") == [5.0, 1.0]


def test_top_m_recall_reports_oracle_rank_by_scorer() -> None:
    rows = [
        _row("proxy_best", hpwl=0.0, bbox=0.0, soft=0.0, changed=0.0, displacement=2.0, cost=10.0),
        _row("middle", hpwl=1.0, bbox=0.0, soft=0.0, changed=0.0, displacement=1.0, cost=8.0),
        _row("oracle", hpwl=2.0, bbox=0.0, soft=0.0, changed=0.0, displacement=0.0, cost=1.0),
    ]
    summary, recall, top_order = summarize_scorer_for_case(
        case_id="case-0",
        rows=rows,
        scorer_name="hpwl_bbox_proxy",
        best_untrained_cost=9.0,
        best_trained_cost=11.0,
        top_m_values=[1, 2, 3],
    )
    assert summary["selected_candidate_id"] == "proxy_best"
    assert summary["oracle_candidate_id"] == "oracle"
    assert summary["oracle_rank_by_scorer"] == 3
    assert recall == {1: False, 2: False, 3: True}
    assert top_order == ["proxy_best", "middle", "oracle"]
