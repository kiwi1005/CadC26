from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_two_stage_reranker.py"
_SPEC = importlib.util.spec_from_file_location("run_two_stage_reranker", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["run_two_stage_reranker"] = _MODULE
_SPEC.loader.exec_module(_MODULE)

normalized_feature_matrix = _MODULE.normalized_feature_matrix
proxy_order = _MODULE.proxy_order
train_pairwise_ranker = _MODULE.train_pairwise_ranker
_select_by_ranker = _MODULE._select_by_ranker


def _row(candidate_id: str, *, hpwl: float, bbox: float, cost: float):
    return {
        "candidate_id": candidate_id,
        "kind": "test",
        "seed": None,
        "checkpoint": None,
        "proxy_features": {
            "proxy_hpwl_total": hpwl,
            "bbox_area": bbox,
            "proxy_total_soft_violations": 0.0,
            "changed_block_fraction": 0.0,
            "shelf_fallback_count": 0.0,
            "mean_displacement": 0.0,
            "semantic_fallback_fraction": 0.0,
            "whitespace_ratio": 0.0,
        },
        "analysis_metrics": {"official_cost": cost},
    }


def test_normalized_features_are_pool_local() -> None:
    rows = [_row("a", hpwl=10.0, bbox=100.0, cost=1.0), _row("b", hpwl=20.0, bbox=300.0, cost=2.0)]
    features = normalized_feature_matrix(rows)
    assert features[0, 0].item() == 0.0
    assert features[1, 0].item() == 1.0
    assert features[0, 1].item() == 0.0
    assert features[1, 1].item() == 1.0


def test_proxy_order_uses_stage1_scorer() -> None:
    rows = [
        _row("bad", hpwl=10.0, bbox=100.0, cost=1.0),
        _row("good", hpwl=0.0, bbox=10.0, cost=2.0),
    ]
    assert [rows[idx]["candidate_id"] for idx in proxy_order(rows, "hpwl_bbox_proxy")] == [
        "good",
        "bad",
    ]


def test_pairwise_ranker_can_learn_oracle_reordering_inside_top_m() -> None:
    train_cases = {
        "case-a": [
            _row("proxy_best", hpwl=0.0, bbox=0.0, cost=5.0),
            _row("oracle", hpwl=0.5, bbox=0.0, cost=1.0),
        ],
        "case-b": [
            _row("proxy_best", hpwl=0.0, bbox=0.0, cost=6.0),
            _row("oracle", hpwl=0.5, bbox=0.0, cost=2.0),
        ],
    }
    ranker = train_pairwise_ranker(
        train_cases,
        scorer_name="hpwl_bbox_proxy",
        top_m=2,
        epochs=100,
        lr=0.1,
        weight_decay=0.0,
    )
    rows = [
        _row("proxy_best", hpwl=0.0, bbox=0.0, cost=5.0),
        _row("oracle", hpwl=0.5, bbox=0.0, cost=1.0),
    ]
    selected_idx, _shortlist, _scores = _select_by_ranker(
        rows, scorer_name="hpwl_bbox_proxy", top_m=2, ranker=ranker
    )
    assert rows[selected_idx]["candidate_id"] == "oracle"
