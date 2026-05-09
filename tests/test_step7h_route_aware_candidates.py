from __future__ import annotations

from puzzleplace.alternatives.route_aware_candidates import (
    candidate_diversity_report,
    decision_for_step7h,
    generate_route_aware_candidates,
    pareto_report,
    predict_candidates,
    synthetic_probe_candidates,
    synthetic_probe_report,
)


def test_route_aware_candidates_produce_non_global_diversity() -> None:
    candidates = generate_route_aware_candidates(
        case_ids=[1], block_counts={1: 40}, min_slack={1: 10.0}
    )
    predictions = predict_candidates(candidates)
    diversity = candidate_diversity_report(predictions)

    assert diversity["candidate_count_by_class"]["local"] >= 1
    assert diversity["candidate_count_by_class"]["regional"] >= 1
    assert diversity["candidate_count_by_class"]["macro"] >= 1
    assert diversity["candidate_count_by_class"]["global"] >= 1
    assert diversity["non_global_candidate_rate"] > 0.0
    assert diversity["invalid_local_attempt_rate"] == 0.0


def test_synthetic_probes_cover_all_router_classes() -> None:
    report = synthetic_probe_report(predict_candidates(synthetic_probe_candidates()))

    assert report["pass_count"] == 4
    assert report["router_class_confusion"] == []


def test_pareto_report_is_original_inclusive_and_preserves_step7g_safe_cases() -> None:
    predictions = predict_candidates(
        generate_route_aware_candidates(case_ids=[19], block_counts={19: 40}, min_slack={19: 8.0})
    )
    report = pareto_report(predictions, preserved_step7g_safe_cases=[19])

    assert report["per_case"]["19"]["original_included"] is True
    assert report["pareto_front_non_empty_count"] == 1
    assert report["useful_regional_macro_candidate_count"] >= 1
    assert report["safe_improvement_preservation"]["lost_cases"] == []


def test_decision_promotes_when_diverse_and_safe() -> None:
    predictions = predict_candidates(
        generate_route_aware_candidates(case_ids=[1], block_counts={1: 40}, min_slack={1: 10.0})
    )
    diversity = candidate_diversity_report(predictions)
    pareto = pareto_report(predictions, preserved_step7g_safe_cases=[1])
    synthetic = synthetic_probe_report(predict_candidates(synthetic_probe_candidates()))

    assert decision_for_step7h(diversity, pareto, synthetic) == (
        "promote_route_aware_iteration_to_step7c"
    )
