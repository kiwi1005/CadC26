#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _resolve_shared_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / 'AGENT_step4.md').exists() and (
            candidate / 'artifacts' / 'research'
        ).exists():
            return candidate
    return start


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = _resolve_shared_root(WORKTREE_ROOT)
RESEARCH = SHARED_ROOT / 'artifacts' / 'research'

EXPECTED_CASES = [
    'validation-14',
    'validation-18',
    'validation-17',
    'validation-15',
    'validation-11',
]
EXPECTED_BEST_UNTRAINED = '19.784'
EXPECTED_BEST_TRAINED = '23.651'
EXPECTED_ORACLE_MEAN_COST = 18.355
EXPECTED_PROXY_MEAN_COST = 19.316
EXPECTED_DISPLACEMENT_MEAN_COST = 37.174


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ''


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _extract_top5_cases(text: str) -> list[str]:
    return re.findall(r'validation-\d+', text)


def _close_enough(value: float, expected: float, *, tol: float = 1e-3) -> bool:
    return abs(value - expected) <= tol


def _best_of_k_checks(best_of_k_text: str, best_of_k_json: dict[str, Any] | None) -> dict[str, bool]:
    if not best_of_k_text or best_of_k_json is None:
        return {
            'best_of_k_has_required_sections': False,
            'best_of_k_json_has_required_keys': False,
            'best_of_k_aggregate_matches_expected_oracle': False,
            'best_of_k_aggregate_matches_expected_proxy': False,
            'best_of_k_aggregate_matches_expected_displacement': False,
        }

    aggregate = best_of_k_json.get('aggregate_by_rule', {})
    oracle = aggregate.get('oracle_best', {})
    proxy = aggregate.get('hpwl_bbox_proxy_best', {})
    displacement = aggregate.get('displacement_best', {})
    required_sections = [
        '## Executive Summary',
        '## Per-case Oracle Table',
        '## Aggregate Oracle Table',
        '## Scorer Comparison',
        '## Decision Recommendation',
    ]
    required_keys = {
        'per_case_candidates',
        'per_case_best_by_rule',
        'aggregate_by_rule',
        'oracle_vs_baseline',
        'final_recommendation',
        'commands_used',
    }
    return {
        'best_of_k_has_required_sections': _contains_all(best_of_k_text, required_sections),
        'best_of_k_json_has_required_keys': required_keys.issubset(best_of_k_json.keys()),
        'best_of_k_aggregate_matches_expected_oracle': _close_enough(
            float(oracle.get('mean_cost', 0.0)), EXPECTED_ORACLE_MEAN_COST
        ),
        'best_of_k_aggregate_matches_expected_proxy': _close_enough(
            float(proxy.get('mean_cost', 0.0)), EXPECTED_PROXY_MEAN_COST
        ),
        'best_of_k_aggregate_matches_expected_displacement': _close_enough(
            float(displacement.get('mean_cost', 0.0)), EXPECTED_DISPLACEMENT_MEAN_COST
        ),
    }


def main() -> None:
    agent = _read(SHARED_ROOT / 'AGENT_step4.md')
    followup = _read(RESEARCH / 'generalization_followup_smallcheckpoints.md')
    delta = _read(RESEARCH / 'cost_semantics_and_trained_vs_untrained_delta.md')
    top5 = _read(RESEARCH / 'top5_loss_drift_audit.md')
    best_of_k = _read(RESEARCH / 'best_of_k_oracle_reranking.md')
    best_of_k_json = _read_json(RESEARCH / 'best_of_k_oracle_reranking.json')

    checks = {
        'top5_artifact_exists': bool(top5),
        'best_of_k_artifact_exists': bool(best_of_k),
        'best_of_k_json_exists': best_of_k_json is not None,
        'followup_best_costs_match_expected': _contains_all(
            followup,
            [EXPECTED_BEST_UNTRAINED, EXPECTED_BEST_TRAINED],
        ),
        'delta_top5_cases_match_expected': _contains_all(delta, EXPECTED_CASES),
        'top5_best_costs_match_expected': _contains_all(
            top5,
            [EXPECTED_BEST_UNTRAINED, EXPECTED_BEST_TRAINED],
        ),
        'top5_cases_match_expected': _contains_all(top5, EXPECTED_CASES),
        'agent_requests_best_of_k': 'best_of_k_oracle_reranking' in agent,
    }
    checks.update(_best_of_k_checks(best_of_k, best_of_k_json))

    top5_cases_in_doc = [case for case in EXPECTED_CASES if case in top5]
    overall_status = 'pass'
    required_top5 = (
        'top5_artifact_exists',
        'followup_best_costs_match_expected',
        'delta_top5_cases_match_expected',
        'top5_best_costs_match_expected',
        'top5_cases_match_expected',
    )
    if not all(checks[key] for key in required_top5):
        overall_status = 'fail'
    elif not checks['best_of_k_artifact_exists'] or not checks['best_of_k_json_exists']:
        overall_status = 'partial'
    elif not all(
        checks[key]
        for key in (
            'best_of_k_has_required_sections',
            'best_of_k_json_has_required_keys',
            'best_of_k_aggregate_matches_expected_oracle',
            'best_of_k_aggregate_matches_expected_proxy',
            'best_of_k_aggregate_matches_expected_displacement',
        )
    ):
        overall_status = 'fail'

    payload = {
        'shared_root': str(SHARED_ROOT),
        'artifacts_checked': {
            'followup': str(
                (RESEARCH / 'generalization_followup_smallcheckpoints.md').relative_to(
                    SHARED_ROOT
                )
            ),
            'delta': str(
                (RESEARCH / 'cost_semantics_and_trained_vs_untrained_delta.md').relative_to(
                    SHARED_ROOT
                )
            ),
            'top5': str((RESEARCH / 'top5_loss_drift_audit.md').relative_to(SHARED_ROOT)),
            'best_of_k_md': str(
                (RESEARCH / 'best_of_k_oracle_reranking.md').relative_to(SHARED_ROOT)
            ),
            'best_of_k_json': str(
                (RESEARCH / 'best_of_k_oracle_reranking.json').relative_to(SHARED_ROOT)
            ),
        },
        'expected': {
            'best_untrained_mean_cost': EXPECTED_BEST_UNTRAINED,
            'best_trained_mean_cost': EXPECTED_BEST_TRAINED,
            'top5_cases': EXPECTED_CASES,
            'oracle_best_mean_cost': EXPECTED_ORACLE_MEAN_COST,
            'hpwl_bbox_proxy_mean_cost': EXPECTED_PROXY_MEAN_COST,
            'displacement_best_mean_cost': EXPECTED_DISPLACEMENT_MEAN_COST,
        },
        'observed': {
            'top5_cases_present_in_doc': top5_cases_in_doc,
            'top5_cases_found_anywhere_in_doc_order': _extract_top5_cases(top5),
            'best_of_k_rules_present': sorted(
                best_of_k_json.get('aggregate_by_rule', {}).keys()
            )
            if best_of_k_json is not None
            else [],
        },
        'checks': checks,
        'overall_status': overall_status,
        'summary': {
            'top5_validation': (
                'consistent with AGENT_step4.md and tracked reports'
                if overall_status != 'fail' or checks['top5_artifact_exists']
                else 'inconsistent'
            ),
            'best_of_k_validation': (
                'pending artifact generation'
                if not checks['best_of_k_artifact_exists'] or not checks['best_of_k_json_exists']
                else 'artifact present and validated'
            ),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
