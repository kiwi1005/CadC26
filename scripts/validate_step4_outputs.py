#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


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


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ''


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _extract_top5_cases(text: str) -> list[str]:
    return re.findall(r'validation-\d+', text)


def main() -> None:
    agent = _read(SHARED_ROOT / 'AGENT_step4.md')
    followup = _read(RESEARCH / 'generalization_followup_smallcheckpoints.md')
    delta = _read(RESEARCH / 'cost_semantics_and_trained_vs_untrained_delta.md')
    top5 = _read(RESEARCH / 'top5_loss_drift_audit.md')
    best_of_k = _read(RESEARCH / 'best_of_k_oracle_reranking.md')

    checks = {
        'top5_artifact_exists': bool(top5),
        'best_of_k_artifact_exists': bool(best_of_k),
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

    top5_cases_in_doc = [case for case in EXPECTED_CASES if case in top5]
    overall_status = 'pass'
    if not all(
        checks[key]
        for key in (
            'top5_artifact_exists',
            'followup_best_costs_match_expected',
            'delta_top5_cases_match_expected',
            'top5_best_costs_match_expected',
            'top5_cases_match_expected',
        )
    ):
        overall_status = 'fail'
    elif not checks['best_of_k_artifact_exists']:
        overall_status = 'partial'

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
            'best_of_k': str(
                (RESEARCH / 'best_of_k_oracle_reranking.md').relative_to(SHARED_ROOT)
            ),
        },
        'expected': {
            'best_untrained_mean_cost': EXPECTED_BEST_UNTRAINED,
            'best_trained_mean_cost': EXPECTED_BEST_TRAINED,
            'top5_cases': EXPECTED_CASES,
        },
        'observed': {
            'top5_cases_present_in_doc': top5_cases_in_doc,
            'top5_cases_found_anywhere_in_doc_order': _extract_top5_cases(top5),
        },
        'checks': checks,
        'overall_status': overall_status,
        'summary': {
            'top5_validation': (
                'consistent with AGENT_step4.md and tracked reports'
                if overall_status != 'fail'
                else 'inconsistent'
            ),
            'best_of_k_validation': (
                'pending artifact generation'
                if not checks['best_of_k_artifact_exists']
                else 'artifact present'
            ),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
