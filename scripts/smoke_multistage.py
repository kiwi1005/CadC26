"""Smoke-test the multi-stage active-soft processor on case 79."""

from __future__ import annotations

import json
import time
from pathlib import Path

from puzzleplace.data.floorset_adapter import adapt_validation_batch
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets
from puzzleplace.repair.multistage_active_soft import multistage_active_soft_postprocess
from puzzleplace.repair.active_soft_postprocess import active_soft_postprocess

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts" / "research"


def run_comparison(case_id: int = 79):
    print(f"Loading case {case_id}...")
    cases = load_validation_cases(ROOT, [case_id])
    case = cases[case_id]

    # Use target positions as baseline
    positions = positions_from_case_targets(case)
    before = evaluate_positions(case, positions, runtime=1.0)
    print(f"  Baseline quality: cost={before['quality']['cost']:.4f}, "
          f"feasible={before['quality']['feasible']}")

    # Run single-stage (existing)
    print("\n--- Single-stage active_soft_postprocess ---")
    t0 = time.perf_counter()
    ss_positions, ss_report = active_soft_postprocess(case, positions)
    ss_time = time.perf_counter() - t0
    ss_after = evaluate_positions(case, ss_positions, runtime=1.0)
    print(f"  Applied: {ss_report['active_soft_applied']}")
    print(f"  Candidates evaluated: {ss_report['active_soft_candidates_evaluated']}")
    print(f"  Strict winners: {ss_report['active_soft_strict_winners']}")
    print(f"  Time: {ss_time:.2f}s")
    print(f"  After quality: cost={ss_after['quality']['cost']:.4f}")

    # Run multi-stage (new)
    print("\n--- Multi-stage multistage_active_soft_postprocess ---")
    t0 = time.perf_counter()
    ms_positions, ms_report = multistage_active_soft_postprocess(case, positions)
    ms_time = time.perf_counter() - t0
    ms_after = evaluate_positions(case, ms_positions, runtime=1.0)
    print(f"  Applied: {ms_report['multistage_applied']}")
    print(f"  Winning stage: {ms_report.get('multistage_winning_stage', 'none')}")
    print(f"  Stages run: {ms_report.get('multistage_stages_run', [])}")
    for key in sorted(ms_report):
        if 'stage' in key and key != 'multistage_stages_run':
            print(f"  {key}: {ms_report[key]}")
    print(f"  Time: {ms_time:.2f}s")
    print(f"  After quality: cost={ms_after['quality']['cost']:.4f}")

    # Compare
    print("\n--- Comparison ---")
    ss_cost = ss_after['quality']['cost']
    ms_cost = ms_after['quality']['cost']
    orig_cost = before['quality']['cost']
    print(f"  Original cost: {orig_cost:.4f}")
    print(f"  Single-stage cost: {ss_cost:.4f} (delta: {ss_cost - orig_cost:+.6f})")
    print(f"  Multi-stage cost:  {ms_cost:.4f} (delta: {ms_cost - orig_cost:+.6f})")

    return {
        "single_stage": ss_report,
        "multi_stage": ms_report,
    }


if __name__ == "__main__":
    results = run_comparison(79)
    # Also try other cases
    for cid in [24, 51, 76]:
        print(f"\n{'='*60}")
        try:
            run_comparison(cid)
        except Exception as e:
            print(f"  Error on case {cid}: {e}")
