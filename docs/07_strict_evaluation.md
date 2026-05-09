# 07 — Strict Evaluation

## Purpose

Strict evaluation is the final gate before submission.
It answers one question: can the repaired layout pass the official hard constraints without hiding policy failure behind fallback packing?

## Required previous steps

- Semantic rollout must have produced a provisional layout.
- Repair / finalizer must have produced a repaired layout and report.
- The contest path must be able to surface fallback usage explicitly.

## Pipeline role

```text
semantic rollout -> repair / finalizer -> strict hard check -> official evaluator -> submission
```

Strict evaluation should reject layouts that are not overlap-free or that violate hard dimension / area / fixed / preplaced constraints.
It should also report how much of the final answer came from policy versus fallback.

## Key data / contracts

### Final strict contract

```text
candidate_mode = strict
rollout_mode = strict
```

### Submission / report fields

```text
semantic_completed
semantic_placed_fraction
hard_feasible_before_repair
hard_feasible_after_repair
repair_displacement
repair_success
fallback_fraction
cost
hpwl_gap
area_gap
violations_relative
```

## Smoke commands

These are historical pivot smoke targets kept as conceptual examples for the strict gate; verify current script availability before running them.

### Pivot smoke target

```bash
python scripts/evaluate_contest_optimizer.py
```

## Expected outputs

- The repaired layout passes the hard check, or fallback is triggered explicitly.
- The summary reports whether the policy or the fallback produced the final answer.
- The evaluator output is consistent with the contest optimizer output.

## Key metrics to inspect

- `final_hard_feasible_rate`
- `cost`
- `hpwl_gap`
- `area_gap`
- `violations_relative`
- `final_fallback_fraction`
- `quick_validate`
- `num_feasible`
- `avg_cost`

## Common failure modes

- Semantic coverage is low but strict evaluation is blamed first.
- Repair returns a feasible layout internally, but the official evaluator rejects the submission format.
- Fallback packing hides that the policy never contributed.
- The evaluation report omits `final_fallback_fraction`.
- The optimizer uses target positions directly instead of predicted intent.

## How to debug

1. Compare the repair report to the official evaluator result.
2. Check whether the final layout shape matches the contest contract exactly.
3. Inspect fallback usage before reading the score.
4. Verify that the strict path uses the repaired layout, not the provisional one.
5. If possible, run the official evaluator on a minimal case before the full batch.

## Next step

If strict evaluation is stable, use `docs/08_known_failure_modes.md` to triage regressions quickly.
