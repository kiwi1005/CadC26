# 08 — Known Failure Modes

## Purpose

This page is the fast triage map for Sprint 2 Pivot regressions.
Use it when a run looks plausible but the metrics do not improve.

## Required previous steps

- Candidate generation, rollout, repair, and strict evaluation must all be wired into the pivot path.
- You should already know which mode failed: semantic, relaxed, repair, or strict.

## Pipeline role

```text
candidate generation -> semantic / relaxed rollout -> repair / finalizer -> strict evaluation
```

This page groups the most common failure modes by stage so you can inspect the right metric first.

## Key data / contracts

Look at these artifacts first:

- candidate coverage report
- semantic / relaxed rollout report
- repair report
- contest optimizer or strict evaluator summary

## Smoke commands

Use the smoke command that matches the failing stage. The following are historical pivot target runners kept as conceptual examples; verify current script availability before running them:

```bash
python scripts/check_candidate_coverage.py
python scripts/rollout_validate.py
python scripts/repair_validate.py
python scripts/evaluate_contest_optimizer.py
```

## Expected outputs

- Candidate coverage is high enough for semantic learning.
- Rollout completes for every validation case.
- Repair reduces infeasibility without erasing intent.
- Strict evaluation reports fallback use explicitly.

## Key metrics to inspect

- `semantic_candidate_coverage`
- `relaxed_candidate_coverage`
- `strict_candidate_coverage`
- `semantic_rollout_completion_rate`
- `avg_semantic_placed_fraction`
- `overlap_pair_count`
- `total_overlap_area`
- `repair_success_rate`
- `intent_preservation_rate`
- `final_fallback_fraction`
- `policy_contribution_ratio`

## Common failure modes

| Failure mode | What it usually means | First thing to inspect |
| --- | --- | --- |
| semantic candidate coverage low | semantic mode is pruning too aggressively | candidate masks and primitive normalization |
| relaxed candidate coverage low | relaxed mode still behaves like strict mode | overlap / area filters and catastrophic-action guards |
| strict candidate coverage low but semantic okay | the strict gate is too narrow for finalization | strict-only pruning and projection rules |
| rollout terminates after 1 block | no-progress logic or over-eager stopping | `semantic_placed` updates and `stopped_reason` |
| `SET_SHAPE` never followed by placement | shape setup is not feeding the action loop | candidate sequencing and executor state updates |
| `FREEZE` selected too early | the policy is locking anchors before it has enough structure | freeze policy, action scoring, and anchors |
| candidate exists but masked out | the candidate generator produced it, but the mask removed it | mask logic and candidate-to-mask alignment |
| executor applies action but `semantic_placed` not updated | state bookkeeping bug | executor state mutation and rollout bookkeeping |
| semantic rollout complete but impossible to repair | provisional layout is too entangled | overlap area, anchor placement, and shelf fallback |
| repair destroys all policy intent | finalizer behaves like silent packing | displacement and intent-preservation counters |
| fallback packing hides policy failure | final path is doing too much work | `final_fallback_fraction` and `policy_contribution_ratio` |
| official evaluator mismatch | local assumptions do not match contest rules | submission shape and evaluator contract |

## How to debug

1. Start with the first failing metric, not the final score.
2. Compare semantic, relaxed, repair, and strict reports side by side.
3. Check whether a mode is accidentally using strict pruning too early.
4. Confirm that intent-preservation metrics are not being dropped from the report.
5. If fallback is high, treat the policy as the likely blocker until proven otherwise.

## Next step

After you identify the stage, return to the matching doc:

- `docs/05_semantic_rollout.md`
- `docs/06_repair_finalizer.md`
- `docs/07_strict_evaluation.md`
