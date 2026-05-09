# Step7C-thin: Route-Aware Layout Edit Loop

## Plan

Step7C-thin follows Step7H's decision:

```text
promote_route_aware_iteration_to_step7c
```

Step7H proved route-aware proxy descriptors can produce local, regional, macro,
and global candidate lanes. Step7C-thin tests the missing link: whether a small
subset of those lanes can be converted into actual rectangle layout edits while
preserving Step7G routing safety.

## Scope

- sidecar only
- no contest runtime integration
- no finalizer semantics change
- no RL / trained ML
- no full iterative optimization
- no silent discard of regional/macro/global candidates
- no sending global candidates to bounded-local repair
- original-inclusive Pareto reporting

## Implemented lanes

- local: `vacancy_aware_local_insertion`
- regional: `adjacent_region_reassignment`
- macro: `mib_group_closure_macro`
- global: `legacy_step7g_global_move` report-only baseline
- original: `original_layout`

The current implementation uses deterministic synthetic rectangle layouts sized
from Step7H case descriptors. This makes Step7C-thin a real geometry/edit test,
but still not a contest-runtime floorplanner.

## Required artifacts

- `artifacts/research/step7c_thin_layout_edit_candidates.json`
- `artifacts/research/step7c_thin_actual_routes.json`
- `artifacts/research/step7c_thin_feasibility_report.json`
- `artifacts/research/step7c_thin_pareto_report.json`
- `artifacts/research/step7c_thin_confusion_report.json`
- `artifacts/research/step7c_thin_decision.md`
- `artifacts/research/step7c_thin_visualizations/`

## Decision options

- `promote_to_step7c_iterative_loop`
- `refine_real_edit_generators`
- `pivot_to_coarse_region_planner`
- `pivot_to_macro_level_move_generator`
- `revisit_route_aware_proxy_assumptions`
- `inconclusive_due_to_real_edit_quality`


## Run result

Generated from `scripts/step7c_thin_run_route_aware_layout_edit_loop.py`.

- Descriptor candidates consumed: 72 from Step7H.
- Actual layout-edit candidates: 40.
- Actual routes: local 16, regional 8, macro 8, global 8.
- `actual_non_global_candidate_rate`: 0.8.
- `invalid_local_attempt_rate`: 0.0.
- `actual_hard_feasible_rate`: 0.8; the only hard-invalid candidates are the
  intentionally global report-only baselines.
- `actual_safe_improvement_count`: 24.
- `regional_macro_preservation_count`: 16.
- `global_report_only_count`: 8.
- Descriptor-to-actual route stability: 1.0; no descriptor collapsed to global.
- Original-inclusive Pareto front non-empty: 8 / 8 cases.

## Decision

```text
promote_to_step7c_iterative_loop
```

Reason: the route-aware proxy lanes survived conversion into deterministic real
rectangle edits in this sidecar setting. Local edits remain feasible under the
bounded-local lane, regional/macro edits are preserved in their reporting lanes,
global edits remain report-only and are not sent to bounded-local repair, and
original-inclusive Pareto reports remain non-empty.

Remaining caveat: these are synthetic rectangle edits sized from Step7H case
descriptors, not yet official FloorSet layout transformations. Step7C proper
should now implement the same route-aware loop on real case placements and
official-aligned metrics.
