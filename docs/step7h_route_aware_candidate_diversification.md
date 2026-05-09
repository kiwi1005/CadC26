# Step7H: Route-Aware Candidate Diversification

## Plan

Step7H follows Step7G's decision:

```text
promote_locality_routing_to_step7c
```

Step7G proved the router can prevent globally disruptive candidates from being
sent to bounded-local repair, but it also showed the existing Step7F candidate
surface was all-global. Step7H therefore tests whether a deterministic sidecar
candidate surface can produce local, regional, macro, and global lanes while
keeping the Step7G router as the safety layer.

## Scope

- sidecar only
- no contest runtime integration
- no finalizer semantics change
- no RL / trained ML
- no hard discard gate from locality routing
- no silent dropping of regional, macro, or global moves
- preserve original layout as an explicit candidate

## Candidate families

The minimal deterministic set is:

- `original_layout`
- `vacancy_aware_local_insertion`
- `connected_slack_hole_move`
- `adjacent_region_reassignment`
- `occupancy_balanced_block_swap`
- `mib_group_closure_macro`
- `staged_regional_decomposition`
- `pin_community_guided_region_relocation`
- `legacy_step7g_global_move`

These candidates are proxy/sidecar descriptors, not contest runtime layout
edits. They intentionally test routing class diversity before a Step7C geometry
loop consumes them.

## Required artifacts

- `artifacts/research/step7h_candidate_diversity.json`
- `artifacts/research/step7h_route_predictions.json`
- `artifacts/research/step7h_pareto_report.json`
- `artifacts/research/step7h_synthetic_probe_report.json`
- `artifacts/research/step7h_decision.md`
- `artifacts/research/step7h_visualizations/`

## Decision options

- `promote_route_aware_iteration_to_step7c`
- `pivot_to_coarse_region_planner`
- `pivot_to_macro_level_move_generator`
- `refine_candidate_generation_more`
- `revisit_locality_router_calibration`
- `inconclusive_due_to_candidate_quality`


## Run result

Generated from `scripts/step7h_run_route_aware_candidate_diversification.py`.

- Candidate count: 72 across 8 Step7G cases.
- Class counts: local 32, regional 24, macro 8, global 8.
- `non_global_candidate_rate`: 0.8889.
- `local_candidate_rate`: 0.4444.
- `regional_candidate_rate`: 0.3333.
- `macro_candidate_rate`: 0.1111.
- `global_candidate_rate`: 0.1111.
- `invalid_local_attempt_rate`: 0.0.
- Pareto front non-empty count: 8 / 8 cases.
- Useful regional/macro candidate count: 32.
- Step7G safe-improvement preservation: 3 / 3 cases preserved (`19`, `24`, `25`).
- Synthetic controlled probes: 4 / 4 pass; no under-predicted or over-predicted globality.
- Router-too-conservative cases: none in this proxy run.

## Decision

```text
promote_route_aware_iteration_to_step7c
```

Reason: the all-global Step7G candidate-surface failure is not inherent to the
router. A deterministic route-aware proxy surface can produce local, regional,
and macro candidates while keeping global candidates out of bounded-local repair.
This is not yet proof of final solver quality; Step7C must turn these proxy lanes
into actual layout edits and verify legality/quality after repair.
