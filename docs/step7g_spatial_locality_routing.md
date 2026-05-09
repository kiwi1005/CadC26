# Step7G: Spatial Locality Map and Move Routing

## Plan

Step7G follows the Step7F decision:

```text
pivot_to_move_generation_constraints
```

The goal is to classify candidate moves before repair as local, regional,
macro, or global, then route them to the right downstream path. This remains a
sidecar study and does not add a hard runtime gate.

## Scope

- sidecar only
- no contest runtime integration
- no RL / full NSGA-II
- no new region placer
- no hard locality gate
- no direct Step7C iterative loop
- preserve original layout as an alternative

## Outputs

- `artifacts/research/step7g_locality_maps.json`
- `artifacts/research/step7g_move_locality_predictions.json`
- `artifacts/research/step7g_routing_results.json`
- `artifacts/research/step7g_calibration_report.json`
- `artifacts/research/step7g_visualization_audit.json`
- `artifacts/research/step7g_decision.md`
- `artifacts/research/step7g_visualizations/`

## Implementation notes

- `src/puzzleplace/diagnostics/spatial_locality.py` builds sidecar locality
  maps at coarse and adaptive resolutions. Each resolution emits the required
  channels: occupancy, free space, fixed/preplaced, pin density, net-community
  demand, slack, hole fragmentation, boundary ownership, MIB/group closure, and
  repair reachability.
- `src/puzzleplace/alternatives/locality_routing.py` converts candidate impact
  estimates into local / regional / macro / global classes. The output is a
  route recommendation, not a rejection gate.
- `scripts/step7g_run_spatial_locality_routing.py` reuses Step7E locality
  artifacts and Step7F weak labels so the replay remains sidecar-only and does
  not recompute or mutate contest runtime behavior.
- Visualization output is grid-based PNG heatmaps plus an HTML index. The
  Step7F arrow debug audit is carried forward to prevent unreadable autoscale
  regressions from being mistaken for algorithm behavior.

## Replay result

- Candidate coverage: 8 Step7F candidate cases.
- Map coverage: 9 Step7E locality cases; case002 has maps but no Step7F move
  candidate in this replay.
- Prediction calibration: 8/8 correct global against Step7F weak labels.
- Under-predicted globality: 0.
- Routing result: all 8 candidates route to
  `global_route_not_local_selector`, which means Step7C should promote
  locality-aware routing before trying bounded local repair as the selector
  default.
- Invalid local-repair attempt rate falls from `1.0` before routing to `0.0`
  after routing because no candidate is sent to `bounded_repair_pareto`.
- Safe improvements are preserved in reporting/non-local follow-up:
  cases `19`, `24`, and `25` still have non-empty Pareto fronts and original
  rollback candidates. They would be lost only if a downstream consumer treated
  non-local routing as rejection, which Step7G explicitly forbids.
- Move-source diversity is retained inside the global route:
  `posthoc_shape_probe:role_aware_cap` (5),
  `construction_shape_policy_replay:group_macro_aspect_regularized` (2), and
  `construction_shape_policy_replay:role_aware_cap` (1).
- Step7F visualization audit: OK with `trace_confidence = reconstructed`.
  Arrow endpoint debug exists, drawn endpoints equal raw after-block centers,
  block id matching is true, raw distances match raw before/after centers,
  after-frame protrusion is measured, suspicious case099/case091 PNGs exist,
  and drawn arrows are clipped/normalized so the plot is not autoscaled into
  unreadability. The exact construction trace is not present, so the audit does
  not overclaim exact trace provenance.

## Decision options

- `promote_locality_routing_to_step7c`
- `pivot_to_coarse_region_planner`
- `pivot_to_macro_level_move_generator`
- `pivot_to_visualization_or_trace_repair`
- `inconclusive_due_to_prediction_quality`

## Decision

```text
promote_locality_routing_to_step7c
```

Reason: locality prediction is calibrated on the available Step7F weak labels
(`correct_global = 8`, no under-predicted globality), it removes invalid
bounded-local repair attempts while preserving safe improvements and non-empty
Pareto fronts in the non-local reporting path, and the visualization/trace audit
is trustworthy enough for routing calibration when labeled as reconstructed
rather than exact.
