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
- Step7F visualization audit: OK. Arrow endpoint debug exists, block id matching
  is true, after-frame protrusion is measured, and raw max displacement remains
  explicitly visible.

## Decision options

- `promote_locality_routing_to_step7c`
- `pivot_to_coarse_region_planner`
- `pivot_to_macro_level_move_generator`
- `pivot_to_visualization_or_trace_repair`
- `inconclusive_due_to_prediction_quality`
