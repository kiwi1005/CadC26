# Step7F: Bounded Large-Scale Legalizer / Repair Radius Study

## Plan

Step7F follows the Step7E decision:

```text
pivot_to_large_scale_legalizer_repair
```

The study asks whether large/XL local alternatives become safe when repair is
bounded by geometry, region cells, graph hops, or macro components instead of
allowing an accidental global reshuffle.

## Scope

- sidecar only
- no contest runtime integration
- no finalizer semantics change
- no complete new global legalizer
- no Step7C iterative loop
- no RL / full NSGA-II
- no silent fallback to global repair

## Repair modes

- `current_repair_baseline`
- `geometry_window_repair`
- `region_cell_repair`
- `graph_hop_repair`
- `macro_component_repair`
- `cascade_capped_repair`
- `rollback_to_original`

Hard-invalid outputs do not enter the Pareto front. Radius-exceeded outputs are
marked with `repair_radius_exceeded` instead of being silently promoted.

## Required artifacts

- `artifacts/research/step7f_repair_candidates.json`
- `artifacts/research/step7f_bounded_repair_results.json`
- `artifacts/research/step7f_repair_radius_metrics.json`
- `artifacts/research/step7f_failure_attribution.json`
- `artifacts/research/step7f_pareto_repair_selection.json`
- `artifacts/research/step7f_decision.md`
- `artifacts/research/step7f_visualizations/`

## Decision options

- `promote_bounded_repair_to_step7c`
- `pivot_to_macro_level_legalizer`
- `pivot_to_region_replanner`
- `pivot_to_move_generation_constraints`
- `inconclusive_due_to_surrogate_or_trace_gap`
