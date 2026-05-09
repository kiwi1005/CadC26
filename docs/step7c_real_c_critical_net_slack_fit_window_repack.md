# Step7C-real-C Critical-Net Slack-Fit Window Repack

## Scope

Step7C-real-C is a sidecar-only research pipeline. It does **not** integrate with the contest runtime solver, does **not** change finalizer semantics, does **not** start multi-iteration search, and does **not** train or use RL/ML. The purpose is a one-step validation of whether small deterministic local-window repacks can improve official-like metrics more densely than Step7C-real-B single-block metric-directed edits.

The pipeline preserves the Step7G contract:

1. generate a real placement edit candidate from real FloorSet validation cases;
2. apply Step7G locality routing after the edit;
3. keep global candidates report-only and out of bounded-local repair;
4. run official-aligned hard feasibility and metric reporting where available;
5. include the original layout in every Pareto comparison.

## Research question

Can critical-net slack-fit window repacking increase official-like improvement density while preserving Step7G route safety and official-aligned feasibility?

Step7C-real-B showed route and feasibility safety were healthy but metric improvement density was still low:

- official-like cost improving candidates: 3;
- official-like cost improvement density: 0.0536;
- official-like hard feasible rate: 0.875;
- invalid local attempt rate: 0.0.

Step7C-real-C tests whether a bounded constructive edit around source/target slack windows improves that signal.

## Candidate strategies

Implemented in `src/puzzleplace/alternatives/window_repack_edits.py`:

- `original_layout`: original-inclusive Pareto anchor.
- `critical_net_slack_fit_window`: ranks movable blocks by HPWL/pin pressure, builds a nearest-neighbor local window, then tries seed nudge, window shift, local slot reassignment, and compatible local swaps.
- `bbox_owner_inward_window`: ranks blocks on the bounding-box extremes, builds local windows, and tries inward shifts/slot assignments while reporting HPWL and bbox tradeoffs separately.
- `balanced_window_swap`: selects pressure-ranked local windows and tries compatible footprint-preserving swaps and slot rotations.
- `macro_or_regional_nonnoop_probe`: uses MIB/cluster groups where present and attempts a bounded multi-block shift/slot assignment as a non-noop route-lane probe.
- `legacy_step7g_global_report`: broad report-only baseline; never routed to bounded-local repair.

Window construction is deterministic and JSON-serializable: artifacts record changed real block ids, window block ids, macro closure ids, slot assignments, construction status, internal trial count, feasible trial count, and selected internal official-like cost delta.

## Generated artifacts

The runner `scripts/step7c_real_c_run_window_repack_edits.py` writes:

- `artifacts/research/step7c_real_c_window_candidates.json`
- `artifacts/research/step7c_real_c_route_report.json`
- `artifacts/research/step7c_real_c_feasibility_report.json`
- `artifacts/research/step7c_real_c_metric_report.json`
- `artifacts/research/step7c_real_c_pareto_report.json`
- `artifacts/research/step7c_real_c_failure_attribution.json`
- `artifacts/research/step7c_real_c_strategy_ablation.json`
- `artifacts/research/step7c_real_c_decision.md`
- `artifacts/research/step7c_real_c_visualizations/`

## Current result

Latest full run on the eight Step7 validation cases produced:

- decision: `refine_window_repack_generator`
- real cases: 8
- window candidates: 48
- route counts: local 30, regional 2, macro 8, global 8
- non-global candidate rate: 0.8333
- invalid local attempt rate: 0.0
- official-like hard feasible rate: 0.8333
- original-inclusive Pareto non-empty: 8 / 8
- average non-global/non-original window block count: 4.44
- max non-global/non-original window block count: 7
- safe/proxy improving candidates: 6
- official-like cost improving candidates: 1
- official-like cost improvement density: 0.0250

## Interpretation

The Step7G safety layer still works: global candidates remain report-only, invalid local attempts remain zero, and Pareto coverage remains healthy. The window generator also creates non-global real edits and preserves macro/regional route lanes without route collapse.

However, official-like metric improvement did **not** exceed Step7C-real-B. The strongest strategy remains `critical_net_slack_fit_window`, which produced the only official-like cost improvement and four safe/proxy improvements, but most windows are still poor-targeting or metric-tradeoff failures. Bbox-owner windows often increase bbox area or soft pressure, and macro/regional probes are non-noop but not improving.

Therefore Step7C-real-C is not strong enough to promote to multi-iteration search. The next useful refinement is to improve window target/slot scoring and repack selection rather than widening to full iterative search.
