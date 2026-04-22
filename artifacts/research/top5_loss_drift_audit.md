# Top-5 Loss Drift Audit

## Executive Summary
- This audit is a **tracked-source reconstruction**: it uses the current research reports plus tracked code inspection because stage-A..E snapshots were not persisted in the current artifacts.
- The top-5 trained loss cases remain `validation-14/18/17/15/11`, matching `AGENT_step4.md` and the current cost-delta report.
- In 3/5 cases (`validation-14/18/17`), trained `awbc seed 1` has **higher final cost with lower mean repair displacement**, which is strong evidence that lower displacement is not rescuing poorer HPWL/bbox structure.
- The tracked finalizer (`normalize_shapes -> resolve_overlaps -> shelf_pack_missing -> resolve_overlaps`) is legality-oriented and not HPWL/bbox-aware, so the current evidence points more to **proposal/objective mismatch** than to a cost-improving repair stage.

## Consistency Check
- `AGENT_step4.md` best untrained mean cost: `19.784`; current report: `19.784`.
- `AGENT_step4.md` best trained mean cost: `23.651`; current report: `23.651`.
- Top-5 case ids match exactly: `validation-14, validation-18, validation-17, validation-15, validation-11`.

## Top-5 Case Table (best untrained vs best trained)

| Case | Untrained cost | Trained cost | Δ cost | Δ HPWL gap | Δ bbox gap | Δ soft viol. | Δ mean disp. | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| validation-14 | 30.544 | 47.091 | 16.547 | 1.659 | 3.870 | -0.143 | -1.200 | finalizer_preserves_bad_structure |
| validation-18 | 24.310 | 38.190 | 13.880 | 1.105 | 1.784 | 0.000 | -1.436 | finalizer_preserves_bad_structure |
| validation-17 | 29.531 | 41.021 | 11.489 | 0.706 | 1.559 | 0.000 | -6.211 | finalizer_preserves_bad_structure |
| validation-15 | 19.489 | 28.053 | 8.564 | 0.082 | 0.158 | 0.156 | 12.361 | soft_violation_driver |
| validation-11 | 21.555 | 29.848 | 8.293 | 0.405 | 0.580 | 0.071 | 12.188 | proposal_bbox_bad |

## Stage Coverage
- **Observed:** Stage F final official metrics from `generalization_followup_smallcheckpoints.json` and `cost_semantics_and_trained_vs_untrained_delta.json`.
- **Missing:** Stage A semantic proposal, Stage B shape-normalized, Stage C overlap-resolved, Stage D shelf/packing intermediate, and Stage E final coordinates. The code path exists, but current artifacts do not persist those snapshots.
- **Action traces:** also missing from tracked artifacts, so primitive histograms / placement-order diagnostics cannot yet be proved from data.

## Per-Case Diagnosis
### validation-14
- Diagnosis: `finalizer_preserves_bad_structure`
- trained final quality is worse on both HPWL and bbox even though repair displacement is not higher.
- tracked finalizer code is legality-oriented and does not optimize HPWL/bbox, so it is likely preserving poor proposal geometry rather than correcting it.
- Final deltas (trained - untrained): cost `16.547`, HPWL gap `1.659`, bbox gap `3.870`, soft violations `-0.143`, mean displacement `-1.200`.
- Additional trained variants on this case:

| Variant | Cost | HPWL gap | bbox gap | Violations rel. | Mean disp. |
| --- | ---: | ---: | ---: | ---: | ---: |
| untrained seed 0 | 30.544 | 1.489 | 1.923 | 0.857 | 14.400 |
| bc seed 0 | 56.610 | 4.301 | 7.959 | 0.679 | 5.743 |
| bc seed 1 | 54.445 | 3.781 | 6.092 | 0.750 | 8.914 |
| awbc seed 0 | 66.026 | 5.497 | 10.379 | 0.643 | 0.686 |
| awbc seed 1 | 47.091 | 3.148 | 5.793 | 0.714 | 13.200 |

### validation-18
- Diagnosis: `finalizer_preserves_bad_structure`
- trained final quality is worse on both HPWL and bbox even though repair displacement is not higher.
- tracked finalizer code is legality-oriented and does not optimize HPWL/bbox, so it is likely preserving poor proposal geometry rather than correcting it.
- Final deltas (trained - untrained): cost `13.880`, HPWL gap `1.105`, bbox gap `1.784`, soft violations `0.000`, mean displacement `-1.436`.
- Additional trained variants on this case:

| Variant | Cost | HPWL gap | bbox gap | Violations rel. | Mean disp. |
| --- | ---: | ---: | ---: | ---: | ---: |
| untrained seed 0 | 24.310 | 1.701 | 1.578 | 0.767 | 17.282 |
| bc seed 0 | 93.201 | 7.194 | 11.134 | 0.767 | 3.179 |
| bc seed 1 | 32.774 | 2.366 | 2.240 | 0.800 | 22.128 |
| awbc seed 0 | 92.741 | 7.339 | 13.785 | 0.700 | 1.308 |
| awbc seed 1 | 38.190 | 2.807 | 3.362 | 0.767 | 15.846 |

### validation-17
- Diagnosis: `finalizer_preserves_bad_structure`
- trained final quality is worse on both HPWL and bbox even though repair displacement is not higher.
- tracked finalizer code is legality-oriented and does not optimize HPWL/bbox, so it is likely preserving poor proposal geometry rather than correcting it.
- Final deltas (trained - untrained): cost `11.489`, HPWL gap `0.706`, bbox gap `1.559`, soft violations `0.000`, mean displacement `-6.211`.
- Additional trained variants on this case:

| Variant | Cost | HPWL gap | bbox gap | Violations rel. | Mean disp. |
| --- | ---: | ---: | ---: | ---: | ---: |
| untrained seed 0 | 29.531 | 1.643 | 2.288 | 0.818 | 20.026 |
| bc seed 0 | 62.382 | 5.292 | 9.458 | 0.667 | 2.368 |
| bc seed 1 | 40.257 | 2.455 | 3.076 | 0.848 | 20.211 |
| awbc seed 0 | 66.411 | 4.776 | 10.134 | 0.697 | 3.474 |
| awbc seed 1 | 41.021 | 2.349 | 3.846 | 0.818 | 13.816 |

### validation-15
- Diagnosis: `soft_violation_driver`
- soft-violation growth dominates while HPWL/bbox deltas stay comparatively small.
- Final deltas (trained - untrained): cost `8.564`, HPWL gap `0.082`, bbox gap `0.158`, soft violations `0.156`, mean displacement `12.361`.
- Additional trained variants on this case:

| Variant | Cost | HPWL gap | bbox gap | Violations rel. | Mean disp. |
| --- | ---: | ---: | ---: | ---: | ---: |
| untrained seed 0 | 19.489 | 1.536 | 1.451 | 0.656 | 26.222 |
| bc seed 0 | 33.289 | 1.990 | 3.555 | 0.719 | 15.639 |
| bc seed 1 | 25.849 | 1.772 | 1.392 | 0.781 | 45.583 |
| awbc seed 0 | 45.510 | 2.810 | 6.176 | 0.688 | 5.583 |
| awbc seed 1 | 28.053 | 1.618 | 1.609 | 0.812 | 38.583 |

### validation-11
- Diagnosis: `proposal_bbox_bad`
- bbox-area degradation is materially larger than the HPWL delta.
- Final deltas (trained - untrained): cost `8.293`, HPWL gap `0.405`, bbox gap `0.580`, soft violations `0.071`, mean displacement `12.188`.
- Additional trained variants on this case:

| Variant | Cost | HPWL gap | bbox gap | Violations rel. | Mean disp. |
| --- | ---: | ---: | ---: | ---: | ---: |
| untrained seed 0 | 21.555 | 1.895 | 1.606 | 0.786 | 19.312 |
| bc seed 0 | 46.276 | 3.476 | 6.330 | 0.786 | 10.219 |
| bc seed 1 | 29.887 | 2.172 | 2.395 | 0.857 | 32.906 |
| awbc seed 0 | 57.989 | 4.298 | 9.442 | 0.750 | 2.750 |
| awbc seed 1 | 29.848 | 2.300 | 2.186 | 0.857 | 31.500 |

## Aggregate Diagnosis
- Driver counts: `{'finalizer_preserves_bad_structure': 3, 'soft_violation_driver': 1, 'proposal_bbox_bad': 1}`.
- Cases where trained cost is higher **despite** lower mean repair displacement: `3/5`.
- Cases where trained soft violations are higher: `2/5`.
- Mean deltas over the top-5 (trained - untrained): cost `11.755`, HPWL gap `0.791`, bbox gap `1.590`, violations `0.017`, mean displacement `3.140`.
- Read together with the tracked finalizer code, the failure pattern is more consistent with **bad proposal geometry that the legality-only finalizer does not fix** than with a finalizer that independently creates the quality loss.

## Missing Metrics / Instrumentation Gaps
- `stagewise_positions_A_to_E` — tracked reports only persist final evaluated metrics, not per-stage coordinates Next touchpoints: `scripts/run_generalization_followup.py, src/puzzleplace/repair/finalizer.py`.
- `stagewise_HPWL_total_HPWLint_HPWLext` — HPWL is only persisted after the final evaluator call Next touchpoints: `src/puzzleplace/eval/official.py, src/puzzleplace/repair/finalizer.py`.
- `bbox_width_height_whitespace_aspect_per_stage` — no tracked artifact saves intermediate stage geometry snapshots Next touchpoints: `src/puzzleplace/repair/finalizer.py`.
- `max_overlap_area_and_overlap_area_per_stage` — tracked reports only keep final overlap-pair count after repair Next touchpoints: `src/puzzleplace/repair/finalizer.py, src/puzzleplace/eval/violation.py`.
- `moved_block_count_changed_block_fraction_shelf_fallback_block_count_anchor_moved_count` — finalizer report is not serialized in the current research artifacts Next touchpoints: `src/puzzleplace/repair/finalizer.py, scripts/run_generalization_followup.py`.
- `attach_boundary_pin_pull_group_compactness_intent_counters` — measure_intent_preservation currently exposes only displacement and x-order preservation Next touchpoints: `src/puzzleplace/repair/intent_preserver.py, src/puzzleplace/rollout/semantic.py`.
- `primitive_histogram_placement_order_first_hub_and_boundary_steps` — semantic action traces are not persisted in tracked reports Next touchpoints: `scripts/run_generalization_followup.py, src/puzzleplace/rollout/semantic.py`.

## Recommended Immediate Next Experiment
- Instrument and persist stage A-E snapshots for the same top-5 cases, then compare semantic proposal vs post-finalizer HPWL/bbox drift before attempting any new training run.
- Exact code files to change next:
  - `src/puzzleplace/repair/finalizer.py`
  - `src/puzzleplace/rollout/semantic.py`
  - `scripts/run_generalization_followup.py`
  - `scripts/generate_cost_semantics_and_trained_vs_untrained_delta.py`
- Experiments to avoid:
  - Do not use mean repair displacement as the primary model-selection metric.
  - Do not spend another training cycle on the same AWBC objective before stagewise HPWL/bbox evidence is available.
  - Do not interpret lower repair motion as proof that the trained proposal is better.
