# Step7O-TDP Plan: Training-Derived Topology/Wire-Demand Prior

## Why this lane exists
Step7N-ALR Phase0 stopped archive/reservoir widening: it normalized 7507 rows, found 824 exact comparable replay rows, and found 0 strict meaningful non-micro winners. Step7M also stopped with zero meaningful winners. The next lane should therefore not generate more archive/reservoir variants.

The best remaining signal is the Step7ML-G training-backed data mart. It contains FloorSet training-derived layout-prior and region-heatmap labels, while keeping Step7 candidate-quality labels separate. Step7ML-I/K show that deterministic geometry decoders can make candidates legal, but quality/winner count remains narrow. Step7O tests whether training-derived target-demand priors can explain or reduce those objective regressions.

## Architecture

```text
Step7N stop guard
  + Step7ML-G training prior data mart
  + Step7ML-I/J/K decoded candidate outcomes
  + Step7N-I constrained archive rows
    -> Step7O Phase1 training-demand atlas
    -> Step7O Phase2 offline prior calibration
    -> Step7O Phase3 bounded masked replay only if gates pass
    -> later selector/ranker PRD only if replay corpus becomes broad
```

## Core rule
Step7O is not a new generator. It is a prior/calibration lane. A request deck may only be emitted after the prior proves it preserves known winners and reduces regression exposure on existing decoded/archive candidates.

## Source boundaries
Phase1 atlas may use only:
- `step7ml_g_layout_prior_examples.json`
- `step7ml_g_region_heatmap_examples.json`
- `step7ml_g_schema_report.json`
- `step7ml_g_data_inventory.json`
- static validation geometry/provenance available before labels

Phase1 must not use candidate-quality examples, Step7ML-I/J/K outcomes, Step7N-I winners/archive rows, Step7N reservoir artifacts, `target_positions`, `fp_sol`, or `tree_sol`.

Route/locality tags in Phase1 must be derived only from the allowlisted prior artifacts or static validation provenance. The Phase1 summary must include `source_ledger` and `rejected_sources[]` so a verifier can audit that the denylist held.

Phase2 may use outcome rows only for calibration. It must tag metric confidence and keep Step7ML winner baseline `2` separate from Step7N-I archive preservation `3/3` winners and `5/5` Pareto.

## Feature families
- Training closure geometry priors from Step7ML-G layout-prior examples.
- Coarse region priors from Step7ML-G region heatmap examples.
- Visible net/terminal pressure proxies from validation case inputs.
- Route/locality tags from existing Step7 artifacts.
- Candidate quality outcomes from Step7ML-I/J/K and Step7N-I, used only for calibration/ranking labels.

## Red lines
- Do not reopen Step7N reservoir Phase1/2 after `stop_no_archive_signal`.
- Do not use validation `target_positions` or `fp_sol` as generation labels.
- Do not train GNN/RL in Step7O.
- Do not promote HPWL-only gains when bbox/soft regress.
- Do not hide rejected/dominated/regressing rows.
- Do not shrink top-budget arbitrarily. Phase2 top-budget is fixed at 6 Step7ML-I/K exact comparable rows, at most 2 per case, with deterministic tie-breaks.
- If the prior-safe budget remains case024/case025-only or largest-case share exceeds 0.70, Phase3 stays closed or report-only.

## Expected useful failure
If Step7O cannot preserve known winners while reducing bbox/soft regression exposure, the correct outcome is another explicit stop signal. That would mean the project needs either a different external source of target opportunities or a research closeout, not more candidate widening.

## Execution handoff
Start with `$ralph` Phase0-1 only:

```text
$ralph Step7O-TDP Phase0-1 only. Build input inventory, stop guard, and training-demand atlas. Do not implement calibration, replay, generator widening, runtime/finalizer changes, or GNN/RL.
```
