# Step7N-I Quality-Filtered Macro Slot/Repack Integration

## Scope

Step7N-I is a sidecar-only integration of Step7N-H findings back into the
Step7N-G macro/topology artifact layer. It does not modify contest runtime code
or finalizer semantics.

## Inputs

- Step7N-G integrated candidates and metric report.
- Step7N-H-C constrained Pareto quality-filter artifacts.
- Step7N-H-D retrieval target calibration artifacts.
- Step7N-H-A failure taxonomy artifacts.
- Optional Step7N-H-E sensitivity artifacts as weak rescue evidence.

## Method

The integration keeps the H-C constrained dominance result as the primary
selector. Rows retained only to preserve local-starved recovery are kept as
report-only evidence unless they also belong to the non-anchor Pareto front or
are official-like winners. H-D target calibration is attached as provenance and
used to explain target-region/internal-repack/slot failure buckets.

No scalar penalty threshold is introduced. All rejected, dominated, report-only,
and metric-regressing candidates remain present in JSON artifacts.

## Expected decision

`promote_quality_filter_to_step7n_g_sidecar` means the quality filter is ready to
be wired into the Step7N-G sidecar archive/reporting path, not into the contest
runtime solver.
