# Step7D: Representative Large/XL Coverage Replay

## Plan

Step7D runs before Step7C. Its job is to decide whether Step7B's construction-time shape-policy signal generalizes beyond the 50..60-block focus cases.

Scope:

- sidecar only
- no contest runtime integration
- no hard global aspect gate
- no RL or full NSGA-II
- preserve original as an alternative
- use Pareto comparison

## Representative suite

The runner profiles the available validation cases and selects one representative per requested category when possible:

- small-good
- small-bad
- medium-good
- medium-aspect-bad
- large-boundary-bad
- large-aspect-bad
- large-MIB/group-heavy
- XL-sparse
- XL-fragmented

If a category is missing, the suite artifact reports it explicitly.

## Replay

For each selected case the runner compares original-inclusive Step7B shape-policy alternatives across both posthoc and construction-time tracks, then reports Pareto representatives and a decision:

- `promote_shape_policy_to_step7c`
- `pivot_to_region_topology_before_step7c`
- `inconclusive_due_to_coverage_or_artifact_gap`

The replay keeps Step7B logic reusable through
`puzzleplace.experiments.shape_policy_replay` instead of copying the old
focus-case script. Large/XL cases use a sidecar fast-surrogate baseline and
fast replay path so the coverage pass can finish before Step7C without touching
contest runtime code.

## Required outputs

The script writes:

- `artifacts/research/step7d_case_suite.json`
- `artifacts/research/step7d_case_profiles.json`
- `artifacts/research/step7d_replay_results.json`
- `artifacts/research/step7d_bucket_summary.json`
- `artifacts/research/step7d_pathology_summary.json`
- `artifacts/research/step7d_decision.md`
- `artifacts/research/step7d_visualizations/`

Pathology summaries and visualizations include both measured profile labels and
selected-suite category labels. This preserves explicit sparse / fragmented
coverage evidence even when the raw large/XL fallback case was selected because
the dataset did not expose a stricter pathology label.
