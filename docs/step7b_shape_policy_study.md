# Step7B: Role-aware Shape Policy Smoke Test

## Plan

Step7B tests whether Step7A aspect pathology is causal enough to shape policy, without turning it into a runtime rule.

Tracks:

1. `posthoc_shape_probe`: cap existing Step6P representative layouts, legalize, and measure whether aspect improves cheaply.
2. `construction_shape_policy_replay`: replay construction with per-block shape-bin caps to test whether policy changes work before topology is fixed.

Policies:

- `original_shape_policy`
- `mild_global_cap`
- `role_aware_cap`
- `filler_only_extreme`
- `boundary_strict_cap`
- `boundary_edge_slot_exception`
- `MIB_shape_master_regularized`
- `group_macro_aspect_regularized`

Focus cases: `29, 32, 33, 36, 39` from Step7A. Large/XL remains Step7D scope.

## Acceptance

- Preserve original as a Pareto alternative.
- Emit role cap reasons for changed blocks.
- Compare posthoc vs construction replay.
- Report candidate-family usage changes.
- Keep sidecar only; no runtime integration.
