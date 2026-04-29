# Step7E: Region / Topology Failure Diagnosis

## Plan

Step7E follows the Step7D decision:

```text
pivot_to_region_topology_before_step7c
```

This step is diagnostic only. It does not implement a new region placer, does
not change contest runtime integration, and does not add new hard gates or
penalty terms.

## Scope

- sidecar only
- no contest runtime integration
- no finalizer semantics change
- no direct Step7C iterative loop
- no RL / full NSGA-II
- preserve Step6 / Step7A-D code
- original layout remains the baseline alternative

## Focus cases

Use the Step7D representative suite first:

```text
2, 19, 24, 25, 79, 51, 76, 99, 91
```

The comparison is:

- cases where Step7D found shape-policy or MIB/group-policy signal
- large/XL cases where Step7D reported `no safe improvement`

## Diagnostics

The runner emits structural evidence for:

- region occupancy and capacity imbalance
- pin / terminal density regions
- deterministic net-community clusters
- expected-vs-actual block region assignment
- cluster spread and assignment entropy
- free-space fragmentation
- reconstructed candidate ordering trace
- repair-radius audit over original-inclusive Step7B replay alternatives

Exact Step6G construction-order traces are not persisted, so Step7E labels the
ordering evidence as `trace_confidence: reconstructed`.

## Required artifacts

- `artifacts/research/step7e_region_occupancy.json`
- `artifacts/research/step7e_pin_density_regions.json`
- `artifacts/research/step7e_net_community_clusters.json`
- `artifacts/research/step7e_block_region_assignment.json`
- `artifacts/research/step7e_free_space_fragmentation.json`
- `artifacts/research/step7e_candidate_ordering_trace.json`
- `artifacts/research/step7e_repair_radius_audit.json`
- `artifacts/research/step7e_failure_attribution.json`
- `artifacts/research/step7e_decision.md`
- `artifacts/research/step7e_visualizations/`

## Decision options

Step7E must end with one of:

- `promote_cluster_first_region_planner`
- `pivot_to_macro_level_MIB_group_planner`
- `pivot_to_candidate_ordering_policy`
- `pivot_to_free_space_topology_generator`
- `pivot_to_large_scale_legalizer_repair`
- `inconclusive_due_to_trace_gap`
