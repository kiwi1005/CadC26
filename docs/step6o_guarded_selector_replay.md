# Step6O Guarded Selector Replay

## Scope

Step6O is a sidecar replay only. It does not change runtime selection, does not fill
large/XL coverage, does not tune MIB/group moves, and does not promote the guards as final
rules.

Inputs:

- `artifacts/research/step6m_report.json`
- `artifacts/research/step6n_case_pathology_report.json`

Applied Step6N guard candidates to accepted `simple_compaction` alternatives:

```text
hpwl_regression_per_boundary_gain > 40
spatial_balance_worsening > 0.10
```

## Artifacts

- `artifacts/research/step6o_guarded_selector_replay.json`
- `artifacts/research/step6o_guarded_selected_alternatives.json`
- `artifacts/research/step6o_guarded_eval_rows.json`
- `artifacts/research/step6o_profile_summary.md`
- `artifacts/research/step6o_suspicious_visualizations/`

## Replay result

```json
{
  "selected_move_counts_before": {
    "original": 15,
    "simple_compaction": 16,
    "group_boundary_touch_template": 2,
    "boundary_edge_reassign": 7
  },
  "selected_move_counts_after": {
    "original": 18,
    "simple_compaction": 6,
    "group_boundary_touch_template": 2,
    "boundary_edge_reassign": 12,
    "soft_shape_stretch": 1,
    "local_region_repack": 1
  },
  "suspicious_simple_compaction_count_before": 12,
  "suspicious_simple_compaction_count_after": 2,
  "original_fallback_count_before": 15,
  "original_fallback_count_after": 18,
  "false_rejection_case_ids": []
}
```

The guard removes most suspicious `simple_compaction` picks, but it also exposes a second
selector issue: after the bad compaction is filtered, the next accepted alternative often has
less boundary gain. That is why this should not yet be treated as a clean pass into runtime.

Mean delta comparison over the same 40-case prefix:

```json
{
  "before": {
    "boundary_delta": 0.06785920342837493,
    "bbox_delta": -1649.7578876412724,
    "hpwl_delta": 0.8918116037424314,
    "soft_delta": -1.175
  },
  "after": {
    "boundary_delta": 0.04578400262622567,
    "bbox_delta": -452.50516120880195,
    "hpwl_delta": 0.47124417110088085,
    "soft_delta": -0.8
  }
}
```

Interpretation:

- HPWL regression improves: mean HPWL delta drops from `+0.892` to `+0.471`.
- Boundary improvement weakens: mean boundary delta drops from `+0.0679` to `+0.0458`.
- BBox and soft improvements also weaken because many strong compactions are filtered.
- No previously selected non-suspicious case was falsely rejected.

## Suspicious case outcomes

```json
{
  "guard_rejects_but_worse_alternative_selected": 7,
  "guard_reverts_to_original": 3,
  "guard_keeps_because_benefit_justified": 2
}
```

Kept because the Step6N guards did not fire:

- case 31
- case 32

Reverted to original:

- case 21
- case 28
- case 33

Rejected but lower-quality next alternative selected:

- case 8
- case 10
- case 14
- case 15
- case 23
- case 37
- case 38

## Decision

`guard_blocks_compaction_but_next_selector_tradeoff_needs_review`

Step6O validates that the two Step6N guard candidates are useful for blocking suspicious
compaction without obvious false rejection of good selected cases. However, replay also shows
that the fallback/next-choice selector needs review before treating this as a clean Step6P
handoff.
