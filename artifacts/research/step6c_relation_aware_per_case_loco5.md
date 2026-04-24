# Step6C Hierarchical Action-Q Audit

- purpose: `Step6C hierarchical action-Q leave-one-case-out generalization audit`
- evaluation mode: `leave_one_case_out`
- feature mode: `relational_state_pool_no_raw_logits`
- feature normalization: `per_case`
- encoder kind: `relation_aware`
- ranker kind: `scalar`
- target kind: `oracle_ce`
- pairwise loss weight: `quality_delta`
- pairwise listwise loss weight: `0.0`
- hybrid scalar loss weight: `1.0`
- hybrid pairwise score weight: `0.5`
- cases: `5`
- policy seeds: `[0, 1, 2]`
- collections: `15`
- pools: `120`
- workers requested: `15`
- ranker epochs: `200`
- split count: `5`
- mean held-out quality rank: `5.6833`
- mean held-out quality regret: `0.9444`
- mean held-out oracle-top1 fraction: `0.1667`

## Leave-One-Case-Out Gate

- mean selected quality rank < 4.0: `False`
- mean oracle-top1 selected fraction > 0.30: `False`
- no individual oracle-top1 fraction is 0.0: `False`
- pass: `False`

## Per-Held-Out Case

- validation-0: rank `5.7500`, top1 `0.1250`, regret `0.3782`, HPWL delta `0.0863`, area delta `-0.0047`, violation delta `0.0091`, top feature shift `case_block_count_norm=-4.04, case_total_area_log=2.45, layout_height_norm=-0.00`
- validation-1: rank `3.4583`, top1 `0.2083`, regret `0.2464`, HPWL delta `0.0710`, area delta `0.0470`, violation delta `0.0000`, top feature shift `case_block_count_norm=-0.58, case_total_area_log=0.28, layout_width_norm=-0.00`
- validation-2: rank `9.9583`, top1 `0.0000`, regret `2.8688`, HPWL delta `0.1531`, area delta `0.4576`, violation delta `0.0556`, top feature shift `case_block_count_norm=0.90, case_total_area_log=-0.42, rel_unplaced_neighbor_count_norm=0.00`
- validation-3: rank `4.5417`, top1 `0.3750`, regret `0.6821`, HPWL delta `0.0194`, area delta `0.0862`, violation delta `0.0160`, top feature shift `case_block_count_norm=0.90, case_total_area_log=0.28, is_fixed=0.02`
- validation-4: rank `4.7083`, top1 `0.1250`, regret `0.5465`, HPWL delta `-0.0058`, area delta `0.0927`, violation delta `0.0139`, top feature shift `case_total_area_log=-2.98, case_block_count_norm=0.90, layout_bbox_area_fraction=-0.00`

Interpretation: this is a leave-one-case-out action-Q generalization diagnostic. It tests whether the candidate features and ranker transfer across validation cases, and reports feature shift plus objective-component regret for the held-out predictions.
