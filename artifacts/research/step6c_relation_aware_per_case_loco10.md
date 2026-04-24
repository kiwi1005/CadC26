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
- cases: `10`
- policy seeds: `[0, 1, 2]`
- collections: `30`
- pools: `240`
- workers requested: `30`
- ranker epochs: `200`
- split count: `10`
- mean held-out quality rank: `6.2542`
- mean held-out quality regret: `1.1657`
- mean held-out oracle-top1 fraction: `0.1042`

## Leave-One-Case-Out Gate

- mean selected quality rank < 4.0: `False`
- mean oracle-top1 selected fraction > 0.30: `False`
- no individual oracle-top1 fraction is 0.0: `False`
- pass: `False`

## Per-Held-Out Case

- validation-0: rank `7.7500`, top1 `0.0000`, regret `0.9381`, HPWL delta `0.0987`, area delta `0.1085`, violation delta `0.0199`, top feature shift `case_block_count_norm=-2.86, case_total_area_log=1.15, layout_width_norm=0.00`
- validation-1: rank `5.3333`, top1 `0.1250`, regret `1.0970`, HPWL delta `0.0880`, area delta `0.0993`, violation delta `0.0123`, top feature shift `case_block_count_norm=-1.06, case_total_area_log=0.14, resolved_w_norm=0.00`
- validation-2: rank `7.1250`, top1 `0.1250`, regret `2.1195`, HPWL delta `0.0708`, area delta `0.4519`, violation delta `0.0386`, top feature shift `case_total_area_log=-0.33, case_block_count_norm=0.11, rel_unplaced_neighbor_count_norm=0.00`
- validation-3: rank `7.6250`, top1 `0.0000`, regret `1.4589`, HPWL delta `-0.0138`, area delta `0.3021`, violation delta `0.0288`, top feature shift `case_total_area_log=0.14, case_block_count_norm=0.11, block_sqrt_area_fraction=-0.00`
- validation-4: rank `3.5000`, top1 `0.1667`, regret `0.1905`, HPWL delta `-0.0098`, area delta `-0.0389`, violation delta `0.0097`, top feature shift `case_total_area_log=-1.39, case_block_count_norm=0.11, layout_width_norm=0.00`
- validation-5: rank `6.1667`, top1 `0.0000`, regret `2.0223`, HPWL delta `-0.0163`, area delta `0.0647`, violation delta `0.0236`, top feature shift `case_total_area_log=-2.09, case_block_count_norm=0.11, layout_height_norm=0.00`
- validation-6: rank `6.3750`, top1 `0.1667`, regret `0.8143`, HPWL delta `-0.0001`, area delta `0.0662`, violation delta `0.0193`, top feature shift `case_total_area_log=1.15, case_block_count_norm=0.11, rel_unplaced_neighbor_count_norm=0.00`
- validation-7: rank `9.4583`, top1 `0.0417`, regret `2.4483`, HPWL delta `0.0715`, area delta `0.4362`, violation delta `0.0328`, top feature shift `case_total_area_log=-1.39, case_block_count_norm=0.11, rel_connected_unplaced_area_fraction=0.00`
- validation-8: rank `4.5000`, top1 `0.2500`, regret `0.4872`, HPWL delta `-0.0387`, area delta `-0.1514`, violation delta `0.0179`, top feature shift `case_total_area_log=1.15, case_block_count_norm=0.11, resolved_w_norm=-0.00`
- validation-9: rank `4.7083`, top1 `0.1667`, regret `0.0803`, HPWL delta `0.0381`, area delta `0.6944`, violation delta `0.0060`, top feature shift `case_block_count_norm=3.50, case_total_area_log=1.15, h_norm=-0.00`

Interpretation: this is a leave-one-case-out action-Q generalization diagnostic. It tests whether the candidate features and ranker transfer across validation cases, and reports feature shift plus objective-component regret for the held-out predictions.
