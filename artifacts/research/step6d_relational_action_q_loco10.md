# Step6C Hierarchical Action-Q Audit

- purpose: `Step6C hierarchical action-Q leave-one-case-out generalization audit`
- evaluation mode: `leave_one_case_out`
- feature mode: `relational_state_pool_no_raw_logits`
- feature normalization: `per_case`
- encoder kind: `relation_aware`
- ranker kind: `relational_action_q`
- target kind: `soft_quality`
- pairwise loss weight: `quality_delta`
- pairwise listwise loss weight: `0.0`
- hybrid scalar loss weight: `1.0`
- hybrid pairwise score weight: `0.5`
- relational pairwise loss weight: `0.0`
- cases: `10`
- policy seeds: `[0, 1, 2]`
- collections: `30`
- pools: `240`
- workers requested: `30`
- ranker epochs: `200`
- split count: `10`
- mean held-out quality rank: `5.4250`
- mean held-out quality regret: `0.8632`
- mean held-out oracle-top1 fraction: `0.1458`

## Leave-One-Case-Out Gate

- mean selected quality rank < 4.0: `False`
- mean oracle-top1 selected fraction > 0.30: `False`
- no individual oracle-top1 fraction is 0.0: `False`
- pass: `False`

## Per-Held-Out Case

- validation-0: rank `6.8333`, top1 `0.0417`, regret `0.7848`, HPWL delta `0.2688`, area delta `0.0706`, violation delta `0.0091`, top feature shift `case_block_count_norm=-2.86, case_total_area_log=1.15, layout_width_norm=0.00`
- validation-1: rank `5.3750`, top1 `0.0000`, regret `0.5787`, HPWL delta `0.0125`, area delta `0.1264`, violation delta `0.0046`, top feature shift `case_block_count_norm=-1.06, case_total_area_log=0.14, layout_width_norm=-0.00`
- validation-2: rank `5.2083`, top1 `0.1667`, regret `1.1096`, HPWL delta `0.1405`, area delta `0.0997`, violation delta `0.0231`, top feature shift `case_total_area_log=-0.33, case_block_count_norm=0.11, rel_unplaced_neighbor_count_norm=0.00`
- validation-3: rank `3.3333`, top1 `0.3333`, regret `0.2805`, HPWL delta `-0.0359`, area delta `0.0145`, violation delta `0.0112`, top feature shift `case_total_area_log=0.14, case_block_count_norm=0.11, layout_height_norm=0.00`
- validation-4: rank `5.4167`, top1 `0.0000`, regret `1.2763`, HPWL delta `0.0044`, area delta `0.0933`, violation delta `0.0306`, top feature shift `case_total_area_log=-1.39, case_block_count_norm=0.11, resolved_w_norm=0.00`
- validation-5: rank `5.1667`, top1 `0.3333`, regret `1.7107`, HPWL delta `-0.0343`, area delta `0.0063`, violation delta `0.0217`, top feature shift `case_total_area_log=-2.09, case_block_count_norm=0.11, h_norm=0.00`
- validation-6: rank `7.2500`, top1 `0.0000`, regret `1.0496`, HPWL delta `-0.0001`, area delta `0.1897`, violation delta `0.0193`, top feature shift `case_total_area_log=1.15, case_block_count_norm=0.11, layout_width_norm=-0.00`
- validation-7: rank `5.1250`, top1 `0.1250`, regret `0.9293`, HPWL delta `0.0178`, area delta `0.1085`, violation delta `0.0164`, top feature shift `case_total_area_log=-1.39, case_block_count_norm=0.11, rel_connected_unplaced_area_fraction=0.00`
- validation-8: rank `5.8750`, top1 `0.2500`, regret `0.8895`, HPWL delta `-0.0201`, area delta `-0.0888`, violation delta `0.0223`, top feature shift `case_total_area_log=1.15, case_block_count_norm=0.11, resolved_w_norm=-0.00`
- validation-9: rank `4.6667`, top1 `0.2083`, regret `0.0226`, HPWL delta `0.0035`, area delta `-0.1057`, violation delta `0.0000`, top feature shift `case_block_count_norm=3.50, case_total_area_log=1.15, block_sqrt_area_fraction=0.00`

Interpretation: this is a leave-one-case-out action-Q generalization diagnostic. It tests whether the candidate features and ranker transfer across validation cases, and reports feature shift plus objective-component regret for the held-out predictions.
