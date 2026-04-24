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
- cases: `5`
- policy seeds: `[0, 1, 2]`
- collections: `15`
- pools: `120`
- workers requested: `15`
- ranker epochs: `200`
- split count: `5`
- mean held-out quality rank: `4.8083`
- mean held-out quality regret: `0.4905`
- mean held-out oracle-top1 fraction: `0.2167`

## Leave-One-Case-Out Gate

- mean selected quality rank < 4.0: `False`
- mean oracle-top1 selected fraction > 0.30: `False`
- no individual oracle-top1 fraction is 0.0: `True`
- pass: `False`

## Per-Held-Out Case

- validation-0: rank `4.7917`, top1 `0.2500`, regret `0.4979`, HPWL delta `0.1412`, area delta `0.0751`, violation delta `0.0054`, top feature shift `case_block_count_norm=-4.04, case_total_area_log=2.45, layout_height_norm=-0.00`
- validation-1: rank `4.5000`, top1 `0.2500`, regret `0.5226`, HPWL delta `0.0461`, area delta `0.0508`, violation delta `0.0062`, top feature shift `case_block_count_norm=-0.58, case_total_area_log=0.28, layout_width_norm=-0.00`
- validation-2: rank `3.2917`, top1 `0.2083`, regret `0.2724`, HPWL delta `0.1084`, area delta `-0.0189`, violation delta `0.0046`, top feature shift `case_block_count_norm=0.90, case_total_area_log=-0.42, resolved_h_norm=0.00`
- validation-3: rank `6.7500`, top1 `0.0833`, regret `0.8696`, HPWL delta `-0.0153`, area delta `0.0930`, violation delta `0.0240`, top feature shift `case_block_count_norm=0.90, case_total_area_log=0.28, is_fixed=0.02`
- validation-4: rank `4.7083`, top1 `0.2917`, regret `0.2900`, HPWL delta `0.0616`, area delta `0.0860`, violation delta `0.0014`, top feature shift `case_total_area_log=-2.98, case_block_count_norm=0.90, layout_width_norm=0.00`

Interpretation: this is a leave-one-case-out action-Q generalization diagnostic. It tests whether the candidate features and ranker transfer across validation cases, and reports feature shift plus objective-component regret for the held-out predictions.
