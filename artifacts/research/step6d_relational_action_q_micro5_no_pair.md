# Step6C Hierarchical Action-Q Audit

- purpose: `Step6C hierarchical action-Q candidate-pool micro-overfit audit`
- evaluation mode: `micro_overfit`
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
- ranker epochs: `80`
- train pools: `120`
- eval pools: `120`
- train mean selected quality rank: `2.9917`
- train oracle-top1 selected fraction: `0.3000`
- eval mean selected quality rank: `2.9917`
- eval mean selected quality regret: `0.2100`
- eval oracle-top1 selected fraction: `0.3000`

Interpretation: this is a micro-overfit action-Q/ranking diagnostic. It shows whether the current hierarchical action representation can express quality preferences over candidate pools. It does not modify contest runtime, repair/finalizer behavior, reranker settings, or proxy weights.
