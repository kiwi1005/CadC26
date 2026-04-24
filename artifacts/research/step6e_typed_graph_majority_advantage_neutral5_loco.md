# Step6E Majority Advantage Ranker

- purpose: `train from majority cross-continuation pairwise advantage labels`
- cases: `[0, 1, 2, 3, 4]`
- policy seeds: `[0, 1]`
- continuation policies: `['policy_greedy', 'immediate_oracle', 'policy_topk_sample']`
- feature mode: `relational_state_pool_no_raw_logits`
- ranker input: `candidate_features`
- objective kind: `majority_pairwise`
- feature normalization: `per_case`
- pool count: `40`
- mean selected quality rank: `3.5250`
- mean selected quality regret: `0.1899`
- oracle top1 selected fraction: `0.1500`
- micro gate pass: `False`

## Leave-One-Case-Out

- mean selected quality rank: `3.2000`
- mean selected quality regret: `0.1747`
- oracle top1 selected fraction: `0.1000`
- LOCO gate pass: `False`

## Gate

- mean rank < 4: `True`
- top1 > 0.30: `False`

Interpretation: this is a method probe for robust advantage learning. Passing micro-overfit would justify widening to LOCO; failing would push the next branch toward a true typed constraint graph/state encoder.
