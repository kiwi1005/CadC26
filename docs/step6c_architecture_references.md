# Step6C Architecture References — Candidate Quality Generalization

Date: 2026-04-24
Scope: CadC26 Step6C sidecar research only. Do not use this note to unlock contest runtime, Wave 2 scorer/reranker/finalizer integration, or BMAD work.

## Current Project Finding

Step6C has already separated three questions:

1. **Semantic learnability**: hierarchical block-first policy can learn semantic actions.
2. **Rollout control**: trained hierarchical logits change semantic rollout decisions.
3. **Quality generalization**: shallow candidate Action-Q rankers still fail leave-one-case-out transfer.

Latest local best sidecar result before this note:

| feature mode | ranker | target | LOCO rank | LOCO regret | LOCO top1 | gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `legacy` | `scalar` | `oracle_ce` | `7.0250` | `1.6351` | `0.0750` | false |
| `state_pool_no_raw_logits` | `scalar` | `oracle_ce` | `6.0000` | `0.8371` | `0.1250` | false |

Interpretation: raw cross-case policy/logit scale is harmful, but independent per-candidate MLP scoring is still not enough. The next architecture should compare candidates **within the same state/pool**.

## References Reviewed

### Chip Placement with Deep Reinforcement Learning

- URL: <https://arxiv.org/abs/2004.10746>
- Key idea: learn transferable chip placement representations by grounding the policy/value encoder in a supervised placement-quality prediction task.
- CadC26 implication: quality prediction should be a representation-learning objective, not a thin post-hoc scorer over raw policy logits.

### ChiPFormer: Transferable Chip Placement via Offline Decision Transformer

- PMLR: <https://proceedings.mlr.press/v202/lai23c.html>
- arXiv: <https://arxiv.org/abs/2306.14744>
- Key idea: cast placement as offline RL to learn transferable policies from fixed offline designs and finetune faster on unseen circuits.
- CadC26 implication: our current dataset/audit shape is closer to offline RL or offline ranking than online PPO; trajectory/state/action context matters.

### Effective Analog ICs Floorplanning with Relational GNNs and RL

- URL: <https://arxiv.org/abs/2411.15212>
- Key idea: relational graph convolution encodes circuit features and positional constraints to improve transfer across designs with different topologies/constraints.
- CadC26 implication: fixed/preplaced/MIB/cluster/boundary, block-block edges, and pin-block edges should be typed relational context, not only scalar metadata.

### Set Transformer

- URL: <https://arxiv.org/abs/1810.00825>
- Key idea: attention over sets models interactions while respecting permutation invariance/equivariance.
- CadC26 implication: candidate pools are sets; scoring each candidate independently ignores within-pool competition and relative evidence.

### Attention, Learn to Solve Routing Problems!

- URL: <https://arxiv.org/abs/1803.08475>
- Key idea: attention-based encoder/decoder learns constructive combinatorial heuristics and selects from available decisions at each step.
- CadC26 implication: placement candidate selection should be contextual over the available action set, not an absolute global score.

### RankNet / Learning to Rank using Gradient Descent

- URL: <https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf>
- Key idea: train on pairs from the same query/list to learn relative preference without requiring absolute rank calibration across unrelated lists.
- CadC26 implication: learn pairwise candidate preference inside each state/pool using quality_cost_runtime1 and component deltas.

## Recommended Architecture

### Pairwise Within-State Geometry Comparator

Do **not** start with full PPO or a larger independent MLP. Start with an audit-only comparator:

```text
FloorSet case/state features
        +
candidate_i state-pool features
        +
candidate_j state-pool features
        +
phi_i - phi_j / geometry deltas
        ↓
Pairwise comparator: P(i better than j)
        ↓
Aggregate pairwise wins inside each pool
        ↓
Selected candidate
```

Training label:

```text
candidate_i wins iff quality_cost_runtime1(i) < quality_cost_runtime1(j)
```

Primary gate:

- LOCO mean held-out selected quality rank `< 4.0`
- LOCO mean oracle-top1 selected fraction `> 0.30`
- no held-out case with oracle-top1 selected fraction `0.0`
- regret improves beyond the current best baseline `0.8371`

Boundary:

- Sidecar audit only.
- No contest runtime integration.
- No Wave 2 scorer/reranker/finalizer unlock until held-out evidence passes.

## Implementation Trial Result

The first sidecar implementation added a `pairwise_set` ranker:

- candidate rows use `state_pool_no_raw_logits` features;
- candidates in the same pool are contextualized by self-attention;
- ordered candidate pairs are trained with a RankNet/Bradley-Terry-style binary loss;
- candidate selection aggregates pairwise win probabilities within the pool.

LOCO comparison after implementation:

| feature mode | ranker | pair weight | listwise aux | LOCO rank | LOCO regret | LOCO top1 | gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `legacy` | `scalar` | - | `0.0` | `7.0250` | `1.6351` | `0.0750` | false |
| `state_pool_no_raw_logits` | `scalar` | - | `0.0` | `6.0000` | `0.8371` | `0.1250` | false |
| `state_pool_no_raw_logits` | `pairwise_set` | `quality_delta` | `0.0` | `6.1750` | `0.8843` | `0.2250` | false |
| `state_pool_no_raw_logits` | `pairwise_set` | `uniform` | `0.0` | `7.2500` | `1.3333` | `0.0500` | false |
| `state_pool_no_raw_logits` | `pairwise_set` | `quality_delta` | `1.0` | `7.2750` | `1.2762` | `0.0500` | false |

Interpretation: pairwise set attention improves oracle-top1 coverage
(`0.2250`, highest so far) and eliminates zero-top1 held-out cases, but it does
not beat the scalar no-raw-logit model on rank/regret. The next design should
combine the scalar model's regret stability with pairwise comparator top1
coverage, or train on substantially more diverse pools before moving to runtime.
## Lane 11 Hybrid Selector Trial Result

Lane 11 implemented two hybrid variants:

1. `hybrid_set`: shared candidate-set attention encoder with scalar and pairwise heads.
2. `late_fusion`: independent scalar MLP plus pairwise-set comparator, mixed at inference by standardized score fusion.

LOCO comparison:

| ranker | pairwise score weight | mean rank | regret | top1 | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `scalar` | - | `6.0000` | `0.8371` | `0.1250` | false |
| `pairwise_set` | - | `6.1750` | `0.8843` | `0.2250` | false |
| `late_fusion` | `0.25` | `6.2750` | `0.8403` | `0.1000` | false |
| `late_fusion` | `0.50` | `6.3500` | `0.9611` | `0.1250` | false |
| `late_fusion` | `0.75` | `6.6750` | `1.1861` | `0.2000` | false |

Interpretation: late fusion nearly matches the scalar baseline on regret at low pairwise weight, but it does not improve rank/top1 enough to pass the gate. The pairwise-only model still gives the strongest top1 signal, while scalar remains best for regret. The next move should not be more fusion-weight tuning on the same 40 pools; it should increase candidate-pool diversity or add a stronger relational/state encoder before revisiting fusion.

## Lane 12 Candidate-Pool Diversity / Relational-State Trial

Lane 12 tested whether Action-Q generalization is data-limited or representation-limited.

Implemented:

- `--policy-seeds` in `scripts/run_step6c_hierarchical_action_q_audit.py` to collect one candidate-pool trajectory per case/seed.
- `relational_state_pool_no_raw_logits` feature mode with block-block, pin-block, placed-neighbor, target-neighbor, and connected-area features.

LOCO comparison:

| feature mode | ranker | cases | policy seeds | pools | mean rank | regret | top1 | gate |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `state_pool_no_raw_logits` | `scalar` | `5` | `0` | `40` | `6.0000` | `0.8371` | `0.1250` | false |
| `state_pool_no_raw_logits` | `scalar` | `5` | `0,1,2` | `120` | `5.5583` | `0.6641` | `0.0833` | false |
| `relational_state_pool_no_raw_logits` | `scalar` | `5` | `0,1,2` | `120` | `5.3417` | `0.6915` | `0.1667` | false |
| `relational_state_pool_no_raw_logits` | `scalar` | `10` | `0,1,2` | `240` | `5.9375` | `1.3659` | `0.1167` | false |

Interpretation:

- More policy-seed pools reduce regret and rank on validation `0..4`, so the old 40-pool result was partly data-limited.
- Relational state features improve rank/top1 and remove zero-top1 held-out cases on validation `0..4`, so representation also matters.
- Extending to validation `0..9` regresses, especially validation `1`, `5`, and `6`; simply adding cases is not sufficient without better normalization or a stronger relation-aware model.
- Wave 2 / contest runtime remain locked.


## Lane 13 Per-Case Normalization + Relation-Aware Encoder Trial

This lane explicitly stops fusion-weight tuning and tests the next three blockers:

1. per-case/per-scale normalization;
2. a stronger relation-aware state encoder;
3. simultaneous 5-case and 10-case LOCO validation.

Implemented:

- `RelationAwareGraphStateEncoder` in `src/puzzleplace/models/encoders.py` with per-case/per-scale normalized block/placement features plus typed block-block and pin-block message channels.
- `HierarchicalSetPolicy(..., encoder_kind="relation_aware")` so Step6C audits can swap the stronger encoder without changing contest/runtime paths.
- `--feature-normalization per_case` in `scripts/run_step6c_hierarchical_action_q_audit.py` to standardize candidate-ranker feature rows within each case before ranker training/evaluation.
- `--encoder-kind relation_aware` in the Step6C action-Q and rollout-control audit surfaces.

Validation commands:

```bash
.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py \
  --case-ids 0 1 2 3 4 \
  --leave-one-case-out \
  --policy-seeds 0 1 2 \
  --feature-mode relational_state_pool_no_raw_logits \
  --feature-normalization per_case \
  --encoder-kind relation_aware \
  --ranker-kind scalar \
  --policy-epochs 120 \
  --ranker-epochs 200 \
  --workers 48 \
  --output artifacts/research/step6c_relation_aware_per_case_loco5.json

.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py \
  --case-ids 0 1 2 3 4 5 6 7 8 9 \
  --leave-one-case-out \
  --policy-seeds 0 1 2 \
  --feature-mode relational_state_pool_no_raw_logits \
  --feature-normalization per_case \
  --encoder-kind relation_aware \
  --ranker-kind scalar \
  --policy-epochs 120 \
  --ranker-epochs 200 \
  --workers 48 \
  --output artifacts/research/step6c_relation_aware_per_case_loco10.json
```

LOCO comparison:

| feature mode | normalization | encoder | ranker | cases | policy seeds | pools | mean rank | regret | top1 | gate |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `relational_state_pool_no_raw_logits` | `per_case` | `relation_aware` | `scalar` | `5` | `0,1,2` | `120` | `5.6833` | `0.9444` | `0.1667` | false |
| `relational_state_pool_no_raw_logits` | `per_case` | `relation_aware` | `scalar` | `10` | `0,1,2` | `240` | `6.2542` | `1.1657` | `0.1042` | false |

Interpretation:

- Per-case normalization plus the relation-aware sidecar encoder is wired and test-covered, but it does **not** pass the current LOCO gate.
- The 5-case result improves top1 relative to the previous 5-case relational scalar lane (`0.1667` vs `0.1667` unchanged) but worsens rank/regret (`5.6833` / `0.9444` vs prior `5.3417` / `0.6915`).
- The 10-case result improves over the prior 10-case relational scalar lane in regret (`1.1657` vs prior `1.3659`) but still regresses in rank/top1 (`6.2542` / `0.1042` vs prior `5.9375` / `0.1167`).
- The blocker remains cross-case generalization / representation, especially held-out cases with zero top1. Do not return to fusion-weight tuning from this evidence alone.

## Lane 14 Step6D Relational Action-Q Implementation

Lane 14 implements the Step6D B-first experiment: a pool-local relation-aware action-Q ranker plus bounded target ablation.

Implemented:

- `CandidateRelationalActionQRanker` in `src/puzzleplace/models/hierarchical.py`.
  - projects candidate state/action feature rows;
  - contextualizes candidates with self-attention;
  - performs pairwise candidate message passing;
  - scores each candidate with a graph-context action-Q head;
  - exposes pairwise logits and component logits for future diagnostics.
- `--ranker-kind relational_action_q` in `scripts/run_step6c_hierarchical_action_q_audit.py`.
- `--relational-pairwise-loss-weight` to keep pairwise auxiliary loss explicit and disabled when it harms micro-overfit.
- Step6D artifacts:
  - `artifacts/research/step6d_target_soft_quality_loco5.{json,md}`
  - `artifacts/research/step6d_relational_action_q_micro5_no_pair.json`
  - `artifacts/research/step6d_relational_action_q_loco5.{json,md}`
  - `artifacts/research/step6d_relational_action_q_loco10.{json,md}`
  - `artifacts/research/step6d_experiment_summary.{json,md}`

Aggregate comparison:

| run | cases | ranker | target | pair aux | mean rank | regret | top1 | zero-top1 | gate |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Lane 13 baseline | 5 | `scalar` | `oracle_ce` | n/a | `5.6833` | `0.9444` | `0.1667` | `[2]` | false |
| target-only ablation | 5 | `scalar` | `soft_quality` | n/a | `6.4750` | `1.0369` | `0.1667` | `[2]` | false |
| relational action-Q | 5 | `relational_action_q` | `soft_quality` | `0.0` | `4.8083` | `0.4905` | `0.2167` | `[]` | false |
| Lane 13 baseline | 10 | `scalar` | `oracle_ce` | n/a | `6.2542` | `1.1657` | `0.1042` | `[0,3,5]` | false |
| relational action-Q | 10 | `relational_action_q` | `soft_quality` | `0.0` | `5.4250` | `0.8632` | `0.1458` | `[1,4,6]` | false |

Interpretation:

- Target-only `soft_quality` worsened the scalar ranker, so target brittleness alone is not the main blocker.
- Relational action-Q improved both 5-case and 10-case over Lane 13 on rank, regret, and top1, so the B-first architecture direction is validated.
- The current variant still fails the LOCO gate; 10-case zero-top1 moved to validation `1`, `4`, and `6` rather than disappearing.
- Pairwise auxiliary loss hurt early same-case micro-overfit, so keep it disabled until pairwise target calibration is separately diagnosed.
- Next gate should be rollout-return label diagnostics on zero-top1 held-out cases before widening model capacity or re-enabling pairwise loss.

## Lane 15 Step6D Rollout-Return Label Horizon Diagnostic

Lane 15 tests whether the remaining Step6D zero-top1 cases are caused by one-step immediate action labels that disagree with bounded rollout-return action values.

Implemented:

- `scripts/run_step6d_rollout_return_label_diagnostic.py`.
  - collects candidate pools on selected cases/seeds;
  - compares each candidate's immediate `quality_after_action` score against bounded continuation quality after greedy hierarchical rollout;
  - reports oracle agreement, immediate-best rollout rank, and immediate-best rollout regret.

Validation command:

```bash
.venv/bin/python scripts/run_step6d_rollout_return_label_diagnostic.py \
  --case-ids 1 4 6 \
  --policy-seeds 0 1 2 \
  --encoder-kind relation_aware \
  --policy-epochs 120 \
  --max-steps 8 \
  --max-candidates-per-step 16 \
  --continuation-horizon 8 \
  --workers 48 \
  --output artifacts/research/step6d_rollout_return_label_diagnostic_1_4_6.json
```

Result:

| cases | collections | pools | oracle agreement | mismatch | immediate-best rollout rank | immediate-best rollout regret | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1,4,6` | `9` | `72` | `0.1667` | `0.8333` | `5.2917` | `1.6095` | false |

Per-case:

| case | agreement | mismatch | immediate-best rollout rank | immediate-best rollout regret |
| --- | ---: | ---: | ---: | ---: |
| `validation-1` | `0.1250` | `0.8750` | `5.4583` | `2.2061` |
| `validation-4` | `0.1667` | `0.8333` | `4.2083` | `0.7698` |
| `validation-6` | `0.2083` | `0.7917` | `6.2083` | `1.8524` |

Interpretation:

- Immediate one-step labels are not reliable on the zero-top1 held-out cases.
- The immediate oracle agrees with the bounded rollout-return oracle only `16.67%` of the time.
- The immediate-best action averages rollout-return rank `5.29` with rollout-return regret `1.61`.
- This shifts the primary blocker from ranker capacity to label horizon / action-value target design.
- Next experiment should train the action-Q ranker on rollout-return labels for the same candidate pools before widening the model or re-enabling pairwise auxiliary loss.

## Lane 16 Step6E Naive Rollout-Return Action-Q Labels

Lane 16 implements the first Step6E experiment: use bounded rollout-return `quality_cost_runtime1` as the action-Q training/evaluation label via `--label-kind rollout_return` in `scripts/run_step6c_hierarchical_action_q_audit.py`.

Result:

| run | label | target | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6D baseline | `immediate` | `soft_quality` | 5 | 120 | `2.9917` | `0.2100` | `0.3000` | same-case micro baseline |
| Step6E | `rollout_return` | `soft_quality` | 5 | 120 | `7.5583` | `1.4614` | `0.1667` | failed micro-overfit |
| Step6E | `rollout_return` | `oracle_ce` | 5 | 120 | `7.4583` | `1.5361` | `0.0667` | failed harder |

Interpretation:

- The label-horizon diagnosis remains valid: immediate labels disagree with rollout-return labels on zero-top1 cases.
- However, naively replacing labels with bounded rollout-return values is not sufficient; the current candidate features/ranker cannot micro-fit those labels.
- This points to continuation-policy noise and/or missing typed constraint-relation representation.
- Do not proceed to 5/10 LOCO for this naive label variant. Next branch should stabilize rollout-return labels with top-k/advantage/contrastive targets or implement a typed constraint graph for boundary + MIB + cluster + preplaced/fixed-anchor interactions.

### Lane 16b Fixed Top-k Rollout-Return Target

A fixed uniform top-3 target over bounded rollout-return labels was tested to reduce brittleness without tuning weights.

| run | label | target | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E1b | `rollout_return` | `topk_quality` | 5 | 120 | `6.9083` | `1.7438` | `0.0917` | failed micro-overfit |

Conclusion: top-k target smoothing does not fix the rollout-return label path. The next research branch should move to typed constraint-relation representation and/or label denoising with multiple continuations, not parameter tuning.

## Lane 17 Step6E Typed Constraint-Relation Feature Trial

Lane 17 adds `constraint_relation_pool_no_raw_logits`, a sidecar feature mode that appends typed scalar features for same-cluster, same-MIB, boundary compatibility, nearest fixed/preplaced anchors, and group edge weights.

| run | feature mode | cases | pools | mean rank | regret | top1 | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6D relational action-Q baseline | `relational_state_pool_no_raw_logits` | 5 | 120 | `4.8083` | `0.4905` | `0.2167` | false |
| Step6E typed constraint-relation | `constraint_relation_pool_no_raw_logits` | 5 | 120 | `6.0833` | `1.0742` | `0.1500` | false |

Interpretation: scalar-appended typed constraint features regress LOCO transfer. The constraint information is relevant, but flat concatenation appears to overfit case-specific patterns rather than produce transferable reasoning. Next typed-constraint work should use a structured edge-aware/heterogeneous graph encoder or contrastive diagnostics, not more scalar feature appends or parameter tuning.

### Lane 17b Constraint-Token Action-Q Ranker

A `constraint_token_action_q` ranker was tested to avoid flat scalar append: the final constraint-relation feature segment is encoded as typed relation tokens before candidate-set ranking.

| run | ranker | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E2b | `constraint_token_action_q` | 5 | 120 | `7.8250` | `1.7466` | `0.0250` | failed micro-overfit |

Conclusion: tokenizing the appended scalar relation extras does not fix transfer or even same-case fitting. Future typed-relation work should construct graph/node/edge embeddings directly from constraints and current state rather than post-hoc scalar extras in the candidate feature row.

### Lane 18 Step6E Rollout Label Stability Diagnostic

Lane 18 tests whether bounded rollout-return labels are themselves stable across continuation policies before using them as an action-Q target. The diagnostic compares `policy_greedy`, `immediate_oracle`, and `policy_topk_sample` continuations on the hard zero-top1 cases `1,4,6`.

| run | cases | seeds | steps | horizon | all-policy agreement | mean unique rollout oracles | stable gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E3 slice | 3 | 2 | 4 | 4 | `0.0000` | `2.4583` | false |

Pairwise agreement was also low: `policy_greedy__immediate_oracle = 0.0000`, `policy_greedy__policy_topk_sample = 0.2917`, and `immediate_oracle__policy_topk_sample = 0.2500`.

Conclusion: rollout-return labels are policy-dependent and noisy, not a clean replacement for one-step labels. The next method should use cross-continuation robust advantages/consensus labels or a true typed constraint graph encoder, not fusion-weight tuning or more flattened scalar features.

### Lane 19 Step6E Consensus Advantage Diagnostic

Lane 19 converts the unstable multi-continuation rollout labels into pairwise preference evidence. This tests a paper-like advantage-learning direction: use robust relative preferences instead of brittle hard top1 rollout labels.

| run | pools | unanimous pair | majority pair | split pair | consensus-top hits any oracle | strict usable gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E4 | 24 | `0.3348` | `0.6369` | `0.0283` | `0.9167` | false |

Conclusion: hard consensus is sparse, but majority pairwise advantage signal is not empty. The next non-tuning experiment should train a soft/majority cross-continuation pairwise objective, then validate first on micro-overfit before any 5/10-case LOCO widening.

### Lane 20 Step6E Majority Advantage Ranker

Lane 20 trains directly on majority cross-continuation pairwise advantages. This avoids brittle hard rollout top1 labels and uses pairwise relative evidence across `policy_greedy`, `immediate_oracle`, and `policy_topk_sample`.

| run | cases | pools | micro rank | micro regret | micro top1 | LOCO rank | LOCO top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step6E E5 | 3 | 24 | `1.9167` | `0.0423` | `0.5833` | `4.5000` | `0.0833` |

Conclusion: majority pairwise advantage learning works on the hard-case micro slice, but the current feature representation still does not transfer. This narrows the next bottleneck to case-invariant structured representation, not label absence and not parameter/fusion-weight tuning.

### Lane 21 Step6E Typed Constraint Graph Encoder

Lane 21 introduces a true typed constraint graph/state encoder. Unlike Lane 17 scalar appends and Lane 17b scalar-token encoding, `typed_constraint_graph` sends explicit messages over b2b, p2b, same-cluster, same-MIB, boundary-side, and fixed/preplaced-anchor relation channels.

| run | encoder | micro rank | micro top1 | LOCO rank | LOCO top1 | gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E5 baseline | `relation_aware` | `1.9167` | `0.5833` | `4.5000` | `0.0833` | false |
| E6 typed graph | `typed_constraint_graph` | `1.9167` | `0.4583` | `3.4583` | `0.2500` | false |

Conclusion: the typed graph improves hard-case transfer but remains below the top1 gate. This is positive evidence for structured representation, not a final solution. Continue with typed-relation ablation and graph-action representation work before 5/10 LOCO expansion.

### Lane 22 Step6E Typed Relation Channel Ablation

Lane 22 ablates E6 typed-graph channels. Removing cluster/MIB/boundary group messages collapses LOCO transfer (`rank 5.0833`, `top1 0.0417`) despite strong micro fit, and removing anchor messages also regresses (`rank 4.1250`, `top1 0.1667`). This supports the hypothesis that transferable signal comes from structured group/anchor relations rather than scalar feature additions or loss tuning.

Next: build a graph-action ranker that consumes typed graph block/action embeddings directly instead of only using the typed encoder to generate candidate pools.

### Lane 23 Step6E Naive Graph-Action Embedding Ranker

Lane 23 makes the first graph-action ranker attempt by feeding candidate block embedding, target embedding, graph embedding, primitive one-hot, and target-presence flag from the typed graph policy encoder into the majority-advantage ranker.

| run | input | micro rank | micro top1 | LOCO rank | LOCO top1 |
| --- | --- | ---: | ---: | ---: | ---: |
| E6 baseline | candidate features | `1.9167` | `0.4583` | `3.4583` | `0.2500` |
| E8 naive graph-action | frozen graph/action embeddings | `3.9167` | `0.1667` | `3.7917` | `0.2500` |

Conclusion: direct frozen policy embeddings are not enough. The next graph-action model must be action-conditioned and trainable in the ranker path, while preserving the group+anchor typed relation channels identified by Lane 22.

### Lane 24 Step6E Joint Typed Graph-Action Ranker

Lane 24 makes the graph-action ranker trainable in the ranker path: node projection, typed relation messages, and action pairwise comparison are learned from majority cross-continuation advantages.

| run | input | micro rank | micro top1 | LOCO rank | LOCO top1 |
| --- | --- | ---: | ---: | ---: | ---: |
| E6 candidate features | `candidate_features` | `1.9167` | `0.4583` | `3.4583` | `0.2500` |
| E8 frozen graph/action | `graph_action_embeddings` | `3.9167` | `0.1667` | `3.7917` | `0.2500` |
| E9 joint graph/action | `joint_typed_graph_action` | `3.7917` | `0.2500` | `4.2083` | `0.1667` |

Conclusion: the static graph-action formulation fails micro and regresses transfer. Future graph-action work should encode action-conditioned next-state relation deltas and legality/geometry consequences, not just pre-action graph node relations. Execution should also be parallelized to use 48 cores for independent slice/split jobs.

### Lane 25 Step6E Action Delta Features and Parallel LOCO

Lane 25 parallelizes LOCO split training using spawn-safe multiprocessing and tests a minimal action-conditioned next-state delta representation. The run used `--workers 48`; the slice exposed `12` collection jobs and `3` LOCO split jobs.

| run | input | seeds | pools | micro rank | micro top1 | LOCO rank | LOCO top1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E6 candidate features | `candidate_features` | 2 | 24 | `1.9167` | `0.4583` | `3.4583` | `0.2500` |
| E10 action deltas | `action_delta_features` | 4 | 48 | `3.8958` | `0.2292` | `3.7708` | `0.1042` |

Conclusion: hand-built next-state deltas fail. Future delta work should be learned and relation-aware. Keep the parallel harness, but default to serial smoke runs and reserve `--workers 48` for independent case/seed or LOCO split jobs after the relevant gate passes.
