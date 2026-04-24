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
