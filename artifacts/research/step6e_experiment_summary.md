# Step6E Experiment Summary

- status: `e1_micro_failed`
- scope: rollout-return action-Q labels for `relational_action_q`
- conclusion: naive bounded rollout-return labels are not directly learnable by the current candidate feature/ranker setup.

## Runs

| run | mode | label | target | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `step6d_relational_action_q_micro5_no_pair` | micro | immediate | soft_quality | 5 | 120 | 2.9917 | 0.2100 | 0.3000 | current immediate-label micro baseline |
| `step6e_rollout_return_action_q_micro5` | micro | rollout_return | soft_quality | 5 | 120 | 7.5583 | 1.4614 | 0.1667 | failed micro-overfit |
| `step6e_rollout_return_action_q_micro5_ce` | micro | rollout_return | oracle_ce | 5 | 120 | 7.4583 | 1.5361 | 0.0667 | failed harder with hard top1 CE |

## Finding

The prior diagnostic proved immediate labels disagree with rollout-return labels on zero-top1 cases, but this experiment shows that simply swapping `quality_values` to bounded rollout-return is not enough. The current feature/ranker representation cannot even micro-fit the rollout-return labels on the 5-case training pools.

## Interpretation

Likely failure classes:

1. **Continuation-policy noise**: rollout-return value is induced by a greedy continuation policy, so small first-action changes can create discontinuous downstream quality labels.
2. **State/action representation gap**: current candidate rows may not encode enough of the future constraint-group state needed to predict rollout return.
3. **Label target brittleness**: hard rollout-return top1 is worse than soft quality, but soft quality is still too noisy for current features.
4. **Need structured labels**: prefer contrastive/top-k/advantage labels or averaged rollout-return labels before using raw return as a direct scalar/listwise target.

## Next Research Branch

Do not widen model blindly and do not tune parameters. Next branch should either:

- stabilize rollout-return labels with top-k/advantage/contrastive targets and/or multiple continuation seeds; or
- implement typed constraint-relation graph features so the model can represent boundary + MIB + cluster + preplaced/fixed-anchor interactions before trying rollout-return labels again.

## E1b Fixed Top-k Rollout-Return Target

| run | label | target | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E1b | `rollout_return` | `topk_quality` | 5 | 120 | `6.9083` | `1.7438` | `0.0917` | failed micro-overfit |

Interpretation: a fixed top-3 rollout-return target does not rescue the naive rollout-return label path. This strengthens the conclusion that the current candidate features/ranker cannot represent the bounded continuation value, or the bounded continuation labels are too noisy without stronger state/constraint structure.

## E2 Typed Constraint-Relation Features

| run | feature mode | label | target | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6D baseline micro | `relational_state_pool_no_raw_logits` | immediate | soft_quality | 5 | 120 | `2.9917` | `0.2100` | `0.3000` | best prior micro |
| Step6E E2 micro | `constraint_relation_pool_no_raw_logits` | immediate | soft_quality | 5 | 120 | `3.9167` | `0.4121` | `0.3083` | similar top1, worse rank/regret |
| Step6D baseline LOCO | `relational_state_pool_no_raw_logits` | immediate | soft_quality | 5 | 120 | `4.8083` | `0.4905` | `0.2167` | best prior 5-case LOCO |
| Step6E E2 LOCO | `constraint_relation_pool_no_raw_logits` | immediate | soft_quality | 5 | 120 | `6.0833` | `1.0742` | `0.1500` | regressed transfer |

Finding: simply appending typed constraint-relation scalar features for boundary/MIB/cluster/preplaced/fixed-anchor interactions regresses LOCO transfer. The information is relevant, but the current concatenated-feature ranker likely overfits case-specific constraint patterns. Next should move from scalar append features to a structured typed relation graph / edge-aware encoder, or use constraint features only in a contrastive diagnostic rather than direct LOCO ranker input.

## E2b Constraint-Token Action-Q Ranker

| run | ranker | feature mode | label | target | cases | pools | mean rank | regret | top1 | interpretation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E2b | `constraint_token_action_q` | `constraint_relation_pool_no_raw_logits` | immediate | soft_quality | 5 | 120 | `7.8250` | `1.7466` | `0.0250` | failed micro-overfit |

Interpretation: tokenizing the appended constraint scalars as typed relation tokens did not rescue the scalar feature path; it failed even same-case micro-overfit. The failure suggests the issue is not just how the appended scalar extras are encoded. A future structured graph attempt should derive typed edge/node embeddings from the underlying case graph directly rather than from flattened candidate rows.

## E3 Rollout Label Stability Across Continuation Policies

A bounded diagnostic tested whether rollout-return oracle labels are stable under three materially different continuation policies: `policy_greedy`, `immediate_oracle`, and `policy_topk_sample`. This is a label-quality experiment, not a parameter sweep.

| run | cases | seeds | steps | horizon | all-policy agreement | mean unique rollout oracles | stable gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E3 slice | 3 | 2 | 4 | 4 | `0.0000` | `2.4583` | `False` |

Pairwise rollout-oracle agreement:

- `policy_greedy__immediate_oracle`: `0.0000`
- `policy_greedy__policy_topk_sample`: `0.2917`
- `immediate_oracle__policy_topk_sample`: `0.2500`

Interpretation: raw rollout-return labels are not just hard to fit; they are continuation-policy dependent. On the hard cases, different plausible continuation policies almost never identify the same best first action. Therefore the next research path should not tune the current ranker or fusion weights. It should either learn from robust cross-continuation advantage/consensus labels, or build a true typed constraint graph/state encoder and then revisit rollout labels.

## E4 Cross-Continuation Consensus Advantage Diagnostic

Since raw rollout top1 labels are unstable, E4 asks whether the same multi-continuation evidence still contains pairwise advantage signal. It converts the E3 candidate pools into pairwise preferences across `policy_greedy`, `immediate_oracle`, and `policy_topk_sample`.

| run | pools | unanimous pair | majority pair | split pair | consensus-top hits any oracle | strict usable gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Step6E E4 | 24 | `0.3348` | `0.6369` | `0.0283` | `0.9167` | `False` |

Interpretation: strict unanimous pairwise labels are not dense enough, but majority pairwise signal is much stronger than raw top1 agreement. This suggests a real research path: train a soft/majority cross-continuation pairwise advantage objective, while treating hard rollout top1 labels as unreliable. If that fails, the bottleneck is likely representation/rollout-policy design rather than ranker loss shape.

## E5 Majority Cross-Continuation Pairwise Advantage Ranker

E5 trains a pool-local pairwise ranker from majority cross-continuation advantage labels rather than hard rollout top1 labels. This directly tests the E4 finding that top1 rollout labels are unstable but pairwise majority signal exists.

| run | cases | pools | feature mode | micro rank | micro regret | micro top1 | micro gate | LOCO rank | LOCO top1 | LOCO gate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Step6E E5 | 3 | 24 | `relational_state_pool_no_raw_logits` | `1.9167` | `0.0423` | `0.5833` | `True` | `4.5000` | `0.0833` | `False` |

Interpretation: majority advantage labels are learnable; this validates that E4 was not just noise. However, the learned advantage is still case-specific under the current flattened relational features and fails leave-one-case-out transfer. The next non-tuning move should be representation-side: true typed constraint graph/state encoder, or a case-invariant advantage representation, before 5/10-case LOCO widening.

## E6 Typed Constraint Graph Encoder + Majority Advantage

E6 replaces the policy-side state encoder with `typed_constraint_graph`, which constructs explicit typed messages for b2b, p2b, same-cluster, same-MIB, boundary-side, and fixed/preplaced-anchor relations. This is the first true graph/state representation attempt after scalar append and scalar-token trials failed.

| run | encoder | cases | pools | micro rank | micro regret | micro top1 | micro gate | LOCO rank | LOCO regret | LOCO top1 | LOCO gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| E5 baseline | `relation_aware` | 3 | 24 | `1.9167` | `0.0423` | `0.5833` | `True` | `4.5000` | `0.2820` | `0.0833` | `False` |
| Step6E E6 | `typed_constraint_graph` | 3 | 24 | `1.9167` | `0.0537` | `0.4583` | `True` | `3.4583` | `0.1898` | `0.2500` | `False` |

Interpretation: typed constraint graph representation improves LOCO transfer substantially, but does not yet pass the top1 gate. This supports the representation-side research direction while showing the current graph/action interface is still incomplete. The next step should be typed-relation/channel ablation and a more case-invariant graph-action ranker, not parameter tuning.

## E7 Typed Constraint Graph Channel Ablation

E7 ablates typed relation channels from E6 to identify whether the LOCO improvement is coming from real constraint structure or incidental model variation.

| variant | micro rank | micro top1 | micro gate | LOCO rank | LOCO regret | LOCO top1 | LOCO gate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| full | `1.9167` | `0.4583` | `True` | `3.4583` | `0.1898` | `0.2500` | `False` |
| no_anchor | `2.8750` | `0.2083` | `False` | `4.1250` | `0.2356` | `0.1667` | `False` |
| no_groups | `1.6250` | `0.5417` | `True` | `5.0833` | `0.3531` | `0.0417` | `False` |
| no_boundary | `2.9167` | `0.1667` | `False` | `3.4583` | `0.2238` | `0.2500` | `False` |

Interpretation: the transfer gain is not random. Group relations are critical for LOCO even though micro still fits without them; anchor relations also matter. Boundary alone is not the main transfer driver on this slice. Next should keep group+anchor typed relations and expose typed graph/action embeddings directly to the ranker.

## E8 Naive Graph-Action Embedding Ranker

E8 tests whether the ranker can consume direct typed-graph block/target/graph action embeddings from the trained policy encoder, instead of the existing candidate feature rows. This is a representation experiment, not a fusion-weight tuning run.

| run | ranker input | cases | pools | micro rank | micro regret | micro top1 | micro gate | LOCO rank | LOCO top1 | LOCO gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| E6 baseline | `candidate_features` | 3 | 24 | `1.9167` | `0.0537` | `0.4583` | `True` | `3.4583` | `0.2500` | `False` |
| Step6E E8 | `graph_action_embeddings` | 3 | 24 | `3.9167` | `0.2679` | `0.1667` | `False` | `3.7917` | `0.2500` | `False` |

Interpretation: naively exporting frozen policy encoder graph/action embeddings to the ranker is insufficient and fails even micro. E6/E7 remain valid: typed relations help, especially group+anchor channels, but the next architecture should be a joint action-conditioned graph ranker rather than a frozen-embedding side channel.

## E9 Joint Typed Graph-Action Ranker

E9 moves typed graph/action encoding into the ranker path instead of feeding frozen policy embeddings. The ranker trains its own node projection, typed relation messages, and action-conditioned pairwise comparator from majority cross-continuation advantage labels.

| run | ranker input | cases | pools | micro rank | micro regret | micro top1 | micro gate | LOCO rank | LOCO top1 | LOCO gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| E6 baseline | `candidate_features` | 3 | 24 | `1.9167` | `0.0537` | `0.4583` | `True` | `3.4583` | `0.2500` | `False` |
| E8 frozen graph/action | `graph_action_embeddings` | 3 | 24 | `3.9167` | `0.2679` | `0.1667` | `False` | `3.7917` | `0.2500` | `False` |
| Step6E E9 | `joint_typed_graph_action` | 3 | 24 | `3.7917` | `0.2489` | `0.2500` | `False` | `4.2083` | `0.1667` | `False` |

Interpretation: making the graph/action encoder trainable inside the ranker is still not enough in this static form. The missing ingredient appears to be action-conditioned next-state relation deltas / legality geometry, not merely typed relation messages over the pre-action graph. Also, this run exposed a throughput issue: collection uses only the number of case/seed jobs, and LOCO ranker training is sequential; the next script change should parallelize LOCO/slice runs to use the available 48 cores.

## E10 Action-Conditioned Delta Features + Parallel LOCO Harness

E10 first fixes the execution bottleneck: LOCO split training is now parallelized with spawn-safe multiprocessing. The run was launched with `--workers 48`; this slice had `12` independent collection jobs and `3` LOCO split jobs, so it used all available independent work without hitting the PyTorch autograd/fork crash.

E10 also tests a minimal action-conditioned next-state delta representation: candidate action geometry, immediate group/MIB/boundary placed-fraction deltas, target relation matches, anchor/group flags, primitive one-hot, and immediate objective components.

| run | input | cases | seeds | pools | micro rank | micro top1 | micro gate | LOCO rank | LOCO top1 | LOCO gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| E6 baseline | `candidate_features` | 3 | 2 | 24 | `1.9167` | `0.4583` | `True` | `3.4583` | `0.2500` | `False` |
| E10 | `action_delta_features` | 3 | 4 | 48 | `3.8958` | `0.2292` | `False` | `3.7708` | `0.1042` | `False` |

Interpretation: simple hand-built action deltas are not enough and underperform the E6 candidate-feature path. The useful part of this round is the execution harness fix plus the negative evidence: the next architecture needs richer learned next-state relation deltas, not manually added scalar deltas or loss/fusion tuning.

## E13 Neutral 5-Case Typed-Graph Majority Advantage Validation

After the no-manual/no-case-specific guardrail, E13 reruns the best active learned route on a neutral 5-case slice (`0,1,2,3,4`) instead of the hard-case-only slice. This uses typed constraint graph candidate generation plus majority cross-continuation pairwise advantage ranking; no hand-built delta features are active.

| run | cases | seeds | pools | micro rank | micro top1 | micro gate | LOCO rank | LOCO top1 | LOCO gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| E13 neutral5 | 5 | 2 | 40 | `1.9750` | `0.4500` | `True` | `4.0500` | `0.1500` | `False` |

Per-heldout LOCO top1: case 0 `0.0000`, case 1 `0.0000`, case 2 `0.1250`, case 3 `0.3750`, case 4 `0.2500`.

Interpretation: the majority-advantage signal still learns on a neutral slice, so the label direction is not purely hard-case overfitting. However, transfer is still weak. Future work must remain learned/case-agnostic and improve LOCO, not add manual features or specialize to these heldout cases.

## E14 Case-Invariance Probe on Neutral5 Features

E14 is a learned diagnostic, not a manual feature edit. It trains a small probe to predict case identity from the neutral5 candidate feature rows used by E13.

| probe | rows | dim | chance | eval accuracy |
| --- | ---: | ---: | ---: | ---: |
| random row split case-ID probe | 320 | 99 | `0.2000` | `0.9792` |

Interpretation: the current candidate feature representation makes case identity almost trivially recoverable. This explains the recurring pattern: majority-advantage labels micro-fit, but LOCO transfer fails. The next valid learned direction is case-invariant representation/objective learning, such as domain-adversarial or leave-case contrastive regularization, not hand-crafted feature removal or case-specific branches.

## E15 Learned Case-Adversarial Pairwise Objective

E15 is the first learned invariance objective after E14 found strong case identity leakage. It adds a gradient-reversal case classifier on the ranker encoding while keeping the majority pairwise advantage task. This is not manual feature removal and does not specialize to cases.

| run | objective | cases | pools | micro rank | micro top1 | micro gate | LOCO rank | LOCO top1 | LOCO gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| E13 baseline | `majority_pairwise` | 5 | 40 | `1.9750` | `0.4500` | `True` | `4.0500` | `0.1500` | `False` |
| E15 | `case_adversarial_pairwise` | 5 | 40 | `4.1500` | `0.1000` | `False` | `4.1500` | `0.1000` | `False` |

Interpretation: case identity leakage is real, but a naive adversarial objective is too destructive for the task signal. The next learned-invariance attempt should not tune adversarial weights; it should use a better architecture/objective, such as leave-case contrastive/meta-learning that preserves pairwise advantage structure while reducing case-local shortcuts.
