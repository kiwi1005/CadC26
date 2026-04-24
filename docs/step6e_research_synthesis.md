# Step6E Research Synthesis: Label Horizon, Typed Relations, and Graph-Action Failures

Date: 2026-04-24

## Research framing

Step6E is a research branch, not a product hardening branch. The goal is to identify which modeling assumptions are wrong and which signals are still worth pursuing. The rule for this branch is: do not tune fusion weights or sweep parameters as the main activity; prefer architecture/label/representation experiments with rollbackable evidence.

## Current thesis

The current bottleneck is not a missing scalar weight. It is a coupled label/representation problem:

1. one-step immediate labels are too short-horizon on the hard cases;
2. raw rollout-return top1 labels are continuation-policy dependent;
3. majority cross-continuation pairwise advantage labels contain real signal;
4. transferable signal depends on structured group/anchor constraint relations;
5. static graph-action embeddings and hand-built action deltas do not yet encode the needed next-state consequences.

## Evidence table

| Lane | Method | Micro result | LOCO / diagnostic result | Decision |
| --- | --- | --- | --- | --- |
| Lane 15 | immediate vs rollout-return diagnostic on cases 1/4/6 | n/a | immediate/rollout oracle agreement `0.1667`; mismatch `0.8333`; immediate-best rollout rank `5.2917` | immediate labels are horizon-biased on hard cases |
| Lane 16 | naive rollout-return labels | rank `7.5583`, top1 `0.1667` | not widened | raw rollout labels fail micro; do not use hard replacement |
| Lane 16b | fixed top-k rollout target | rank `6.9083`, top1 `0.0917` | not widened | simple target smoothing does not fix rollout labels |
| Lane 17 | scalar typed constraint features | micro rank `3.9167`, top1 `0.3083` | LOCO rank `6.0833`, top1 `0.1500` | scalar append overfits / regresses transfer |
| Lane 17b | tokenized scalar constraint extras | rank `7.8250`, top1 `0.0250` | not widened | tokenizing flattened extras fails micro |
| Lane 18 | rollout label stability | n/a | all-policy agreement `0.0000`, mean unique rollout oracles `2.4583` | rollout top1 labels are policy-dependent |
| Lane 19 | consensus/advantage diagnostic | n/a | majority pair fraction `0.6369`, consensus-top hits any oracle `0.9167` | pairwise majority signal exists |
| Lane 20 / E5 | majority pairwise advantage ranker | rank `1.9167`, top1 `0.5833` | 3-case LOCO rank `4.5000`, top1 `0.0833` | signal is learnable but case-specific |
| Lane 21 / E6 | typed constraint graph encoder + advantage | rank `1.9167`, top1 `0.4583` | LOCO rank `3.4583`, top1 `0.2500` | typed graph improves transfer but not enough |
| Lane 22 / E7 | typed-channel ablation | full rank `1.9167`, top1 `0.4583` | no-groups LOCO rank `5.0833`, top1 `0.0417`; no-anchor LOCO rank `4.1250`, top1 `0.1667` | group + anchor relations drive transfer |
| Lane 23 / E8 | frozen graph-action embeddings | rank `3.9167`, top1 `0.1667` | LOCO rank `3.7917`, top1 `0.2500` | frozen policy embeddings are insufficient |
| Lane 24 / E9 | trainable static graph-action ranker | rank `3.7917`, top1 `0.2500` | LOCO rank `4.2083`, top1 `0.1667` | static graph-action ranker still misses action consequence |
| Lane 25 / E10 | hand-built action delta features + parallel LOCO | rank `3.8958`, top1 `0.2292` | LOCO rank `3.7708`, top1 `0.1042` | scalar deltas fail; harness parallelized |

## What is ruled out

### 1. Do not replace immediate labels with raw rollout-return top1 labels

The hard-case diagnostic shows immediate labels are wrong on cases 1/4/6, but raw rollout-return labels are not stable enough to be a direct target. Different continuation policies select different first-action oracles almost all the time.

### 2. Do not keep appending scalar constraint features

Scalar typed features and tokenized scalar extras both failed. They expose useful information but do not create transferable abstraction.

### 3. Do not treat graph embeddings as a simple side channel

Frozen graph/action embeddings failed micro. A trainable static graph-action ranker also failed. The missing part is not merely access to node embeddings; it is action-conditioned consequence modeling.

### 4. Do not tune fusion weights as the next move

The strongest results came from label/representation changes, not weight selection. E6 improved transfer through typed group/anchor relations; E10 showed scalar deltas are not enough.

## What remains promising

### Majority pairwise advantage labels

E5 proved the majority cross-continuation advantage signal is learnable on the hard-case micro slice. This remains the best label direction, but it needs a representation that transfers.

### Typed group and anchor relations

E7 is the clearest representation evidence:

- removing group channels collapses LOCO despite good micro fit;
- removing anchor channels also regresses;
- boundary alone is not the main transfer driver on this slice.

This suggests the next model should keep explicit group/anchor relation structure.

### Learned action-conditioned next-state relation deltas

E9 and E10 failed because they used either static pre-action graph relations or hand-built scalar deltas. The next real architecture should learn relation deltas caused by applying the candidate action, especially:

- group completion / fragmentation changes;
- anchor-relative placement consequences;
- target relation compatibility after placement;
- legality/geometry consequences after the action, not just before it.

## Execution/harness finding

E10 fixed a practical research bottleneck: LOCO split training now uses spawn-safe multiprocessing to avoid PyTorch autograd + fork failures. The E10 run was launched with `--workers 48`; the slice exposed 12 independent collection jobs and 3 LOCO split jobs.

Future wider validations should use:

```bash
--workers 48
```

and should increase independent jobs via more cases/seeds/splits when the machine should be fully saturated.

## Recommended next experiment

### E11: learned relation-delta ranker, not scalar deltas

Design a ranker that builds both pre-action and post-action typed relation summaries for each candidate, then learns the delta representation. Keep the E7 relation evidence:

- include cluster/MIB group relation channels;
- include fixed/preplaced anchor relation channels;
- boundary can remain but should be ablated again.

Acceptance shape:

1. first gate: hard-case micro should beat E6/E5 micro top1 or at least pass micro gate;
2. second gate: 3-case LOCO should beat E6 `rank 3.4583 / top1 0.2500`;
3. only after that widen to 5-case and 10-case LOCO with `--workers 48`.

If E11 fails micro, stop architecture implementation and analyze candidate generation / label-policy mismatch again. If E11 passes micro but fails LOCO, the representation still encodes case identity rather than reusable group/anchor constraints.
