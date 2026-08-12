# HCFP-5090 P3.4 Runtime B*-Tree Results

## Hypothesis

A small tree-supervised head can recover enough runtime topology to capture part of the usable gold-tree oracle headroom, even when exact serialized edge recovery is low.

## Minimal implementation

- Direct FloorSet-Lite streaming now carries optional `tree_sol` edges in `DataSample`.
- Added optional B*-Tree root and joint `(parent, branch)` heads.
- Added a `btree` training stage that freezes the existing model and trains only six new head parameters.
- Added a deterministic hard decoder that starts from the predicted root and only attaches an unconnected node to an available branch of the connected tree. Every decoded result is rooted, connected, acyclic, and binary.
- Added `LearnedConfig.btree_seeds` / `HCFP_BTREE_SEEDS` and an additive B*-Tree candidate family. The incumbent remains in the portfolio.

## Training smoke

```text
large samples: 300
trainable tensors: 6 B*-Tree head parameters
steps: 300
loss: 10.802401 -> 3.736812
```

Checkpoint:

`artifacts/checkpoints/hcfp5090-p34-btree-head-s300-seed7301.pt`

## Runtime-available topology audit

On eight large samples, using model-predicted trees rather than gold trees:

| Metric | Current portfolio | Model-tree oracle |
| --- | ---: | ---: |
| Weighted cost | 8.535005 | 8.073323 |
| Unique wins | - | 5/8 |
| Hard feasible | 8/8 | 8/8 |
| Mean exact tree-edge accuracy | - | 13.0% |
| Root accuracy | - | 100% |

Low edge accuracy with positive candidate QoR confirms that serialized edge accuracy is not the promotion metric.

## Integrated large15 benchmark

The exact benchmark used the frozen P2.5 settings plus four additive B*-Tree seeds per case.

| Metric | P2.5 | P3.4 runtime | Delta |
| --- | ---: | ---: | ---: |
| Weighted cost | 7.332108 | 7.095689 | -0.236419 |
| Below cap | 14/15 | 14/15 | unchanged |
| Hard feasible | 15/15 | 15/15 | unchanged |
| Improved / tied / regressed | - | 3 / 12 / 0 | zero regressions |

Changed cases:

```text
85: 7.272539 -> 6.923185
92: 8.291613 -> 7.641153
98: 9.797765 -> 8.017462
```

Case 89 remains the only capped large15 case.

## Decision

**KEEP.** Runtime B*-Tree candidates add measurable QoR with four candidates per case and no hard-feasibility regression. Keep the family additive; the next stage is selector/replay calibration and targeted work on case-89-like generation gaps.

Artifacts:

- `artifacts/benchmarks/hcfp5090-p34-model-tree-large8.json`
- `artifacts/benchmarks/hcfp5090-p34-btree-runtime-large15.json`
