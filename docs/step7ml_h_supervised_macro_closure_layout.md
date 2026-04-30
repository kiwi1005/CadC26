# Step7ML-H Supervised Macro Closure Layout Training

Step7ML-H trains a small supervised closure-internal layout prior from
Step7ML-G `layout_prior_examples`. The target is FloorSet training `fp_sol`
normalized internal block coordinates only. Step7 candidate quality labels are
reserved for downstream evaluation/ranking and are not used as layout targets.

## Baselines

- `area_rank_template_mean`: deterministic normalized template by closure type,
  block-count bucket, and block area rank.
- `mlp_set_block_baseline`: small MLP over per-block geometry/fixity/context
  features that predicts normalized internal `(x, y)`.

## Evaluation

1. Training/validation layout-prior metrics: coordinate MAE, pairwise order
   accuracy, closure-type and block-count bucket breakdowns.
2. Step7P-style payload evaluation: model predictions are snapped into the
   closure bbox, fixed/preplaced blocks are preserved, overlap/non-noop/fixity
   are checked, and Step7P metric/gate provenance is joined only after geometry
   screening.

This remains sidecar-only and does not modify runtime solver or finalizer
semantics.
