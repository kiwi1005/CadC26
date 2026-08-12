# HCFP-5090 P5 selector / replay experiment

## Hypothesis

Explicit treemap and B*-Tree family identity plus more near-cap replay should let the repair-aware ranker recover useful structured candidates in its top four.

## Changed

- Replay now preserves `treemap` and `btree` provenance.
- Ranker features use five explicit candidate-family bits instead of folding treemap and B*-Tree into `learned`.
- The counterfactual audit accepts treemap and B*-Tree seed counts and remains output-neutral.
- Generated 16-case and 64-case large training replays; official visible cases were used only for shadow evaluation.

## Experiment

All runs used 16 topology, 16 constraint, 1 treemap and 4 B*-Tree seeds. The selector remained counterfactual-only and the existing candidate-funnel proxy selected the returned placement.

| Ranker | Replay | Top-1 exact-source recall | Top-4 exact-source recall | Pareto accept | Hard feasible |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 folded-family | 16 cases | 0/15 | not meaningful | 0/15 | 15/15 |
| v5 explicit-family | 16 cases | 1/15 | 10/15 | 0/15 | 15/15 |
| v5 explicit-family | 64 cases | 1/15 | 10/15 | 0/15 | 15/15 |

The v5 representation fixed the obvious family collapse: its 60 top-four slots contained 45 B*-Tree, 12 treemap and 3 topology candidates on the 16-case replay model. Expanding replay to 64 cases did not improve exact-source recall or produce a Pareto-dominating counterfactual selection.

Artifacts:

- `artifacts/replay/hcfp5090-p5-btree-nearcap-16.pt`
- `artifacts/replay/hcfp5090-p5-btree-nearcap-64.pt`
- `artifacts/checkpoints/hcfp5090-p5-btree-ranker-v5-s200-seed7501.pt`
- `artifacts/checkpoints/hcfp5090-p5-btree-ranker-v5-r64-s400-seed7502.pt`
- `artifacts/benchmarks/hcfp5090-p5-btree-ranker-v5-counterfactual-large15.json`
- `artifacts/benchmarks/hcfp5090-p5-btree-ranker-v5-r64-counterfactual-large15.json`

## Decision

**REJECT ranker promotion; KEEP replay, family identity and shadow audit.**

The current observable proxy remains the active selector. More iterations on the same small candidate-only ranker are not justified until candidate-specific topology or exact post-tail targets provide additional signal.

