# HCFP-5090 P2 exact-area treemap experiment — 2026-08-12

## Hypothesis

Replacing loose residual seeds with recursive slicing inside a P1 latent outline
should reduce bbox area and projection displacement while the existing exact tail
preserves feasibility.

## Minimal implementation

- structured geometry supplies deterministic leaf order only;
- cluster members are packed as compound slicing units;
- inferred-outline slack becomes a whitespace leaf;
- preplaced and fixed-shape blocks are placed first as exact obstacles;
- soft blocks tile the remaining free rectangles;
- `--treemap-seeds` replaces residual slots and does not increase population;
- the experiment is default-off.

## Representative result

Checkpoint and search settings match the Q2 large checkpoint control. Cases 86,
92, and 99 were run separately because another GPU job was active.

| Metric | Control | P2 treemap4 |
| --- | ---: | ---: |
| hard feasible | 3/3 | 3/3 |
| case 86 cost | 9.999999 | 9.999999 |
| case 92 cost | 8.588517 | 8.588517 |
| case 99 cost | 9.991075 | 9.991075 |

The useful geometry signal is visible inside case 86: three raw treemap seeds
are hard feasible with zero overlap, bbox area `1.025641`, and zero projection
displacement. They are not selected because soft violation remains about
`0.945`, worse than the current structured incumbent. Constraint post-processing
reduced some soft terms but made the dense candidates overlap, so raw feasible
seeds remain in the slot pool.

A follow-up boundary-biased leaf-order ablation left boundary violations at
`25` and increased MIB violations from `3` to `4`; it was rejected and removed.

Artifacts:

- `artifacts/benchmarks/hcfp5090-p2-constraint-treemap4-case86.json`
- `artifacts/benchmarks/hcfp5090-p2-treemap4-case92.json`
- `artifacts/benchmarks/hcfp5090-p2-treemap4-case99.json`

## Decision

`MODIFY` — keep the default-off candidate family because exact packing works,
but do not expand to large15 yet. The next experiment must preserve cluster
abutment and boundary/MIB construction during slicing; ordinary post-hoc
constraint translation destroys the dense packing advantage.
