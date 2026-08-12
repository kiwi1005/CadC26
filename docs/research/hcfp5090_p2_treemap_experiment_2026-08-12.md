# HCFP-5090 P2 exact-area treemap experiment — 2026-08-12

## Hypothesis

Replacing loose residual seeds with recursive slicing inside a P1 latent outline
should reduce bbox area and projection displacement while the existing exact tail
preserves feasibility.

## Final minimal implementation

- structured geometry supplies deterministic leaf order only;
- cluster members are packed as compound slicing units;
- inferred-outline slack becomes a whitespace leaf;
- preplaced and fixed-shape blocks are placed first as exact obstacles;
- soft blocks tile the remaining free rectangles;
- compatible MIB groups share one exact shape and are packed as one obstacle unit;
- boundary-aware region choice preserves requested perimeter contact when possible;
- `--treemap-seeds` adds an explicit challenger without deleting residual seeds;
- treemap provenance is mapped to exact merged-candidate indices;
- the old soft-first selector cannot silently select a treemap; a feasible raw
  treemap must beat the repaired incumbent on an input-observable area/soft
  proxy without increasing HPWL;
- the experiment is default-off.

## Experiment result

Checkpoint and search settings match the Q2 large checkpoint control:
`flow=0`, `collective=0`, seed `6501`, 16 topology seeds, 16 constraint seeds,
and one treemap challenger.

| Metric | Control | P2 final |
| --- | ---: | ---: |
| large15 weighted cost | 8.822391 | 7.492749 |
| hard feasible | 15/15 | 15/15 |
| below cap | 7/15 | 12/15 |
| regressed cases | — | 0 |
| case 86 cost | 9.999999 | 5.935267 |
| case 92 cost | 8.588517 | 8.588517 |
| case 96 cost | 5.511786 | 5.511786 |
| case 99 cost | 9.991075 | 5.714720 |

Case 86 exposed the selector bug directly: the raw treemap was hard feasible,
zero-overlap, had `area_gap=-0.0107`, and official cost `6.383008`, but the old
selector rejected it because it had two additional boundary violations. The
post-tail treemap result improves further to `5.935267` because grouping repair
restores the incumbent soft ratio while retaining dense geometry.

An intermediate overwrite experiment regressed case 96 from `5.511786` to
`6.213367` because the treemap deleted the original winning residual candidate.
It was rejected. Keeping the treemap as an explicit challenger and excluding it
from the legacy soft-first merge restored case 96 exactly while preserving the
case 86/99 gains.

Artifacts:

- `artifacts/benchmarks/hcfp5090-p2-treemap-incumbent-guard-large15.json`
- `artifacts/benchmarks/hcfp5090-p2-treemap-final-smoke-86-96.json`

## Decision

`KEEP` — exact packing plus incumbent-aware selection gives a large strict-score
gain with no large15 regression. The next experiment should target the three
still-capped cases, especially grouping/contact construction, without changing
this treemap incumbent guard.
