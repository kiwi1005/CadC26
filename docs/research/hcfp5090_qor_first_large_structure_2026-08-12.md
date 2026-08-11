# HCFP-5090 large-case QoR-first result — 2026-08-12

## Decision

For 106--120 block cases, use the large-case structure checkpoint with 16
Sequence-Pair topology seeds and 16 constraint seeds. Keep the smaller-case
checkpoint unchanged.

The useful gain came from training the structure that controls candidate
geometry, then enabling the already implemented constraint construction. More
absolute-center regression and overlap-only fine-tuning did not improve the
active topology lane.

## Training recipe

```text
parent:      hcfp5090-q2-constraints-s1000-seed5090.pt
data:        FloorSet training source only
block range: 106--120
stage:       structure
steps:       3000
population:  8
learning rate: 3e-5
EMA:         off
seed:        6501
```

Checkpoint:

```text
artifacts/checkpoints/hcfp5090-q2-structure-large-s3000-seed6501.pt
```

The structure loss fell from `1.663877` to `0.895539`.

## Exact public-large result

Command configuration:

```text
cases:            85--99
topology seeds:   16
constraint seeds: 16
flow:             off
collective:       off
```

Report:

```text
artifacts/benchmarks/
  hcfp5090-q2-structure-large-s3000-seed6501-
  constraints16-official-large15-exact.json
```

Results:

| Metric | Analytic | Learned structured |
| --- | ---: | ---: |
| Hard feasible | 15/15 | 15/15 |
| Weighted capped cost | 9.999999 | 8.822391 |
| Cases below cap | 0 | 7 |
| Runtime p50 | 0.390 s | 6.015 s |
| Runtime p95 | 0.799 s | 7.803 s |

The seven uncapped learned cases were 85, 87, 91, 92, 95, 96, and 99.
Case 96 reached the best observed cost in this batch at `5.511786`.

## Ablation result

With topology seeds but no constraint seeds, the same checkpoint remained
capped on 14/15 cases and had weighted cost `9.998999`. Enabling constraint
seeds reduced the soft-violation ratio enough to cross the cap on seven cases.

The absolute-initializer experiment learned target centers successfully, but
its large-case topology oracle remained behind the Q2 topology checkpoint.
The ineffective aspect-clamp experiment was discarded instead of entering the
runtime path.

## Runtime policy

`HCFP_CHECKPOINT` remains the general checkpoint. For cases with at least 106
blocks, runtime uses `HCFP_LARGE_CHECKPOINT` when set and enables 16 topology
plus 16 constraint seeds by default. The threshold and seed counts remain
environment-overridable for contest sweeps.

## Next QoR target

Do not add a larger model yet. Train/replay the near-cap and still-capped large
cases, with priority on reducing grouping, boundary, and MIB violations while
preserving the current topology geometry. Runtime reduction comes after the
next cap-cross gain.
