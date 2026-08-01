# HCFP-5090 runnable framework effect report — 2026-08-01

## Scope and decision

Commit `de93922a005a34f66c5cfbcc37ffe3c2a66faa21` closes the runnable analytic
framework loop:

```text
official tensors
  -> exact-safe fallback
  -> deterministic population seeds
  -> typed FP32 dynamics
  -> active-pair BDP outer rebuild
  -> safe / fast / exact incumbent tiers
  -> exact-compatible verifier
  -> official weighted and bucket benchmark
  -> deterministic SVG / HTML visualization
```

The framework is executable, testable, and inspectable. The analytic lane
remains **HOLD** for contest promotion: all validation cases are feasible and
several exact metrics improve, but all 100 costs remain capped at `9.999999`
and runtime exceeds the fallback gate.

## Exact command

```bash
PYTHONPATH=src python -B scripts/benchmark_hcfp.py \
  --optimizer fallback=scripts/audit_fallback_optimizer.py \
  --optimizer analytic=submission/optimizer.py \
  --baseline fallback \
  --data-path artifacts/floorset-v10 \
  --cases all \
  --device cuda \
  --output artifacts/benchmarks/hcfp-de93922-validation100.json \
  --visualize-dir artifacts/benchmarks/hcfp-de93922-validation100 \
  --visualize-cases 0,50,99
```

The machine-readable report SHA256 is:

```text
cc323d629540252d027228a6c1c65461b73e37851abd42e7b2a156cc9ca588c8
```

## Official exact results

| Metric | Fallback | Analytic framework |
| --- | ---: | ---: |
| Hard-feasible | 100/100 | 100/100 |
| 106–120 hard-feasible | 15/15 | 15/15 |
| Weighted cost | 9.999999 | 9.999999 |
| Capped feasible cases | 100 | 100 |
| Runtime p50 | 0.158948 s | 0.492917 s |
| Runtime p95 | 0.498124 s | 1.318761 s |
| Runtime p99 | 0.738522 s | 1.686754 s |
| Runtime max | 0.833473 s | 1.799699 s |

Per-case analytic-minus-fallback effects:

| Exact metric | Mean delta | Improved | Regressed | Tied |
| --- | ---: | ---: | ---: | ---: |
| HPWL gap | -4.323762 | 79 | 5 | 16 |
| Area gap | -1.600589 | 49 | 35 | 16 |
| Relative soft violation | -0.038913 | 69 | 0 | 31 |
| Runtime | +0.389683 s | 3 | 97 | 0 |

This is a real geometric effect but not yet a score improvement. The correct
promotion result is therefore `HOLD`, not `PROMOTE`.

## Visualization evidence

The benchmark emitted self-contained comparison pages:

```text
bc47de2f0ab04ab2593bce3ab186bb12353b5cf900ec328daee87c54d3423067  case_0.html
65aced05b3981f80ddbf6d503979558ef303f9564ac7acd06175749e1dcdfb1e  case_50.html
feabe7d759241302b011b33f4e3c0b3f9ed26faabace145c852c82f0b98e13ec  case_99.html
```

Each page embeds fallback and analytic placements with deterministic block
labels, constraint coloring, bounding boxes, exact cost/gap metadata, and no
external JavaScript, CSS, or plotting dependency.

## Verification

- 95 repository tests passed.
- Ruff and `compileall` passed.
- The official one-case integration benchmark is part of the test suite when
  the pinned local validation cache is present.
- An independent review found one medium promotion-gate issue. It was fixed by
  enforcing hard feasibility, capped-case, large-bucket, p50, and p95 gates.

## Remaining research gate

Learned SCENE/initializer/controller/ranker modules and data shards remain
default-off until their checkpoint/data contracts exist and post-BDP exact
metrics beat this analytic framework. No validation case may be used for
training or threshold fitting.
