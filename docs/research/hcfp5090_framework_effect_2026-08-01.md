# HCFP-5090 runnable framework effect report — 2026-08-01

## Scope and decision

Commit `bb0e5d0b190c44dfdba4622018c41aebfc457904` closes the runnable analytic,
learned-research, evaluation, and visualization framework loop:

```text
official tensors
  -> exact-safe fallback
  -> deterministic population seeds
  -> typed FP32 dynamics
  -> active-pair BDP outer rebuild
  -> safe / fast / exact incumbent tiers
  -> exact-compatible verifier
  -> audited data shards + supervised training smoke
  -> schema/hash/normalization checked learned initializer (default-off)
  -> official weighted and bucket benchmark
  -> synchronized fixed-bucket CUDA profiler
  -> deterministic SVG / HTML visualization
```

The framework is executable, testable, and inspectable. Learned checkpoints
are strictly opt-in and always pass through the same analytic/BDP/exact tail;
missing or incompatible checkpoints fall back without weakening hard safety.
The analytic lane remains **HOLD** for contest promotion: all validation cases
are feasible and several exact metrics improve, but all 100 costs remain
capped at `9.999999` and runtime exceeds the fallback gate.

## Exact command

```bash
PYTHONPATH=src python -B scripts/benchmark_hcfp.py \
  --optimizer fallback=scripts/audit_fallback_optimizer.py \
  --optimizer analytic=submission/optimizer.py \
  --baseline fallback \
  --data-path artifacts/floorset-v10 \
  --cases all \
  --device cuda \
  --output artifacts/benchmarks/hcfp-bb0e5d0-validation100.json \
  --visualize-dir artifacts/benchmarks/hcfp-bb0e5d0-validation100 \
  --visualize-cases 0,50,99
```

The machine-readable report SHA256 is:

```text
be2d7d05932267c3afcb9cedb6754b0b60999856daf715b8b368c9e457a096f1
```

## Official exact results

| Metric | Fallback | Analytic framework |
| --- | ---: | ---: |
| Hard-feasible | 100/100 | 100/100 |
| 106–120 hard-feasible | 15/15 | 15/15 |
| Weighted cost | 9.999999 | 9.999999 |
| Capped feasible cases | 100 | 100 |
| Runtime p50 | 0.157180 s | 0.493083 s |
| Runtime p95 | 0.492210 s | 1.323080 s |
| Runtime p99 | 0.729586 s | 1.679867 s |
| Runtime max | 0.823637 s | 1.801551 s |

Per-case analytic-minus-fallback effects:

| Exact metric | Mean delta | Improved | Regressed | Tied |
| --- | ---: | ---: | ---: | ---: |
| HPWL gap | -4.379549 | 79 | 6 | 15 |
| Area gap | -1.659402 | 50 | 35 | 15 |
| Relative soft violation | -0.038513 | 69 | 0 | 31 |
| Runtime | +0.389543 s | 2 | 98 | 0 |

This is a real geometric effect but not yet a score improvement. The correct
promotion result is therefore `HOLD`, not `PROMOTE`.

## Visualization evidence

The benchmark emitted self-contained comparison pages:

```text
584b15d2941e2a0ea9f9ae27ec2611938840f9be365b4f83051a5c7cc1d39a64  case_0.html
7feaf3cbb08576e6517746dd42569eca1b3c875434aa03c0401b79a3c8a028bc  case_50.html
5bec215d9497eb3cc788961882913af1d86bb2733b173a899a810f995a337caf  case_99.html
```

Each page embeds fallback and analytic placements with deterministic block
labels, constraint coloring, bounding boxes, exact cost/gap metadata, and no
external JavaScript, CSS, or plotting dependency.

## RTX 5090 fixed-bucket profile

The real `solve_case` fast path was synchronized on CUDA with `N=120`, `K=32`,
four dynamics steps, eight projection iterations, and a direction beam of two:

| Metric | Result |
| --- | ---: |
| Raw/projected candidates | 65 / 65 |
| Exact-feasible candidates | 10 |
| Selected incumbent | `candidate_2` |
| Runtime p50 | 0.957574 s |
| Runtime p95 | 0.962069 s |
| Runtime max | 0.962568 s |
| CUDA peak allocation | 76,457,984 bytes |

Profile report:
`artifacts/reports/profile-bb0e5d0-n120-k32-cuda.json`, SHA256
`923e1f48564d9fffc0369d8e9a43314a2eeecb53e128d344756d0bee7dd31284`.
Telemetry was collected once outside the timed repeats.

## Data and training smoke

A synthetic 32-block sample completed the full
`DataSample -> tar/manifest -> stream loader -> CUDA training -> checkpoint`
path. The shard SHA256 is
`b045c558464182bd7aace8b1430e47e29e0e9b40baac4e82fe128eb0c42d40b5`;
the checkpoint state hash is
`27bd952bad8fc8772070681a0b9a996fbcafadd8f3c4d03fe0641cfd3e3e39dc`.
This proves execution and serialization, not learned QoR.

## Verification

- 119 repository tests passed.
- Ruff and `compileall` passed.
- The official test suite includes analytic/fallback and strict positive
  learned-checkpoint benchmark paths against the pinned local validation cache.
- Independent review returned `ACCEPT` after validation leakage, shard
  provenance, streaming, normalization, and checkpoint-attribution issues were
  repaired.

## Remaining research gate

The model/data/checkpoint contracts exist, but the 1M training inventory,
contact/event/repair labels, trained oracle@K, and controller/ranker calibration
remain future promotion work. Learned output stays default-off until post-BDP
exact metrics beat the analytic framework. No validation case may be used for
training or threshold fitting.
