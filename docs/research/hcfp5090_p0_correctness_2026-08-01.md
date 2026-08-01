# HCFP-5090 P0 correctness evidence — 2026-08-01

## Decision

- **P0 correctness: PROMOTE.** The official 100-case validation replay is
  100/100 hard-feasible for both the full HCFP runtime and the isolated safe
  fallback. The 106–120 block subset is 15/15 hard-feasible in both runs.
- **P1 quality: HOLD.** All 100 cases in both runs have `cost >= 9.99`. The
  current analytic lane is a correctness baseline, not a competitive QoR
  baseline.
- The local evaluator fixes `RuntimeFactor=1.0`; these measurements do not
  predict the formal cross-submission runtime factor.

## Provenance

| Item | Value |
| --- | --- |
| Branch | `feat/hcfp5090-greenfield` |
| Full-runtime source commit | `d3fc9997c49e56ede19b02b10f7315bc97458ea9` |
| Fallback audit adapter commit | `9d6f961d1361364e15678477342dffd7901bd220` |
| FloorSet checkout commit | `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa` |
| Evaluator SHA256 | `64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498` |
| Validation inventory | 100 litedata + 100 litelabel, `config_21` through `config_120` |
| GPU | NVIDIA GeForce RTX 5090, driver 610.43.02, 32607 MiB |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |

The evaluator checkout and replay JSON files are ignored local evidence, not
submission payload. Their hashes below make the exact runs auditable without
committing the upstream dataset or generated placements.

## Exact invocations

Full HCFP runtime:

```bash
HCFP_DEVICE=cuda PYTHONPATH=src python -B \
  artifacts/floorset-v10/iccad2026contest/iccad2026_evaluate.py \
  --evaluate submission/optimizer.py \
  --data-path artifacts/floorset-v10 \
  --output artifacts/reports/hcfp-d3fc999-validation100.json
```

Fallback-only audit:

```bash
HCFP_DEVICE=cpu PYTHONPATH=src python -B \
  artifacts/floorset-v10/iccad2026contest/iccad2026_evaluate.py \
  --evaluate scripts/audit_fallback_optimizer.py \
  --data-path artifacts/floorset-v10 \
  --output artifacts/reports/hcfp-9d6f961-fallback100.json
```

## Results

| Metric | Full HCFP runtime | Fallback only |
| --- | ---: | ---: |
| Tests | 100 | 100 |
| Hard-feasible | 100 | 100 |
| Evaluator errors | 0 | 0 |
| 106–120 block feasible | 15/15 | 15/15 |
| Average cost / total score | 9.999999 | 9.999999 |
| Cases with `cost >= 9.99` | 100 | 100 |
| Average runtime | 0.513950 s | 0.202637 s |
| Runtime p50 | 0.413986 s | 0.157940 s |
| Runtime p95 | 1.118387 s | 0.496697 s |
| Runtime p99 | 1.441782 s | 0.734643 s |
| Runtime max | 1.560965 s | 0.830374 s |

Report hashes:

```text
d1f291ae9be6bc7076bdd4eced3945c4e125b2bbc54db5839fc313af53f3bf56  hcfp-d3fc999-validation100.json
a0528c2a7f48a60dee4c86193a7f0f7f9356c754f91914d5ba0b7dae9631c18c  hcfp-9d6f961-fallback100.json
```

## Supporting correctness checks

- Boundary codes 0–15 match the pinned official evaluator predicates.
- MIB rounding compatibility, grouping edge contact, fixed-shape, and
  preplaced target tolerance have regression fixtures.
- CPU output is exact across ten repeats. CUDA repeatability and CPU/CUDA
  agreement are tolerance-bounded.
- The official optimizer constructor accepts the evaluator's `verbose`
  argument.
- Candidate telemetry is opt-in; the official fast path does not compute it.
- The full repository suite passed 78 tests before this report was committed.

## Known gaps and next gate

- The official result format does not expose whether the full runtime selected
  its analytic candidate or retained the safe incumbent. The isolated
  fallback-only replay proves fallback correctness, but not the full-run
  fallback selection rate.
- Runtime was measured locally on RTX 5090. A100 portability and formal
  cross-submission runtime normalization remain untested.
- P1 must now improve post-projection exact cost. The next implementation batch
  is structured per-candidate projection results, active-pair rebuilding,
  bounded direction search, incumbent tiers, and exact baseline comparisons.
