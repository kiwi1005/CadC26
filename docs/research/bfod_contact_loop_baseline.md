# Contact-Driven Obligation Loop v1 — regression baseline

Pinned configuration for the frozen v1 experiment (2026-08-14):

```text
beam=4  max_rounds=6  top_experts=2  proposals_per_operator=4
patch_sizes=4,8,12,16  exact_decode_cap=96  runtime_ceiling=30s (official-style) / 240s (untruncated)
```

Scope: P8/P7 incumbents -> S0 hard protection -> S3 contact generator -> S5 common
loop. Production `solve()` untouched; every candidate passes the exact verifier.

Artifacts are gitignored; numbers below are the reference for the Generator v2 gate.

## Ceiling 240s (decode-cap-bound, time does not truncate)

Source: `artifacts/experiments/bfod_v1_contact_only_c240/` (v1 contact-only)
and `artifacts/experiments/bfod_v1_c240/` (full BFOD; identical winners on
70/90/94/97, worse on 89).

| Case | Baseline cost | v1 contact-only cost | Final G | Decodes | Runtime |
|---:|---:|---:|---:|---:|---:|
| 70 | 10.0000 | 7.2916 | 13 | 72 | 7.2s |
| 89 | 9.6928 | 9.3338 | 17 | 0 (bootstrap only) | 1.6s |
| 90 | 7.9484 | 7.7116 | 29 | 0 (bootstrap only) | 1.5s |
| 94 | 8.0341 | 6.2509 | 17 | 96 | 10.9s |
| 97 | 5.9782 | 4.2802 | 5 | 70 | 12.2s |

All winners hard-feasible; bbox and area_gap bit-identical to baseline on
every accepted step; grouping is the only soft-debt mover.

## Ceiling 30s (official-style budget)

Source: `artifacts/experiments/bfod_v1_contact_only_c30/`.

| Case | Full BFOD 30s cost (G) | v1 contact-only 30s cost (G) |
|---:|---:|---:|
| 70 | 8.9526 (19) | 7.2916 (13) |
| 89 | 9.6803 (18) | 9.3338 (17) |
| 90 | 7.7116 (29) | 7.7116 (29) |
| 94 | 7.2967 (22) | 6.2509 (17) |
| 97 | 5.2183 (11) | 4.2802 (5) |

## Case70 trajectory signature (audit, ceiling 240)

`artifacts/experiments/bfod_v1_audit/case70/audit.json` + `audit_report.md`.

7 accepted steps, all contact family, G 23->13, cost 10.0000->7.2916:

- groups: G4 x3, G3 x2, G2 x1, G1 x1
- bridge always a single block; moving component singleton in 5/7
- patch size 2 in 5/7, 4 in 2/7
- horizontal 5/7, vertical 2/7
- HPWL neutral (|Δhpwl_gap| <= 6e-3); bbox/area untouched
- oracle == selected every round (no ranking gap; generator is the bottleneck)

## v2 gate

Generator v2 (Group-first, patch 2/4) must keep QoR >= these numbers
(case70 ~7.29) with significantly fewer generator repack attempts / runtime.
