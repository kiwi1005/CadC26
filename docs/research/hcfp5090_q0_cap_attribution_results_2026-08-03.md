# HCFP-5090 Q0 cap-attribution results

Date: 2026-08-03  
Implementation branch: `feat/hcfp5090-qor-first`

## Result

Q0 is implemented as additive analysis code. It does not import from or alter
runtime candidate generation, selection, verification, or submission paths.

The exact unit fixtures reconstruct the pinned official-v10 uncapped cost in
both multiplicative and log domains. `cap_margin` is `log(10) - J`; positive
values are below the official cap.

## Validation-100 attribution

Input evidence:
`artifacts/benchmarks/hcfp5090-pareto-guard-final-validation100.json`

Command:

```bash
PYTHONPATH=src python scripts/report_cap_sources.py \
  --input artifacts/benchmarks/hcfp5090-pareto-guard-final-validation100.json \
  --lane learned \
  --output artifacts/reports/hcfp5090-q0-cap-attribution-learned-validation100.json
```

Observed learned-lane summary:

| Metric | Result |
|---|---:|
| cases | 100 |
| hard feasible | 100 |
| capped | 100 |
| mixed dominated | 93 |
| quality dominated | 7 |
| hard/soft/projection dominated | 0 / 0 / 0 |

This confirms that the immediate problem is not hard feasibility. Most current
cases require both constraint and quality improvement; seven remain capped even
with no soft penalty and are quality-only targets.

## Evidence limitation

The historical benchmark schema stores `violations_relative` but not the raw
boundary, grouping, MIB, or maximum-soft counts. Q0 therefore reports those
per-constraint contributions and integer `k_req` values as unavailable for this
legacy artifact. Exact breakdown is verified in unit tests and will be present
when future benchmark rows retain the raw official soft counts.

The historical lane report also lacks paired raw/projected/post-repair records,
so it cannot classify projection dominance. The Q0 staged-schema tests cover
both cap crossing and uncapped `J` regression; Q1/Q4 replay must preserve the
three stages to make the classification material.

## Verification

```text
tests/test_score_attribution.py: 10 passed
official cost parity target: passed
Ruff: passed
git diff --check: passed
validation report: 100 cases parsed deterministically
```

