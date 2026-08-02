# HCFP-5090 learned-tail activation phase

Date: 2026-08-02

## Outcome

Learn when the learned split tail is likely to improve the protected
analytic-plus-Pareto result, so the runtime can skip model-tail work on clear
analytic-retention cases without fitting to visible validation or weakening
any exact gate.

The implementation is staged as training-only paired replay, then a shadow
policy, and only then an optional fail-open runtime gate.

## Fresh evidence

### Runtime headroom

After exact-overlap vectorization, clean validation 100 reports:

- analytic p50/p95: `0.215 / 0.475 s`;
- learned p50/p95: `0.367 / 0.505 s`;
- learned/analytic ratios: `1.707x / 1.062x`.

The p95 gate now passes. The remaining issue is median overhead on the many
cases where the learned tail does not change the final result.

### Existing ranker is not an activation classifier

The current ranker is trained on per-case standardized candidate targets. It
can order learned candidates inside one case, but its absolute scores do not
answer whether the learned tail beats the protected analytic result.

A diagnostic-only visible-validation audit confirms this limitation:

```text
artifacts/benchmarks/hcfp5090-activation-feature-audit-validation100.json
```

- preserving all 11 current gains with a learned-versus-analytic ranker score
  threshold activates `97/100` cases;
- the best single pre-tail feature still activates more than half the cases.

Visible validation is evidence only and must not set a threshold or train a
policy.

### Learned tail is useful, but sparse

An analytic-only plus raw Pareto-guard attribution audit matches the complete
learned output exactly on `89/100` cases:

```text
artifacts/benchmarks/hcfp5090-analytic-guard-attribution-validation100.json
```

The remaining cases are `1, 4, 45, 49, 51, 54, 68, 71, 77, 86, 91`; six of
the current 11 quality gains require the learned tail. Therefore an
always-skip policy is invalid, while a high-recall activation policy has
material theoretical headroom.

A training-only 64/32 pilot produced only one positive paired example in 96
score-aware samples:

```text
artifacts/benchmarks/hcfp5090-activation-pilot-train64-heldout32.json
```

This is insufficient to train or promote a runtime gate. The first framework
must support rare-positive collection and shadow evaluation rather than force
an unreliable classifier into the runtime.

## Phase A: paired activation replay

Add a versioned replay record that contains only information available before
the learned tail plus an exact paired outcome:

```text
sample_id
source checkpoint hash
candidate/runtime config
pre-tail feature vector + feature version
analytic-plus-guard exact metrics
full learned split-tail exact metrics
tail_needed label and quality margin
analytic/learned runtime
```

Pre-tail features:

- normalized block count and constraint/connectivity densities;
- analytic-tail selected candidate features and telemetry;
- learned-initial candidate feature min/mean/std;
- learned-initial minus analytic feature margins;
- current ranker score order statistics and margins.

The label is positive only when the full learned path is raw-feasible and
improves the runtime-independent official v10 objective over analytic plus its
raw Pareto guard. Infeasible or failed learned paths are negative. Runtime is
recorded for policy analysis but is not allowed to override feasibility or
quality.

Generation uses FloorSet training workers only. One seeded stream is split
without replacement so train, calibration, and held-out sample IDs are
provably disjoint.

## Phase B: activation policy

Start with a standardized linear/logistic policy stored as a separate,
hash-verified JSON artifact. This avoids changing the current model checkpoint
schema and keeps the policy independently removable.

Training requirements:

- class-weighted binary loss;
- deterministic seed and bounded steps;
- train statistics only for normalization;
- threshold calibrated on the internal calibration split;
- exact sample-ID overlap rejection;
- checkpoint/config/feature-version compatibility checks;
- reject training if either train or calibration has no positive examples.

Do not add a nonlinear head until a linear policy has proven that the available
features contain useful held-out signal.

## Phase C: shadow mode

Shadow mode always executes the learned tail and records the policy decision.
It must report:

- positive recall, false-skip count, activation rate, and precision;
- results by `N` bucket and constraint density;
- all large cases and OOD/invalid features forced active;
- counterfactual p50/p95 from measured paired runtimes;
- exact list of any would-be quality regressions.

Promotion to a live gate requires:

- internal held-out positive recall `100%`;
- at least 32 positive train records and 16 positive calibration records;
- safe skip rate greater than `50%` overall;
- all `106--120` cases forced active during the first live experiment;
- visible validation retains all 11 gains, 100/100 feasibility, zero new
  `cost=10`, and no large-case regression;
- learned p50 no worse than analytic p50 and learned p95 no worse than
  `analytic p95 x 1.10`.

## Phase D: fail-open runtime integration

Only after shadow promotion:

```text
run_tail = large_case or policy_missing or policy_invalid or OOD
           or policy_probability >= calibrated_threshold
```

When `run_tail` is false, return the protected analytic-plus-raw-Pareto result.
Any policy load, hash, feature, or inference failure runs the full learned path.

## Verification gates

- Replay JSONL roundtrip and hash/provenance tests.
- Feature determinism and no post-tail information leakage tests.
- Train/calibration/held-out sample overlap rejection.
- Policy checkpoint/config/feature-version mismatch rejection.
- Rare-positive/no-positive training rejection.
- Shadow decisions are output-neutral and byte-identical.
- Live-gate tests cover skip, forced-large, OOD, missing policy, corrupt policy,
  and exception fail-open paths.
- Full test, Ruff, compileall, material replay, determinism 88/97, and clean
  validation 100 before any promotion.

## Stop conditions

- Do not train or calibrate from visible validation.
- Do not use testcase IDs as features or rules.
- Do not lower projection beam, iterations, population, or verifier coverage.
- Do not activate a policy trained from the current 96-sample pilot.
- Do not implement live skipping before paired replay and shadow gates pass.
