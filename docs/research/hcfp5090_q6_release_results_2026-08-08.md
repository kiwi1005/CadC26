# HCFP-5090 Q6 release verification results

Date: 2026-08-08
Branch: `feat/hcfp5090-qor-first`
Evidence commit: `88b78c3`
Decision: **QoR evidence PASS; runtime and active-ranker promotion HOLD**

## Outcome

The Q0--Q5 framework now runs end to end on the 100 visible official cases with
three deterministic execution seeds. The preserved exact verifier and Pareto
guard produce the same output geometry for all three seeds:

- 100/100 final placements are hard feasible;
- 32 cases improve over the analytic incumbent, 68 tie, and none regress;
- score-weighted capped cost improves from `9.999999` to `9.724486415`;
- the 106--120-block subset improves from `9.999999` to `9.814156972`;
- all 100 output geometry hashes are seed-stable.

This is useful QoR evidence, but it is not submission promotion evidence. On the
RTX 5090 host, the learned lane takes roughly `2.55--2.58 s` at p50 and
`6.45--6.59 s` at p95, while the analytic lane takes about `0.236 s` at p50 and
`0.508 s` at p95. A100 evidence is unavailable. Ranker selection also remains
counterfactual-only and its checkpoint metadata keeps `capabilities.ranker`
false.

## Official visible 100 x 3 seeds

| Seed | Hard feasible | Improved / tied / regressed | Weighted cost | 106--120 weighted cost | p50 | p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 7001 | 100/100 | 32 / 68 / 0 | 9.724486415 | 9.814156972 | 2.54595 s | 6.55456 s |
| 7002 | 100/100 | 32 / 68 / 0 | 9.724486415 | 9.814156972 | 2.56097 s | 6.59304 s |
| 7003 | 100/100 | 32 / 68 / 0 | 9.724486415 | 9.814156972 | 2.58043 s | 6.44682 s |

The official local evaluator fixes `RuntimeFactor=1`, so the weighted cost result
above is a local QoR comparison rather than a prediction of the final
cross-submission runtime factor.

## Counterfactual ranker audit

The 300-row audit is fail-closed:

- all 300 existing outputs remain hard feasible;
- the selected output hash is deterministic per case;
- no counterfactual ranker choice would be accepted;
- cases 65 and 89 have zero exact-eligible learned-initial candidates for all
  three seeds.

Those six rows are candidate-coverage failures, not missing instrumentation.
Commits `80f73cb`, `8051c2c`, and `ceb1c15` make that distinction explicit and
prevent partial evidence from passing the audit.

## Q5 5,000-record replay and v5 training

The production replay is complete and independently checksum-validated:

```text
samples:             2,500 unique training samples
records:             5,000 (2,500 initial + 2,500 post-relax)
shards:              40
manifest SHA-256:    34b38bafc43d8879bd78c7291ed7d6043557a04bf3b9c8e2582cf6a65dece859
run fingerprint:     ad64e06616862f80299fce48da33423a4421a1d3de2b67298c734c8ac617b8a3
sample-ID SHA-256:   98d5818cdfed518c56810e550ebc173be2e4e6bf3f499fa6bb12b62662e09c26
```

The initial-stage Q5 sampler uses overlapping draw pools. Its 30,000-step draw
schedule is 40% actual hard-negative records, 25% near-cap records, 20%
106--120-block records, and 15% successful positives. These are draw quotas,
not mutually exclusive unique-record proportions. All three seeds have zero
bucket shortfall and zero fallback draws.

| Seed | Checkpoint SHA-256 | State hash | Loss window | Dev16 initial top-1 / top-4 / false |
|---:|---|---|---:|---:|
| 5104 | `6d52aff1...d2a8d84` | `8fe63484...447c58f` | 2.15167 -> 2.00967 | 10/16 / 15/16 / 1 |
| 5105 | `752f6bcd...e8c3c5` | `3f04282e...cd7323` | 2.15218 -> 2.02359 | 13/16 / 15/16 / 1 |
| 5106 | `09353bba...810f8d` | `1d588062...d76cc` | 2.16185 -> 2.02122 | 10/16 / 15/16 / 1 |

None of the v5 checkpoints passes the unfiltered pre-tail promotion gate on the
16-case pilot. All three errors occur on the same case,
`worker_97/layouts_8400.th:46`: the scalar ranker prefers exact-infeasible
candidate 6 over exact-feasible candidate 7. The current post-tail shadow masks
candidate 6 with the exact `hard_feasible & projection_ok` eligibility result,
so existing runtime output remains safe. The dormant pre-tail `tail_topk` path
does not have that protection and must remain disabled.

The diagnostic visualization is at:

```text
artifacts/visualizations/hcfp5090-q5-ranker-v5-dev16-seed5105-false-promotion/
index SHA-256:    ce44b89c87fc40a38b4033e29b011089dceb605cba35234046694c3fd942cce2
manifest SHA-256: 59827ebe5d7787cb98e3c7719595b8a589be9648de337d81b81941dabab2b0cd
```

The disjoint 512-case broad evaluation is complete:

```text
samples:             512 unique development samples
records:             1,024 (512 initial + 512 post-relax)
shards:              8
manifest SHA-256:    c42be1bc4be41066c96057e169d28f8f0746a93853f456e91c12cd33d036b184
run fingerprint:     7a1ca59466ac6df26ee6fc89f7350570c6d9c3d3e8a2a1f6b77ed0d258ca533a
sample-ID SHA-256:   ccf44027125ad6d9150279e60a9e3e7e9e076b4367bfeef78dbd33df6e00f3c5
training overlap:    0
legacy replay overlap: 0
```

| Checkpoint | Initial top-1 | Initial top-4 | False promotion | Weighted rank regret |
|---|---:|---:|---:|---:|
| v4 seed 5104 | 226/512 (44.1%) | 362/512 (70.7%) | 4 | 1.5527 |
| v4 seed 5105 | 222/512 (43.4%) | 358/512 (69.9%) | 2 | 1.7358 |
| v5 seed 5104 | 252/512 (49.2%) | 369/512 (72.1%) | 2 | 2.1557 |
| v5 seed 5105 | 242/512 (47.3%) | 368/512 (71.9%) | 2 | 2.2018 |
| v5 seed 5106 | 249/512 (48.6%) | 368/512 (71.9%) | 1 | 1.9787 |

The 5,000-record training replay improves broad top-1 by roughly five percentage
points and top-4 by one to two points over v4, but misses the required 75% and
93.75% rates by a wide margin. It also does not achieve zero false promotion.
The small dev16 pilot therefore overestimated generalization.

A targeted sampler probe redefined hard negatives as lists where a worse
feasibility tier has a lower scalar `J` than the best tier. This reduced the
eligible hard-negative pool from 2,500 to 2,025 lists, but seed 5205 regressed
to 235/512 (45.9%) top-1, 361/512 (70.5%) top-4, and one false promotion. The
sampler semantic change was rejected and removed from the branch; its
checkpoint remains an experiment artifact only.

Broad diagnostic visualizations:

```text
artifacts/visualizations/hcfp5090-q5-ranker-v5-dev512-false-promotion/
index SHA-256:    e2852ed61e4041af0d6dc10a7221d2b8dda858c992964bf29ab778c40ab3a6ab
manifest SHA-256: dad8b350e154195964d9c5f2e47e478615b01801bfb2674929e160978291ee11

artifacts/visualizations/hcfp5090-q5-ranker-v5-dev512-worst-top4-miss/
index SHA-256:    9426e640b6028505c40018e6cff941a80e67fd297f622312db233d31cef7dec0
manifest SHA-256: 7cb5fcce3e9fc057417a21f51deb7e1c253c7050200d6f24e544d68200c30c8a
```

## Release blockers

1. Learned p50 and p95 exceed the analytic runtime gates by a wide margin.
2. No A100 cold-load, peak-memory, or runtime profile is available.
3. Active pre-tail ranker pruning is unproven and remains disabled.
4. The v5 ranker does not pass the unfiltered dev16 false-promotion gate.
5. Q2 group connectivity and compatible-MIB legalization gates remain partial.

## Preserved invariants

- exact raw-coordinate hard verification is authoritative;
- fixed and preplaced geometry is copied exactly;
- the analytic incumbent and Pareto guard remain available;
- missing, corrupt, incompatible, or untrained rankers fail closed;
- official visible validation is evaluation-only and never enters replay
  training;
- no activation classifier has been promoted.

## Evidence artifacts

```text
artifacts/benchmarks/hcfp5090-q6-shadow-official100-seed7001.json
artifacts/benchmarks/hcfp5090-q6-shadow-official100-seed7002.json
artifacts/benchmarks/hcfp5090-q6-shadow-official100-seed7003.json
artifacts/benchmarks/hcfp5090-q6-summary-official100-seeds7001-7003.json
artifacts/benchmarks/hcfp5090-q6-counterfactual-official100-seeds7001-7003-v3.json
artifacts/replay/hcfp5090-q5-dagger5000-manifest.json
artifacts/benchmarks/hcfp5090-q5-ranker-v5-dev16-dagger5000-q5sampler-s30000-seeds5104-5106.json
artifacts/replay/hcfp5090-q5-dev512-manifest.json
artifacts/benchmarks/hcfp5090-q5-ranker-dev512-v4-v5-comparison.json
artifacts/benchmarks/hcfp5090-q5-ranker-v6-confusable-dev16-dev512-seed5205.json
```

The Q6 summary SHA-256 is
`b73fbf763244743459517aef02d67400ec2e892483fc56d94798dddce4c79be6`.
The counterfactual v3 SHA-256 is
`1d506153bd8a2663f900aa87724850a60c1b9ba4c5a503e95700e097ec5f716b`.
The broad v4/v5 comparison SHA-256 is
`99b6dc53245bcd72a00c901e4e4377b051870f9f94509926b63209c7e10cc973`.
