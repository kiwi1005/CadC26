# HCFP-5090 learned-tail activation shadow results (2026-08-02)

## Decision

The learned-tail activation policy remains **shadow-only and disabled at runtime**.
The data/replay/training/evaluation framework is operational and fail-closed, but
the first disjoint held-out result missed both promotion gates:

- held-out positive recall: `75%` (`6/8`), required `100%`;
- held-out skip rate: `44.14%`, required greater than `50%`.

No threshold, feature, or sample from the visible validation set was used. All
records came from the FloorSet-Lite training corpus.

## Closed-loop implementation

The activation lane now provides:

1. versioned `pre_tail_v2` features and schema-2 replay records;
2. exact official-coordinate raw selection shared with the contest runtime;
3. auditable negative records for learned-tail exceptions;
4. deterministic interleaved train/calibration/held-out assignment;
5. bounded cross-file sampling without loading the million-case corpus;
6. hash-bound policy, training report, and held-out exclusion verification;
7. a class-weighted linear shadow policy with CPU-safe threshold calibration;
8. explicit estimated-runtime warnings that prohibit promotion without a live benchmark.

The relevant implementation commits are:

- `172b0f5` — paired replay, policy training, evaluation, and exact selector reuse;
- `fb7f3d9` — preserve exact official targets instead of reversing normalized FP32;
- `33e84e1` — interleave splits across source-file buckets;
- `66b3956` — bound layouts read from each source file;
- `5f86589` — preserve score-aware semantics by counting examined layouts.

## Replay evidence

The accepted diagnostic replay used uniform training-only sampling, at most 16
examined layouts per source file, and seed `20260806`:

| Split | Records | Positives | Learned failures | 1-32 | 33-64 | 65-96 | 97-105 | 106-120 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1024 | 22 | 0 | 128 | 288 | 304 | 128 | 176 |
| Calibration | 512 | 7 | 0 | 64 | 144 | 152 | 64 | 88 |
| Held-out | 512 | 8 | 0 | 64 | 144 | 152 | 64 | 88 |

All 37 observed positives were in `N=21-78`; no positive appeared in `N>=97`.
This explains why score-aware sampling was counterproductive for the activation
classifier: large cases are already forced active and supplied no positive skip
decision evidence.

The normal promotion minimum of 32 train positives and 16 calibration positives
was not met. A deliberately lower `16/4` gate was used only to test whether the
feature/policy loop had held-out signal. The resulting artifact is not eligible
for runtime integration.

## Shadow policy result

Training reduced class-weighted loss from `1.35651` to `0.62067`. After the
persisted CPU weights were used for calibration, train and calibration recall
were both `100%`.

The isolated held-out result was:

| Metric | Result | Promotion gate | Status |
| --- | ---: | ---: | --- |
| Positive recall | 75% (6/8) | 100% | Fail |
| Activation rate | 55.86% | below 50% | Fail |
| Skip rate | 44.14% | above 50% | Fail |
| Learned failures | 0/512 | 0 | Pass |
| False skips | 2 | 0 | Fail |

False-skipped held-out sample IDs:

- `worker_1/layouts_4816.th:74`
- `worker_57/layouts_8512.th:94`

The replay-only component estimates were analytic p50 `0.1813 s`, selected
counterfactual p50 `0.3186 s`, and selected p95 `0.3996 s`. These values are not
official live runtime evidence and must not be used for promotion.

## Artifact provenance

| Artifact | SHA256 |
| --- | --- |
| `hcfp5090-activation-uniform-cap16-r2048-v3.report.json` | `e87a9235fe45bd93dc315e49ebda6e04afd05d5e715556c2dbc1a81041a0b082` |
| train replay | `34f7f08bb07ba52a8ac1fa7dfcff2c1785e91a4c10f57ec9cf2f627039549f70` |
| calibration replay | `7ec8e701716c5daf9e1f564be404b59f4ad56a80cb7cfb3afe8a3d1033e031b6` |
| held-out replay | `bd28b90f1c2ff50215a5e335d0185c9a500a55895a5505124650a1ef001f5b61` |
| shadow policy | `5da3e9c92f1a03fc67074b4e72a463f35f91393f731e9403787e9b595f32722a` |
| training report | `f812561efc4e9bba04802c7fbcaa8b93d3db5c21d7767e1aa35626d1355507fa` |
| held-out evaluation | `4309ea0c2c61f0e5ed2b055fb021bee25d28264b2504e1f32744e60865db5a6a` |

## Placement and visualization smoke

A fresh official-data smoke on cases `0,50,99` completed with both fallback and
analytic lanes at `3/3` hard feasibility. Both lanes remained at the cost cap,
so the promotion decision was `HOLD`. The comparison report and deterministic
visualizations were generated successfully:

| Artifact | SHA256 |
| --- | --- |
| three-case comparison JSON | `7b125cf9c0e30eb00452a2441376f329a6ff1f15ba23d6afe0f8c6a71621ef79` |
| case 0 comparison HTML | `584b15d2941e2a0ea9f9ae27ec2611938840f9be365b4f83051a5c7cc1d39a64` |
| case 50 comparison HTML | `7feaf3cbb08576e6517746dd42569eca1b3c875434aa03c0401b79a3c8a028bc` |
| case 99 comparison HTML | `5bec215d9497eb3cc788961882913af1d86bb2733b173a899a810f995a337caf` |
| case 50 analytic SVG | `65168823d7effdd8c4f6a3b857f0fd82d0bc2b0ad481a7154f169ff56c91ef91` |
| case 50 analytic PNG | `58e468c885514a5fa9514b34fb82d5ad7548cf052cdd945f6aa66d6752f6c6a1` |

The PNG was visually inspected: all 71 labeled blocks and pin markers render
inside the bounding outline, and the image agrees with exact hard-feasibility
reported by the benchmark.

## Final contract verification

- Pinned FloorSet checkout: `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`.
- Pinned evaluator SHA256: `64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498`.
- A detached repo-local audit worktree passed with `clean=true` and `ok=true`.
- The JSON binary smoke passed from both the repository root and the
  `submission/` working directory.
- Both binary invocations returned three rectangles and copied the preplaced
  target exactly as `[10.0, 5.0, 2.0, 2.0]`.

The primary FloorSet checkout intentionally contains untracked training data,
so it is not used as clean-reference evidence and was not modified or cleaned.

## Next action

Do not add activation inference to the contest entrypoint. Preserve this lane as
an offline diagnostic and return optimization effort to output-neutral runtime
work and placement quality. If activation is revisited, collect more disjoint
`N<=96` training-only positives and evaluate a richer policy against the same
untouched natural held-out protocol; never lower the recall gate.
