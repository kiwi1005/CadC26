# CadC26 / HCFP-5090 — Experiment-First Research Contract

This repository is an ICCAD 2026 FloorSet Challenge research solver, not a general-purpose product. Its default objective is to discover and validate QoR improvements as quickly as possible.

Subject to the solver boundary invariants below, use this priority order:

1. Improve official/strict FloorSet QoR.
2. Improve the high-weight 106--120 block cases.
3. Increase the number of cases below the official cost cap.
4. Preserve hard feasibility.
5. Identify the active geometry, topology, or constraint bottleneck.
6. Run informative ablations quickly.
7. Optimize runtime only after a QoR method has demonstrated value.

Do not treat software cleanliness, general-purpose architecture, audit completeness, or release readiness as primary goals unless the user explicitly switches to Release Mode.

## 老莊式研究紀律

老莊思想在本專案中是可執行的研究約束，不是裝飾性引文：

- **為學日益，為道日損：**證據可以增加，系統假設與自由度應減少。遇到失敗，先刪除資料、表示與 exact truth 之間的矛盾，再考慮增加模型容量。
- **無為而無不為：**不要讓 learned model 重學 deterministic geometry 已知的真理。模型只提出不確定的結構 action；decoder、verifier 與 hard constraints 保持確定性權威。
- **庖丁解牛，依乎天理：**沿既有 obligation、component、decoder 與 verifier 的自然邊界下刀；一次只處理一個可量測 bottleneck，不為一次實驗造通用框架。
- **得魚忘筌：**action label、checkpoint 與 heuristic 都是工具，不是目的；最終判準仍是 exact outcome、QoR 與 hard feasibility。未經 gate，不因 proxy 變好便擴張方法。
- **知止不殆：**小 gate 失敗即停止擴張，保留 incumbent，先說清失敗屬於資料、表示、optimization、generalization 或 decoder ceiling，再決定下一刀。

因此每輪工作都應能回答：這次刪除了哪個混淆因素，留下了哪個可證偽問題？

## Default mode: Experiment Mode

Use this loop by default:

```text
Observed contradiction -> causal hypothesis -> smallest removal -> metric -> decision
```

Before editing, ask: what is the smallest change that can tell us whether this algorithmic idea works?

### Hypothesis

State the expected causal improvement briefly, for example:

```text
grouping violations down -> V_rel down -> more large cases cross cap
```

### Minimal implementation

- Modify an existing function before creating a framework.
- Prefer a small flag, CLI option, environment variable, or one-off script.
- Do not refactor unrelated working code.
- Do not add abstractions for hypothetical future experiments.
- Temporary experimental code is acceptable when it makes the result available sooner and does not violate solver boundary invariants.
- Prefer twenty useful lines today over a reusable subsystem that delays the experiment.

### Fast evaluation

Run the smallest evaluation that can distinguish the hypothesis:

- Representative large cases when the issue is large-case QoR.
- The affected capped or near-cap cases when the issue is localized.
- Short training runs or small replay subsets before full training.
- `large15` before full validation when it is sufficiently informative.
- Expand only after the direction shows measurable gain.

### Decision

Every experiment ends with `KEEP`, `MODIFY`, or `REJECT` based on metrics. Stop extending methods that do not produce measurable benefit.

## Research priorities

### Current scope lock: P12 G1 Model-First Direct Generation Gate

This is the only intentionally time-sensitive section of `AGENT.md`. Update it when the active decisive experiment changes.

目前只回答一個問題：在沒有 solver initial placement 的情況下，既有 learned structure model 能否只根據 FloorSet case、preplaced anchors、fixed shapes 與 constraints，直接產生 K=4 完整 Sequence-Pair / aspect / outline programs，並以一次性 exact compile 在 source-held-out 106–120 block cases 至少打平 analytic incumbent 且產生 cap crossing。

執行順序：

1. 先完成並評估進行中的 `hcfp5090-q2-structure-large-s10000-pool100k-seed6501.pt`，不要重跑同一個 scaling 實驗。
2. #25 只把既有 `SceneEncoder`、`StructureHeads`、`DualPermutationHead`、aspect/outline outputs 接成 anchor-only full-mask direct path，重用 `pack_sequence_pair_with_anchors`、`exact_shape_projection` 與 verifier。
3. #26 固定 10K train / 2K source-held-out，106–120 blocks 為主桶、40–80 為對照桶，測 K=4 one-shot direct generation。
4. G1 結束前，不建立第二套模型、generic generative framework、iterative diffusion、Boundary/MIB/Contact residual heads、HPWL critic、router、DAgger、offline RL、1M scaling 或 production integration。
5. PASS：held-out large model compile hard-feasible >=95%，weighted uncapped cost <= analytic incumbent，且 K=4 至少一個 below-cap case。
6. MODIFY：hard feasibility 通過但 cost 輸 analytic，只檢查一個已量測的 aspect / outline / program 表示問題，不先加容量。
7. REJECT：model compile hard-feasible <80%，回到 partial reconstruction 約 `t=0.6`；不靠更多資料、大模型或 post-model solver 搶救。

目前 P11 結論：structured corruption 可教會 Contact ordering，但在 P8 single-block bridge action space 中 canonical order 已中位數兩個 decodes 命中，因此停止 repair-ranker integration。P11 code 可作為後續 residual-learning evidence與 auxiliary工具，但不是 P12 的正常 initial solution。

## Solver boundary invariants

Experiment velocity never authorizes breaking these constraints:

- Preserve the official `solve()` contract.
- Preserve hard feasibility.
- Preserve preplaced `(x, y, w, h)` exactly.
- Preserve fixed-shape `(w, h)` exactly.
- Preserve legal area and non-overlap.
- Do not memorize validation case IDs or solution fingerprints.
- Never use official validation solutions as training data.
- Preserve a known-working incumbent/fallback unless the experiment explicitly studies that behavior.

Keep these checks at the solver boundary. Do not turn them into production-grade policy machinery throughout the codebase.

## Do not do by default

Unless required to run the active experiment, do not proactively:

- perform a large refactor or cleanup sweep;
- build a generic framework or reusable infrastructure for a one-off study;
- introduce speculative abstractions or dependencies;
- add broad defensive validation, fail-closed policy layers, or exhaustive error handling;
- build provenance, fingerprint, manifest, promotion, or release systems;
- rewrite an existing working pipeline;
- modify unrelated modules;
- add exhaustive edge-case tests;
- write long plans or large documentation packages before measuring QoR.

If a workaround safely answers the research question, use it.

Observed issues that do not block the current experiment should be recorded in one line as:

```text
Observed but not blocking current experiment: ...
```

Then continue the active experiment.

## Test budget in Experiment Mode

Do not run the full test suite after every research edit. Default verification:

1. Import, syntax, or compile smoke for changed files.
2. Targeted tests directly covering the modified path.
3. One to three representative cases.
4. The benchmark directly tied to the hypothesis.

If the experiment metric does not improve, stop; do not spend time completing an exhaustive regression pass for rejected code.

Run full pytest, Ruff, compileall, device parity, official full benchmarks, packaging, or provenance checks only when:

- the user asks for merge, release, submission, or complete verification;
- creating a formal checkpoint;
- changing the official interface;
- changing hard-feasibility, fixed/preplaced semantics, or widely shared geometry primitives.

Research iteration and release verification are separate activities.

## Small Change Mode

When the user says `小改模式`:

- make only the requested change;
- do not refactor, broaden scope, or add abstractions;
- do not modify unrelated solver logic;
- run only a targeted smoke/static check;
- stop immediately after reporting the requested result.

## Release Mode

Enter strict engineering mode only when the user explicitly says `Release mode`, `完整驗證`, `準備提交`, or `準備 merge`.

Release Mode may include full pytest, Ruff, compileall, CUDA/device parity, official benchmark replay, runtime regression, submission packaging, and checkpoint/provenance verification.

## Execution behavior

When enough information exists, execute directly:

```text
inspect -> modify -> run -> compare -> decide
```

Do not replace execution with a long implementation plan. Fix newly discovered issues only when they block the current experiment or violate a solver boundary invariant.

When architecture elegance conflicts with a simple experiment that can produce metrics today, run the simple experiment. When runtime conflicts with QoR and runtime is still acceptable, prove QoR first.

## Experiment report format

End each research iteration with only:

**Hypothesis**

What the experiment tried to improve.

**Changed**

What was actually modified.

**Experiment**

Which cases, training run, or benchmark was executed.

**Result**

The important before/after metrics, including hard feasibility when relevant.

**Decision**

`KEEP`, `MODIFY`, or `REJECT`.

**Next experiment**

The most informative immediate follow-up.

Do not use software-engineering activity as a substitute for QoR evidence.
