# P12 — Model-First Structured E2E Floorplanning

日期：2026-08-15  
分支：`research/model-first-e2e-floorplanner`  
基準：`main@8e4e588b98adc93aaa47aa62d0fac50691944ab9`

## 1. 研究目標

P11 已證明 FloorSet structured corruption 可以教會小模型真正的 Contact repair ordering；但在目前 P8 single-block bridge action space 中，canonical order 在真 solver states 已經中位數 2 decodes 命中，因此 learned ranker 沒有實際 integration surface。

P12 不再把模型定位成傳統 solver 的排序助手，而是重新定義主問題：

> **模型直接從 FloorSet instance 與 hard anchors 生成完整 floorplan program；演算法只負責把 program 精確編譯為合法幾何、驗證 hard constraints，並執行模型提出的少量 residual actions。**

目標流程：

```text
FloorSet instance
  -> anchor-only / full-mask generation state
  -> model generates complete floorplan program
  -> one-shot exact compiler
  -> exact verifier and QoR measurement
  -> same model observes residual state
  -> bounded Top-K structural repair actions
  -> exact local decoder
  -> final placement
```

這不是 `solver -> model repair`，而是：

```text
model generate -> algorithm compile -> model refine -> algorithm verify
```

## 2. 責任邊界

### 模型負責

- 全局 block topology；
- Sequence-Pair 正負排列；
- movable soft-block aspect ratio；
- outline / utilization prior；
- boundary/perimeter ordering prior；
- group contact prior；
- HPWL-aware adjacency；
- 低信心 program token 的重新生成；
- decode 後 residual structural action。

### 演算法負責

- 將 model program 一次性轉成 `(x,y,w,h)`；
- exact area projection；
- preplaced `(x,y,w,h)` 與 fixed `(w,h)`；
- non-overlap by construction；
- exact verifier；
- exact QoR measurement；
- 執行 model-selected bounded local action；
- 所有 model samples 失敗時的 contest safety fallback。

### 演算法不得負責

- 重新搜尋另一棵全局 topology；
- 大量枚舉 Sequence-Pair / B*-Tree；
- 在模型輸出後用傳統 solver 重新完成整張 placement；
- 以 P8 incumbent 作為正常生成起點；
- 隱性取代模型的全局決策。

安全 fallback 可以存在，但不得成為主要 QoR來源。

## 3. 初始解與修復的統一

初始解不再由 `safe_shelf` 或 P8 供應。模型使用同一種 `GenerationState` 處理三種情境：

### FULL_MASK

```text
preplaced geometry: known
fixed shape: known
movable topology: masked
movable aspect: masked
outline: masked
```

用途：直接 E2E 生成完整初始解。

### PARTIAL_PROGRAM

```text
部分 Sequence-Pair ranks 已知
部分 shape 已知
低信心 tokens masked
```

用途：structured denoising、partial reconstruction、iterative refinement。

### DECODED_RESIDUAL

```text
完整 program + compiled placement
boundary/group/MIB/HPWL residual features
部分低信心 program tokens 或 local action slots masked
```

用途：由同一 shared encoder 進行 residual repair。

因此「生成初始解」與「修復殘局」不是兩套完全獨立模型，而是同一個模型在不同 mask ratio / task token 下的行為：

```text
100% movable mask -> full generation
50% mask         -> partial reconstruction
local corruption -> repair
```

## 4. Floorplan Program 表示

第一版 program 只包含足以產生合法全局 placement 的最小資訊。

```python
@dataclass(frozen=True)
class FloorplanProgram:
    positive_assignment: Tensor   # [N,N] soft block-to-rank assignment
    negative_assignment: Tensor   # [N,N]
    log_aspect: Tensor            # [N]
    outline: Tensor               # [4] = width, height, utilization, ratio
    token_confidence: Tensor      # program-token confidence
```

Hard decode 後：

```python
@dataclass(frozen=True)
class HardFloorplanProgram:
    positive: Tensor              # [N] block permutation
    negative: Tensor              # [N]
    log_aspect: Tensor            # [N]
    outline: Tensor               # [4]
```

第二階段才加入：

```text
boundary order / perimeter slot
contact obligations
MIB canonical-shape token
region assignment for preplaced fragmentation
```

不要在第一個 direct-generation gate 同時加入所有 constraint heads。

## 5. GenerationState

```python
@dataclass(frozen=True)
class GenerationState:
    case: FloorplanCase
    positive_assignment: Tensor
    negative_assignment: Tensor
    log_aspect: Tensor
    outline: Tensor
    positive_known: Tensor
    negative_known: Tensor
    aspect_known: Tensor
    outline_known: Tensor
    task_kind: str                # FULL_MASK / PARTIAL / RESIDUAL
    noise_level: float
    round_index: int
    sample_id: int
    decoded_placement: Tensor | None
    residual_features: Tensor | None
```

`FloorplanCase` 維持 immutable official input。不要把 mutable generation state 塞回 `FloorplanCase`。

## 6. 模型架構

## 6.1 Static Case Encoder

重用現有 `scene_node_features(case)` 與 pair-biased Graph Transformer概念。

Static node features：

- log area / sqrt area；
- b2b weighted degree；
- pin weight、centroid、spread；
- fixed / preplaced；
- boundary bits；
- group / MIB membership；
- target geometry validity。

Static pair bias：

- normalized b2b weight；
- same group；
- same MIB；
- preplaced/fixed relation；
- pin affinity；
- allowed preplaced precedence relations。

## 6.2 Program-State Encoder

每個 block 額外加入：

- current positive-rank embedding；
- current negative-rank embedding；
- current log aspect；
- known/masked flags；
- decoded center / dimensions，若已有 placement；
- residual boundary/group/MIB/HPWL features，若為 residual state；
- noise-level embedding；
- round embedding；
- sample/population embedding。

模型輸入：

```text
static block representation
+ partial program representation
+ task/noise/sample tokens
```

## 6.3 Shared Pair-Biased Transformer

第一個 debug model：

```text
d_model = 96
layers = 3
heads = 4
FFN multiplier = 4
```

第一個 real model：

```text
d_model = 192
layers = 4
heads = 6
FFN multiplier = 4
```

只有小模型在 direct-generation gate 顯示 capacity不足，且資料/label/compiler已排除，才測 `256x6x8`。

## 6.4 Generation Heads

### DualPermutationHead

重用現有 Sinkhorn `DualPermutationHead`：

```text
block embeddings -> positive [N,N]
                 -> negative [N,N]
```

Hard decode 重用 `hard_permutation()`。

### ShapeHead

```text
block embedding -> log_aspect
```

- preplaced/fixed shape直接使用 target；
- movable shape經 `exact_shape_projection` 還原 exact area；
- optional output uncertainty供 iterative mask使用。

### OutlineHead

```text
pooled scene token -> width, height, utilization, ratio
```

第一版作 auxiliary guidance / loss，不讓 outline head直接破壞 exact compiler。

### ConfidenceHead

預測：

- positive assignment confidence；
- negative assignment confidence；
- aspect confidence。

之後 masked iterative generation只 remask低信心 tokens。

### Auxiliary Constraint Heads

第一個 gate只作 supervision，不直接改 geometry：

- pairwise contact type；
- boundary order score；
- MIB canonical log-aspect；
- pairwise HPWL affinity/value。

若 auxiliary loss沒有提升 decoded QoR，移除，不為完整架構保留。

## 7. Exact Program Compiler

新增一個非常薄的 compiler，不做全局搜尋。

```python
compile_program(case, hard_program) -> CompileResult
```

流程：

1. validate permutations；
2. project exact dimensions；
3. decode Sequence-Pair relations；
4. no-preplaced case使用 `pack_sequence_pair`；
5. preplaced case使用 `pack_sequence_pair_with_anchors`；
6. exact hard verifier；
7. emit typed failure reason與 exact metrics。

```python
@dataclass(frozen=True)
class CompileResult:
    placement: Tensor | None
    hard_feasible: bool
    failure: CompileFailure | None
    bbox_area: float | None
    hpwl: float | None
    boundary: int | None
    grouping: int | None
    mib: int | None
```

Compiler 可以拒絕 model program，但不能自行搜索另一套 topology。

## 8. Preplaced anchor 問題

直接 Sequence-Pair 可能與 preplaced anchors 衝突。分兩階段處理。

### Gate A：無 preplaced / anchor-free

先證明模型能直接生成有 QoR價值的全局 topology。這個 gate使用：

- 40–80 blocks；
- `preplaced_count = 0`；
- fixed-shape允許；
- grouping/boundary/MIB仍可存在；
- one-shot compiler。

若 anchor-free direct generation都沒有學習價值，不進 anchor工程。

### Gate B：anchor-aware program

只有 Gate A KEEP後才做：

- 從 preplaced target推導 allowed pair relations；
- 在 positive/negative assignment上加入 anchor precedence constraints；
- 先固定 preplaced相對順序，再插入 movable blocks；
- compile failure回報 anchor-chain overflow / anchor contradiction；
- 必要時加入 region token，但不先建立通用 free-space framework。

不得以 P8 placement作初始解規避 anchor generation問題。

## 9. 推論流程

## 9.1 第一個 decisive version：One-Shot K-sample

```text
case + full mask
  -> model forward K times through sample embedding
  -> K FloorplanPrograms
  -> hard decode
  -> exact compile K
  -> exact score feasible candidates
  -> select best
```

建議：

```text
K = 1 / 4 / 8
```

模型為主，因為 K 個全局 program全部由模型產生；演算法只編譯與評分。

## 9.2 第二階段：Masked Iterative Generation

Gate A/B通過後才加入：

```text
full mask
-> predict all program tokens
-> keep high-confidence tokens
-> remask low-confidence tokens
-> repeat 4/8/12 rounds
```

這比直接上 Gaussian coordinate diffusion更符合 discrete topology。

## 9.3 Residual Model Refinement

完成 initial compile後：

```text
placement + residual debt
-> same shared encoder with RESIDUAL task token
-> Top-K local structural actions
-> exact local decoder
-> select best verified action
```

第一個 residual head可直接重用已完成的 Contact `RepairAction` machinery。

但 residual stage不得成為主要 floorplanner；建議：

```text
max rounds = 1–3
Top-K = 4
```

## 10. 訓練資料

FloorSet-Lite已提供：

- `fp_sol` clean placement；
- `metrics_sol` baseline area/HPWL；
- `tree_sol` topology label；
- full case conditioning。

從 `fp_sol` derive：

- partial allowed pairwise precedence；
- centers / log-aspect；
- outline；
- boundary order；
- exact contacts；
- group connectivity；
- MIB shape targets。

## 11. Loss

第一版 one-shot：

```text
L = λ_rel * partial sequence-pair relation NLL
  + λ_anti * antisymmetry
  + λ_perm * Sinkhorn assignment regularization
  + λ_shape * log-aspect SmoothL1
  + λ_outline * outline SmoothL1
  + λ_contact * contact auxiliary loss
  + λ_boundary * boundary-order auxiliary loss
  + λ_mib * MIB auxiliary loss
```

關鍵：

- 不要求唯一 Sequence-Pair label；
- 使用由 clean geometry推導的 allowed relation set；
- 不直接回歸最終 `(x,y)`；
- decoded QoR為主要 gate，training loss只是診斷。

第二階段 masked denoising：

```text
L_masked = only on masked program tokens
```

第三階段 preference：

- 模型生成 K programs；
- exact compiler/scorer給 listwise ranking；
- 用 listwise preference或 offline RL微調；
- 不在 direct-generation gate之前做。

## 12. 初始解與模型引導如何結合

### 正常 model-first path

```text
full-mask GenerationState
-> complete model program
-> exact compiler
```

完全不需要 solver initial placement。

### Optional warm-start path

同一模型也可以接 partial program：

```text
現有 placement
-> derive approximate program
-> mask low-confidence relations
-> model regenerate
```

這只是 optional fallback / ablation，不是正常依賴。

### Hard failure handling

若 K個 model programs全部 compile失敗：

```text
safe fallback
```

但報告必須區分：

- generator compile success；
- fallback output hard feasibility。

不可用 fallback成功率掩蓋 model generation failure。

## 13. 程式架構

```text
src/hcfp/generative/
  __init__.py
  schema.py              FloorplanProgram / GenerationState / CompileResult
  features.py            static + partial-program features
  model.py               ModelFirstFloorplanner
  heads.py               permutation / shape / outline / confidence
  compiler.py            one-shot exact compile
  losses.py
  dataset.py             clean labels + mask curriculum
  inference.py           K-sample and masked iterative generation
  preference.py          later, exact-score preference

scripts/
  audit_direct_generation_bucket.py
  train_direct_generator.py
  eval_direct_generation.py
  experiment_direct_generation_gate.py

 tests/
  test_generative_schema.py
  test_program_compiler.py
  test_direct_generation_model.py
  test_direct_generation_loss.py
  test_direct_generation_gate.py
```

重用而不複製：

- `scene_node_features`；
- pair-biased Transformer block concept；
- `DualPermutationHead` / `hard_permutation`；
- `decode_sequence_pair`；
- `pack_sequence_pair`；
- `pack_sequence_pair_with_anchors`；
- `exact_shape_projection`；
- exact verifier；
- existing Contact RepairAction/decoder for residual stage。

不要重寫第二套 verifier或通用 floorplanner。

## 14. Decisive gates

## P12.1 — Compiler/label parity

- source-held-out clean program可重建；
- anchor-free clean label compile hard-feasible = 100%；
- fixed shape/area exact；
- no unique-SP assumption；
- compiler不搜尋。

## P12.2 — One-shot direct-generation gate

固定：

```text
10K train sources
2K source-held-out
40–80 blocks
preplaced = 0
model 96x3x4 first
K = 1/4/8
```

比較：

- canonical index Sequence-Pair；
- random Sequence-Pair；
- current learned topology head；
- model-first generator；
- clean `fp_sol` ceiling；
- optional analytic incumbent as practical reference。

主要判準：

- generator compile success；
- hard feasibility；
- official-style cost；
- area/HPWL gap；
- boundary/group/MIB；
- K-sample unique wins。

KEEP條件：

- K=4 generator hard-feasible compile rate >=99%；
- learned program明顯優於 canonical/random program；
- source-held-out至少有可重複的 QoR gain或 unique wins；
- gain來自 model topology，而非 algorithmic cleanup。

若不過，停止，不上 diffusion、不上1M。

## P12.3 — Anchor-aware direct generation

只在 P12.2 KEEP後。

- 先測少量 preplaced；
- anchor constraints加入 assignment masks；
- compile success / anchor contradiction分開報；
- 不以 solver incumbent替代 full-mask generation。

## P12.4 — Unified masked generation and repair

只在 one-shot有QoR後：

- 0–100% program mask curriculum；
- 4/8/12 iterative rounds；
- full generation與partial repair共用模型；
- 比較 one-shot vs iterative的 decoded QoR。

## P12.5 — Constraint-aware heads and residual refinement

只加入已證明能改變 decoded QoR的 head。

順序：

1. Contact auxiliary / residual；
2. Boundary；
3. MIB；
4. region/topology only if anchors demand。

## P12.6 — Data scaling and exact-score preference

只有 direct model已經在source-held-out有QoR價值才跑：

```text
10K -> 50K -> 200K -> 1M
```

每一級使用同一 frozen held-out bucket畫 scaling curve。

若10K→50K無明顯成長，停止1M。

## P12.7 — Large-case and production qualification

- 80–105；
- 106–120；
- large15；
- full100；
- K/runtimes；
- exact fallback rate；
- model contribution vs algorithm tail。

## 15. Stop rules

停止或修改方向，如果：

- direct model只會背唯一座標/排列；
- model program不優於 canonical/random topology；
- QoR主要來自 compiler後的 algorithmic search；
- anchor問題只能靠P8 seed規避；
- 10K資料已飽和仍直接跑1M；
- residual repair成為主要搜尋器；
- adding heads改善loss但不改善decoded QoR；
- hard-feasible output主要由fallback提供；
- 中型direct-generation gate未過便開始大型架構。

## 16. 最小研究主張

第一個需要證明的不是「完整 diffusion system」，而是：

> **在沒有 solver initial placement 的情況下，模型能否只根據 FloorSet instance與hard anchors，直接產生比 canonical/random topology更好的完整 Sequence-Pair program，並由一次性 exact compiler得到合法 placement。**

這一題成立後，才有資格擴成 masked diffusion、multi-constraint heads與1M資料規模。
