# P12 — Model-First Structured E2E Floorplanning Implementation Spec

日期：2026-08-15  
分支：`research/model-first-e2e-floorplanner`  
基準：`main@6924307223b771e349ef1e41c384486be21d3104`  
Canonical plan：`docs/research/model_first_e2e_floorplanning_plan_2026-08-15.md`

本文件是 canonical plan 的**實作補充**，不是第二套架構。若文字衝突，以 canonical plan 與最新實驗證據為準。

## 1. 研究目標

P11 的結論已收斂：structured corruption 確實教會小模型 Contact ordering，但在 P8 single-block bridge action space 中，canonical order 中位數兩次 decode 即命中，因此 repair ranker 沒有 integration surface。

P12 將 learned capacity移到真正的全局決策：

> **模型從 FloorSet instance、preplaced anchors 與 constraints 直接產生完整 Sequence-Pair、aspect 與 outline；既有 deterministic decoder 一次性編譯成 placement。演算法只 compile、verify、score 與 bounded cleanup，不搜尋另一套 topology。**

```text
case + anchors
-> existing HCFP structure model
-> complete floorplan program
-> one-shot exact Sequence-Pair compile
-> exact verify / official QoR
-> optional bounded model-selected residual action
-> final placement
```

## 2. 不重蓋既有資產

本 repo 已經有 P12 G1 需要的大部分零件：

| 功能 | 既有實作 | P12 用法 |
| --- | --- | --- |
| Static graph encoding | `SceneEncoder` / pair-biased transformer in `src/hcfp/model.py` | 直接重用 |
| SP precedence / outline | `StructureHeads` | 直接重用 |
| Soft Sequence-Pair | `DualPermutationHead` | 直接重用 |
| Aspect prediction | existing initializer / structure lane aspect output | 只取 shape 決策，不取 solver seed |
| Hard SP decode | `hard_permutation`, `decode_sequence_pair` | 直接重用 |
| One-shot packing | `pack_sequence_pair_with_anchors` | 直接重用 |
| Exact area | `exact_shape_projection` | 直接重用 |
| Exact legality | `verify.py` | 直接重用 |
| 1M streaming | `scripts/train_hcfp.py --floorset-lite-root` | 直接重用 |
| Large direct checkpoint | `hcfp5090-q2-structure-large-s3000-seed6501.pt` | G1 incumbent |
| In-flight scale evidence | `hcfp5090-q2-structure-large-s10000-pool100k-seed6501.pt` | 完成後先評估 |
| Residual Contact | P11 `ContactRepairModel` + decoder | 只作後期 auxiliary |

**G1 不建立新的 `ModelFirstFloorplanner`、第二套 Graph Transformer 或通用 generative framework。**

第一個新增核心只有：

```text
anchor-conditioned generation-state adapter
+ direct-generation experiment runner
```

## 3. 模型責任與演算法責任

### 模型決定

- positive / negative Sequence-Pair；
- movable soft-block aspect；
- outline prior；
- HPWL-aware adjacency prior；
- boundary/group/MIB conditioning；
- 後續 structured denoising 時的 masked token補全；
- 後續 residual action。

### 演算法決定

- soft assignment轉 hard permutation；
- exact dimensions；
- one-shot anchor-aware coordinates；
- hard feasibility；
- official exact metrics；
- 執行 bounded model-selected local action；
- 所有 model programs失敗時的安全 fallback。

### 演算法禁止

- decode後重新搜尋 Sequence-Pair/B*-Tree；
- 大量 topology seed枚舉；
- 用 P8 / safe_shelf placement補完 model program；
- 讓 post-model repair成為主要 QoR來源。

## 4. 初始解：Anchor-only Full Mask

正常路徑不需要 solver初始 placement。

已知：

```text
area
b2b / p2b / pins
group / MIB
boundary bits
fixed shape
preplaced xywh
```

未知：

```text
movable x/y
movable aspect
positive / negative Sequence-Pair
contact topology
```

對現有 static `SceneEncoder` 而言，movable geometry本來就不是必要輸入。因此 G1 先確認：

> **現有 structure lane 是否已經等價於 anchor-only direct generation，只差一條明確 one-shot output/compile path。**

只有當 masked/partial curriculum需要區分已知與未知 program tokens時，才加入 `GenerationState` mask features。

## 5. 最小 GenerationState

G1 只需要很薄的 adapter：

```python
@dataclass(frozen=True)
class GenerationState:
    case: FloorplanCase
    task_kind: str              # FULL_MASK / PARTIAL / RESIDUAL
    noise_level: float
    round_index: int
    sample_index: int
    positive_known: Tensor | None = None
    negative_known: Tensor | None = None
    aspect_known: Tensor | None = None
    decoded_placement: Tensor | None = None
```

### G1 FULL_MASK

- `positive_known = None`
- `negative_known = None`
- `aspect_known = fixed/preplaced only`
- `decoded_placement = None`

G1 不需要新 partial-program embedding；先用 existing model output。

### G2 之後

只有 G1 PASS後才擴：

- partial rank embeddings；
- mask flags；
- noise level；
- confidence；
- decoded residual features。

## 6. Floorplan Program Adapter

G1 不建立大 schema framework，只建立 experiment-side adapter：

```python
@dataclass(frozen=True)
class DirectFloorplanProgram:
    positive: Tensor          # hard permutation [N]
    negative: Tensor          # hard permutation [N]
    log_aspect: Tensor        # [N]
    outline: Tensor           # [4]
    sample_index: int
```

來源：

```text
HCFPModel output
-> hard_permutation(positive_permutation)
-> hard_permutation(negative_permutation)
-> aspect from existing learned output
-> outline from existing StructureHeads
```

不要要求唯一 gold Sequence-Pair；training沿用 partial precedence supervision。

## 7. One-shot Exact Compiler

新增的 compiler只做接線：

```text
DirectFloorplanProgram
-> exact_shape_projection
-> pack_sequence_pair_with_anchors
-> verify_feasible / exact_metrics
```

```python
@dataclass(frozen=True)
class DirectCompileResult:
    placement: Tensor | None
    hard_feasible: bool
    failure: str | None
    metrics: ExactMetrics | None
```

允許的 failure：

```text
PROGRAM_INVALID
ANCHOR_CONTRADICTION
PACK_FAILED
HARD_INFEASIBLE
NONFINITE
```

Compiler不得：

- 枚舉另一個program；
- 呼叫 analytic/P8 solver；
- 對整張 placement做搜索式 repair。

## 8. G1 模型輸出與 K-sample

現有 HCFPModel已有 population/sample embedding機制。G1 應先沿用現有 population邏輯產 K 個 program，而不是新增 diffusion sampler。

```text
case
-> existing model K outputs
-> K hard programs
-> K one-shot compiles
-> exact score
-> best feasible model program
```

固定：

```text
K = 4
```

可附帶 K=1/8 作曲線，但決策以 K=4 為主。

## 9. G1 Direct Generation Gate

### 為什麼 106–120 是主桶

不要 medium-first。現有實驗已顯示 direct-generation 的唯一明確勝場在 106–120：analytic large15 全 cap，而 learned structure checkpoint可使 8/15 穿 cap並將 weighted cost降至8.822391。中型 analytic headroom反而較少。

因此：

```text
主桶：106–120 blocks
對照：40–80 blocks
```

### 資料

```text
10K train sources
2K source-held-out sources
source-level split
106–120主桶
40–80對照桶
```

先完成並評估進行中的：

```text
pool100K / 10K-step structure checkpoint
```

它是 Phase 1 資料 scaling evidence，不另開新模型。

### 輸入

- movable geometry 100% unknown；
- preplaced/fixed/constraints可見；
- 不使用 safe_shelf / P8作輸入。

### 對照

- analytic incumbent；
- current large structure checkpoint；
- in-flight 100K checkpoint；
- safe fallback只報 hard safety；
- clean `fp_sol`作 ceiling。

### 凍結判準

PASS：

```text
held-out 106–120 model compile hard-feasible >=95%
weighted uncapped cost <= analytic incumbent
K=4至少一個 below-cap case
```

MODIFY：

```text
hard-feasible過，但 cost輸 analytic
-> 只檢查 shape/outline/program表達，不放大模型
```

REJECT：

```text
hard-feasible <80%
-> direct full-mask representation不足
-> 回到 partial reconstruction t≈0.6
-> 不硬堆1M/大模型
```

## 10. G2 Structured Denoising

只有 G1 PASS或明確 MODIFY才做。

同一 HCFP model加入最小 mask conditioning：

```text
t=0.0  C0/C1/C2-like local repair
t=0.3  mask 2–4 blocks
t=0.6  mask one group/subtree
t=1.0  full movable mask
```

Mask是 program-level：

- rank token；
- aspect token；
- local relation；

不是 Gaussian `(x,y)` noise。

推論比較：

```text
one-shot
4 rounds
8 rounds
16 rounds
```

MaskGIT式保留高信心 tokens，只補低信心 tokens。

## 11. G3 Exact-score Preference

只有 supervised/denoising direct model已有decoded QoR價值才做。

```text
K model programs
-> exact compile / score
-> listwise preference
```

先用現有 replay/listwise基礎；不要先建 PPO/RL framework。

## 12. G4 Residual Repair

CCRL資產保留，但不再作全局主線。

```text
direct model placement
-> exact residual state
-> frozen/shared Contact head Top-4
-> exact local decode
->最多1–3 rounds
```

只有 exact attribution顯示 grouping仍是主 bottleneck時才接。

Boundary/MIB/HPWL heads一律依 residual QoR逐一開，不做完整multi-expert框架。

## 13. 程式變更最小集合

G1 active files建議：

```text
src/hcfp/generation_state.py
src/hcfp/direct_generation.py
scripts/experiment_direct_generation_gate.py
scripts/build_direct_generation_split.py
 tests/test_direct_generation.py
```

必要時對 `model.py` 做最小 optional mask/sample輸入擴充，必須保持現有 checkpoint/runtime相容。

G1 不新增：

```text
src/hcfp/generative/ 大型子系統
第二套 encoder
第二套 verifier
通用 diffusion framework
新 RL framework
```

## 14. Issue graph

```text
#25 G1 plumbing / one-shot direct path
-> #26 G1 decisive 106–120 gate
   -> PASS/MODIFY: #27 G2 structured denoising
                   -> #28 G3 exact-score preference
                   -> #29 G4 bounded residual only if measured
                   -> #30 scaling / large / full100 qualification
   -> REJECT: close P12
```

## 15. Stop rules

- 不重建已有 StructureHeads/encoder/decoder；
- direct QoR未過，不做 1M；
- 中型好但大型無 headroom，不把中型當主勝場；
- fallback成功不得算 model compile成功；
- post-model solver不得提供主要gain；
- loss改善但 exact decoded QoR不動即停；
- G1未完成前，不加 multi-expert、router、DAgger、RL；
- in-flight 100K checkpoint先完成評估，不重複訓練相同實驗。

## 16. 最小論文主張

第一個要證明的主張是：

> **在沒有 solver initial placement 的情況下，既有 learned structure model能否只看 FloorSet case與anchors，直接產生完整 Sequence-Pair/aspect/outline，並以K=4 one-shot exact decode在106–120 source-held-out cases至少打平 analytic incumbent且產生cap crossing。**

只有此主張成立，才擴到 structured denoising、exact preference與1M資料。
