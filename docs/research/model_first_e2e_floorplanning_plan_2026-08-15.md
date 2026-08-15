# Model-First Structured E2E Floorplanning — 主線方案

日期：2026-08-15
狀態：**主線**（取代 P11.4 Contact Gate D scope lock；CCRL 線收斂為輔助資產）
決策鏈：P11.5/5b/5c 三場 battle（corruption 92.2% → safe_shelf 打平 → P8 打平）證明
「repair 排序」在真 solver 接觸面上無紅利 → 轉向 direct generation。

## 0. 一句話

> 以共享的條件式結構生成模型，從只含 preplaced anchors 與 constraints 的全遮罩
> 狀態，structured denoising 出完整 Sequence-Pair、aspect 與 outline；一次性
> exact decoder 編譯成合法 placement；同一模型依 residual state 提出少量局部
> repair action。演算法只負責精確幾何、hard constraints、驗證與 bounded
> cleanup——**不搜尋**。

## 1. 方法立場：Model-first，Algorithm-assisted

| 項目 | 模型 | 演算法（一次性、確定性） |
| --- | :---: | :---: |
| 全域 topology / Sequence-Pair | ✓ | |
| Aspect / shape 決策 | ✓ | 精確 area projection |
| Outline hypothesis | ✓ | 相容性檢查 |
| Boundary arrangement、group contact plan | ✓ | 精確 side/contact 構造 |
| x/y 精確座標 | 間接（經結構） | ✓ one-shot decode |
| Non-overlap、fixed/preplaced、area、hard feasibility | | ✓ |
| 候選合法性與 exact QoR | | ✓ |
| Residual 局部修復 | ✓ 提 action | ✓ 執行 + 驗證 |
| **全域 topology 搜尋、大量 candidate enumeration** | **不做** | **不做** |

硬界線：**decode 之後演算法不得重新搜尋另一套全域 topology。** 否則回到
solver 主導，本方案失效。演算法只能是 compile / verify / small bounded
cleanup / fallback。

為什麼不直接回歸座標：連續 (x,y,w,h) 回歸對 floorplanning 殘酷——0.001 的
預測差即 overlap、fixed/preplaced/area 浮點失敗。模型決定**結構**，deterministic
decoder 保證幾何（無為而無不為：不讓模型重學幾何已知的真理）。

## 2. 與既有資產的對接（不重蓋）

本 repo 已有 direct-generation lane 的全部零件：

| 方案零件 | 既有實作 | 狀態 |
| --- | --- | --- |
| SP precedence heads + aspect | `StructureHeads`、`DualPermutationHead`、`RectifiedFlowHead`（`hcfp/model.py`） | 已訓練 |
| One-shot SP decoder | `pack_sequence_pair_with_anchors`（`hcfp/topology/longest_path_pack.py`） | 已存在 |
| Exact verifier / official parity | `hcfp/verify.py`、`score_attribution.py` + parity tests | 已存在 |
| 1M streaming trainer | `scripts/train_hcfp.py --floorset-lite-root`（score-aware） | 已驗證 |
| 106–120 直生勝場 | `hcfp5090-q2-structure-large-s3000-seed6501.pt`：large15 cost 9.999999→8.822391、8/15 穿 cap | **incumbent** |
| Outline 推論 | `outline_inference.py`（P1 已過：oracle@K area err med <1%） | 已存在 |
| 全遮罩訓練狀態 | `FloorplanCase` 不可變 + **新增** masked generator state | 待建 |
| Residual repair head | CCRL `ContactRepairModel`（Top-4 92.2% gain on corruption states） | 保留為輔助 |

**新增的核心只有一塊：anchor-conditioned masked generation state + curriculum。**
其餘是接線。

## 3. 核心機制

### 3.1 Anchor-only 全遮罩初始狀態

```text
已知：area、nets、group/MIB、boundary bits、fixed shape、preplaced xywh、pins
未知：movable blocks 的位置、aspect、Sequence-Pair、contact topology
```

Movable blocks = masked tokens。不再需要 safe_shelf / P8 當主要初始解；
它們降級為 fallback 與對照組。

### 3.2 Structured denoising curriculum（把 CCRL 放大成生成）

同一個模型、同一套 corruption 語言，corruption 深度連續化：

```text
t=0.0  幾乎完整（= 現有 C0/C1/C2，local repair）
t=0.3  遮 2–4 blocks
t=0.6  遮一個 group / subtree
t=1.0  全部 movable blocks 遮蔽（full generation）
```

修復與生成是同一個 structured denoising model 在不同 t 的行為。
Noise 是 floorplanning 語義的（mask block / break contact / remove
subtree / corrupt aspect / shuffle local order），不是 Gaussian 座標。

### 3.3 Floorplan Program（模型的輸出語言）

第一版最小集合（依既有 head 能力）：

```text
Sequence-Pair（rank+ / rank−）
aspect ratio
outline hypothesis
```

Boundary slot/order、group contact edge、MIB shared shape 先做 conditioning
與 auxiliary loss，不急著變獨立 head（為學日益者資料；為道日損者自由度）。

### 3.4 推論流程

```text
case → anchor-only state
  → masked structured denoising（MaskGIT 式 8–16 輪，每輪補一批高信心 tokens）
  → K=4/8 個完整 Floorplan Programs
  → one-shot exact decode（無 beam、無枚舉）
  → exact verify + official QoR
  → 殘差回饋 shared encoder → Top-4 residual repair actions
  → exact local decode + verify
  → final placement
```

無 topology seed 枚舉、無 BDP 重投影當主路徑；K 個樣本只被 decode 與
score，不被搜尋。

## 4. G1 — Direct Generation Gate（第一個決定性實驗）

> **模型能否在沒有任何 solver 初始解的情況下，只看 case 與 anchors，
> 直接生成 hard-feasible 且有競爭力的完整 Sequence-Pair placement？**

### 設計

- 資料：10K train / 2K held-out **source-level split**（沿用 ccrl-source-v1
  split 規範與 denylist 慣例）。
- Block 範圍：**106–120 為主桶**（理由見 §5），40–80 同跑作對照桶。
- 輸入：movable geometry 100% masked；只有 preplaced/fixed/constraints 可見。
- 輸出：Sequence-Pair + aspect + outline。
- 推論：K=4 → one-shot decode → exact verify。
- 平行：沿用已沉澱的 Pool 模式（states 間無共享）。

### 評估（五欄，預先凍結）

| 指標 | 對照 |
| --- | --- |
| hard-feasible rate | safe_shelf、analytic |
| official-style cost | analytic、fp_sol reference |
| HPWL / bbox | analytic、fp_sol |
| boundary/grouping/MIB soft debt | analytic |
| below-cap count | analytic incumbent |

### 決策規則（預先凍結，不跑完再定）

- **PASS**：held-out 106–120 hard-feasible ≥ 95%，且 uncapped weighted cost
  ≤ analytic incumbent（即至少打平最強 deterministic 初始解），且 K=4 內
  至少一個 below-cap case（證明分佈內有 cap-crossing 能力）。
- **MODIFY**：hard-feasible 過但 cost 輸 analytic → 問題在 shape/outline
  語言，不是結構能力；修表示。
- **REJECT**：hard-feasible < 80% → 表示不足以支撐生成；回到 curriculum
  中段（t=0.6 partial reconstruction），不硬堆資料/容量。

### G1 與舊規劃的差異（記錄為何不 medium-first）

舊討論曾提「40–80 過了再上大型」。**不採用**：此 repo 的既有證據是
direct-generation 的唯一勝場在 106–120（analytic 全 cap → learned 8/15
穿 cap），而中型 analytic 已近優、headroom 不足。難度軸是 headroom，
不是 block 數。中型只作對照桶。

## 5. 為什麼現在是對的時機

1. CCRL 已證明模型能從 FloorSet 學結構規律（Top-4 92.2% exhaustive gain）。
2. CCRL 也證明了 repair-ordering 紅利在真 solver states 不存在（P11.5c）。
3. Direct lane 的零件全在：heads、one-shot decoder、verifier、trainer、
   outline 推論。
4. 資料紅利未開採：incumbent checkpoint 只消費了 3K samples（1M 的 0.3%）。
   in-flight：pool 100K / 10K steps 的 structure 訓練正在跑（見 §7）。

## 6. 訓練總規劃（四相，每相有 gate）

```text
Phase 1  supervised structure pretraining（fp_sol 直接監督）
         從 anchor-only 全遮罩預測 SP/aspect/outline
Phase 2  structured denoising curriculum（t: 0 → 1 漸進）
         CCRL corruption 語言連續化；小 t 收斂到現有 C0–C2 行為
Phase 3  exact-score preference（K 個 programs → decode → exact cost →
         listwise/DPO-style 偏好）；只有 QoR gate 過了才做
Phase 4  residual repair head（凍結 CCRL contact head 接上，shared encoder）
```

擴張順序（每步都有 go/no-go）：

```text
G1（10K, 106–120 主桶）
  → PASS → 80–105 桶 + 50K pool
  → PASS → 40–80 對照 + 200K pool
  → 每桶以 large15-式 exact 評測 + source-held-out 評測
  → 1M 全量只在曲線仍上升時
```

## 7. 進行中與立即下一步

- in-flight：`hcfp5090-q2-structure-large-s10000-pool100k-seed6501.pt`
  （同 parent、structure stage、pool 100K、10K steps；完成後跑同一條
  large15 exact 命令對比 8.822391 / 8-of-15）。此實驗屬 Phase 1 資料
  scaling，結果直接餵 G1 的起點 checkpoint 選擇。
- 立即下一步（G1 實作面）：
  1. `src/hcfp/generation_state.py`：anchor-only masked state builder
     （複用 `build_repair_state` 的 exact-contact 慣例：continuous 特徵
     float32、contact/component truth 用 raw float64）。
  2. masked-state 過 `StructureHeads` 的小驗證（現有 head 是否已能吃
     masked 輸入；若不行，最小擴充是 input mask flag，不是新架構）。
  3. `scripts/experiment_direct_generation_gate.py`：G1 全流程
     （Pool 平行 decode/verify/score）。
  4. 10K/2K split manifest（沿用 source_pool 產生器的慣例，另開新檔，
     不覆寫 ccrl-source-v1）。

## 8. 不變的邊界（承襲 solver boundary invariants）

- `solve()` 官方契約、hard feasibility、preplaced/fixed exact、area、
  non-overlap：永遠由 exact tail 保證。
- validation cases 僅 read-only 作為輸入；解與指紋不得入訓練。
- 每個 phase 保留已知可行的 incumbent/fallback（analytic + P8 guarded
  portfolio 不拆，直到新線在 gate 上超越它）。
- 任何 gate 失敗即停（知止），先歸因（資料/表示/optimization/
  generalization/decoder ceiling）再決定下一刀。

## 9. CCRL 資產歸檔

- `ContactRepairModel` + statefix checkpoint：保留，定位為 Phase 4 residual
  head 與 curriculum 低 t 端的起點。
- 三場 battle 腳本（`experiment_ccrl_*.py`）：保留為 eval pattern。
- P11 系列文檔：完結，不再擴充。

## 10. 成功判準（論文敘事）

主線主張：**floorplanning 可以從「每 case 重優化」轉成「從大量 layouts 學
placement prior，直接生成、精確編譯」。**

量化錨點（隨 gate 更新）：

```text
G1:      held-out 106–120 direct generation ≥ analytic incumbent
Phase 3: K=4 exact-score 選優後，large15 uncapped cost < 8.822391
終局:    full100 加權成本 < 6.414040（P8 guarded）且 runtime 可接受
```
