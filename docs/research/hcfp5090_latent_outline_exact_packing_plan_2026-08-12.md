# HCFP-5090 Latent-Outline Exact Packing 執行計畫

日期：2026-08-12

狀態：P0、P1 完成；P2 可開始

基準 commit：`891c191`

基準 checkpoint：`artifacts/checkpoints/hcfp5090-q2-structure-large-s3000-seed6501.pt`

## 1. 目前基準與下一階段目標

目前 large15 精確評測已證明 structured topology／constraint seeds 有效：

| 指標 | Analytic | Learned structured |
| --- | ---: | ---: |
| Hard feasible | 15/15 | 15/15 |
| Weighted capped cost | 9.999999 | 8.822391 |
| 穿過 exact cap | 0/15 | 8/15 |
| Runtime p50 | 0.390 s | 6.015 s |
| Runtime p95 | 0.799 s | 7.803 s |

已穿過 exact cap 的 cases 為 `85, 87, 91, 92, 95, 96, 97, 99`。先前
`cost >= 9.99` 的保守 competitiveness threshold 將 case 99 算在 cap-side，
因此舊報告顯示 7/15；本計畫以 evaluator 的 `9.999999` exact cap 為準。
本階段不再擴大
模型、增加 random flow samples 或做 constraint-only fine-tune；主線改為：

```text
正確的 bbox / cap attribution
  -> latent outline hypotheses
  -> exact-area treemap candidates
  -> tree_sol-supervised B*-Tree candidates
  -> generator-inversion anchors
  -> mask-guided refinement + instance TTO
  -> near-cap replay
  -> candidate pruning + submission freeze
```

核心目標是讓模型只決定 `outline hypothesis、tree、split、order、contact、
whitespace`，由確定性幾何 decoder 保證 area、non-overlap 與緊密 packing。

## 2. 不可混淆的幾何語義

以下規則是所有實作與圖表的共同 contract：

1. `official_candidate_bbox` 是輸出 blocks 的 extrema；block 不可能位於此 bbox
   外，且 official input 沒有固定 canvas。
2. `inferred_latent_outline` 是從 FloorSet generator 訊號反推的候選 envelope，
   不是官方 hard constraint。若 hypothesis 與 preplaced／fixed geometry 不相容，
   應 reject 該 hypothesis，而不是宣告 official infeasible。
3. `model_temporary_outline` 是現有 structure head 的預測，只能作 conditioning／
   ranking feature。
4. `pin_perimeter_hypothesis` 由 `pins_pos` 推定；pin 是二維點，沒有寬高。
5. `gold_outline` 僅能由 training `fp_sol` 建立，用於 audit／supervision；不得在
   validation 或 submission runtime 使用。
6. official boundary predicate 永遠相對於最終 `official_candidate_bbox` 重算。

既有 `src/hcfp/verify.py`、`src/hcfp/score_attribution.py` 與 official parity tests
是 exact metric source of truth；新模組不得另寫一套 evaluator。

## 3. 共通執行原則

- Candidate budget 固定為 32；以較強 candidate family 替換弱 seeds，不擴成
  64／128。
- 永遠保留目前 analytic incumbent 與 large structured incumbent。
- 每一階段先跑 internal training held-out，再跑 visible large15；visible cases
  不得回灌模型權重或 replay。
- 每個 candidate 保存 source provenance：`analytic`、`topology`、`constraint`、
  `treemap`、`btree`、`mask`、`tto`。
- 每個 promotion report 同時報 capped cost 與 uncapped attribution；禁止只報平均
  loss 或單一 aggregate score。
- 所有 geometry、overlap、bbox、fixed/preplaced replay 保持 FP32／exact tail
  原生 precision。
- 本階段不新增第三方 dependency；先使用 PyTorch、Python standard library 與
  現有投影器。

所有 QoR phase 共用下列保底 gate：

```text
large15 hard feasible = 15/15
目前 8 個 exact-uncapped cases 不得退回 cap
任何新 candidate family 失敗時仍可回傳既有 incumbent
candidate count <= 32
```

## 4. 任務 DAG

```text
P0 Evidence semantics
  -> P1 Latent outline beam
       -> P2 Exact-area treemap
       -> P3 tree_sol + B*-Tree
            -> P4 Generator inversion + mask/TTO
                 -> P5 Near-cap replay
                      -> P6 Full evaluation/runtime/freeze
```

`P2` 與 `P3` 在 `P1` audit gate 通過後可以平行實作；其他 phase 依賴前一階段
產生的真實 candidates 與 evidence，不提前訓練。

## 5. P0 — Evidence semantics 與 large15 attribution

時程：8/12

訓練：禁止

目的：先把每個 capped case 的主因與圖中每一個框的語義說清楚。

### P0.1 修正視覺化語義

責任範圍：

- `src/hcfp/visualize.py`
- `scripts/visualize_hcfp.py`
- `tests/test_visualize.py`

工作：

- 將目前單一 `.bbox` 改名並標示為 `official_candidate_bbox`。
- 支援選配 overlays：
  - `inferred_latent_outline`：虛線；
  - `model_temporary_outline`：點線；
  - `pin_perimeter_hypothesis`：不同顏色虛線；
  - `gold_outline`：僅 diagnostic mode；
  - 超出 inferred outline 的 blocks：紅色外框；
  - whitespace leaves：半透明。
- 圖例與 JSON schema 必須明確區分上述五種幾何。
- summary 新增：
  - `utilization`；
  - `outline_overflow_area`；
  - `blocks_outside_inferred_outline`；
  - `pin_side_coverage`；
  - boundary satisfied／total；
  - group connected components；
  - MIB distinct-shape count；
  - raw→projected displacement。

### P0.2 large15 精確歸因

優先擴充既有：

- `scripts/report_cap_sources.py`
- `scripts/audit_hcfp_oracle.py`
- `src/hcfp/score_attribution.py`

不另建重複 scorer。對 cases 85–99 產生：

```text
hard feasibility
quality_factor
hpwl_gap / area_gap
V_boundary / V_grouping / V_mib / V_relative
uncapped cost / capped cost / cap margin
utilization
raw / projected / final cap margin
projection displacement
candidate source
dominant blocker
```

分類順序：

1. hard dominated；
2. soft dominated；
3. area dominated；
4. HPWL dominated；
5. projection dominated；
6. mixed。

### P0 驗收

- [x] exact attribution 可重建 official local cost，逐 case 誤差在既有 parity tolerance
  內。
- [x] 15/15 cases 均有唯一 primary blocker 與 secondary contributions。
- [x] 圖上每一種 bbox／outline 都有圖例與不同 stroke style。
- [x] `official_candidate_bbox` 永遠等於輸出 blocks extrema。
- [x] 產出 `artifacts/benchmarks/hcfp5090-large15-attribution-v2.json`。
- [x] 重新產出 15 張逐 case PNG。

P0 未完成前，不啟動新 checkpoint training。

## 6. P1 — Latent outline beam 與 100k recovery audit

時程：8/13

目的：只用正式 input 反推 FloorSet generator 的原始 envelope hypotheses。

### 最小實作面

新增：

- `src/hcfp/outline_inference.py`
- `scripts/audit_outline_recovery.py`
- `tests/test_outline_inference.py`

暫不拆出 `outline_beam.py`；hypothesis generation、scoring 與 deterministic
deduplication 先放在同一模組。

### P1.1 Hypothesis generator

每個 case 產生 4–8 個 hypotheses，來源包括：

- pin coordinate modes／robust side-line fitting；
- total block area 與 utilization `0.95–1.00` prior；
- preplaced／fixed anchor span；
- pin-spread aspect ratio；
- square、horizontal、vertical fallback variants。

每個 hypothesis 保存：

```text
x_left / x_right / y_bottom / y_top
source
pin_residual
area_prior_residual
anchor_residual
pin_side_assignment
confidence
```

不相容 preplaced／fixed geometry 的 hypothesis 在生成階段 reject。推論順序與
tie-break 必須 deterministic。

### P1.2 Training-only recovery audit

從 FloorSet training source 串流抽取至少 100,000 個 106–120 block cases：

- gold bbox 僅由 `fp_sol` 建立；
- inference 僅能讀 official input fields；
- 報 top-1 與 oracle@K：width／height／area error、side recovery、pin-side
  coverage、gold blocks outside rate；
- 以 block count、pin count、preplaced density、boundary density 分 bucket。

### P1.3 Runtime 接點

- outline hypotheses 只進 candidate conditioning／generation；
- 對既有 structured seeds 建立 inside-envelope variant；
- hypothesis 不確定時保留 unconstrained structured incumbent；
- 不改 official exact verifier 語義。

### P1 驗收

- [x] 100k large audit 無 validation／visible path。
- [x] oracle@K outline area relative error median `< 1%`、p95 `< 3%`。
- [x] 四邊 recovery `>= 95%`；top-1 未達、oracle@K 達標，因此保留 beam。
- [x] deterministic hypothesis IDs 與 byte-stable audit summary。
- [x] 所有 preplaced rectangles 位於被接受的 hypothesis 內。
- [x] 產出 `artifacts/benchmarks/hcfp5090-outline-recovery-large100k.json`。

若 oracle@K 仍未達 gate，先修 inference／audit，不讓錯誤 outline 成為 hard
envelope，也不提前開始 outline-conditioned training。

## 7. P2 — Exact-area treemap candidate family

時程：8/14

目的：以 by-construction geometry 直接消除鬆散 area、overlap 與大量 repair
displacement。

### 最小實作面

新增：

- `src/hcfp/treemap.py`
- `src/hcfp/constraint_partition.py`
- `tests/test_treemap.py`
- `tests/test_constraint_partition.py`

whitespace token 先定義於 `treemap.py`；除非實作後確有兩個以上 consumers，
否則不新增獨立 `whitespace_tokens.py`。

### P2.1 Recursive exact partition

- 對任意 rectangle 與 block set 遞迴做 horizontal／vertical split。
- split ratio 完全由兩側 area sums 決定。
- 一般 soft leaf 的 `w*h` 必須等於 area target。
- 加入 1–4 個 dummy whitespace leaves，使總面積等於 selected latent outline。
- 輸出忽略 whitespace leaves，但 provenance 保存其位置與面積。

### P2.2 Constraint-aware partition

- 同 cluster 先形成 compound subtree，避免將成員分散到不相鄰區域。
- corner／boundary blocks 指派到對應 perimeter path。
- compatible MIB group 使用 shared shape variables。
- fixed／preplaced obstacle 複雜 case 先由 eligibility predicate 排除，繼續走既有
  structured lane，不強塞普通 treemap。

### P2.3 Candidate integration

- 初版產生 8 個 treemap candidates。
- 將現有較弱 random／near-duplicate seeds 減少 8 個，總候選維持 32。
- raw exact-feasible treemap 可跳過不必要的 overlap BDP，只進 exact verifier 與
  constraint／HPWL refinement。

### P2 驗收

- [ ] eligible cases raw overlap count = 0。
- [ ] 一般 soft block area error 在 exact verifier tolerance 內。
- [ ] eligible large15 utilization median `>= 0.97`。
- [ ] projection displacement 相較目前 structured candidates 降低 `>= 30%`。
- [ ] merged portfolio 維持 hard feasible 15/15，且既有 7 個 uncapped cases不退步。
- [ ] 產出 `artifacts/benchmarks/hcfp5090-treemap-large15.json`。

## 8. P3 — 解封 tree_sol 並建立 B*-Tree teacher lane

時程：8/15–8/16

目的：直接使用官方 training payload 的結構 supervision，取代只從 `fp_sol`
間接猜測 topology。

### 已確認的前置斷點

目前 `src/hcfp/floorset_lite.py` 的官方 Lite payload 使用：

```text
payload[4] = tree_sol
payload[5] = fp_sol
payload[6] = metrics_sol
```

但 loader 目前跳過 `payload[4]`，`DataSample` 與 training path 都拿不到
`tree_sol`。因此 P3.1 是 decoder 的硬依賴。

### 最小實作面

修改：

- `src/hcfp/floorset_lite.py`
- `src/hcfp/data.py`
- `src/hcfp/model.py`
- `src/hcfp/training.py`

新增：

- `src/hcfp/topology/btree_labels.py`
- `src/hcfp/topology/btree.py`
- `tests/test_btree_labels.py`
- `tests/test_btree.py`

初版將 pointer decode 與 contour packing 放在同一 `btree.py`；只有 profiler／
maintainability 證明需要時才拆成多檔。

### P3.1 tree_sol schema audit

- 將 raw `tree_sol` 以明確 schema 保存到 `DataSample`。
- 建立 padding／block-count validation、round-trip 與 D4 augmentation contract。
- 對 training corpus 抽樣驗證 tree decode 後與 `fp_sol` 的結構一致性。
- shard serialization 與 checkpoint schema 必須版本化，舊 checkpoint 能明確
  fail closed 或走無 B*-Tree lane。

### P3.2 B*-Tree decoder

模型預測：

```text
root
parent[i]
child_side[i] = LEFT_CHILD / RIGHT_CHILD
block aspect ratio
optional whitespace attachment
```

訓練順序：

1. teacher forcing；
2. tree validity／round-trip；
3. beam width 8 contour packing；
4. scheduled sampling；
5. post-packing HPWL／area／constraint loss。

### P3.3 Candidate integration

- 8 個 B*-Tree beam candidates 取代 8 個較弱 topology／constraint duplicates。
- 使用 P1 outline hypotheses 對齊與篩選，但始終保留 unconstrained incumbent。
- exact evaluator 比較 B*-Tree、treemap、current structured 三個 family 的
  oracle 與 selected result。

### P3 驗收

- [ ] raw tree labels schema validation = 100%。
- [ ] hard-decoded tree 無 orphan、duplicate parent 或 cycle。
- [ ] 同 seed 完全 deterministic。
- [ ] B*-Tree oracle@8 在 large internal held-out 與 large15 均優於目前 topology
  seed oracle；任一資料集未提升就不 promotion。
- [ ] merged portfolio 維持共通保底 gate。
- [ ] 產出 `artifacts/benchmarks/hcfp5090-btree-oracle8.json`。

## 9. P4 — Generator inversion、mask refiner 與 instance TTO

時程：8/17

目的：在已緊密、合法的 structured candidate 上恢復 gold-like adjacency 與
HPWL，不再從鬆散 absolute coordinates 開始。

### P4.1 Generator-inversion distance head

從 training `fp_sol` 建立 block-block／pin-block Manhattan distance labels：

```text
input: connectivity weights, area, degree, constraints, pin-side
target: gold center Manhattan distance
```

預測距離只用於：

- treemap bipartition scoring；
- B*-Tree parent／side scoring；
- contact-tree ranking；
- mask placement order／cost map；
- TTO anchor。

不得直接當 final absolute coordinate output。只有在 internal held-out 明確提升
split／tree oracle 時才進 runtime。

### P4.2 Mask-guided constructive refiner

最小新增：

- `src/hcfp/mask_canvas.py`
- `src/hcfp/mask_refine.py`
- `tests/test_mask_refine.py`

先做 deterministic imitation／beam，不做 RL：

- 64x64 occupancy／legal mask；top-k region 再局部細化；
- integral image 做 rectangle occupancy query；
- masks 合併 inside、free、boundary、group contact、MIB shape；
- fail-first 放置順序；
- beam width 8；mask 為空時回退 branch，而不是製造非法 candidate。

### P4.3 Instance test-time optimization

```text
32 candidates
  -> cheap exact-compatible metrics
  -> top 8
  -> 64–128 GPU steps
  -> 每 16/32 steps exact projection
  -> 保存所有時刻 best verified incumbent
```

TTO variables 限於 movable positions、soft shapes、tree/local topology 與
whitespace placement；preplaced 完全 freeze，fixed dimensions freeze。當 candidate
已達 `V_rel=0` 且 `area_gap<=0`，停止壓 area，將剩餘 steps 只用於 HPWL。

### P4 驗收

- [ ] mask lane raw overlap = 0。
- [ ] exact-feasible candidate density 顯著高於 current structured population。
- [ ] group connected ratio `>= 95%`，boundary satisfaction `>= 95%`。
- [ ] compatible MIB groups violation = 0。
- [ ] large15 至少 12/15 穿過 cap，weighted capped cost `< 5`。
- [ ] 產出 `artifacts/benchmarks/hcfp5090-mask-tto-large15.json`。

若 mask/TTO 未達 QoR gate，保留 treemap／B*-Tree portfolio，不以增加 rollout
steps 掩蓋 candidate construction 問題。

## 10. P5 — Near-cap replay 與低 learning-rate fine-tune

時程：8/18

前置：P2、P3 至少一個新 candidate family 通過 promotion gate。

目的：訓練模型修正其實際產生的 near-cap／post-repair states，而非重複擬合
鬆散 coordinate corruption。

### Replay composition

只從 1.008M training source 的 internal split 收集：

```text
30% uncapped cost 7–10
25% V_rel=0、但 area 或 HPWL 仍高
20% raw 良好、projection 後退步
15% 106–120 block difficult cases
10% 已穿 cap positives
```

每筆保存：

```text
outline hypotheses
candidate family / tree / split / whitespace
raw / post-TTO / post-projection geometry
hpwl_gap / area_gap / soft attribution
projection displacement
uncapped J / cap margin
teacher repair delta
```

### Target 與 curriculum

禁止回歸 capped score。主要連續 target：

```text
J = log(quality_factor) + 2 * V_relative
cap_margin = log(10) - J
```

curriculum：

1. hard feasible；
2. boundary／group／MIB 全部為零；
3. `area_gap<=0`；
4. 最小化 positive HPWL gap；
5. 最後才做 runtime-aware pruning。

freeze 基礎 scene encoder，低 learning rate 更新 outline、tree、split／mask 與
contact heads；不再單獨 fine-tune constraint head。

### P5 驗收

- [ ] replay 不包含 visible validation IDs／paths。
- [ ] checkpoint hash、case hash、candidate provenance 與 teacher geometry 完整。
- [ ] internal held-out cap-cross count 與 oracle@K 提升。
- [ ] large15 weighted capped cost `< 2.5` 作為第二階段 target。
- [ ] `V_rel=0` cases `>= 12/15`。
- [ ] full validation Pareto regressions = 0。

## 11. P6 — Full evaluation、runtime pruning 與 submission freeze

時程：8/19–8/21

目的：QoR 達標後再壓 runtime，完成 A100／packaging evidence。

### P6.1 8/19 full evaluation

執行：

```text
large15 x 3 deterministic seeds
full100 x 3 deterministic seeds
106–120 weighted subset
candidate-family ablation
checkpoint cold load
missing checkpoint fallback
CPU/CUDA precision differential
```

必報：

```text
hard feasibility
weighted capped cost / uncapped J
cap-cross count
oracle@K / selected regret
constraint attribution
utilization / projection displacement
p50 / p95 runtime
checkpoint load time / peak memory
```

### P6.2 8/20 runtime prune

- cheap scorer 將 32 candidates prune 到 top 6–8；
- raw-feasible treemap／mask candidates 跳過多餘 BDP；
- cache static encoding；
- 合併 candidate-family batches；
- `V=0`、`area_gap<=0` candidates early stop area optimization；
- 只有 HPWL 仍可改善的 candidates 進 TTO。

Promotion runtime gate：

```text
p50 <= current learned p50
p95 <= current learned p95 * 1.10
no uncontrolled large-case tail
```

### P6.3 8/20 freeze、8/21 upload

8/20 後只允許：packaging、dependency、path、checkpoint loading、deterministic
seed、wrapper compatibility 與 cold-start correction。8/21 只修 submission blocker。

## 12. Candidate portfolio 演進

每一階段總量維持 32，具體配額以 oracle evidence 決定：

| 階段 | Candidate portfolio |
| --- | --- |
| Current | 16 topology + 16 constraint |
| P2 pilot | 8 treemap + 12 topology + 12 constraint |
| P3 pilot | 8 treemap + 8 B*-Tree + 8 topology + 8 constraint |
| P4 final candidate | 8 treemap + 8 B*-Tree + 8 current structured + 8 mask/TTO variants |

若某 family 在 internal held-out 與 large15 都沒有提供 oracle gain，刪除該 family，
不以增加總數保留。

## 13. 每日交付與決策點

| 日期 | Phase | 必須交付 | Go／Hold 判斷 |
| --- | --- | --- | --- |
| 8/12 | P0 | large15 attribution v2、語義正確 PNG | 15 cases 可歸因才 Go |
| 8/13 | P1 | outline recovery 100k audit | oracle@K 達標才接 runtime |
| 8/14 | P2 | exact treemap large15 report | area／overlap／displacement 達標才保留 |
| 8/15–16 | P3 | tree_sol audit、B*-Tree oracle@8 | held-out 與 large15 同時提升才 promotion |
| 8/17 | P4 | mask/TTO large15 report | 12/15 cap-cross、weighted cost <5 |
| 8/18 | P5 | near-cap checkpoint | weighted cost <2.5 stretch gate |
| 8/19 | P6.1 | full100 x3 report | zero Pareto regression |
| 8/20 | P6.2–3 | A100 profile、frozen package | cold-start／wrapper pass |
| 8/21 | Final | final upload | 只修 blocker |

## 14. 明確不做

1. 不把 inferred latent outline 當 official fixed outline。
2. 不增加 model hidden size。
3. 不只增加 flow steps 或 random candidates。
4. 不在 exact packing family 接入前做 constraint-only fine-tune。
5. 不用 capped score 當 ranker／replay regression target。
6. 不在 visible validation 上訓練、replay 或記住 case-specific labels。
7. 不重做完整 RL；mask 先使用 imitation、deterministic costs 與 beam search。
8. 不另寫 scorer、legalizer 或 overlap predicate取代現有 exact source of truth。
9. 不移除 Pareto guard、analytic incumbent 或 exact verifier。
10. 不為暫時 hypothesis 新增 dependency 或複雜 abstraction。

## 15. 完成定義

本階段只有在以下條件全部成立時才算完成：

```text
hard feasible = 100%
official bbox / latent outline semantics 可由圖與 telemetry 驗證
至少一個 by-construction candidate family 通過 held-out + large15 gate
large15 >= 12/15 穿 cap
large15 weighted capped cost < 5；stretch < 2.5
V_rel=0 cases >= 12/15（stretch）
full100 Pareto regressions = 0
A100 cold-start、fallback、wrapper 與 deterministic replay 通過
```

若 `< 5` 未達成，仍以 P2／P3 oracle attribution 判斷是 outline、packing、HPWL
或 selection 問題；不得退回「放大模型／增加 samples」作為預設答案。
