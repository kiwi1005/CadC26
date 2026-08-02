# HCFP-5090 階段性執行路線圖

> 狀態基準：2026-08-01，分支 `feat/hcfp5090-greenfield`

本文件把 [`HCFP5090_完整技術報告.md`](../../HCFP5090_完整技術報告.md)
與 [`HCFP5090_中文完整方案_2026-08-01`](../../HCFP5090_中文完整方案_2026-08-01/)
轉成可逐項執行、驗證與停止的工程任務。高階架構仍以
[`hcfp5090_greenfield_plan.md`](hcfp5090_greenfield_plan.md) 為準；本文件只負責
任務順序、依賴、產物與 promotion gate。

## 1. 執行原則

- 僅在 `CadC26` 與本分支內開發；不讀寫其他專案的實作或產物。
- `src/hcfp/` 是唯一 runtime 主線，不恢復舊 `puzzleplace`／`reformplace` 路徑。
- 每個候選先通過 exact-compatible verifier，才可取代目前安全 incumbent。
- hard geometry、overlap predicate 與投影計算維持 FP32；模型層日後才可用 BF16。
- fixed-shape 與 preplaced 值直接重播 `target_positions`，不得由模型自由回歸。
- learned、compiled 或 GPU 特化路徑預設關閉，達成該階段 gate 才可升級。
- 不為未通過前置 gate 的階段預先建立抽象層、依賴或空殼模組。

狀態標記：

- `[x]` 已有實作且具本地驗證證據。
- `[~]` 已有第一版，但尚未完成該階段 promotion gate。
- `[ ]` 尚未開始。
- `[blocked]` 缺少資料、環境或上游產物，不能合理開工。

## 2. 目前基線快照

| 能力 | 狀態 | 現況與缺口 |
| --- | --- | --- |
| Greenfield branch／package boundary | `[x]` | 新 runtime 位於 `src/hcfp/`，舊 runtime 已從工作樹移除。 |
| Official seven-input adapter | `[x]` | 已有 case validation、padding/mask 處理與 submission surface。 |
| Exact-compatible hard verifier | `[x]` | 已對 pinned FloorSet v10 evaluator 建立 parity tests。 |
| Safe fallback／incumbent guard | `[x]` | exception、non-finite 或不合法候選不會覆蓋安全解。 |
| FP32 typed analytic dynamics | `[~]` | 已有 deterministic 第一版與 100-case runtime；exact QoR 仍全部落在 cost cap。 |
| BDP | `[x]` | 已有 bounded active-pair outer rebuild、方向 beam 與 per-candidate status。 |
| Official 100-case replay | `[x]` | 完整 runtime 與 fallback-only 均為 100/100 hard-feasible；見 [`hcfp5090_p0_correctness_2026-08-01.md`](hcfp5090_p0_correctness_2026-08-01.md)。 |
| Data shards／labels | `[~]` | 已盤點 1.008M cases、direct streaming、100-sample raw audit、score-aware sampling；stratified internal split 尚未執行。 |
| SCENE／POP-INIT | `[~]` | structure 1k + all-head 3k 正式短訓練完成；100-case attribution 顯示 learned oracle 0 wins，gate REJECT。 |
| Rectified flow／controller／ranker | `[~]` | 6-step flow、32-record exact replay、ranker 500-step/top-4 已閉環；QoR/runtime gate HOLD。 |
| Learned runtime | `[~]` | additive analytic+learned sidecar、raw official replay gate與 100/100 feasibility 已完成；預設關閉。 |
| Benchmark／visualization | `[x]` | 官方 weighted/bucket report、promotion decision 與 deterministic SVG/HTML 已完成。 |
| Runtime profile | `[~]` | N120/K32 CUDA profile 已通過穩定性／記憶體 gate；cold-start 與 A100 profile 尚待補。 |
| Submission freeze／package proof | `[~]` | 正式 100-case replay 與 hashes 已完成；payload freeze、portable smoke 與 dry run 尚未做。 |

因此目前正確定位是：**P0 correctness gate 已通過，P1–P5 的第一個正式資料
closed loop 已完成；trained sidecar 在 100 個 validation case 為 100/100
feasible，但 weighted cost 仍為 `9.999999`、runtime p95 為 `3.100891s`，所以
維持 HOLD。**完整證據見
[`hcfp5090_training_closed_loop_2026-08-01.md`](hcfp5090_training_closed_loop_2026-08-01.md)。

## 3. 階段依賴與停止點

```text
P0 合約與安全基線
  -> P1 Analytic population + BDP 強化
  -> P2 資料、labels 與 shards
  -> P3 SCENE + POP-INIT
  -> P4 HiCoDy one-step -> multi-step
  -> P5 CAL + ETR + learned BDP direction + PVR
  -> P6 Runtime profiling + portable acceleration
  -> P7 Full replay + freeze + submission
  -> P8 DAgger／RL（選配，非 contest 必要路徑）
```

不可跨越的依賴：

- P0 未完成，不開始模型訓練。
- P1 未證明 post-BDP 改善，不增加 learned dynamics 複雜度。
- P2 未通過 leakage／transform audit，不開始 P3。
- P3 的 oracle@K 未優於 analytic initializer 時，P4 只能做 bounded、default-off
  實驗，不擴大步數或升為正式路徑。
- 沒有 exact replay 產生的 event／repair labels，不訓練 ETR；PVR/ranker 只能用
  已存在的 candidate/outcome labels 做第一版校準。
- 未經 profiler 證明的 kernel，不改寫 Triton／CUDA。
- P0–P7 任一新路徑失敗時，正式輸出仍回退到最後一個 verified incumbent。

## 4. 階段總覽

| 階段 | 建議時窗 | 主要結果 | 退出 gate | 目前狀態 |
| --- | --- | --- | --- | --- |
| P0 | D1–D2 | 官方合約、parity、fallback、determinism | correctness gate 100% | 通過 |
| P1 | D3–D5 | Analytic population、typed dynamics、BDP、telemetry | post-BDP 改善且 N120/K32 穩定 | 框架完成／promotion HOLD |
| P2 | D6–D8 | 訓練 split、labels、shards、corruptions | audit／round-trip／leakage 全通過 | 1.008M direct stream／stratified split 待辦 |
| P3 | D9–D11 | SCENE、POP-INIT、oracle@K | post-BDP oracle@K 優於 analytic seeds | 1k+3k 短訓練完成／gate HOLD |
| P4 | D12–D14 | rectified-flow recovery | exact QoR 改善且 large case 不退步 | 6-step flow 完成／promotion HOLD |
| P5 | D15–D17 | controller、ranker、局部事件 | stagnation／ranker regret 明確改善 | 32 replay + ranker 500／calibration 待辦 |
| P6 | D18 | compile／graph／profile／portable fallback | p95 達標且官方環境可執行 | BF16/profile 完成／runtime HOLD |
| P7 | D19–D20 | full replay、freeze、package、dry run | submission hard gates 全通過 | 100-case replay/可視化完成／未 freeze |
| P8 | gate 後選配 | DAgger／local policy optimization | full-case Pareto improvement | 禁止提前開始 |

時窗是依賴順序，不是允許跳過 gate 的硬日期。若前一階段未過，後一階段的
時間優先用於修復前一 gate，而不是堆疊更多模組。

原技術報告把 D6–D8 留給 BDP v0／direction beam；本分支已先落地 BDP v0，
所以本表把該時窗重排為「完成 P1 強化 gate，同時準備 P2」。這是依目前程式
進度做的壓縮，不代表 BDP promotion evidence 已完成。

## 5. P0 — 官方合約與安全基線

### 目標

鎖定所有會造成 `cost=10`、fixed/preplaced 漂移或錯誤 runtime 判定的基礎
語義，使後續 QoR 實驗不需要懷疑 evaluator、adapter 或 fallback。

### 任務

- `[x] P0.1` 將官方七個輸入轉成 validated `FloorplanCase`。
- `[x] P0.2` 建立 hard constraints、soft violations 與 cost 的 parity tests。
- `[x] P0.3` 建立 deterministic shelf fallback 與 incumbent-preserving guard。
- `[x] P0.4` 提供 Python submission API 與 JSON stdin/stdout executable。
- `[x] P0.5` pin 官方 evaluator commit 與檔案 SHA256。
- `[x] P0.6` 對可取得的官方 validation cases 執行 fallback-only 全量重播。
- `[x] P0.7` 補齊 boundary 16 種 bitmask、MIB 可相容／不可相容、cluster、
  fixed/preplaced obstacle 的 edge-case fixtures。
- `[x] P0.8` 同一輸入重跑至少 10 次，確認 CPU 與 CUDA 各自 bitwise 或
  tolerance-bounded deterministic；記錄允許的差異來源。
- `[~] P0.9` 產生單一 P0 correctness report，包含 case 數、hard feasibility、
  parity mismatch、fallback 使用率、例外與 non-finite 計數。

P0.9 的 correctness report 已完成；官方輸出不含 full-runtime 的 fallback
selection rate，因此該觀測欄位保留為 P1 incumbent manager 的必要輸出。

### 退出條件

- hard feasibility、fixed/preplaced exact replay 與 scorer parity 均為 100%。
- 所有 exception／non-finite／timeout-adjacent failure path 都能回傳安全 incumbent。
- 正式資料若尚不可取得，P0 保持「條件式通過」，不得宣稱 full validation 完成。

## 6. P1 — Analytic population、typed dynamics 與 BDP

### 目標

先證明無學習版本能把多樣化候選推近 exact-feasible 且降低 projection repair
displacement，再把它作為所有模型的可靠 teacher、baseline 與 fallback。

### 任務

- `[~] P1.1` 固定 `N={32,64,96,120}` 與候選數 bucket，建立 deterministic
  population seeds；至少包含 safe、compact、pin-aware 與方向多樣性。
- `[~] P1.2` 完成 net、pin、overlap、precedence、group、boundary、compaction、
  anchor 與 MIB shape 的 FP32 typed channels。
- `[x] P1.3` 增加 candidate telemetry：每步 energy、overlap component、HPWL、
  bbox、soft violation、projection displacement、是否 exact-feasible。

P1.3 已有 opt-in per-candidate metrics、energy history、overlap component 與
incumbent source；telemetry 不放進 submission fast path 計時。
- `[x] P1.4` 將 BDP v0 升級為 per-candidate 結果，明確回傳 success、residual、
  displacement、active-pair 數與 failure reason。
- `[x] P1.5` 加入 active-pair outer rebuild 與 bounded direction beam；只對
  unresolved conflict components 搜尋，不做全域無界離散搜尋。
- `[x] P1.6` 建立 safe／fast-feasible／exact-feasible 三層 incumbent manager；
  exact tier 永遠優先，任何 tier 都不得失去 safe incumbent。
- `[x] P1.7` 比較 fallback 與 analytic+BDP，所有 QoR
  都使用 post-BDP exact metrics，不用 raw differentiable proxy 代替。
- `[~] P1.8` 在 CPU 與 CUDA 執行 `N=120, K=32` 穩定性測試，收集 cold/warm
  p50、p95、p99、peak memory、projection success 與 tail failure。

P1.8 的 RTX 5090 warm profile 為 p50 `0.957574s`、p95 `0.962069s`、peak
`76,457,984` bytes，65 個候選有 10 個 exact-feasible；cold-start 與 A100
仍未量測，因此保持 `[~]`。

### 退出條件

- overlap energy 與 projection displacement 對基線有一致改善，不只改善 proxy。
- `N=120, K=32` 無 OOM、NaN、未界定迴圈或單一 case 失控長尾。
- 不新增 `cost=10`，且 exact-feasible incumbent 不會被較差候選覆蓋。
- P1 report 能指出收益來自 seed、dynamics 或 BDP，而非只給總平均。

## 7. P2 — 資料合約、labels、shards 與 corruptions

### 目標

把 1M FloorSet samples 轉成可稽核、可重現且不污染官方 100 validation 的訓練
資料；所有 learned 階段共用同一份 case schema 與 transform contract。

### 任務

- `[x] P2.1` 清點實際可用資料、版本、大小、欄位與 checksum；9,000 files ×
  112 layouts = 1,008,000 cases，官方 validation path 明確拒絕。
- `[ ] P2.2` 依 block count、constraint density、connectivity statistics 與
  topology hash 建立 stratified train/dev/internal-test split。
- `[~] P2.3` 從 solution fixture 抽取 rectangles、center、log-aspect、
  pairwise precedence、H/V DAG、outline、contact tree、MIB 與 boundary labels。
- `[x] P2.4` 沿用 `FloorplanCase` translation／sqrt-total-area canonicalization。
- `[~] P2.5` 實作 D4 mirror／90-degree rotation；同步轉換 pins、
  targets、fixed/preplaced geometry 與 boundary bitmask。
- `[~] P2.6` 建立 deterministic shift／aspect corruption 並硬投影 targets；
  contact/event corruption 等待擴充 exact replay labels。
- `[~] P2.7` 寫成 streaming tar shards、內嵌 sample manifest 與含 source、split、
  denylist checksum、tar SHA256 的 sidecar manifest。
- `[~] P2.8` 已測 transform round-trip、area preservation、target exactness、
  label validation、shard decode 與 100 筆 raw official audit；完整 1M split
  leakage 仍等待 stratified split。

### 退出條件

- official validation leakage 為 0。
- 所有 augmentation round-trip 後 hard geometry 語義不變。
- shard manifest 與實際內容數量、checksum、schema 完全一致。
- 任一 sample 可追溯到來源、label 版本與 split 決策。

## 8. P3 — SCENE 與 POP-INIT

### 目標

只學 static topology prior 與 population initializer，不碰 multi-step controller；
先確認 learned seeds 經同一個 analytic dynamics／BDP tail 後確實更好。

### 任務

- `[~] P3.1` 實作 SCENE static encoder 與 precedence／outline heads；contact head 等待 labels。
- `[ ] P3.2` 實作 cycle-free H/V relation decoding 與少量 structure seeds。
- `[x] P3.3` 實作 POP-INIT residual population generator，維持 hard projections。
- `[x] P3.4` checkpoint contract、warm-start、EMA、periodic checkpoints 與
  schema/hash/normalization failure fallback 已完成。
- `[x] P3.5` 已完成 structure 1k/all-head 3k 短訓練與 100-case raw/post-BDP
  attribution；raw learned 0/1,600 feasible，post-BDP learned oracle 0 wins，gate
  REJECT。
- `[~] P3.6` 已報 106–120 subset 15/15 feasible，並找到 case 88 regression、
  case 97 improvement；constraint-density 分層仍待補。

### 退出條件

- learned initializer 的 post-BDP oracle@K 明確優於 random／shelf／analytic seeds。
- fixed/preplaced replay、area 與 hard feasibility 不退步。
- checkpoint 缺失、損毀或不相容時能無條件回退到 P1。
- 若 P3 未過，停止 learned controller，contest 版本凍結在 P1。

## 9. P4 — HiCoDy recovery 與 multi-step dynamics

### 目標

先學「從已知 corruption 回到 exact solution 附近」的一步 residual，再逐步延長
unroll；learned 輸出只修正 analytic force，不直接取代 projection 或 verifier。

### 任務

- `[~] P4.1` 以 ground-truth residual 與 on-the-fly corruption 建立 rectified-flow
  target；analytic teacher replay 等待資料。
- `[x] P4.2` flow velocity head、hard mobility mask 與第一個正式短訓練
  checkpoint 已完成。
- `[~] P4.3` 已實作並測試 6-step Euler flow；後續 8、16、24 steps 必須在
  oracle@K gate 通過後才展開；每個長度獨立
  檢查 NaN、energy drift、gradient stability 與 projection displacement。
- `[ ] P4.4` 加入 hard projection after each step、best-incumbent checkpoint 與
  stagnation detector。
- `[~] P4.5` 已完成 analytic vs additive learned sidecar 的 100-case ablation；
  cost 未改善且 runtime regression，因此明確 HOLD。

### 退出條件

- one-step 與 multi-step 都在 post-BDP exact metrics 改善。
- 106–120 subset 不退步，且 runtime p95 未超過當期 cap。
- learned path 出現 NaN、checkpoint error 或 QoR regression 時可在 case 內回退。
- 若 multi-step 不穩定，保留 P3 learned initializer + P1 analytic tail。

## 10. P5 — CAL、ETR、方向學習與 PVR

### 目標

只對 analytic／learned dynamics 的真實停滯點增加局部離散決策，並學習預測
post-projection 價值，降低無效 BDP 與 exact tail 次數。

### 任務

- `[ ] P5.1` 從 exact solutions 抽取 cluster contact tree、contact side 與
  latch labels；實作 CAL hysteresis 與 semi-rigid group aggregation。
- `[~] P5.2` 已從 P4 exact tail 產生第一批 32-record candidate/outcome replay；
  conflict component、觸發原因與動作 labels 尚待擴充。
- `[ ] P5.3` 實作 bounded ETR actions：local resample、direction swap、shape
  perturb、component translation、boundary slot change。
- `[ ] P5.4` 先用 exhaustive small-component replay 產生 BDP direction outcome，
  再訓練 direction proposal；exact projector 仍負責最後判定。
- `[x] P5.5` repair-aware ranker 已由 32-record exact replay 訓練 500 steps，
  loss `0.318158 -> 0.125271`；checkpoint 維持 default-off。
- `[~] P5.6` top-M 只裁 learned 增量並永遠保留完整 analytic candidates；top-4
  已降低三案 tail runtime，但 top-1 regret、top-M recall 與 calibration 尚未達 gate。

### 退出條件

- ETR 使 stagnation case 的 exact QoR 或 projection success 改善，且沒有全域抖動。
- PVR top-1 regret 達成目標，top-M 保留 oracle 的比例可接受。
- CAL latch 不製造 group 內 overlap 或不可逆錯誤 topology。
- 稀疏離散步驟有 component size 上限與 deterministic timeout fallback。

## 11. P6 — Runtime profiling 與可攜式加速

### 目標

在包含 process startup 的真實執行模型下控制 p95；RTX 5090 用於本地開發與
訓練，正式 runtime 不依賴 5090-only 能力，保留 A100／CPU fallback。

### 任務

- `[~] P6.1` 已量 N120/K8 training FP32/BF16 與 official 100-case analytic/learned
  p50/p95/max；其餘 bucket、cold-start 與 phase breakdown 待補。
- `[~] P6.2` 已有 deterministic flow-step 與 tail-top-M runtime controls；根據
  uncertainty 與預算制定 candidate policy 仍待補，任何 adaptive policy 都必須
  有 deterministic upper bound。
- `[ ] P6.3` 先使用 vectorized PyTorch；只有 profiler 證明收益後才逐項啟用
  `torch.compile` 或 CUDA Graph，並保留 eager fallback。
- `[ ] P6.4` 只有 pairwise force 等單一 kernel 超過約 20% runtime，且 compile
  仍不足時，才評估 Triton；不為此新增 submission 不可攜依賴。
- `[x] P6.5` neural linear 可選 BF16；coordinates、dimensions、
  overlap、force accumulation 與 exact geometry 維持 FP32／evaluator 原生精度。
- `[x] P6.6` 驗證 GPU 不可用與 checkpoint 不存在／損壞／normalization mismatch
  時均能回退到已驗證 eager analytic 路徑；尚未啟用 compile。

### 退出條件

- median 與 p95 都符合當期 runtime cap，沒有 large-case uncontrolled tail。
- acceleration 關閉時 QoR 語義不變；開啟後無 hard-feasibility regression。
- 正式 A100 類環境不需要 FP8、persistent worker、網路或本機編譯 cache。

## 12. P7 — Full replay、凍結與 submission

### 目標

以正式 evaluator 與正式啟動方式完成最後 promotion；凍結後只接受 packaging
或已證明的 correctness fix，不再加入模型或搜尋策略。

### 任務

- `[ ] P7.1` internal dev／test 各跑固定多 seeds，輸出 per-case exact metrics、
  failure atlas、weighted score、median/p95/p99 與 fallback rate。
- `[~] P7.2` 106–120 bucket 已驗 15/15 hard-feasible並辨識 case 88/97；
  preplaced/group/MIB/boundary-heavy 與 disconnected-net 分層待補。
- `[~] P7.3` 已固定 analytic vs additive flow+ranker ablation與相同 exact tail；
  CAL/ETR、完整 PVR 與 acceleration 尚未進 gate。
- `[x] P7.4` 官方 validation 僅用於 promotion／freeze 證據，不回灌訓練、
  threshold fitting 或 checkpoint selection。
- `[~] P7.5` source commits、三個 model、replay、reference evaluator、benchmark
  config/report 與 visualization hashes 已記錄；submission payload 尚未 freeze。
- `[ ] P7.6` 依官方真實 payload layout 打包，驗證 root entrypoint、權限、
  offline dependencies、clean-host smoke、timeout 與 output format。
- `[ ] P7.7` 執行 final dry run，保存 invocation、host/GPU、exit code、runtime、
  checker 結果與 checksum；之後進入 freeze。

### 退出條件

- post-tail hard feasibility、fixed/preplaced exact replay、scorer parity 為 100%。
- 無新增 `cost=10`，106–120 subset 不退步。
- weighted total cost 達 promotion target，且 median／p95 在 runtime cap 內。
- 任一例外都回傳 verified incumbent；submission payload 可在乾淨離線環境啟動。

## 13. P8 — 選配自我改進

P8 不屬於最低可提交版本，只有 P7 已有穩定候選且仍有明確 headroom 時才啟動。

- `[ ] P8.1` DAgger：candidate -> exact repair -> displacement/event labels -> replay。
- `[ ] P8.2` hard-negative replay，優先補真實 failure atlas，不平均擴增資料。
- `[ ] P8.3` 只對 top-1／top-2 候選做 4–16 步 bounded local policy refinement。
- `[ ] P8.4` offline RL 僅在 supervised/local-search plateau 後評估；若增加 runtime
  tail、seed variance 或 feasibility risk，立即移除。

## 14. 每一階段的 Definition of Done

每個 task 只有同時符合以下條件才可勾選：

1. 實作或資料產物存在，且版本／輸入／輸出 contract 清楚。
2. 有針對該行為的單元或整合測試；hard geometry 相關改動必須有 regression fixture。
3. 使用 exact-compatible verifier 與 post-projection metrics 驗證，不只看 proxy loss。
4. 報告包含 per-case 或 per-bucket 分布、失敗案例與 fallback rate，不只給平均值。
5. 新路徑失敗時的 fallback 已實際測試。
6. 文件、config 與 checkpoint/schema version 同步。
7. Git diff 僅包含該 task 的必要範圍，沒有順手恢復舊系統或加入新依賴。

Promotion 決策只允許三種結果：

- `PROMOTE`：所有 hard gate 通過，且 exact QoR／runtime 達標。
- `HOLD`：correctness 通過但效益或統計證據不足，維持 default-off。
- `REJECT`：hard regression、不可控長尾或無 exact improvement，移除 default path。

## 15. 時程不足時的凍結策略

依序保留最後一個完整通過 gate 的版本：

1. P1 通過：`analytic population -> typed dynamics -> BDP -> verifier`。
2. P3 通過：`SCENE/POP-INIT -> analytic dynamics -> BDP -> verifier`。
3. P4 通過：再加入 learned residual dynamics。
4. P5 通過：再加入 CAL／ETR／PVR；未過者全部 default-off。

禁止為了趕時程犧牲 verifier、fallback、fixed/preplaced exact replay 或 official
payload smoke。模型未成熟時，P1 本身就是最低可提交主線。

## 16. 下一個 promotion 批次

可執行框架已涵蓋 analytic、learned/data/checkpoint、benchmark、profile 與
visualization，且第一輪 1.008M direct-stream training／exact replay 已完成。
下一批只處理目前 exact evidence 指出的 promotion blocker：

| 優先序 | Task | 輸入 | 必須產出 | Done 證據 |
| --- | --- | --- | --- | --- |
| 1 | Raw-safe candidate reselection | 12 raw-infeasible incumbents | candidate-pool retry，不全量重算 | 100/100 feasible 且 p95 下降 |
| 2 | Official-objective selector labels | training `metrics_sol` + exact tail | baseline-normalized outcome labels | 43-case miss 明顯下降 |
| 3 | Learned feasibility supervision | raw overlap／BDP／repair telemetry | hard-negative structure/flow loss | learned raw/post-BDP feasibility 上升 |
| 4 | Soft-constraint force repair | grouping/boundary/MIB telemetry | typed-force ablation | 解除 capped cost 且 large case 不退步 |
| 5 | Stratified internal split | 1.008M training source | source/split/checksum manifest | validation leakage = 0 |

「框架可執行且已短訓練」不等於 learned lane 已 promotion；只有上述 exact
QoR、large-case 與 runtime gates 全通過後，`HCFP_CHECKPOINT` 才能從 opt-in
升為 default。
