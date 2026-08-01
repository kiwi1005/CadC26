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
| FP32 typed analytic dynamics | `[~]` | 已有 deterministic 第一版；尚缺完整 bucket QoR 與 runtime 證據。 |
| BDP v0 | `[~]` | 已有 deterministic projection；尚非完整 active-pair outer-loop／方向搜尋版本。 |
| Official 100-case replay | `[blocked]` | 目前尚未取得並完成官方 validation 全量重播。 |
| 1M data shards／labels | `[ ]` | 尚未建立資料清單、切分、label extractor 與 shard manifest。 |
| SCENE／POP-INIT | `[ ]` | 尚未訓練。 |
| HiCoDy learned residual dynamics | `[ ]` | 尚未訓練；必須先完成 one-step recovery gate。 |
| CAL／ETR／PVR | `[ ]` | 尚未建立 contact/event/projection-value labels。 |
| Submission freeze／package proof | `[ ]` | 尚未做正式資料 replay、portable smoke、hash 與 dry run。 |

因此目前正確定位是：**P0 主體已落地，P1 有 analytic／BDP v0 雛形；P0
的正式資料證據與 P1 的效益、穩定性、長尾 gate 仍未關閉。**

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
- P3 的 oracle@K 未優於 analytic initializer，不開始 P4 multi-step。
- 沒有 exact replay 產生的 event／repair labels，不訓練 ETR 或 PVR。
- 未經 profiler 證明的 kernel，不改寫 Triton／CUDA。
- P0–P7 任一新路徑失敗時，正式輸出仍回退到最後一個 verified incumbent。

## 4. 階段總覽

| 階段 | 建議時窗 | 主要結果 | 退出 gate | 目前狀態 |
| --- | --- | --- | --- | --- |
| P0 | D1–D2 | 官方合約、parity、fallback、determinism | correctness gate 100% | 進行中 |
| P1 | D3–D5 | Analytic population、typed dynamics、BDP、telemetry | post-BDP 改善且 N120/K32 穩定 | 進行中 |
| P2 | D6–D8 | 訓練 split、labels、shards、corruptions | audit／round-trip／leakage 全通過 | 未開始 |
| P3 | D9–D11 | SCENE、POP-INIT、oracle@K | post-BDP oracle@K 優於 analytic seeds | 未開始 |
| P4 | D12–D14 | one-step 與 multi-step HiCoDy | exact QoR 改善且 large case 不退步 | 未開始 |
| P5 | D15–D17 | CAL、ETR、direction learning、PVR | stagnation／ranker regret 明確改善 | 未開始 |
| P6 | D18 | compile／graph／profile／portable fallback | p95 達標且官方環境可執行 | 未開始 |
| P7 | D19–D20 | full replay、freeze、package、dry run | submission hard gates 全通過 | 未開始 |
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
- `[ ] P0.6` 對可取得的官方 validation cases 執行 fallback-only 全量重播。
- `[ ] P0.7` 補齊 boundary 16 種 bitmask、MIB 可相容／不可相容、cluster、
  fixed/preplaced obstacle 的 edge-case fixtures。
- `[ ] P0.8` 同一輸入重跑至少 10 次，確認 CPU 與 CUDA 各自 bitwise 或
  tolerance-bounded deterministic；記錄允許的差異來源。
- `[ ] P0.9` 產生單一 P0 correctness report，包含 case 數、hard feasibility、
  parity mismatch、fallback 使用率、例外與 non-finite 計數。

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
- `[ ] P1.3` 增加 candidate telemetry：每步 energy、overlap component、HPWL、
  bbox、soft violation、projection displacement、是否 exact-feasible。
- `[ ] P1.4` 將 BDP v0 升級為 per-candidate 結果，明確回傳 success、residual、
  displacement、active-pair 數與 failure reason。
- `[ ] P1.5` 加入 active-pair outer rebuild 與 bounded direction beam；只對
  unresolved conflict components 搜尋，不做全域無界離散搜尋。
- `[ ] P1.6` 建立 safe／fast-feasible／exact-feasible 三層 incumbent manager；
  exact tier 永遠優先，任何 tier 都不得失去 safe incumbent。
- `[ ] P1.7` 比較 random、shelf、analytic dynamics、analytic+BDP，所有 QoR
  都使用 post-BDP exact metrics，不用 raw differentiable proxy 代替。
- `[ ] P1.8` 在 CPU 與 CUDA 執行 `N=120, K=32` 穩定性測試，收集 cold/warm
  p50、p95、p99、peak memory、projection success 與 tail failure。

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

- `[ ] P2.1` 清點實際可用資料、版本、大小、欄位與 checksum；官方 100
  validation 明確列入 denylist。
- `[ ] P2.2` 依 block count、constraint density、connectivity statistics 與
  topology hash 建立 stratified train/dev/internal-test split。
- `[ ] P2.3` 從 `fp_sol`／`metrics_sol` 抽取 rectangles、center、log-aspect、
  pairwise precedence、H/V DAG、outline、contact tree、MIB 與 boundary labels。
- `[ ] P2.4` 實作 translation／scale canonicalization；anchor 存在時使用其
  weighted centroid，否則以 ground-truth bbox 原點對齊。
- `[ ] P2.5` 實作 permutation、mirror、90-degree rotation；同步轉換 pins、
  targets、fixed/preplaced geometry 與 boundary bitmask。
- `[ ] P2.6` 建立線上 corruptions：shift、overlap injection、aspect perturb、
  cluster detachment、boundary release、MIB mismatch、SP swap、component move。
- `[ ] P2.7` 寫成 tar shards 與 manifest；每個 shard 記錄 sample count、schema
  version、split、source version 與 checksum。
- `[ ] P2.8` 執行 transform round-trip、area preservation、target exactness、
  split leakage 與 shard decode audit。

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

- `[ ] P3.1` 實作 SCENE static encoder 與 precedence、outline、contact heads。
- `[ ] P3.2` 實作 cycle-free H/V relation decoding 與少量 structure seeds。
- `[ ] P3.3` 實作 POP-INIT residual population generator，維持 hard projections。
- `[ ] P3.4` 訓練與 checkpoint contract：EMA、schema/version、normalization、
  model hash 與 loading failure fallback。
- `[ ] P3.5` 評估 best-of-K diversity、structure accuracy、raw oracle@K 與
  post-BDP oracle@K；後者才是 promotion 主指標。
- `[ ] P3.6` 對 106–120 blocks、high constraint density 與 preplaced-heavy
  subsets 分別報告，不以全體平均掩蓋 large-case regression。

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

- `[ ] P4.1` 以 ground truth correction／analytic teacher 建立 one-step recovery
  dataset 與 exact post-step labels。
- `[ ] P4.2` 訓練 one-step learned residual；限制 velocity、shape mobility 與
  fixed/preplaced mobility。
- `[ ] P4.3` one-step gate 通過後，依序 unroll 8、16、24 steps；每個長度獨立
  檢查 NaN、energy drift、gradient stability 與 projection displacement。
- `[ ] P4.4` 加入 hard projection after each step、best-incumbent checkpoint 與
  stagnation detector。
- `[ ] P4.5` 以 analytic-only、learned-only、analytic+residual 做 ablation；禁止
  只用 training loss 宣稱改善。

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
- `[ ] P5.2` 從 P4 replay 建立 hard-negative event labels，包含 conflict
  component、觸發原因、動作與 exact outcome。
- `[ ] P5.3` 實作 bounded ETR actions：local resample、direction swap、shape
  perturb、component translation、boundary slot change。
- `[ ] P5.4` 先用 exhaustive small-component replay 產生 BDP direction outcome，
  再訓練 direction proposal；exact projector 仍負責最後判定。
- `[ ] P5.5` 訓練 PVR 預測 post-repair feasibility、violations、QoR、repair
  displacement 與 tail runtime。
- `[ ] P5.6` 以 lexicographic hard gate 選 top-M；永遠保留 current incumbent，
  並測量 top-1 regret、top-M recall、calibration 與 tail call reduction。

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

- `[ ] P6.1` 以 `N={32,64,96,120}`、候選數與 constraint density 分桶，量測
  cold/warm p50、p95、p99、peak memory 與各 phase time。
- `[ ] P6.2` 根據 uncertainty 與預算制定 candidate／step／tail-top-M policy；
  任何 adaptive policy 都有 deterministic upper bound。
- `[ ] P6.3` 先使用 vectorized PyTorch；只有 profiler 證明收益後才逐項啟用
  `torch.compile` 或 CUDA Graph，並保留 eager fallback。
- `[ ] P6.4` 只有 pairwise force 等單一 kernel 超過約 20% runtime，且 compile
  仍不足時，才評估 Triton；不為此新增 submission 不可攜依賴。
- `[ ] P6.5` neural linear／attention 可評估 BF16；coordinates、dimensions、
  overlap、force accumulation 與 exact geometry 維持 FP32／evaluator 原生精度。
- `[ ] P6.6` 驗證 GPU 不可用、checkpoint 不存在、compile failure、不同 GPU
  capability 時均能回退到已驗證 eager analytic 路徑。

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
- `[ ] P7.2` 獨立檢查 106–120 bucket、preplaced-heavy、group-heavy、MIB-heavy、
  boundary-heavy 與 disconnected-net cases。
- `[ ] P7.3` 固定 ablation：P1 analytic、+POP-INIT、+HiCoDy、+CAL/ETR、+PVR、
  +acceleration；每項都用相同 exact tail 與 budget。
- `[ ] P7.4` 官方 validation 僅用於 promotion／freeze 證據，不回灌訓練、
  threshold fitting 或 checkpoint selection。
- `[ ] P7.5` 鎖定 source、model、reference evaluator、config 與 payload hashes。
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

## 16. 下一個執行批次

下一批工作先關閉 P0 與 P1，不開始資料訓練：

| 優先序 | Task | 輸入 | 必須產出 | Done 證據 |
| --- | --- | --- | --- | --- |
| 1 | P0.6 正式資料可用性與 fallback sweep | 本地 FloorSet cache／官方格式 | case inventory + correctness report | 所有可用 cases hard-feasible；缺資料明確列為 blocker |
| 2 | P0.7 edge-case parity fixtures | v10 evaluator predicates | boundary/MIB/group/target fixtures | local 與 official predicate 全一致 |
| 3 | P0.8 determinism audit | 固定 seeds、CPU/CUDA | repeatability report | 差異在明確 tolerance 內，無隱藏 randomness |
| 4 | P1.3 telemetry | 現有 analytic runner | per-step/per-candidate metrics | 可定位 overlap、projection displacement 與 tail failure |
| 5 | P1.4–P1.5 BDP 強化 | BDP v0 + conflict components | structured result + bounded outer loop | success/failure 可解釋，無無界迴圈 |
| 6 | P1.6 incumbent tiers | verifier + candidate metrics | safe/fast/exact manager | 任意 failure 都保留 verified incumbent |
| 7 | P1.7–P1.8 benchmark | fixed buckets與基線 seeds | exact QoR + p50/p95/p99 report | P1 promotion gate 可做明確判定 |

完成這批後，先產出 `PROMOTE／HOLD／REJECT` 的 P1 決策，再決定是否進入 P2；
不以「程式已寫完」取代 promotion evidence。
