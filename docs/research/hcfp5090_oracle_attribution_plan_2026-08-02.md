# HCFP-5090 Oracle／Source Attribution 計畫 — 2026-08-02

## 決策目標

目前 learned sidecar 已達 100/100 hard-feasible，但官方 weighted cost 仍被
`9.999999` cap 截斷，且 runtime 高於 analytic。下一步先回答一個問題：

> learned candidates 本身是否優於 analytic candidates，還是只有 ranker／tail
> 沒有選到它們？

在回答前，不延長訓練、不擴模型、不新增 CUDA kernel，也不改 submission
default。

## Phase A — Attribution contract

每個 exact-tail candidate 必須標示來源：

```text
fallback
analytic_initial
learned_initial
analytic_relaxed
learned_relaxed
```

對 raw 與 post-BDP placements 分別保存：

- candidate index 與 source；
- hard feasibility；
- HPWL gap、area gap、soft violation；
- capped official cost；
- 不含 runtime、未截斷的 official-quality objective；
- overall oracle 與各 source oracle；
- incumbent 實際選中的 candidate/source。

未截斷 objective 沿用 v10 quality／violation 公式：

```text
(1 + 0.5 * (max(0, hpwl_gap) + max(0, area_gap)))
* exp(2 * violations_relative)
```

infeasible candidate 不得成為 oracle。這個 objective 只用於 attribution，不能
取代 official scorer 或 submission incumbent gate。

## Phase B — 最小實作

1. 重用 pinned evaluator loader、checkpoint loader、learned analysis 與 exact tail。
2. 新增一個 audit CLI；不修改 submission API，不新增第三方依賴。
3. CLI 支援 `all` 或指定 validation case ids，輸出 deterministic JSON。
4. checkpoint 缺失／不相容時 fail closed，不得把 analytic fallback 記成 learned。
5. 將 candidate/source 對應與 oracle selection 寫成可單元測試的純函式。

## Phase C — 驗證順序

1. synthetic fixture：鎖定 source index layout、infeasible exclusion 與 uncapped
   objective。
2. case `44,51,88,97`：確認 improvement/regression 是哪一類 candidate 造成。
3. official 100 cases：輸出 raw/post-BDP source win counts、learned oracle gain、
   incumbent miss count 與 106–120 subset。
4. 執行 full pytest、Ruff、compileall 與 `git diff --check`。

## Promotion 分支

- 若 learned post-BDP oracle 明顯優於 analytic，但 incumbent／top-k 常 miss：
  擴充 exact replay、加入 raw margin，優先修 ranker。
- 若 learned raw oracle 好、post-BDP oracle 退化：修 BDP direction／repair policy。
- 若 learned raw 與 post-BDP oracle 都沒有改善：修 structure／flow／typed forces，
  不增加 ranker 訓練。
- 若 large-case subset 退步：維持 learned default-off，先建立 bucket-aware policy。

## Done gate

- 100 cases attribution 成功且 checkpoint 確實使用；
- overall/source oracle 數量與 candidate 數對帳；
- raw/post-BDP hard feasibility 可追溯；
- case 44/51/88/97 有 per-case source 證據；
- 不影響現有 100/100 submission hard feasibility；
- repository tests、Ruff、compileall 全通過。

## 執行結果

完整 validation 以 runtime 對齊設定執行：`population=8`、12-step dynamics、
24-step projection、direction beam 4、6-step flow，且 primary oracle 不使用
`tail_topk`。報告：

```text
artifacts/benchmarks/hcfp5090-oracle-attribution-fulltail-validation100.json
SHA256 6f265e2190a25cce40070af59a519fabfdce85266c3f37d2cbb34c7efa4592b6
```

| 指標 | 全部 100 cases | 106–120 subset |
| --- | ---: | ---: |
| raw candidates | 3,300 | 495 |
| raw learned feasible | 0/1,600 | 0/240 |
| post-BDP learned feasible | 111/1,600 | 6/240 |
| learned overall oracle wins | 0 | 0 |
| analytic overall oracle wins | 100 | 15 |
| internal incumbent misses | 43 | 11 |
| internal incumbent raw-infeasible | 12 | 1 |

post-BDP 有 54 cases 同時存在 analytic 與 learned feasible oracle；54/54 都由
analytic 勝出。large-case comparable cases 為 6，learned 也是 0/6。

四個關鍵案例的完整 tail audit 已重跑兩次，SHA256 固定為
`5ab57b06acad7c82ff8e0024a4a682542148a626b81e366bd54b731941e82cd9`。

## 決策

本 gate 結論為 **learned generator REJECT／submission default 不變**。證據同時
指出兩個下一步，順序不可顛倒：

1. 先讓 raw official infeasible 的 normalized incumbent 在既有 candidate pool
   內選下一個合法候選，避免 12-case analytic 全量重算。
2. 使用 training `metrics_sol` 建立 official-objective-aligned selector labels，
   修復 43-case incumbent miss；validation 僅作 promotion，不回灌 threshold。
3. 對 learned raw overlap、post-BDP feasibility 與 repair displacement 加入明確
   supervision，再重訓 structure／flow；目前不擴 ranker replay。
4. learned oracle 能贏過 analytic 後，才重新啟動 top-k calibration 與長訓練。
