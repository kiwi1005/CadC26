# HCFP-5090 訓練與擺置閉環報告 — 2026-08-01

## 結論

本輪已完成可執行的正式資料閉環：

```text
FloorSet-Lite 1.008M direct stream
  -> score-aware supervised training
  -> multi-step rectified-flow sidecar candidates
  -> analytic population + exact BDP tail
  -> exact-tail replay
  -> repair-aware ranker top-k
  -> raw official feasibility gate
  -> official 100-case benchmark + HTML visualization
```

所有 hard gate 都已恢復為 100%，但 trained checkpoint 的正式決策是
**HOLD**：它尚未降低 capped official cost，且 runtime 高於 analytic baseline。
因此 `submission/optimizer.py` 仍維持 analytic default；learned checkpoint 只允許
以 `HCFP_CHECKPOINT` 明確 opt-in。

## 1. 正式資料與資料路徑

- 官方 `floorset_lite/worker_*/layouts*` 共 9,000 個檔案。
- 每檔 112 個 layouts，合計 1,008,000 筆訓練案例。
- 訓練直接逐檔 streaming，不複製成另一份 24 GB shards。
- visible validation/test 路徑會被 adapter 拒絕。
- 抽查 100 筆 raw `fp_sol`：100/100 無 overlap、hard targets 合規、面積相對
  誤差最大值為 0；100/100 normalized labels 通過一致性檢查。
- 已驗證並刪除下載壓縮檔；原始 archive SHA256：
  `dad21c725a31185a826b0bac4c392e8c0d4de0f21d143b431ab70366f75daf82`。

score-aware rejection probability 為：

```text
0.30 + 0.70 * min(exp((n - 80) / 12), 8) / 8
```

檔案與 layout 順序都由 seed 決定，避免短訓練只看到相鄰、同 block-count
案例。

## 2. 訓練路徑優化

### 單次 model forward

`stage=all` 原本為 flow loss 重跑完整 encoder；目前先建立 flow corruption，
再以一次 forward 同時取得 structure、initializer 與 velocity heads。

### Vectorized precedence labels

pairwise precedence 已由 Python `N x N` 迴圈改為 dense tensor predicate，並以
scalar reference regression test 鎖住 `ambiguous/tie` 語義。

在 224 筆官方 samples 上：

| 路徑 | Throughput |
| --- | ---: |
| 舊 Python precedence | 12.4227 samples/s |
| Vectorized precedence | 72.6429 samples/s |
| 改善 | 5.85x |

資料準備已高於目前約 12 training steps/s，暫不引入 cross-case padding 或
多進程 DataLoader。

### RTX 5090 N120/K8 profile

| Compute | p50 | p95 | Steps/s | Peak allocation |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 82.245 ms | 91.123 ms | 11.939 | 41.423 MiB |
| BF16 model layers | 82.594 ms | 87.241 ms | 12.053 | 30.823 MiB |

BF16 的主要收益是約 25.6% peak-memory reduction；本 profile 沒有證明 speedup。
座標、尺寸、overlap 與 force accumulation 仍維持 FP32。

Profile artifacts：

```text
05dc713b88a3a7c87aa5024f2796a86254bdcd85343061a00a7d50dd4fc67c5c  profile-n120-k8-fp32-one-forward.json
f1c9e36fece61406b7abf8f22805012c110cd7abd8d8defcbeb33f5fd758ec86  profile-n120-k8-bf16-one-forward.json
```

## 3. 本輪訓練產物

| 階段 | Steps | First loss | Last loss | State hash |
| --- | ---: | ---: | ---: | --- |
| Structure/outline warm-up | 1,000 | 1.780828 | 0.190650 | `486a99243f52...8079a2` |
| All heads warm-start | 3,000 | 1.620870 | 0.747307 | `070d558890ad...90477` |
| Exact-replay ranker | 500 | 0.318158 | 0.125271 | `5781e52c3a43...1aec30` |

All-head 最後一步的 flow loss 為 `0.442683`。三個 checkpoint 都使用 hash、
schema 與 normalization fail-closed loader；supervised checkpoint 使用 EMA
權重。

File SHA256：

```text
99a3b49ee524f216f19f9b0046b086a40661d9e8051a1a1827f42fb955ef20de  hcfp5090-structure-s1000.pt
c793f4409384546eef398e3d43d2dbaa39e4cd6532cb8d53a998ef358a7ab336  hcfp5090-flow-all-s3000.pt
6b89a287cecae5f12e578216421a72c4c8513602cb0636cf916258b2fd6a5fa3  hcfp5090-ranked-r500-v2.pt
```

Exact tail 產生 32 筆 replay records；JSONL SHA256：
`7e635cad5fb0fc5e0c3e87943a40a9b77c54d108fc091dfe04cfe34c173cae41`。

## 4. Sidecar 與 hard-safety 修正

learned lane 現在保留完整 analytic `K` candidates，再加入 learned candidates；
ranker top-k 只裁 learned 增量，不能移除既有 analytic seeds。

本輪正式資料另外暴露兩個 FP32 邊界：

1. raw 合法的 preplaced touching edges 正規化後可能出現微小 overlap；目前在
   raw adapter 驗證一次並序列化 validation provenance，不全域放寬 tolerance。
2. normalized winner denormalize 後仍可能出現 raw overlap；learned lane 會依序
   replay raw-verified analytic incumbent 與 safe fallback。

修正前的 11 個 learned hard failures 已逐案重跑為 11/11 feasible。projection
beam 也改為 feasibility、fixed-pair status、residual overlap 的 lexicographic
選擇，不再偏好新 fixed-pair failure。

## 5. Official validation 100 結果

Exact command：

```bash
PYTHONPATH=src python scripts/benchmark_hcfp.py \
  --optimizer analytic=submission/optimizer.py \
  --optimizer learned=scripts/audit_learned_optimizer.py \
  --checkpoint learned=artifacts/checkpoints/hcfp5090-ranked-r500-v2.pt \
  --baseline analytic --data-path artifacts/floorset-v10 --cases all \
  --device cuda --flow-steps 6 --tail-topk 4 \
  --output artifacts/benchmarks/hcfp5090-trained-ranked-top4-validation100-final.json \
  --visualize-dir artifacts/benchmarks/hcfp5090-trained-ranked-top4-validation100-final \
  --visualize-cases 0,50,99
```

| Metric | Analytic | Trained sidecar top-4 |
| --- | ---: | ---: |
| Hard-feasible | 100/100 | 100/100 |
| 106–120 hard-feasible | 15/15 | 15/15 |
| Weighted cost | 9.999999 | 9.999999 |
| Runtime p50 | 0.526678 s | 1.277437 s |
| Runtime p95 | 1.305969 s | 3.100891 s |
| Runtime max | 1.903064 s | 4.339856 s |

100 cases 中，93 個的 HPWL/area/soft metrics 在 `1e-8` 內相同。具有實質差異
的主要案例：

- case 44：HPWL gap `-1.9110`、soft violation `-0.02174`，但 area gap
  `+0.38494`。
- case 51：HPWL gap `-2.4595`、area gap `-0.14418`、soft violation
  `-0.03922`。
- case 97（118 blocks）：HPWL gap `-2.4918`、area gap `-1.03546`、soft
  violation `-0.01667`。
- case 88（109 blocks）：HPWL gap `+0.50094`、area gap `+0.55326`、soft
  violation `+0.03175`，為明確 regression。

Top-4 ranker 在三案 smoke 將 case 99 learned runtime 從 `3.9938s` 降到
`3.2548s`，但仍顯著慢於 analytic，且 final 100-case p95 不符合 promotion
gate。

Final report SHA256：
`aa64130942444b658f3d7160a68cacef630437dd1e631918cca640bf9536682e`。

## 6. 視覺化

最終報告含 self-contained HTML：

```text
07459bfdc210b351d02312b0fade6036374e9a9887d6da09e4e93337f80f2f6d  case_0.html
665f23c60d6008e9c03a3460680519e1e9ed3f0dff507e161944123fc00ae6f5  case_50.html
7d9c6f988b229217c54845dc8b62642e67ccd299c8d178dbb17ff9941fdd3c6a  case_99.html
```

頁面包含 analytic/learned placements、block labels、constraints coloring、bbox
與官方 metrics，不需要外部 JavaScript 或 plotting dependency。

## 7. Promotion decision 與下一輪

Decision：**HOLD**。

已通過：

- official hard feasibility 100/100；
- 106–120 subset 15/15；
- fallback/analytic/raw replay 鏈完整；
- checkpoint、replay、benchmark、visualization 可重現；
- repository 135 tests、Ruff 與 compileall 全通過。

未通過：

- weighted cost 沒有改善；
- case 88 large-case regression；
- learned runtime median/p95 高於 analytic；
- replay 只有 32 records，ranker calibration 仍不足。

下一輪只做能直接改善上述 gate 的工作：

1. 先輸出 raw/post-BDP oracle@K 與 candidate-source attribution，確認 learned
   candidate 是否真的贏過 analytic，而非只在 incumbent replay 後持平。
2. 對 case 44/51/88/97 產生 hard-negative replay，擴大到至少數千 records。
3. 將 raw margin 納入 ranker feature，減少 denormalize 後重跑 analytic 的成本。
4. 改善 grouping/boundary/MIB typed forces，優先解除 `cost=9.999999` cap。
5. 只有 oracle@K 與 ranker regret 過 gate 後，才把正式訓練延長到原規劃的
   20k–80k steps；目前不擴模型、不寫 custom CUDA。
