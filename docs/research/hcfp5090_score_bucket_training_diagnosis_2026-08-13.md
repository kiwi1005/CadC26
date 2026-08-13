# HCFP-5090 分數分桶與訓練診斷（2026-08-13）

基準：`hcfp5090-p8-guarded-full100.json` 的 `learned` lane。分數越低越好。

## 分桶結果

| 分組 | Cost | Cases | 平均 N | Vrel | Area gap | HPWL gap | Utilization |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Elite | `< 4.5` | 12 | 49.6 | 0.491 | 0.142 | 0.863 | 0.899 |
| Strong | `4.5–5.5` | 40 | 64.5 | 0.526 | 0.627 | 0.977 | 0.720 |
| Middle | `5.5–6.5` | 25 | 72.4 | 0.585 | 0.656 | 1.110 | 0.701 |
| Weak | `6.5–8.0` | 17 | 91.6 | 0.632 | 0.869 | 1.191 | 0.643 |
| Critical | `>= 8.0` | 6 | 84.8 | 0.726 | 1.027 | 1.384 | 0.649 |

完整逐 case 資料由 `scripts/analyze_hcfp_score_buckets.py` 產生至
`artifacts/analysis/hcfp5090-p8-score-buckets.{json,md}`。

## 低分案例的共同點

這裡的「低分」指表現差、official cost 高的尾端案例。

與 Elite 相比，Critical cases 同時具有：

- `Vrel`: `0.491 -> 0.726`；
- `area gap`: `0.142 -> 1.027`；
- `HPWL gap`: `0.863 -> 1.384`；
- utilization: `0.899 -> 0.649`；
- boundary violations: `12.2 -> 20.0`；
- grouping violations: `4.9 -> 15.7`；
- p2b edges/block: `3.45 -> 12.79`；
- B2B graph density: `0.167 -> 0.088`。

因此最差 cases 的共同壓力是：pin attraction 更強、block-to-block topology 訊號更稀疏、
preplaced/constraint debt 更高，最後同時表現在 HPWL、soft constraints 與 packing 面積上。

## Critical 不是單一失敗型態

### Dense topology/constraint：49、70、89

這三個 case 的 area gap 已經接近或低於零，utilization 約 `0.95–0.975`。問題不是 packing
不夠密，而是 grouping/boundary contact 與 HPWL adjacency 錯誤。延長 B*-Tree topology
訓練有機會改善 adjacency；單純增加 coordinate flow 或 compaction 不會對症。

### Sparse area fragmentation：21、61、93

三個 case 的 area gap 分別為 `1.862/1.180/3.096`，utilization 只有
`0.343/0.440/0.235`。這些 cases 需要 obstacle/anchor-aware topology；若延長訓練只改善
tree edge loss、卻沒有改善這三個 case，下一步應回到 region assignment/subtree split，而不是
繼續堆訓練 steps。

## 是否沒 train 到

現用 B*-Tree checkpoint `hcfp5090-p34-btree-head-s300-seed7301.pt` 有直接的 undertraining
證據：

- 僅 `300 steps / 300 samples`；
- 只看過 `106–120` blocks；
- B*-Tree loss `10.8024 -> 3.7368`，尚未飽和；
- full100 runtime 卻把這個 head 用在 `21–120` blocks。

結論：**B*-Tree head 確實沒 train 夠，但這只能解釋 topology 泛化不足，不能單獨解釋所有
Critical cases。** 下一輪先做 all-size B*-Tree continuation；constraint-only 長訓不是第一選擇，
因為先前 constraint-only fine-tune 已未能超越 parent checkpoint。

## 訓練實驗結果

### 1,000-step all-size pilot

從原 `300-step / 106–120` checkpoint 接續，只解凍 B*-Tree head，改用 `21–120`
score-aware samples：

```text
checkpoint: hcfp5090-p9-btree-all-s1000-seed8131.pt
steps:      1,000
samples:    1,000 unique / 5,000 source limit
lr:         1e-4
precision:  BF16
```

Guarded full100 fresh A/B：

| Metric | P8 old head | 1k all-size head |
| --- | ---: | ---: |
| weighted cost | 6.414040 | **6.249533** |
| capped feasible | 2 | **1** |
| hard feasible | 100/100 | 100/100 |
| p50 runtime | 5.025 s | 4.877 s |
| p95 runtime | 13.464 s | 13.376 s |

以 old-head score bands 分析：

| Band | Weighted old | Weighted 1k | Wins / ties / losses |
| --- | ---: | ---: | ---: |
| Elite | 4.452737 | 4.454003 | 0 / 10 / 2 |
| Strong | 5.147308 | 5.152061 | 6 / 27 / 7 |
| Middle | 5.874841 | 5.818125 | 6 / 12 / 7 |
| Weak | 7.290574 | **6.983841** | 8 / 8 / 1 |
| Critical | 9.507044 | **8.940368** | 5 / 1 / 0 |

最大收益是 case 21 `8.5209 -> 4.9714`、61 `8.3599 -> 5.6264`、49
`8.2881 -> 7.1333`；case 89 也由 cap `9.999999 -> 9.4814`。case 70 仍 capped。

### 延長至 5,000 steps

再追加 4,000 steps、lr `5e-5` 後，supervised loss `3.1523 -> 1.9715`，但 QoR
沒有持續改善。Critical+canary12 weighted cost：

```text
old 300-step: 8.2464
all-size 1k:   7.7059   best
all-size 2k:   critical6 9.4055 vs 1k 9.1409
all-size 3k:   critical6 9.8454
all-size 4k:   critical6 9.6669
all-size 5k:   7.8836 on critical+canary12
```

因此更多 steps 的 supervised loss 下降沒有自動轉化成 placement QoR；最佳點在早期 1k。

## Decision

**KEEP/MODIFY**：保留 1k all-size checkpoint 作 Weak/Critical specialist。它證明原 B*-Tree
head 確實 undertrained，full100 weighted cost 改善 2.56%，且少一個 capped case。

**REJECT**：5k checkpoint 與「只要繼續加 steps 就會更好」的假設。

目前不直接宣稱 1k checkpoint 為零退步 default，因為 full100 是 `25 wins / 58 ties /
17 losses`。下一個最小實驗是依 runtime-visible case pressure routing old/all-size heads；case 70
則繼續用 dense contact patch，不應再靠延長 B*-Tree 訓練。
