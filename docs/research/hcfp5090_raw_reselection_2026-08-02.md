# HCFP-5090 Raw-safe Candidate Reselection — 2026-08-02

## 結論

learned lane 的 normalized incumbent 在正式座標中不合法時，現在先重用已完成
BDP 的 projected candidate pool。只有非-fallback candidate 通過 raw official
hard verifier 才直接回傳；若只能找到 fallback，仍沿用 analytic replay，再由
safe fallback fail closed。

正常 raw-feasible fast path不增加 candidate scan，submission default 仍為
analytic。

## 實作合約

候選先通過 normalized hard-feasibility，再依既有 incumbent key 排序：

```text
(soft_violation, bbox_area + 0.05 * hpwl, stable_candidate_index)
```

每個候選經 `to_official_placements()` 重播 raw fixed/preplaced targets，再以 raw
`verify_feasible()` 判定。index 0 是 fallback；它只作 fail-closed incumbent，
不能為了省 runtime 取代既有 analytic replay。

## 100-case 行為歸因

相同 checkpoint、top-4、12-step dynamics、24-step projection、beam 4：

```text
selected raw-infeasible: 12 cases
non-fallback pool reuse:  7 cases  [4, 36, 49, 54, 68, 77, 91]
analytic replay retained: 5 cases  [1, 44, 45, 51, 71]
normal fast path:         88 cases
```

對 7 個 pool-reuse cases 的同配置直接計時：

```text
before: 15.126396 s
after:  11.235154 s
reduction: 25.72%
```

## Official validation 100

報告：

```text
artifacts/benchmarks/hcfp5090-raw-reselection-qor-safe-validation100.json
SHA256 c945c57b7fe78a223065c4604aa6e19c7d40cd08f0711ecf182a29294ccff907
```

| Metric | Analytic | Learned + raw reselection |
| --- | ---: | ---: |
| hard-feasible | 100/100 | 100/100 |
| 106–120 hard-feasible | 15/15 | 15/15 |
| weighted cost | 9.999999 | 9.999999 |
| runtime p50 | 0.537487 s | 1.254564 s |
| runtime p95 | 1.342992 s | 3.169762 s |

official cost 仍全部被 cap，因此 promotion 維持 **HOLD**。以不含 runtime、未
截斷的 official-quality objective 分析，本次真正改變的 7 個 pool-reuse cases
全部改善，沒有 regression：

| Case | Blocks | Analytic objective | Reselected objective |
| ---: | ---: | ---: | ---: |
| 4 | 25 | 17.315109 | 12.006073 |
| 36 | 57 | 56.283802 | 19.841057 |
| 49 | 70 | 66.988118 | 23.480754 |
| 54 | 75 | 61.718383 | 23.069656 |
| 68 | 89 | 75.543010 | 25.567639 |
| 77 | 98 | 58.385790 | 20.995205 |
| 91 | 112 | 40.026846 | 20.364848 |

整體 learned p95 仍高於 analytic，且 normal fast path 尚有既存 QoR regression；
本修正只解決 raw failure branch，不宣稱 learned lane 已可 promotion。

## 下一步

1. 從 training `metrics_sol` 保存 area/HPWL baseline。
2. 將 exact-tail candidate 轉成 official uncapped objective labels。
3. 訓練 objective-aligned selector，降低先前量到的 43/100 incumbent miss。
4. selector evidence 通過後，再加入 learned raw-feasibility／repair supervision。
