# BFOD v1 / P10 Case70 進度（2026-08-14）

## 定位

這一輪驗證的是 **Boundary-First Obligation-Driven Cooperative
Floorplanning (BFOD)** 的最小 deterministic prototype，而不是把多個模型接入
正式 `solve()`。所有新路徑都是 experiment sidecar；submission 與目前 P8
guarded portfolio 均未改動。

硬限制仍由既有 exact verifier 守住：preplaced 的 `(x, y, w, h)`、fixed-shape
的 `(w, h)`、area legality 與 non-overlap 必須完全正確。

## 你的 v1 規劃與目前落地對照

| 階段 | 你的意圖 | 目前實作 | 狀態 |
| --- | --- | --- | --- |
| S0 Hard geometry | freeze preplaced；freeze fixed shape；建立 mask | 既有 exact verifier + BFOD/P10 state 均保留 protected masks；候選一律 exact admission | 已完成 |
| S1 Boundary-first skeleton | 先確定 corner/side membership，保留 side-order 候選 | 重用 `boundary_skeleton_candidates`，並在 BFOD state 報告 perimeter slots | 部分完成；尚未有 learned side-order policy |
| S2 MIB shape | anchor shape、compatible broadcast、local repack | 新增 `mib_patch.py` 的 anchor-local exact repack，BFOD S2 會嘗試它 | 已實作；Case70 尚未接受 MIB repair |
| S3 Group/contact | residual obligation、joint boundary/group、4/8/12/16 patch、mandatory contact | P10 與 BFOD 都使用 bounded dense contact patch；joint/contact/boundary router 與 exact scoring 已接上 | 已完成（實驗路徑） |
| S4 HPWL topology | connectivity-aware B*-Tree、HPWL critic、局部 tree move | 重用 P8 connectivity-aware B*-Tree beam、exact HPWL critic 與 frame-core cleanup | 部分完成；尚未有 learned tree policy |
| S5 Common loop | residual obligation router、beam 4、最多 6 輪、成果鎖定 | `experiment_bfod_v1.py` 有 beam=4、max-rounds=6；P10 以 beam=4、48 decode cap、30 秒上限做 locked contact loop | 已完成（實驗路徑） |
| Learned experts | 先 deterministic oracle，成功後才訓練對應 expert | 只訓練一個 32-hidden contact-candidate ranker；exact scorer 仍是 final admission | 部分完成；boundary/shape/tree expert 未訓練 |
| Sparse region route | sparse case 走 region assignment | 沿用 P8 實驗結論：目前 QoR gate 為 REJECT，未納入 BFOD/P10 | 未啟用 |
| 正式 promotion | 有多 case、hard/QoR 證據後才改 solver | `solve()` / submission 未改；P8 guarded 仍是目前 promotion | HOLD |

## P10 Case70：單 case 因果實驗

| Variant | Cost | Uncapped | B | G | M | Hard feasible | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| P8 baseline | 9.999999 | 10.272975 | 24 | 23 | 5 | yes | 5.853 s |
| C70-A boundary rejected-best | 9.999999 | 10.272975 | 24 | 23 | 5 | yes | 1.514 s |
| C70-B mandatory contact oracle | 9.914890 | 9.914890 | 24 | 22 | 5 | yes | 0.429 s |
| C70-D locked common loop | 8.940068 | 8.940068 | 24 | 19 | 5 | yes | 1.480 s |
| C70-E locked cleanup winner | **8.642094** | **8.642094** | **23** | **19** | **5** | **yes** | 4.604 s |

- C70-0：baseline、52 個 soft debts 的 obligation CSV、before PNG 已輸出。
- C70-A：0 個 admissible boundary candidate；`REJECT`。
- C70-B：16 個 mandatory-contact proposal 中找到第一個 cap crossing；`KEEP`。
- C70-C：依原定規則，C70-B 穿 cap 後不再跑完整 MIB 同步；MIB local operator 已存在但 Case70 仍為 5。
- C70-D：已驗證接續 repair 可把 grouping 從 22 降至 19。
- C70-E：沒有降低 HPWL，但在不增加 soft debt 的情況下再消一個 boundary debt，最終 cost 降至 8.642094；`MODIFY`，保留為 winner。

產物位於 `artifacts/experiments/p10_case70/`，包括 placement、metrics、provenance、obligations 與 PNG/SVG。

## BFOD v1：把 deterministic action 接成共享 state

`scripts/experiment_bfod_v1.py` 依序執行 S0/S1/S2/S3/S4/S5，並只保留 hard-feasible、exact-scored state。它不以 case ID 改 solver；Case70 只是由實驗 CLI 指定。

在 `70, 89, 90, 94, 97` 的 deterministic smoke 中，五案都 hard feasible；Case70
從 P8 的 `9.999999` 降至 `8.952643`。這是 BFOD driver 的基線，不能和上表使用
不同 common-loop/cleanup 組合的 P10 winner 混作同一條 ablation 線。

## 第一個 learned expert：contact candidate ranker

新增的 `src/hcfp/contact_policy.py` 是小型 candidate ranker，不產生完整 placement：

- 以 24 個 runtime-visible geometry/topology 特徵對最多 16 個 deterministic contact patch 排名。
- teacher actions 只來自官方 `floorset_lite` training stream；visible validation/test root 會被拒絕。
- Case70 只提供 input-only constraint signature 和 checkpoint 評估，**不提供 teacher label**。
- 32 個訓練 state、8 個 held-out state；step 500 的 held-out top-1 / top-4 teacher recall 為 `0.25 / 0.75`。

| BFOD Case70 variant | Cost | B | G | M | Hard feasible | Runtime |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| deterministic baseline | 8.952643 | 24 | 19 | 5 | yes | 28.192 s |
| learned contact, step 500 | **8.646254** | 24 | **18** | 5 | yes | 28.200 s |
| learned contact, step 1000 | 8.646254 | 24 | 18 | 5 | yes | 28.085 s |
| learned contact, step 1500 | 8.963900 | 24 | 19 | 5 | yes | 28.231 s |

step 500 是 best checkpoint；之後兩個 checkpoint 未改善 Case70，依 gate 停止。這支持把已驗證的 contact operator 做成 teacher-action experiment，但尚不足以 promotion 到 production solver。

## 驗證與下一個最小實驗

已通過：14 個 focused tests（contact policy、MIB patch、boundary skeleton、contact patch、B*-Tree）、`py_compile`、Ruff，且 best checkpoint fresh replay 為 hard feasible（`B/G/M = 24/18/5`、cost `8.646254`）。

下一步只需選一個仍未證實的單一 hypothesis：以 P10 winner 為 incumbent，測試 MIB local repair 是否能在不回退 boundary/group 與 HPWL 的條件下減少一個 MIB debt。不要直接開始 boundary/shape/tree Transformer，也不要把這個 sidecar promotion 到 `solve()`。
