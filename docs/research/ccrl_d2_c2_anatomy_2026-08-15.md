# P11.4-D2 — Contact miss anatomy（held-out replay, 2026-08-15）

## Hypothesis

Gate D 的 C2 尾巴（exact recall 83.04% vs hard-feasible 93.57%）主要來自
(a) patch_budget single-label ambiguity、(b) same-component anchor 選錯、
或 (c) evaluation 只認單一 teacher action 的假失敗。逐一量化後決定
CCRL 下一刀是 representation、mask 還是 evaluation。

## Changed

- `scripts/analyze_ccrl_c2_anatomy.py`（新）：對 held-out replay 的 1,184 rows
  跑 model Top-4 → exact decode，分解 miss factors（target / anchor / side /
  patch_budget / role_swap）與 derived factors（teacher/top1 component 關係），
  並對照 teacher decode 幾何與 debt outcome。無模型改動、無 cache 改寫。
- `docs/research/ccrl_gate_d_decision_provenance_2026-08-15.md`（新）：
  記錄 Gate D `historical 99% -> REJECT` 與 `approved 95% -> PASS` 的
  decision provenance，歷史 artifact 不動。

## Experiment

Checkpoint：`train_fixed64_gate_d_statefix/contact_gate_d.pt`（64×2×4，50k steps）。
Replay：`full_2k_512/heldout.replay.jsonl.gz`（512 unseen sources，1,184 rows）。
產物：`artifacts/experiments/p11_ccrl_contact_replay/d2_c2_anatomy/heldout_all_anatomy.json`。

## Result

Recall 階層（Top-4）：

| Kind | exact | budget-canonical | functional | hard-feasible | equals-teacher-debt |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 (501) | 97.01% | 97.01% | 98.40% | 98.40% | 98.20% |
| C1 (512) | 89.65% | 89.65% | 95.90% | 96.68% | 95.51% |
| C2 (171) | 83.04% | 83.04% | 91.81% | 93.57% | 91.23% |
| all (1184) | 91.81% | 91.81% | 96.37% | 96.96% | 96.03% |

C2 的 29 個 exact miss 分解（Top-1 對照 teacher）：

| Factor | Count | Share |
| --- | ---: | ---: |
| role_swap（target↔anchor 互換） | 10 | 34.5% |
| anchor（對 target、錯 anchor） | 9 | 31.0% |
| target（錯 block） | 8 | 27.6% |
| side | 2 | 6.9% |
| patch_budget only | 0 | 0% |

假設檢定：

1. **patch_budget label ambiguity：REJECT。** budget-canonical recall 與 exact
   recall 在三種 kind 完全相等；same-geometry recall 也等於 exact recall
   （top-4 中沒有任何「同幾何、不同 budget」的動作）。teacher 的
   first-success budget 搜索沒有製造假 label。
2. **same-component anchor：REJECT。** teacher anchor 100% 跨 component
   （by construction）；model Top-1 選到 same-component anchor 的只有
   C0=0、C1=7/512、C2=3/171 個 miss。component-aware mask 最多修 ~1.8%。
3. **acceptable-action equivalence：KEEP。** C2 functional 91.81% vs exact
   83.04%（+8.8pp）：miss 裡過半（15/29）的 Top-4 仍含可修復 debt 的替代
   action。exact inverse recall 低估了 capability；下游 QoR 判準本來就是
   exact decode + debt，不是 teacher equality。
4. **role swap：新發現，C2 最大單一 factor。** 34.5% 的 C2 miss 是模型把
   要搬的 component 和該貼的 anchor 兩個角色對調。這是表示問題：兩個 block
   在 pair features 中不對稱資訊不足（誰該動取決於 component 大小/preplaced/
   patch 形狀，模型看得到但顯然權重不足）。

## Decision

- Evaluation 口徑：`MODIFY` — Gate D 之後的 CCRL 評估應同時報 exact /
  functional / hard-feasible 三層（本腳本已支援）。
- patch-budget canonicalization 與 component mask：`REJECT`，不進 codebase。
- role-swap：保留為下一個最小可證偽實驗的對象（見下）。

## Next experiment

單一 hypothesis：**role-swap 表示缺陷能否用零參數代價修復？**
在 `contact_action_masks` 的 anchor mask 加「target 之 component 必須不含
anchor」這條 decoder 已知的不變量是不夠的（只修 1.8%）；正確的刀是
loss 層面——把 (t,a,side) 與其 role-swap 對 (a,t,opposite_side) 視為
different-but-comparable，先量測：若 teacher role-swap 版 action 也
decode 成功且 debt 相同的比例高，則這 10 個 miss 大半是 label 對稱性
問題（eval 修），否則是模型對稱性問題（feature/loss 修）。
預算：一次 anatomy 重跑 + 一次 5 分鐘 loss ablation，不動模型容量。

Observed but not blocking current experiment: `same_geometry_best_rate`
與 exact recall 相等，確認 corruption generator 的 budget 搜索沒有留下
多解 teacher。
