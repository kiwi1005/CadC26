# 初步研究測試計劃：驗證模型是否真的能學到「Instruction-Aware Puzzle Placement」


ref:"https://github.com/IntelLabs/FloorSet/tree/main"

下面是一版可落地的 **research validation plan**。核心設計原則是：不要一開始只看 final floorplan cost，因為那會混在一起看不出模型到底學到的是：

1. constraint legality；
2. action grammar；
3. block role；
4. sequential behavior；
5. feedback-driven correction；
6. 或只是被 rule-based legalizer 補救。

所以我建議把驗證拆成一個 **learning ladder**：
**role 可學 → action 可學 → partial rollout 可學 → full rollout 可學 → feedback 可改進 → scaffold 可淡出**。

---

# 0. 測試目標與核心假設

## 0.1 研究問題

你要驗證的不是「模型能否輸出座標」，而是：

> 模型是否能從資料與 feedback 中學到一套可解釋、受 constraint 約束、可逐步生成 floorplan 的 puzzle placement policy？

因此實驗不應只有 final cost，而要有 **behavior-level metrics**。

---

## 0.2 主要 hypotheses

| Hypothesis | 要驗證什麼                                                                | 成立代表什麼                                                               |
| ---------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| H1         | block role / constraint role 可由 graph + metadata + final layout 統計學出 | anchor、hub、boundary-seeker、shape-locked、follower 不是任意人工命名，而是資料中有可學訊號 |
| H2         | pseudo-trajectory 中的 primitive action 可被穩定模仿                         | `attach / snap / pull / equalize / align / freeze` 不是過度手工規則，而是可學行為單位 |
| H3         | typed action + legality mask 可顯著降低 invalid action rate               | action contract 對 hard constraints 有實際效果                             |
| H4         | role-conditioned policy 比 plain graph policy 更好                      | semantic role 對 placement behavior 有增益                               |
| H5         | free rollout performance 不只是 teacher-forcing accuracy                | 模型真的能逐步解 puzzle，而不是只在給定正確歷史時預測下一步                                    |
| H6         | feedback / offline refinement 可以超越 heuristic teacher                 | 模型不只是模仿 scripted expert，而能根據 verifier/cost 改善                        |
| H7         | scaffold fading 後 performance 不大幅崩潰                                  | 最終方法不是 rule-based patch，而是 learned puzzle policy                     |

---

# 1. Dataset 與 evaluation setting

## 1.1 建議使用 FloorSet-Lite 作主測試平台

FloorSet-Lite 很適合這個研究計劃，因為它有：

* 1M training samples；
* block 數量從 21 到 120；
* optimal-by-construction layouts；
* inter-module connectivity；
* external terminal connectivity；
* soft constraints：grouping、MIB、boundary、preplaced、fixed-shape；
* hard constraints：area / dimension validity、overlap-free；
* validation set 與 evaluator；
* final objective 同時考慮 HPWL gap、bounding-box area gap、soft-constraint violation、runtime；且 infeasible solution 會得到固定 penalty。

這個 setting 很符合你的研究題目，因為它不是單一 geometry regression，而是有 graph、constraint、soft/hard feasibility、multi-objective cost 的結構化決策問題。

---

## 1.2 建議資料切分

不要只做 random split。要同時做 **IID generalization** 和 **size generalization**。

| Split                     | 用途                                   | 建議                                                                            |
| ------------------------- | ------------------------------------ | ----------------------------------------------------------------------------- |
| Train-small               | 快速 prototype                         | block 數 21–40                                                                 |
| Train-mid                 | 中等規模訓練                               | block 數 41–80                                                                 |
| Train-large               | scalability training                 | block 數 81–120                                                                |
| IID validation            | 檢查一般化                                | 每個 size 隨機 hold out                                                           |
| Size-OOD validation       | 檢查 extrapolation                     | train 21–80，test 81–120                                                       |
| Constraint-OOD validation | 檢查 instruction/constraint robustness | hold out 特定 constraint combination                                            |
| Stress set                | 檢查 hard failure                      | 高 grouping density、高 MIB density、高 external connectivity、boundary-heavy cases |

---

# 2. 整體測試架構

我建議把實驗設計成 7 個層級。

```text
E0. Pseudo-trajectory sanity check
E1. Role / constraint grounding learnability
E2. Single-step action imitation
E3. Partial rollout imitation
E4. Full constructive rollout
E5. Feedback / offline RL improvement
E6. Scaffold fading and ablation
E7. Counterfactual instruction tests
```

每一層都回答一個更具體的問題。

---

# 3. E0：Pseudo-trajectory sanity check

## 3.1 目的

先驗證：你從 final placement 反推出來的 pseudo-trajectories 是否合理。
如果 pseudo-trajectory 本身品質很差，後面 behavior cloning 學得再好也沒有意義。

---

## 3.2 輸入與輸出

| 項目                | 內容                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Input             | FloorSet instance + optimal final layout                                                                                              |
| Output            | 多條 pseudo action trajectories                                                                                                         |
| Action primitives | `select_block`, `snap_to_boundary`, `equalize_shape`, `attach`, `pull_to_pin`, `align_cluster`, `freeze`, `repair_overlap`, `compact` |
| Grounding         | 每個 action 必須有 primitive、arg1 block、arg2 target、relation、params、pre/post-condition                                                     |

---

## 3.3 Pseudo-trajectory 生成方法

建議不要只產生一條 trajectory。每個 floorplan 至少產生 3–8 條不同 explanation path。

### A. Anchor-first trace

```text
preplaced/fixed → boundary blocks → high-degree hubs → group members → followers
```

### B. Group-first trace

```text
MIB equalize → grouping attach → align cluster → freeze cluster → place remaining blocks
```

### C. Wirelength-first trace

```text
high external connectivity blocks → high interconnect hubs → chain links → local followers
```

### D. Reverse-peeling trace

從 final layout 反向移除 leaf-like blocks，再把 reverse order 當 constructive order。

### E. Constraint-priority trace

```text
hard shape constraints → boundary/preplaced/fixed → MIB → grouping → HPWL/local compaction
```

---

## 3.4 Sanity metrics

| Metric                         | 意義                                                 | 期望                    |
| ------------------------------ | -------------------------------------------------- | --------------------- |
| Reconstruction success         | 用 pseudo actions 能不能重建 final layout 或近似 layout     | 越高越好                  |
| Hard feasibility rate          | pseudo rollout 是否 overlap-free、area-valid          | 應接近 100%              |
| Action coverage                | final layout 中有多少 block 能被某個 action explanation 覆蓋 | >95% 作為初步目標           |
| Primitive distribution entropy | 是否所有 action 都被某一類主宰                                | 不應極端偏斜                |
| Multi-trace diversity          | 不同 pseudo traces 是否真的不同                            | 越高越能避免 teacher bias   |
| Constraint explanation rate    | boundary/MIB/grouping blocks 是否被對應 action 解釋       | 高才代表 action schema 合理 |
| Candidate coverage             | expert action 是否出現在 candidate generator 裡          | 初期應 >98%              |

---

## 3.5 通過標準

初步可以用這些門檻：

| Gate | Pass condition                                          |
| ---- | ------------------------------------------------------- |
| G0.1 | >95% samples 可產生至少一條 hard-feasible trajectory           |
| G0.2 | >90% constrained blocks 有對應 semantic action explanation |
| G0.3 | expert action candidate coverage >98%                   |
| G0.4 | trajectory replay 後 final HPWL/bbox gap 不明顯惡化           |

如果 E0 過不了，不要急著訓練 neural policy。先修 action schema 或 pseudo-trajectory generator。

---

# 4. E1：Role / constraint grounding learnability

## 4.1 目的

驗證模型是否能從 graph、connectivity、constraint metadata 中學到 block role。

你希望模型理解：

* 哪些 block 應該先處理；
* 哪些 block 是 hub；
* 哪些 block 應該靠 boundary；
* 哪些 block 是 MIB / shape-locked；
* 哪些 block 是 group follower；
* 哪些 block 是 chain-link。

---

## 4.2 Labels

一開始可用 rule-based weak labels。

| Role              | Weak label source                                  |
| ----------------- | -------------------------------------------------- |
| `anchor`          | preplaced、fixed、large area、high graph degree       |
| `hub`             | high weighted interconnect degree                  |
| `boundary-seeker` | boundary constraint 或 strong external connectivity |
| `shape-locked`    | fixed-shape 或 MIB group member                     |
| `group-leader`    | grouping group 中 degree / area / centrality 最大者    |
| `follower`        | grouping member、low degree、near group leader       |
| `chain-link`      | high betweenness 或連接兩個 dense clusters              |

---

## 4.3 Model

```text
Heterogeneous graph encoder
  block nodes
  terminal nodes
  net edges
  constraint nodes
  role query tokens

Heads
  role multi-label classifier
  constraint membership predictor
  placement priority scorer
  local relation predictor
```

---

## 4.4 Metrics

| Metric                         | 意義                                                            |
| ------------------------------ | ------------------------------------------------------------- |
| Role F1 / AUROC                | role 是否可從資料中預測                                                |
| Constraint membership accuracy | block 是否屬於 boundary/MIB/grouping/fixed/preplaced              |
| Priority correlation           | predicted priority 是否接近 pseudo trajectory order               |
| Cross-size generalization      | 在 larger block count 上 role prediction 是否穩定                   |
| Ablation gain                  | 有 role embedding vs 無 role embedding 對後續 action imitation 的增益 |

---

## 4.5 判斷標準

如果 role prediction 做不好，不代表方向錯，但代表：

1. role definition 太 noisy；
2. graph encoder 太弱；
3. role 不應作 hard label，而應作 latent variable；
4. 或 pseudo expert 的 role 設計不合理。

我建議初期不要要求 role label 絕對正確，而是看：

> role embedding 是否能改善 action prediction 和 rollout quality。

---

# 5. E2：Single-step action imitation

## 5.1 目的

驗證模型在給定正確 state history 的情況下，是否能預測下一步 semantic action。

這一步回答：

> action grammar 是否可學？
> primitive / target / relation / params 是否可由盤面 state 推出？

---

## 5.2 Action factorization

每個 action 拆成：

```text
primitive
arg1_type
arg1_id
arg2_type
arg2_id
relation_enum
param_bin / continuous_param
freeze_flag
```

例如：

```text
attach(
  arg1_type = block,
  arg1_id   = b17,
  arg2_type = block,
  arg2_id   = b03,
  side      = RIGHT,
  align     = CENTER
)
```

---

## 5.3 Training objective

[
\mathcal{L}_{BC}
================

CE(o)
+
CE(arg_1)
+
CE(arg_2)
+
CE(relation)
+
NLL(params)
]

加上 auxiliary losses：

[
\mathcal{L}_{aux}
=================

\lambda_{valid} CE(validity)
+
\lambda_{role} CE(role)
+
\lambda_{\Delta} |\widehat{\Delta cost} - \Delta cost|
]

總 loss：

[
\mathcal{L}
===========

\mathcal{L}*{BC}
+
\mathcal{L}*{aux}
]

---

## 5.4 Baselines

| Baseline                          | 說明                                         |
| --------------------------------- | ------------------------------------------ |
| Random legal action               | 從 legal candidates 中隨機選                    |
| Heuristic priority                | 手寫 anchor-first / group-first / HPWL-first |
| Plain GNN + flat action head      | 不使用 typed decoder                          |
| GNN + typed decoder, no role      | 有 action factorization，但無 role             |
| GNN + role + typed decoder        | 你的主要模型                                     |
| GNN + role + typed decoder + mask | 主要模型加 legality mask                        |

---

## 5.5 Metrics

| Metric                       | 看什麼                                           |
| ---------------------------- | --------------------------------------------- |
| Primitive accuracy           | 是否選對 action type                              |
| Arg1 top-1 / top-5 accuracy  | 是否選對 block                                    |
| Arg2 top-1 / top-5 accuracy  | 是否選對 target block/terminal/group              |
| Relation accuracy            | side/alignment/corner 是否正確                    |
| Param error                  | offset、aspect bin、shape params 是否接近           |
| Legal top-1 rate             | top-1 action 是否 legal                         |
| Masked CE                    | 在 legal candidate set 裡的 imitation loss       |
| Candidate ranking AUC        | expert action 是否比 negative actions 分數高        |
| Constraint-specific accuracy | boundary/MIB/grouping cases 的 action accuracy |

---

## 5.6 通過標準

初步目標可以設為：

| Metric                | Small 21–40 | Mid 41–80 | Large 81–120 |
| --------------------- | ----------: | --------: | -----------: |
| Primitive accuracy    |        >85% |      >80% |         >75% |
| Arg1 top-5            |        >95% |      >90% |         >85% |
| Arg2 top-5            |        >90% |      >85% |         >80% |
| Legal top-1           |        >99% |      >98% |         >97% |
| Candidate ranking AUC |        >0.9 |     >0.85 |         >0.8 |

這些不是絕對標準，而是 initial research gate。真正重點是相對於 baselines 是否有明顯提升。

---

# 6. E3：Partial rollout imitation

## 6.1 目的

Teacher-forcing accuracy 高，不代表 free rollout 會好。
E3 要測試模型在部分自動 rollout 下是否會累積錯誤。

---

## 6.2 Rollout modes

| Mode                            | 說明                                | 用途                          |
| ------------------------------- | --------------------------------- | --------------------------- |
| Teacher forcing                 | 每一步 state 都由 expert trajectory 提供 | 測 action prediction         |
| Scheduled sampling              | 一部分使用模型 action，一部分回到 expert       | 測 transition robustness     |
| k-step free rollout             | 從 expert state 開始自由走 k 步          | 測短期 drift                   |
| Constraint-triggered correction | 走錯時由 verifier / expert 修正         | DAgger-like data collection |
| Full free rollout               | 完全由 policy placement              | 真正 end-to-end 測試            |

---

## 6.3 k-step 測試設計

對每個 validation sample，從不同 progress point 開始：

```text
25% placed
50% placed
75% placed
90% placed
```

讓 policy 自由走：

```text
k = 1, 3, 5, 10, until complete
```

---

## 6.4 Metrics

| Metric                           | 意義                              |
| -------------------------------- | ------------------------------- |
| k-step survival rate             | k 步內沒有 hard violation           |
| constraint progress monotonicity | soft violation 是否持續下降           |
| recovery rate                    | 遇到 bad state 是否能修正              |
| divergence from expert           | action trace 與 expert 距離        |
| final partial cost delta         | partial rollout 後 cost proxy 變化 |
| repair dependency                | 有多少錯誤需要 legalizer 補救            |

---

## 6.5 判斷標準

重點不是模型每一步都跟 expert 一樣，而是：

> free rollout 後 state 是否仍然可解、合法、constraint progress 沒有崩潰。

如果模型 action 與 expert 不同，但 final cost 更好，這是正面結果。

---

# 7. E4：Full constructive rollout

## 7.1 目的

這是第一個完整驗證：

> 模型是否能從空盤面逐步完成 floorplan？

---

## 7.2 Inference variants

| Variant                  | 說明                           | 目的                          |
| ------------------------ | ---------------------------- | --------------------------- |
| Greedy top-1             | 每步選最高分 legal action          | 測純 policy 能力                |
| Top-k beam               | 保留 k 條 partial floorplans    | 測 search-guided policy      |
| Policy + verifier prune  | verifier 剪掉 bad states       | 測 constraint-aware decoding |
| Policy + local repair    | action 後做最小 repair           | 測實用落地                       |
| Policy + fallback expert | policy 失敗時交給 scripted expert | 測混合系統上限                     |

---

## 7.3 Final metrics

用 contest evaluator 的概念拆成幾類。

| Metric                 | 類型         | 說明                                              |
| ---------------------- | ---------- | ----------------------------------------------- |
| Feasibility rate       | hard       | overlap-free + area/dimension valid             |
| Hard failure type      | hard       | overlap、area、fixed-shape、preplaced mismatch     |
| Violationsrelative     | soft       | grouping/MIB/boundary/preplaced/fixed violation |
| HPWLgap                | quality    | wirelength gap                                  |
| Areagap_bbox           | quality    | bbox area gap                                   |
| Runtime                | deployment | rollout + verifier + repair time                |
| Total cost             | final      | multi-objective final score                     |
| Action count           | behavior   | 是否用過多步驟                                         |
| Repair count           | behavior   | 是否主要靠 repair                                    |
| Trace interpretability | behavior   | action 是否對應 constraint/role                     |

FloorSet scoring 本身就把 infeasible、HPWL gap、bbox gap、soft violation、runtime 放在同一個 objective 中，因此 final evaluation 可以直接對齊該 evaluator；但研究驗證時仍應拆開看各項指標。

---

## 7.4 主要比較對象

| 方法                             | 用途                         |
| ------------------------------ | -------------------------- |
| Coordinate regression baseline | 證明 direct regression 不足    |
| Scripted heuristic expert      | 測模型是否只是模仿規則                |
| SA / local search baseline     | 測 quality/runtime tradeoff |
| Typed BC without role          | 測 role 的增益                 |
| Typed BC with role             | 主模型 baseline               |
| Typed BC + verifier beam       | 測 constrained decoding     |
| Typed BC + feedback refinement | 測 feedback 是否有效            |
| Typed BC + offline RL          | 測能否超越 teacher              |

---

# 8. E5：Feedback / offline RL improvement

## 8.1 目的

驗證模型是否能從「會模仿」進一步變成「會修正與改善」。

---

## 8.2 Dataset for improvement

建立 replay buffer：

```text
D = {
  expert pseudo trajectories,
  BC policy rollouts,
  beam search rollouts,
  repaired rollouts,
  failed rollouts with verifier labels,
  high-quality self-generated rollouts
}
```

每條 trajectory 保存：

```text
state_t
action_t
legal_mask_t
reward_t
violation_vector_t
delta_cost_t
next_state_t
terminal_cost
```

---

## 8.3 Reward design

建議 reward 分成 hard gate 與 soft shaping。

### Hard gate

```text
if hard violation:
    terminal penalty
    stop or repair
```

### Dense reward

| Reward item       | 訊號                                         |
| ----------------- | ------------------------------------------ |
| overlap margin    | 避免接近 infeasible                            |
| grouping progress | connected components 減少                    |
| MIB progress      | distinct shapes 減少                         |
| boundary progress | required edge/corner distance 減少           |
| HPWL proxy        | weighted center distance / partial HPWL 改善 |
| bbox compactness  | bounding box growth controlled             |
| freeze stability  | 已滿足 cluster 不被破壞                           |
| step penalty      | 避免 unnecessary actions                     |

---

## 8.4 Offline improvement methods

初期不要直接 PPO。建議順序：

| 方法                    | 建議程度 | 原因                                        |
| --------------------- | ---: | ----------------------------------------- |
| Advantage-weighted BC |    高 | 最穩，容易接在 BC 後面                             |
| IQL-style improvement |    高 | 比較保守，適合 offline data                      |
| CQL-style critic      |   中高 | 避免 OOD action overestimation              |
| Decision Transformer  |   中高 | 若 trajectory data 很多，適合 sequence modeling |
| PPO fine-tune         |    中 | 只在 legal rollout rate 很高後使用               |
| SAC                   |   中低 | typed relational action space 要改造很多       |

---

## 8.5 Improvement metrics

| Metric                   | 判斷                                           |
| ------------------------ | -------------------------------------------- |
| Cost improvement over BC | offline RL 是否有效                              |
| Feasibility retention    | improvement 不應犧牲 hard legality               |
| Soft violation reduction | feedback 是否學到 constraint                     |
| HPWL/bbox tradeoff       | 是否只改善一項、犧牲另一項                                |
| Teacher surpass rate     | 有多少 cases 超過 scripted expert                 |
| OOD action rate          | policy 是否開始選未見且危險的 action                    |
| Critic calibration       | predicted value 是否與 final cost correlation 高 |

---

## 8.6 成功標準

一個合理初步目標：

| Gate | Pass condition                           |
| ---- | ---------------------------------------- |
| G5.1 | offline improvement 不降低 feasibility rate |
| G5.2 | final cost 相對 BC 有穩定下降                   |
| G5.3 | soft violation relative score 下降         |
| G5.4 | 在至少部分 size buckets 超越 scripted teacher   |
| G5.5 | improvement 不是來自更多 repair 或更長 runtime    |

---

# 9. E6：Ablation 與 scaffold fading

## 9.1 目的

證明模型不是 rule-based patch。

---

## 9.2 Ablation matrix

| Variant                   | Role | Mask | Verifier |   Repair | Feedback | Offline RL | 用途                                |
| ------------------------- | ---: | ---: | -------: | -------: | -------: | ---------: | --------------------------------- |
| V0 Coordinate regression  |    ✗ |    ✗ | optional |        ✓ |        ✗ |          ✗ | direct regression baseline        |
| V1 Scripted expert        | rule | rule |        ✓ |        ✓ |        ✗ |          ✗ | heuristic baseline                |
| V2 Typed BC only          |    ✗ |    ✗ |        ✗ |        ✗ |        ✗ |          ✗ | action imitation lower bound      |
| V3 Typed BC + mask        |    ✗ |    ✓ |        ✗ |        ✗ |        ✗ |          ✗ | mask contribution                 |
| V4 Role + typed BC + mask |    ✓ |    ✓ |        ✗ |        ✗ |        ✗ |          ✗ | role contribution                 |
| V5 V4 + verifier beam     |    ✓ |    ✓ |        ✓ |        ✗ |        ✗ |          ✗ | constrained decoding contribution |
| V6 V5 + repair            |    ✓ |    ✓ |        ✓ |        ✓ |        ✗ |          ✗ | practical hybrid upper bound      |
| V7 V5 + feedback critic   |    ✓ |    ✓ |        ✓ | optional |        ✓ |          ✗ | feedback contribution             |
| V8 V7 + offline RL        |    ✓ |    ✓ |        ✓ | optional |        ✓ |          ✓ | final policy                      |

---

## 9.3 Scaffold fading tests

逐步移除手工規則：

| Fade target                       | 測試                               |
| --------------------------------- | -------------------------------- |
| hand-written block order          | 改由 learned selector              |
| hand-written role priority        | 改由 role-conditioned policy       |
| deterministic attach side         | 改由 action scorer                 |
| deterministic MIB reference shape | 改由 learned shape scorer          |
| local repair                      | 降低 repair 使用率，看 feasibility 是否保持 |
| verifier beam width               | 減少 beam，看 greedy policy 是否仍穩     |

---

## 9.4 重要指標

| Metric                             | 解讀                                      |
| ---------------------------------- | --------------------------------------- |
| Performance retention after fading | 淡出 scaffold 後 performance 掉多少           |
| Repair dependency ratio            | final solution 有多少來自 repair             |
| Rule override rate                 | learned policy 有多少次覆蓋 heuristic choice  |
| Learned improvement rate           | learned choice 比 heuristic choice 更好的比例 |
| Greedy-vs-beam gap                 | gap 小代表 policy 本身更強；gap 大代表主要靠 search   |

---

# 10. E7：Counterfactual instruction / constraint tests

## 10.1 目的

這是驗證「instruction-aware」最關鍵的一組測試。

你要證明模型不是只 memorizing geometry，而是會根據 instruction/constraint semantics 改變 action。

---

## 10.2 Counterfactual tests

| Test                            | 做法                                             | 期望行為                                |
| ------------------------------- | ---------------------------------------------- | ----------------------------------- |
| Boundary flip                   | 把某 block 的 boundary constraint 從 LEFT 改成 RIGHT | `snap_to_boundary` target 應跟著變      |
| Remove grouping                 | 拿掉某 group constraint                           | `attach` / `align_cluster` 頻率應下降    |
| Add grouping                    | 加入新的 group                                     | policy 應更傾向 group attach            |
| MIB perturb                     | 改 MIB membership                               | `equalize_shape` target 應改變         |
| External terminal weight change | 增加某 terminal edge weight                       | `pull_to_pin` 應更偏該 terminal         |
| Preplaced injection             | 加入 preplaced anchor                            | placement order 和 attach target 應改變 |
| Connectivity rewiring           | 改 interconnect graph                           | hub/chain-link selection 應改變        |
| Instruction masking             | 隱藏 constraint nodes                            | performance 應下降，證明模型有用到 instruction |

---

## 10.3 Metrics

| Metric                      | 意義                                                         |
| --------------------------- | ---------------------------------------------------------- |
| Action sensitivity          | constraint 改變後 action distribution 是否合理改變                  |
| Constraint compliance delta | 改 instruction 後是否滿足新 constraint                            |
| Counterfactual consistency  | 不相關 constraints 改變時 action 不應亂變                            |
| Role shift accuracy         | role 是否跟著 constraint 改變                                    |
| Causal attribution          | attention / gradient / intervention 是否指向相關 constraint node |

---

# 11. 最小可行實驗版本

如果要快速驗證方向是否可行，我建議先做一個 **MVP experiment**，不要一開始就做完整 offline RL。

## MVP scope

| 項目          | 建議                                                                       |
| ----------- | ------------------------------------------------------------------------ |
| Dataset     | 先用 21–40 blocks                                                          |
| Samples     | 10k–50k training samples                                                 |
| Constraints | boundary + grouping + MIB 先做，preplaced/fixed 再加入                         |
| Actions     | `select_block`, `attach`, `snap_to_boundary`, `equalize_shape`, `freeze` |
| Model       | hetero-GNN / graph transformer + typed pointer decoder                   |
| Training    | pseudo-trajectory BC                                                     |
| Inference   | greedy + legality mask；再加 small beam                                     |
| Evaluation  | step accuracy、legal rollout、soft violation、final cost                    |

---

## MVP success criteria

| Question                   | Pass signal                                                 |
| -------------------------- | ----------------------------------------------------------- |
| Action schema 是否合理？        | candidate coverage >95%，pseudo replay feasible              |
| 模型是否學到 primitive？          | primitive accuracy 明顯高於 heuristic/random                    |
| 模型是否學到 target selection？   | arg1/arg2 top-5 accuracy 高                                  |
| rollout 是否穩？               | small cases feasibility rate 高                              |
| role 是否有幫助？                | +role variant 比 no-role cost/violation 更好                   |
| mask 是否必要？                 | no-mask invalid action rate 明顯高                             |
| 是否有 instruction awareness？ | counterfactual boundary/grouping/MIB tests 有正確 action shift |

---

# 12. 建議實驗報告格式

每次實驗不要只報 final score。建議固定報下面這張表。

## 12.1 Behavior-level report

| Model                  | Primitive acc | Arg1 top-5 | Arg2 top-5 | Legal top-1 | k-step survival | Full feasible | Repair ratio |
| ---------------------- | ------------: | ---------: | ---------: | ----------: | --------------: | ------------: | -----------: |
| Random legal           |               |            |            |             |                 |               |              |
| Heuristic              |               |            |            |             |                 |               |              |
| Typed BC               |               |            |            |             |                 |               |              |
| Typed BC + role        |               |            |            |             |                 |               |              |
| Typed BC + role + mask |               |            |            |             |                 |               |              |
| + verifier beam        |               |            |            |             |                 |               |              |
| + feedback             |               |            |            |             |                 |               |              |

## 12.2 Final floorplanning report

| Model                  | Feasible rate | HPWLgap | Areagap_bbox | Violationsrelative | Runtime | Cost |
| ---------------------- | ------------: | ------: | -----------: | -----------------: | ------: | ---: |
| Coordinate regression  |               |         |              |                    |         |      |
| Scripted heuristic     |               |         |              |                    |         |      |
| Typed BC               |               |         |              |                    |         |      |
| Typed BC + role        |               |         |              |                    |         |      |
| Typed BC + role + mask |               |         |              |                    |         |      |
| + verifier beam        |               |         |              |                    |         |      |
| + offline RL           |               |         |              |                    |         |      |

## 12.3 Constraint-specific report

| Model      | Boundary violation | Grouping violation | MIB violation | Fixed/preplaced violation | Overlap failure | Area failure |
| ---------- | -----------------: | -----------------: | ------------: | ------------------------: | --------------: | -----------: |
| Typed BC   |                    |                    |               |                           |                 |              |
| + role     |                    |                    |               |                           |                 |              |
| + mask     |                    |                    |               |                           |                 |              |
| + feedback |                    |                    |               |                           |                 |              |

這張表很重要，因為它可以回答：

> 模型到底學到哪一種 constraint？
> 哪一種 constraint 還只是靠 rule 或 repair？

---

# 13. 具體實驗里程碑

## Milestone 1：Action schema and pseudo-trajectory validation

| 目標                             | 輸出                                      |
| ------------------------------ | --------------------------------------- |
| 定義 action API                  | action grammar document                 |
| 建立 pseudo-trajectory generator | 每個 sample 多條 trace                      |
| replay trajectory              | 檢查 feasibility / reconstruction         |
| negative candidate generator   | legal-poor、illegal、wrong-target samples |

**通過條件：**
pseudo trace replay 大多可行，expert action candidate coverage 很高。

---

## Milestone 2：Single-step policy learnability

| 目標                     | 輸出                               |
| ---------------------- | -------------------------------- |
| 訓練 typed action BC     | primitive / arg / relation heads |
| 加入 role auxiliary task | role prediction                  |
| 加入 legality mask       | masked action CE                 |
| 對比 no-role / no-mask   | ablation                         |

**通過條件：**
action accuracy 明顯高於 random legal / heuristic baseline，且 role 有增益。

---

## Milestone 3：Partial rollout robustness

| 目標                             | 輸出                     |
| ------------------------------ | ---------------------- |
| k-step rollout evaluation      | drift/survival metrics |
| scheduled sampling             | 降低 exposure bias       |
| DAgger-style correction buffer | 收集 bad states          |
| verifier-based relabeling      | 修正 off-policy states   |

**通過條件：**
k-step survival 隨訓練改善，full rollout 開始有 non-trivial feasible rate。

---

## Milestone 4：Full floorplan generation

| 目標                      | 輸出                                        |
| ----------------------- | ----------------------------------------- |
| greedy rollout          | policy-only baseline                      |
| beam rollout            | constrained decoding                      |
| local repair optional   | practical upper bound                     |
| final evaluator scoring | cost / HPWL / bbox / violations / runtime |

**通過條件：**
在 small/mid cases 上可產生大量 feasible solutions，且 soft violation / cost 優於簡單 heuristic。

---

## Milestone 5：Feedback refinement

| 目標                          | 輸出                            |
| --------------------------- | ----------------------------- |
| dense reward design         | violation vector + cost delta |
| value / critic model        | 預測 rollout quality            |
| advantage-weighted BC 或 IQL | policy improvement            |
| compare against BC          | feedback value                |

**通過條件：**
feedback refinement 在不降低 feasibility 的情況下降低 cost 或 soft violation。

---

## Milestone 6：Scaffold fading

| 目標                       | 輸出                           |
| ------------------------ | ---------------------------- |
| 降低 hand-written priority | learned selector takeover    |
| 減少 repair                | policy avoids illegal states |
| 減少 beam width            | policy 本身更強                  |
| report retention         | 證明非 rule patch               |

**通過條件：**
移除大部分 heuristic ranking 後，performance 沒有崩潰；hard legality contract 可保留。

---

# 14. 可能失敗模式與 fallback

| 失敗模式                                       | 可能原因                        | Fallback                                                 |
| ------------------------------------------ | --------------------------- | -------------------------------------------------------- |
| pseudo trajectories replay 不回 final layout | action schema 太粗            | 增加 `place_relative`, `resize_to_area`, `compact_cluster` |
| primitive accuracy 高但 rollout 差            | exposure bias               | scheduled sampling + DAgger                              |
| top-1 target 常錯                            | candidate set 太大            | two-stage selector：block first，再 target                  |
| legal action rate 低                        | mask/contract 不完整           | 強化 precondition checker                                  |
| feasible rate 高但 cost 差                    | policy 只學合法，沒學 quality      | 加 dense reward / critic / beam ranking                   |
| feedback 後變差                               | reward hacking 或 OOD action | IQL/CQL + KL-to-BC + hard verifier                       |
| role 沒幫助                                   | role labels noisy           | 改成 latent role + auxiliary，不作 hard supervision           |
| repair ratio 太高                            | policy 依賴 post-hoc repair   | repair penalty + scaffold fading                         |
| large case 崩潰                              | size generalization 不足      | curriculum 21→40→80→120，graph sparsification             |
| grouping/MIB 學不好                           | action schema 不足            | group-level option / cluster-level decoder               |

---

# 15. 最關鍵的「能不能學到」判斷

我建議你不要只用 final cost 來判斷是否成功，而是用下面五個問題。

## Q1：模型是否學到 action grammar？

看：

* primitive accuracy；
* typed argument accuracy；
* legal top-1 rate；
* invalid action rate；
* expert candidate coverage。

如果這些不過，代表 action API 或 pseudo labels 有問題。

---

## Q2：模型是否學到 constraint behavior？

看：

* boundary counterfactual；
* grouping counterfactual；
* MIB counterfactual；
* constraint-specific violation reduction；
* no-mask vs mask vs learned-mask ablation。

如果只有 mask 有效，而 no-mask model 完全不知道 constraint，代表它主要靠 rule contract，還沒學到 constraint preference。

---

## Q3：模型是否學到 sequential planning？

看：

* k-step survival；
* full rollout feasible rate；
* greedy-vs-beam gap；
* recovery from bad state；
* repair dependency ratio。

如果 teacher forcing 很好但 rollout 很差，代表只是 static imitation，不是 decision policy。

---

## Q4：模型是否能超越 heuristic teacher？

看：

* offline RL / feedback refinement 是否降低 final cost；
* learned action 是否在 beam/evaluator ranking 中打敗 scripted action；
* teacher surpass rate；
* high-quality self-generated rollouts 是否增加。

如果永遠不能超越 teacher，這仍然是 useful imitation system，但不是 strong learned puzzle policy。

---

## Q5：模型是否仍依賴 rule-based patch？

看：

* scaffold fading；
* repair ratio；
* rule override rate；
* removing heuristic priority 的 performance drop；
* reducing beam width 的 performance drop。

如果拿掉 heuristic ranking 後崩潰，代表主要貢獻還在 rule system。
如果只保留 hard legality contract 後仍能維持大部分 performance，才比較接近真正的 learned puzzle policy。

---

# 16. 我會建議的第一版實驗配置

## Model v1：最小可行 learned puzzle policy

```text
Encoder:
  Heterogeneous graph encoder
    block nodes
    terminal nodes
    net edges
    constraint nodes
    role embeddings

Decoder:
  primitive head
  block pointer head
  target pointer head
  relation head
  param head

Safety:
  typed action mask
  geometry precondition checker
  simple projection for coordinates/shape

Training:
  pseudo-trajectory behavior cloning
  role auxiliary loss
  validity auxiliary loss
  candidate ranking loss

Inference:
  greedy legal rollout
  optional beam width = 4 or 8
```

---

## v1 必做 ablations

| Ablation                          | 目的                        |
| --------------------------------- | ------------------------- |
| no role                           | 測 role 是否有用               |
| no instruction / constraint nodes | 測 instruction awareness   |
| no mask                           | 測 hard legality layer 重要性 |
| no candidate ranking loss         | 測 preference learning     |
| no multi-trajectory               | 測 teacher bias            |
| greedy vs beam                    | 測 policy 本身強度             |
| with repair vs without repair     | 測是否靠 legalizer            |

---

# 17. 最推薦的初步結論判準

如果第一版實驗得到以下結果，我會認為這條研究路線值得繼續：

1. pseudo-trajectory replay feasible rate 高；
2. typed action imitation 明顯優於 random legal / heuristic-only baseline；
3. role-conditioned model 優於 no-role model；
4. mask 顯著降低 invalid actions，但 model 在 no-mask 或 relaxed-mask setting 下仍有部分 legality preference；
5. k-step rollout survival 隨訓練進步；
6. full rollout 在 small/mid cases 有 non-trivial feasible rate；
7. verifier beam 或 feedback refinement 能降低 soft violation / final cost；
8. counterfactual instruction tests 中，action distribution 會隨 constraint 改變；
9. scaffold fading 後 performance 沒有完全崩潰。

如果只達到 1–4，表示模型學到 **action imitation**。
如果達到 1–7，表示模型開始學到 **sequential constrained placement behavior**。
如果達到 1–9，才比較能主張它接近 **instruction-aware learned puzzle policy**。

---

# Final recommendation

第一版研究測試計劃不要直接衝 offline RL 或 full contest score。最穩的驗證路徑是：

> **先證明 pseudo action 是合理的 → 再證明 typed policy 能模仿 → 再證明 free rollout 不崩 → 再用 feedback 改善 → 最後用 scaffold fading 證明不是 rule patch。**

近期最值得做的 MVP 是：

```text
FloorSet small cases 21–40
→ derive multi-explanation pseudo-trajectories
→ train role-conditioned typed BC policy
→ evaluate single-step + k-step + full rollout
→ compare no-role / no-mask / no-feedback / heuristic baselines
→ run counterfactual constraint tests
```

這個 MVP 可以最快回答你最關心的問題：

> 這個模型到底有沒有學到「像解拼圖一樣逐步 floorplan」的行為？
> 還是它只是學了一個帶 legalizer 的座標產生器？
