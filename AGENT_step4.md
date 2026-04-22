這份結果已經把下一步壓得很清楚：**現在不是 feasibility 問題，而是 post-finalizer quality objective alignment 問題**。

我會把目前 branch 的結論更新成：

> trained checkpoint 學到的不是「更好的 floorplan」，而是「更容易被 current repair/finalizer 保留的 proposal」。但 current final objective 真正在乎的是 HPWL、bbox area、soft violations 與 runtime；在 validation-0..19 slice 上，trained 的低 displacement 不但沒有幫助，甚至是很差的 proxy。

你的 agent 報告裡已經確認幾個關鍵事實：best untrained mean cost 是 `19.784`，best trained mean cost 是 `23.651`；trained 只贏 `3/20` cases；gap 主要由 HPWL + bbox area 解釋，約 `77.2%`，soft violations 約 `21.9%`，runtime 只有 `0.9%`；而 repair displacement 與 official cost / HPWL / bbox 的相關性在這批資料上是負的，所以 displacement 幾乎可以判定不是好 proxy。

---

# 1. 我對目前結果的更新判讀

## 1.1 feasibility 已經不是主瓶頸

現在 validation-0..19 上所有 variants 都 `20/20 feasible`，代表 hard constraints 路徑已經不是區分 trained / untrained 的因素。這也跟 FloorSet objective 的設計一致：hard infeasible 只是直接 short-circuit 到 fixed penalty，但 feasible 之後仍然會被 HPWL gap、bbox area gap、soft violations 與 runtime 乘上去。

所以後續不要再把主要 effort 放在：

```text
more overlap resolver tuning
lower repair displacement
more feasibility-only training
larger BC/AWBC checkpoint
```

這些目前都不直接對準 loss driver。

---

## 1.2 lower displacement 反而可能是壞訊號

這是最重要的反直覺結果。

你現在看到：

```text
trained seed-0 displacement 更小
但 HPWL gap / area gap 更差
```

這意味著 current finalizer 可能在做：

```text
保留 trained proposal 的相對結構
```

但 trained proposal 的相對結構本身不利於：

```text
wirelength
compactness
soft constraints
```

因此 displacement 小不是好事，可能代表 finalizer **沒有足夠改正 trained policy 的壞 topology**。

我會把目前假設寫成：

> trained policy 產生了比較穩定、比較一致、比較容易 repair 的 layout proposal；但這個 proposal 的 block relative ordering / cluster structure / bbox growth direction / terminal alignment 比 untrained 更差。finalizer 因為 displacement-preserving，所以把這些壞結構保留下來，導致 official cost 變高。

---

## 1.3 目前最需要拆解的是 proposal drift vs repair drift

下一步要回答的不是「trained 為什麼 cost 高」這個泛問題，而是下面這個精確問題：

```text
trained 的 HPWL / bbox 變差，是在 semantic proposal 階段就已經發生？
還是 repair / finalizer 把它修差？
```

這會決定下一步方向：

| 診斷結果                                      | 下一步                                      |
| ----------------------------------------- | ---------------------------------------- |
| trained proposal pre-repair 就 HPWL/bbox 差 | 改 training objective / action scoring    |
| trained proposal 還可以，但 repair 後變差         | 改 finalizer / repair objective           |
| untrained proposal 很差，但 repair 後好         | finalizer 主導解，policy contribution 低      |
| oracle top-k 能找到好解                        | 做 cost-aware reranker / scorer           |
| oracle top-k 也不行                          | candidate/finalizer search space ceiling |

---

# 2. 我現在最想確認的訊息

請下一個 agent 優先回答這些問題。

## A. top-5 loss cases 的 drift 發生在哪一段？

目前 top-5 trained losses 是：

```text
validation-14
validation-18
validation-17
validation-15
validation-11
```

這些 case 應該逐段拆：

```text
semantic proposal
→ shape normalization
→ overlap resolver
→ shelf fallback / packing
→ strict final
→ official evaluation
```

每段要量：

```text
HPWLgap
HPWLint
HPWLext
bbox area
Areagap_bbox
Violationsrelative
boundary / grouping / MIB
overlap pairs
total overlap area
moved block count
mean displacement
changed block fraction
```

---

## B. trained 是 wirelength topology 差，還是 bbox compactness 差？

報告說 HPWL + area 解釋 77.2% gap，但下一步要拆得更細：

```text
HPWLint vs HPWLext
bbox width vs bbox height vs bbox area
terminal-connected blocks 的距離
high-degree blocks 的相對位置
cluster / group 是否過度集中或過度分散
```

如果 HPWLext 差，代表 terminal-aware placement / `pull_to_pin` / boundary-seeking objective 不夠。
如果 HPWLint 差，代表 graph cluster / hub / chain-link placement 錯。
如果 bbox area 差，代表 proposal/finalizer 的 compaction objective 錯。

---

## C. untrained 為什麼會贏？

這個結果很可能是下一篇研究故事的關鍵。

要檢查 untrained 是否：

```text
1. 更接近 randomized broad exploration
2. 產生更分散但更容易被 finalizer compact 的初始 layout
3. 讓 finalizer 有更大自由度重排
4. 沒有過度 preserve bad semantic relation
5. 在某些 cases 只是 seed lucky
```

所以要看：

```text
untrained seeds distribution
trained seeds distribution
per-case paired deltas
action histogram
placement order
bbox growth trajectory
finalizer movement trajectory
```

---

## D. current action + finalizer space 有沒有好解？

這是下一個實驗決策點。

如果 top-k rollouts 裡其實存在比 untrained 更好的 candidate，那代表：

```text
不是 action/finalizer 不行，而是 scorer / training objective 不會選。
```

如果 top-k oracle 也不行，代表：

```text
candidate family 或 finalizer geometry strategy 到 ceiling 了。
```

---

# 3. 建議下一個 agent prompt：Top-5 Loss Drift Audit

下面這份 prompt 我建議你直接丟給 agent。這是現在最值得做的下一份文件。

---

## Prompt：Top-5 Loss Drift Audit

```text
你是一位 EDA floorplanning + ML/RL 實驗分析 agent。請根據目前 repo code 與 artifacts，針對 trained checkpoint 輸給 untrained 的 top-loss cases 做 proposal → repair → final 的 drift audit。

請產生兩份輸出：

1. artifacts/research/top5_loss_drift_audit.md
2. artifacts/research/top5_loss_drift_audit.json

背景：

目前已有報告：
- artifacts/research/cost_semantics_and_trained_vs_untrained_delta.md
- artifacts/research/cost_semantics_and_trained_vs_untrained_delta.json

該報告確認：
- validation-0..19 上 best untrained 是 `untrained seed 0`，mean cost = 19.784。
- best trained 是 `awbc seed 1`，mean cost = 23.651。
- trained 只贏 3/20 cases。
- trained 輸的主因是 HPWL + bbox area，約解釋 77.2% mean gap。
- soft violations 約 21.9%，runtime 只有 0.9%。
- repair displacement 是 poor proxy，與 official cost / HPWL / bbox 沒有正向關係。
- top-5 trained loss cases 是：
  validation-14
  validation-18
  validation-17
  validation-15
  validation-11

你的任務是回答：

trained 的 HPWL / bbox area 變差，到底發生在：
A. semantic proposal 階段？
B. repair / finalizer 階段？
C. proposal-finalizer interaction？
D. official scoring / metric interpretation？

============================================================
Part 1: Reconstruct pipeline stages
============================================================

請針對以下 variants：

1. best untrained: untrained seed 0
2. best trained: awbc seed 1
3. trained seed 0, if available
4. any other trained variants available in generalization_followup_smallcheckpoints.json

針對以下 cases：

- validation-14
- validation-18
- validation-17
- validation-15
- validation-11

重建或讀取每個 variant 的以下階段資料：

Stage A: semantic proposal / provisional layout before repair
Stage B: after shape normalization
Stage C: after overlap resolver
Stage D: after shelf fallback or final packing inside finalizer
Stage E: strict final layout
Stage F: official evaluator result

如果某些 stage 目前沒有 artifact，請：
- 明確寫 missing stage
- 說明目前 code 是否可以 instrument
- 若可以，請新增 instrumentation 但避免大規模改動
- 儲存 stage-wise positions / metrics 到 artifacts/research/top5_loss_drift_audit.json

============================================================
Part 2: Stage-wise metrics
============================================================

每個 case / variant / stage 都要量以下 metrics：

Geometry:
- bbox_width
- bbox_height
- bbox_area
- Areagap_bbox if baseline available
- total block area
- whitespace ratio
- aspect ratio of bbox

Wirelength:
- HPWL total
- HPWLint
- HPWLext
- HPWLgap if baseline available
- terminal-distance weighted sum
- high external connectivity block distance to terminals
- high interconnect hub distance to connected blocks

Hard feasibility:
- hard_feasible
- overlap_pair_count
- total_overlap_area
- max_overlap_area
- area_violation_count
- fixed_dimension_violation_count
- preplaced_violation_count

Soft constraints:
- Violationsrelative
- boundary_violation_count
- grouping_violation_count
- MIB_violation_count
- group connected components
- distinct MIB shapes

Repair / finalizer:
- moved_block_count
- changed_block_fraction
- displacement_sum
- mean_displacement
- max_displacement
- displacement by role if roles available
- shelf_fallback_block_count
- locally_shifted_block_count
- anchor_moved_count

Intent preservation:
- attach_intents_before
- attach_intents_preserved
- attach_intents_destroyed
- boundary_intents_before
- boundary_intents_preserved
- boundary_intents_destroyed
- pin_pull_distance_change
- group_compactness_change

Runtime:
- use runtime-standardized evaluation with runtime_factor = 1.0 if possible
- also record original runtime_factor from artifact

============================================================
Part 3: Drift decomposition
============================================================

請對每個 top-loss case 輸出：

1. proposal_quality_delta:
   trained semantic proposal minus untrained semantic proposal
   in HPWL, bbox area, soft violations, overlap

2. repair_delta:
   repaired layout minus semantic proposal
   for trained and untrained separately

3. final_delta:
   trained final minus untrained final

4. quality driver:
   classify each case as one of:
   - proposal_hpwl_bad
   - proposal_bbox_bad
   - repair_worsens_hpwl
   - repair_worsens_bbox
   - finalizer_preserves_bad_structure
   - untrained_finalizer_advantage
   - soft_violation_driver
   - mixed

請特別回答：
- trained 在 pre-repair 階段就比較差嗎？
- 還是 repair 後才變差？
- lower displacement 是否代表 finalizer 沒有改正 trained 的壞 proposal？
- untrained 是否因為 finalizer 移動更多，反而得到 better quality？

============================================================
Part 4: Visual / structural diagnostics
============================================================

如果 repo 有 plotting utility，請為每個 top-loss case 產生簡單 plots：

- untrained semantic proposal
- untrained final
- trained semantic proposal
- trained final

輸出到：
artifacts/research/plots/top5_loss_drift/

如果不方便畫圖，請至少輸出 structural summaries：

- top 10 largest blocks positions
- top 10 highest b2b-degree blocks positions
- top 10 highest p2b-degree blocks positions
- bbox corners
- terminal distances
- group block bounding boxes

============================================================
Part 5: Action behavior audit
============================================================

請比較 trained vs untrained 的 action trace：

- primitive histogram
- placement order
- first 10 placed blocks
- first high-degree hub placement step
- first boundary block placement step
- first group leader placement step
- number of ATTACH
- number of SNAP_TO_BOUNDARY
- number of PULL_TO_PIN
- number of PLACE_RELATIVE
- number of FREEZE
- no-progress / fallback action count

請回答：
- trained 是否過度 ATTACH / cluster？
- trained 是否太早 FREEZE？
- trained 是否 high-degree block 放太晚？
- trained 是否 boundary / terminal blocks 處理不佳？
- untrained 是否透過 PLACE_RELATIVE / fallback 讓 finalizer 更自由？

============================================================
Part 6: Conclusion and next experiment decision
============================================================

最後請根據 evidence 判斷：

Case 1:
If trained proposal is already worse pre-repair:
  conclusion = training objective / policy scoring problem
  recommend = post-finalizer cost-aware reranking or objective-aware AWBC

Case 2:
If trained proposal is okay but repair worsens it:
  conclusion = finalizer interaction problem
  recommend = cost-aware finalizer, HPWL/bbox-aware repair

Case 3:
If untrained proposal is bad but finalizer makes it good:
  conclusion = finalizer dominates and policy contribution is low
  recommend = finalizer-aware training or policy should optimize finalizer input distribution

Case 4:
If both trained and untrained proposals are poor and finalizer ceiling is low:
  conclusion = candidate / action space ceiling
  recommend = redesign candidate families / topology representation

請輸出：
- final diagnosis
- confidence level
- recommended immediate next experiment
- experiments to avoid
- exact code files that should be changed next

============================================================
Output requirements
============================================================

Markdown report must include:
- executive summary
- top-5 case table
- stage-wise metrics
- diagnosis per case
- aggregate diagnosis
- next experiment recommendation

JSON must include:
- per_case_stage_metrics
- per_case_drift_decomposition
- action_histograms
- repair_metrics
- intent_preservation_metrics
- final_recommendation
- missing_metrics list

請不要只寫文字推論。請盡量用 artifacts / code instrumentation 產生數據。
如果某 metric 無法取得，請明確列出：
- metric name
- why unavailable
- what instrumentation is needed
```

---

# 4. 第二個可並行 prompt：Best-of-K Oracle / Reranking Ceiling

如果你有資源讓另一個 agent 並行，我會讓它做 **best-of-K oracle**。這個能快速回答 current action + finalizer space 裡到底有沒有好解。

---

## Prompt：Best-of-K Oracle and Reranking Ceiling

```text
你是一位 EDA floorplanning + ML experiment agent。請做一個小型 oracle / reranking ceiling 實驗，判斷目前 semantic rollout + finalizer search space 是否存在比 untrained baseline 更好的解。

請輸出：

1. artifacts/research/best_of_k_oracle_reranking.md
2. artifacts/research/best_of_k_oracle_reranking.json
3. scripts/run_best_of_k_oracle_reranking.py

背景：

目前 validation-0..19 上：
- best untrained mean official cost = 19.784
- best trained mean official cost = 23.651
- trained 輸在 HPWL + bbox area
- repair displacement 是 poor proxy

我們需要知道：
目前 action / rollout / finalizer search space 裡，有沒有 high-quality candidate？
如果有，下一步做 reranker / scorer。
如果沒有，下一步要改 candidate 或 finalizer。

============================================================
Experiment design
============================================================

For each validation case 0..19:

1. Generate K rollouts:
   - K = 16 initially
   - if cheap, also run K = 32
   - include:
     - untrained random seeds
     - trained checkpoints if available
     - heuristic semantic rollouts
     - stochastic candidate selection if implemented

2. For each rollout:
   - run finalizer
   - evaluate with official evaluator
   - also evaluate with runtime_factor standardized to 1.0 if possible
   - record:
     - official_cost
     - HPWLgap
     - HPWLint
     - HPWLext
     - Areagap_bbox
     - Violationsrelative
     - runtime_factor
     - hard_feasible
     - repair_displacement
     - changed_block_fraction
     - fallback_used
     - action histogram

3. For each case compute:
   - best_of_K cost
   - mean_of_K cost
   - median_of_K cost
   - std_of_K cost
   - best candidate source
   - best candidate seed
   - gap to best untrained baseline
   - gap to best trained baseline

4. Aggregate:
   - mean best_of_K cost
   - win rate over best untrained
   - win rate over best trained
   - number of cases where oracle improves
   - average improvement on improved cases
   - average degradation if no improvement

============================================================
Scoring variants
============================================================

Please evaluate at least these selection rules:

A. oracle_best
   Choose lowest official cost after finalizer.
   This is not deployable but gives upper bound.

B. displacement_best
   Choose lowest repair displacement.
   This tests whether displacement is a useful scorer.

C. hpwl_bbox_proxy_best
   Choose lowest pre-finalizer or post-finalizer proxy:
     proxy = HPWL_proxy + bbox_area_proxy
   If exact HPWL is available, use exact HPWL.

D. soft_violation_best
   Choose lowest Violationsrelative.

E. combined_proxy_best
   proxy = 0.5*(normalized HPWL proxy + normalized bbox proxy) + soft_violation penalty

Compare all against:
- best untrained seed 0
- best trained awbc seed 1

============================================================
Questions to answer
============================================================

1. Does best-of-K oracle beat best untrained mean cost?
2. If yes, by how much?
3. Does a simple proxy scorer approximate oracle?
4. Is displacement_best bad, confirming displacement is poor proxy?
5. Which cases have oracle improvement?
6. Which cases remain bad even under oracle?
7. Are remaining bad cases due to HPWL, bbox, soft violations, or finalizer fallback?
8. Does top-k diversity come mostly from untrained randomization or trained policy variants?

============================================================
Decision logic
============================================================

If oracle_best significantly beats untrained:
  Recommend:
  - train post-finalizer cost predictor
  - use cost-aware beam / reranking
  - collect cost-labeled rollout dataset

If oracle_best does not beat untrained:
  Recommend:
  - redesign candidate families
  - redesign finalizer with HPWL/bbox objective
  - add topology-aware placement representation

If hpwl_bbox_proxy_best is close to oracle:
  Recommend:
  - implement cheap proxy scorer in rollout loop

If displacement_best performs poorly:
  Recommend:
  - remove displacement as primary training / selection metric
  - keep displacement only as secondary regularizer

============================================================
Output requirements
============================================================

Markdown must include:
- executive summary
- per-case oracle table
- aggregate oracle table
- scorer comparison table
- decision recommendation

JSON must include:
- per_case_candidates
- per_case_best_by_rule
- aggregate_by_rule
- oracle_vs_baseline
- final_recommendation
- commands_used

Please keep this small and reproducible. Do not run full 100-case validation yet.
```

---

# 5. 我建議的下一步順序

## 立即做

1. **Top-5 Loss Drift Audit**
   先確定 trained 的壞 quality 是 proposal 造成，還是 repair/finalizer 造成。

2. **Best-of-K Oracle / Reranking Ceiling**
   判斷目前 search space 裡是否有好解。

這兩個結果合起來會把下一步鎖定成下列之一。

---

# 6. 決策樹

```text
If top-5 audit says:
  trained proposal already HPWL/bbox bad
And oracle says:
  best-of-K can beat untrained
Then:
  build post-finalizer cost-aware reranker / scorer

If top-5 audit says:
  trained proposal okay, repair makes it bad
Then:
  redesign finalizer objective with HPWL/bbox-aware repair

If oracle says:
  best-of-K cannot beat untrained
Then:
  current candidate + finalizer search space has ceiling
  redesign candidate families / placement topology

If displacement_best performs badly again:
  stop using displacement as primary objective
  keep it only as secondary regularizer
```

---

# 7. 目前我會暫停的工作

在上述兩份診斷完成前，我建議暫停：

```text
1. larger checkpoint training
2. more AWBC variants
3. more repair displacement optimization
4. full 100-case validation
5. adding complex model architecture
6. online RL / offline RL
```

因為現在還不知道問題是：

```text
policy scoring
finalizer interaction
candidate/action space ceiling
```

直接擴訓很可能只是把錯誤 proxy 學得更穩。

---

# 8. 暫定研究方向更新

下一階段我會改名成：

> **Objective-Aligned Puzzle Placement**

核心不是再證明可行性，而是證明：

```text
semantic action policy 可以被 official objective 對齊
```

初步目標變成：

```text
semantic rollout
→ finalizer
→ cost decomposition
→ oracle / reranking ceiling
→ objective-aware scorer
→ cost-weighted policy improvement
```

一句話：

> 現在不是「怎麼讓 trained layout 更容易 repair」，而是「怎麼讓 policy 產生 repair 後 HPWL / bbox / soft violations 更好的 layout」。
