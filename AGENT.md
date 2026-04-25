# AGENT.md

## Step7 專案共識

本專案目標是解 ICCAD 2026 FloorSet-Lite floorplanning。請把問題理解成：

```text
有約束的可變形矩形拼圖最佳化
= 離散拓樸決策
+ 連續形狀調整
+ 合法化
+ 多目標 tradeoff
+ 迭代式局部重構
```

不要再把它視為：

```text
NN coordinate regression
或一串 repair/gate 疊加
```

Step7 開始，專案主架構改成：

```text
Diagnose
-> Generate Alternatives
-> Legalize
-> Pareto Select
-> Refine / Iterate
```

簡稱：

```text
DGLPR loop
```

也就是：

1. 先產生一個合法或接近合法的 layout。
2. 診斷它的病徵。
3. 只針對病徵產生少量 alternatives。
4. 每個 alternative 都通過 deterministic legalization / hard checks。
5. 用 Pareto selector 比較，不再用 boundary-first gate 疊加。
6. 若仍有主要病徵，進下一輪局部重構。

最重要的一句話：

```text
The model and heuristics propose legal puzzle moves.
Diagnostics explain what is wrong.
Legalizers guarantee hard constraints.
Pareto selection chooses among real tradeoffs.
Iteration repairs the remaining pathology.
```

## 類比范式

這題抽象後最像幾個領域的混合。

### EDA Placement Flow

標準 EDA placement 常見流程：

```text
global placement
-> legalization
-> detailed placement
```

本題對應：

```text
constructive packing / puzzle policy
-> hard legalization / frame / hull
-> local alternatives / compaction / shape refine / Pareto select
```

重要教訓：

```text
global stage 不可製造後面救不回來的病態，
例如大量極端細長 blocks、破碎洞、被 regular block 偷走 hull。
```

### Bilevel / Alternating Optimization

外層：

```text
block order
relative topology
group template
MIB shape-master decision
move sequence
```

內層：

```text
coordinate placement
soft block width/height
compaction
overlap removal
boundary/hull ownership
```

所以合理流程是：

```text
choose topology / move
-> solve geometry / shape
-> diagnose
-> revise topology / shape policy
-> repeat
```

### Large Neighborhood Search

大 case 不能每次全域重排。應該：

```text
freeze most blocks
select a failing local neighborhood
destroy/repack only that region
legalize
Pareto compare with original
```

local neighborhood 由 attribution 決定：

```text
failed boundary block
hull owner / hull stealer
nearby blockers
same MIB/group members
top connected neighbors
```

### Multi-objective / Pareto Optimization

這題不是單目標：

```text
hard feasibility
boundary/group/MIB soft violations
HPWL
bbox
shape regularity
hole fragmentation
disruption
runtime
```

不要用一串固定 priority gate 代替最佳化。  
Step7 應以 Pareto front / representative alternatives 為核心。

## Step6 已學到什麼

以下是 Step6G-P 的濃縮結論，避免重走舊路。

### Step6G: Puzzle Policy Sidecar

學到：

```text
policy 確實能改善 HPWL
但會變成 wirelength-driven placer
```

問題：

```text
bbox 變差
boundary soft 變差
block body 可能爆出合理區域
```

### Step6H: Virtual Frame

學到：

```text
virtual frame 可防止 protrusion
frame containment solved
```

問題：

```text
spatial assignment unsolved
blocks 可擠在 frame 一側
```

### Step6I/J: Boundary Frame -> Hull Ownership

學到：

```text
boundary 應 own final compact hull
not virtual frame
```

問題：

```text
predicted hull 可被 final bbox 改寫
regular/non-boundary block 可能偷 hull edge
```

### Step6K/L: Attribution + Shape/Group Probes

學到：

```text
不能假設 boundary failure 一定是 edge segment 問題
MIB/group/shape 可能影響是否擺得進去
```

### Step6M/N/O: Move Library + Guard Replay

學到：

```text
simple_compaction 有價值，但不能全域套用
guard 能擋壞 compaction
但 guard 後 second-best selector 仍可能選壞 move
```

### Step6P: Pareto Alternative Selector

學到：

```text
boundary-first selector 是錯誤架構
Pareto selector + original candidate 更自然
applicability filter 可將 no-effect moves 清掉
```

Step6P 指標：

```text
no-effect moves: 1462 -> 0
original closest-to-ideal: 20 / 40
mean Pareto front size: ~3
case 15 bad second choice 被修掉
```

剩餘風險：

```text
depth-2 combo 仍是 estimated combo
large/XL coverage 尚未完成
large cases 出現大量細長 blocks
shape policy 可能是新瓶頸
```

## Step7 核心轉向

Step7 不再是：

```text
再加一個 guard
再加一個 repair
```

Step7 是把整個 sidecar 改造成可迭代架構：

```text
initial layout
-> diagnostics
-> targeted alternatives
-> legalization
-> Pareto select
-> repeat if needed
```

品質選擇應由：

```text
Pareto / multi-objective comparison
```

而不是：

```text
boundary desc -> bbox -> HPWL -> soft
```

## Step7 模組邊界

後續 agent 請把程式逐步整理成下列模組。  
目前不要一次大搬家；先讓新 Step7 檔案按這個結構新增，舊 Step6 檔案可留在 `research/legacy_step6` 或保持原位但不再擴張。

### Core

建議：

```text
src/puzzleplace/core/
  problem.py
  layout.py
  geometry.py
  metrics.py
  constraints.py
  roles.py
```

職責：

- parse problem-like data structures
- represent rectangles / layouts
- compute hard legality
- compute HPWL / bbox / soft metrics
- infer deterministic roles

核心資料結構：

```text
Problem
Layout
BlockRect
ConstraintSet
MetricBundle
RoleBundle
```

### Construction

建議：

```text
src/puzzleplace/construction/
  candidate_generator.py
  constructive_packer.py
  virtual_frame.py
  predicted_hull.py
```

職責：

- initial layout generation
- legal candidate generation
- virtual frame / predicted hull
- constructive placement

### Diagnostics

建議：

```text
src/puzzleplace/diagnostics/
  boundary.py
  hull.py
  aspect.py
  holes.py
  hpwl.py
  profiles.py
  attribution.py
```

職責：

- explain what is wrong
- produce pathology labels
- identify local neighborhoods
- output per-case reports

重要診斷：

```text
boundary failure type
hull drift / hull stealer
aspect pathology
hole fragmentation
internal vs external HPWL
terminal target distance
MIB/group role conflicts
candidate family bias
```

### Alternatives

建議：

```text
src/puzzleplace/alternatives/
  base.py
  move_library.py
  compaction.py
  boundary_moves.py
  shape_moves.py
  mib_moves.py
  group_moves.py
  local_repack.py
  sequence.py
```

職責：

- generate targeted alternatives
- apply executable moves
- record disruption / deltas
- avoid no-effect moves using applicability filters

核心資料結構：

```text
MoveSpec
Alternative
AlternativeResult
MoveApplicability
MoveExecutionTrace
```

### Legalization

建議：

```text
src/puzzleplace/legalization/
  hard_check.py
  overlap.py
  compaction.py
  frame.py
  repair.py
```

職責：

- no overlap
- area tolerance
- fixed/preplaced exactness
- frame containment
- optional local repair

Hard invalid alternatives 不進 Pareto front。  
這是 legality filter，不是 quality gate。

### Selection

建議：

```text
src/puzzleplace/selection/
  pareto.py
  representatives.py
  objectives.py
  risk.py
```

職責：

- normalize objectives
- compute Pareto fronts
- choose representative candidates
- compare original vs alternatives

不要把 Step6 boundary-first selector 繼續擴張。

### Experiments

建議：

```text
scripts/
  step7a_run_diagnostics.py
  step7b_run_aspect_policy_study.py
  step7c_run_iterative_loop.py
  step7d_run_large_xl_replay.py
```

Artifacts：

```text
artifacts/research/step7*/
```

Docs：

```text
docs/step7_architecture.md
docs/step7_aspect_policy.md
docs/step7_iterative_loop.md
```

Tests：

```text
tests/test_step7_*.py
```

## Step7 Immediate Priority

使用者觀察到：

```text
small cases mostly OK
larger medium cases start to show many skinny blocks
large cases likely worse
```

目前最可疑的瓶頸：

```text
shape policy / aspect pathology
```

所以 Step7 第一刀不是 NSGA-II，不是 runtime integration，而是：

```text
Step7A: Aspect Pathology and Role-Aware Shape Policy Study
```

## Step7A: Aspect Pathology

目標：確認大 case 的問題是否真的來自大量細長 blocks，以及這些細長 blocks 是在 construction、move、還是 compaction 之後出現。

### 必要輸出

Per case：

```text
n_blocks
selected representative
median_abs_log_aspect
p90_abs_log_aspect
max_abs_log_aspect
extreme_aspect_count
extreme_aspect_area_fraction
extreme_aspect_by_role
extreme_aspect_by_move
extreme_aspect_by_candidate_family
pre_move_aspect_stats
post_move_aspect_stats
shape_changed_by_move
```

Role buckets：

```text
boundary
MIB
grouping
fixed/preplaced
terminal-heavy
core
regular
filler
```

Candidate family buckets：

```text
boundary
placed_block left/right/top/bottom
free_rect
pin_pull
group_mate
fallback
```

### Aspect thresholds

不要直接定罪。先 report 多個 threshold：

```text
abs(log(w/h)) > 1.5
abs(log(w/h)) > 2.0
abs(log(w/h)) > 3.0
```

也要 area-weighted：

```text
sum(area of extreme blocks) / total area
```

### Correlations

輸出 correlation / grouped summary：

```text
extreme_aspect_count vs hole_fragmentation
extreme_aspect_count vs occupancy_ratio
extreme_aspect_count vs bbox_delta_norm
extreme_aspect_count vs hpwl_delta_norm
extreme_aspect_count vs boundary failure
extreme_aspect_area_fraction vs selected move
```

### Artifacts

```text
artifacts/research/step7a_aspect_pathology.json
artifacts/research/step7a_aspect_by_role.json
artifacts/research/step7a_aspect_by_candidate_family.json
artifacts/research/step7a_aspect_correlation.md
artifacts/research/step7a_visualizations/
```

## Step7B: Role-aware Shape Policy Alternatives

如果 Step7A 顯示 shape pathology 是主要瓶頸，再做 Step7B。

不要全局禁止長條。長條可能合理：

```text
filler 長條可能合理
boundary edge-slot 長條可能合理
core / high-degree / MIB / group member 長條較可疑
```

### Shape policy alternatives

產生 sidecar alternatives：

```text
original_shape_policy
mild_global_cap
role_aware_cap
filler_only_extreme
boundary_edge_slot_exception
MIB_shape_master_regularized
group_macro_aspect_regularized
```

建議初始 ranges：

```text
core: abs(log_r) <= 1.5
terminal-heavy: abs(log_r) <= 1.5 unless pin/edge justified
boundary: abs(log_r) <= 1.2 unless edge-slot justified
regular: abs(log_r) <= 2.0
filler: abs(log_r) <= 3.0
MIB master: abs(log_r) <= 1.5 when compatible
group macro: template-level aspect cap
```

這些是 alternatives，不是硬改主線。

### Compare by Pareto

每個 shape policy alternative 都要 legalize，再進 Pareto comparison。

Objectives：

```text
boundary violation delta
HPWL delta norm
bbox delta norm
aspect pathology score
hole fragmentation
disruption
```

### Artifacts

```text
artifacts/research/step7b_shape_policy_alternatives.json
artifacts/research/step7b_pareto_shape_policy.json
artifacts/research/step7b_shape_policy_summary.md
```

## Step7C: Iterative DGLPR Loop

在 Step7A/B 後，把流程變成最多 2-3 輪迭代：

```text
layout_0 = constructive layout

for iter in 1..K:
    diagnosis = diagnose(layout)
    alternatives = generate_targeted_alternatives(diagnosis)
    legal_alternatives = legalize(alternatives)
    pareto_front = select_pareto(original + legal_alternatives)
    layout_next = choose_representative(pareto_front)
    stop if no meaningful pathology improvement
```

### Stop conditions

```text
hard infeasible appears -> reject alternative
no Pareto improvement over original
pathology score no longer improves
max iterations reached
runtime budget reached
```

### What not to do

不要每輪全域重排。  
不要每輪跑所有 moves。  
alternatives 必須由 diagnostics targeting 生成。

## Step7D: Large/XL Coverage

目前 Step6M-P 主要是 21..60 prefix。Step7 必須補：

```text
61..120 coverage
especially 80, 90, 100, 110, 120
```

報表要按 bucket：

```text
small
medium
large
xl
```

輸出：

```text
selected representative counts
aspect pathology stats
Pareto front size
hard invalid rate
HPWL/bbox/boundary/soft deltas
runtime estimate
```

不要在 large/XL 上先調參；先 replay / diagnose。

## Pareto Selector Details

Step7 selector 應保持 original-inclusive：

```text
original = zero-delta candidate
```

Normalized objectives：

```text
boundary_violation_delta_norm
hpwl_delta_norm
bbox_delta_norm
disruption_cost_norm
optional aspect_pathology_delta
optional hole_fragmentation_delta
```

不要一次塞太多 objectives。  
如果 objectives 超過 5 個，front 可能失去選擇壓力。

Representatives：

```text
min_disruption
closest_to_ideal
best_boundary
best_hpwl
best_shape_regularized
```

不要立刻把 closest_to_ideal 當最終 rule。先比較 representatives。

## Move Library Rules

Step7 move generation 必須有 applicability filter。

Examples：

```text
MIB moves only if target has MIB role
group moves only if target has grouping role
soft shape moves only if non-fixed and non-preplaced
boundary moves only if boundary failure / hull ownership conflict exists
local repack only if local neighborhood is attribution-defined
```

No-effect moves 不該進 selector。

Global disruptive moves 必須逐步拆小：

```text
simple_compaction
-> edge-local compaction
-> axis-specific compaction
-> hull-owner compaction

local_region_repack
-> connectivity-preserving local repack
-> group/MIB-aware local repack
```

## ML / RL / NSGA-II Roadmap

不要現在 full RL。

建議順序：

```text
Step7: deterministic diagnostics + Pareto alternatives
Step8: collect move dataset
Step9: learned move-ranker / contextual bandit
Step10: limited NSGA-II / mutation search if needed
Step11: RL only after move definitions stable
```

### Move dataset

收集：

```text
case profile
diagnosis
move type
target roles
before metrics
after metrics
Pareto rank
hypervolume contribution
runtime cost
accepted representative
```

Move-ranker 目標：

```text
predict promising move families / targets
seed Pareto alternatives
prune expensive low-value moves
```

不要讓 ML 直接繞過 legalizer。

## Hard Constraints

永遠 hard check：

```text
no overlap
area tolerance
fixed dimensions exact
preplaced x/y/w/h exact
frame containment if enabled
```

Hard invalid alternatives 不進 Pareto front。  
這不是 gate 疊加，而是合法性。

## Project Directory Target

後續整理目錄時，目標結構如下：

```text
src/puzzleplace/
  core/
  construction/
  diagnostics/
  alternatives/
  legalization/
  selection/
  experiments/
  research/legacy_step6/

scripts/
  step7a_run_aspect_pathology.py
  step7b_run_shape_policy_study.py
  step7c_run_iterative_loop.py
  step7d_run_large_xl_replay.py

docs/
  step7_architecture.md
  step7_aspect_policy.md
  step7_iterative_loop.md

tests/
  test_step7_*.py
```

目前不要做大規模搬檔，除非該 agent 明確負責 refactor。  
先用新目錄新增 Step7 檔案，舊 Step6 sidecar 保留可追溯性。

## Immediate Next Task

如果使用者沒有另行指定，下一個 agent 應做：

```text
Step7A: Aspect Pathology and Role-Aware Shape Diagnosis
```

Scope：

```text
sidecar only
no runtime integration
no full NSGA-II
no RL
no large refactor beyond new diagnostics files
```

Deliverables：

```text
docs/step7a_aspect_pathology.md
src/puzzleplace/diagnostics/aspect.py
scripts/step7a_run_aspect_pathology.py
tests/test_step7a_aspect_pathology.py
artifacts/research/step7a_*.json/md/png
```

Key question：

```text
Are large-case failures driven by role-specific extreme aspect ratios,
candidate-family bias,
or post-move/compaction distortion?
```

## Verification

每個 sidecar step：

```text
ruff touched files
mypy new typed modules
pytest relevant tests
full pytest if cheap
artifact sanity check
per-case visualization for representative failures
```

不要只報平均值。  
必須包含：

```text
good cases
bad cases
large/XL coverage status
per-role / per-profile breakdown
```

## References

Relevant families:

```text
FloorSet
PARSAC / constraint-aware SA
EDA global placement -> legalization -> detailed placement
2D bin packing / cutting stock pattern generation
Large Neighborhood Search
Pareto / NSGA-II
contextual bandit / learned move ranking
```
