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

### Step7A Observed Evidence

Step7A 已完成第一輪 40-case smoke test。

已知 coverage：

```text
small: 20 cases
medium: 20 cases
large/XL: not covered yet
```

目前 worst aspect cases 集中在較大的 medium prefix：

```text
case 36: n=57, extreme aspect count=34, extreme area fraction=0.468
case 29: n=50, extreme aspect count=32, extreme area fraction=0.547
case 39: n=60, extreme aspect count=32, extreme area fraction=0.484
case 32: n=53, extreme aspect count=32, extreme area fraction=0.445
case 33: n=54, extreme aspect count=32, extreme area fraction=0.443
```

目前 correlation：

```text
extreme_aspect_count vs hpwl_delta_norm: +0.360
extreme_aspect_count vs occupancy_ratio: -0.189
extreme_aspect_count vs boundary_failure: +0.164
```

Interpretation：

```text
aspect pathology looks real enough to investigate,
but correlation is not causality.
Step7B must test whether shape policy changes improve layout,
not merely report that bad layouts have skinny blocks.
```

## Step7B: Role-aware Shape Policy Smoke Test

Step7B 由 Step7A 結果觸發。Step7B 不是直接限制長條，而是回答：

```text
極端 aspect 是因果還是伴隨？
限制哪些 role 的 aspect 有幫助？
是在 construction 階段修才有效，還是 posthoc 就夠？
會不會傷 boundary / MIB / group / HPWL？
candidate family bias 是否仍存在？
```

### 原則

不要全局禁止長條。長條可能合理：

```text
filler 長條可能合理
boundary edge-slot 長條可能合理
core / high-degree / MIB / group member 長條較可疑
```

所有 policy 都是 sidecar alternatives，保留 original。

### Two-track Evaluation

Step7B 必須分兩種：

```text
posthoc_shape_probe:
    對既有 layout 做局部 shape probe，便宜，用於診斷。

construction_shape_policy_replay:
    在 candidate generation / construction 階段限制 shape bins，才是真正 policy 測試。
```

如果 posthoc 有效但 construction replay 無效，代表問題可能是 candidate placement / topology，不只是 shape。

### Shape Policy Alternatives

產生 sidecar alternatives：

```text
original_shape_policy
mild_global_cap
role_aware_cap
filler_only_extreme
boundary_strict_cap
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

### Role Reasons

role-aware cap 必須輸出每個被 cap 的 block：

```text
block_id
original_w_h
new_w_h
original_abs_log_aspect
new_abs_log_aspect
role_trigger
role_reason
boundary/MIB/group/fixed/preplaced flags
```

role 定義先 deterministic，不用 ML：

```text
filler = low degree + small area + no boundary/MIB/grouping
core = high b2b degree or high area
terminal-heavy = high external_ratio
boundary/MIB/group = explicit constraints
```

### Boundary Exceptions

boundary cap 不可一刀切。請比較：

```text
boundary_strict_cap
boundary_edge_slot_exception
```

exception 條件不是硬 gate，而是生成兩個 alternatives 讓 Pareto 比較。

### MIB / Group Handling

MIB：

```text
exact compatible -> capped shared shape master
not exact compatible -> shared aspect cap or subgroup cap
```

需報：

```text
mib_exact_compatible_count
mib_subgroup_count
mib_alternative_invalid_count
```

Group：

```text
member aspect pathology
group_bbox_aspect pathology
group_template_type
group_connected_components
```

group regularization 應是 template/macro-level，不只是 member-level。

### Candidate Family Impact

Step7A candidate-family attribution 目前只是 usage，不是 causality。Step7B 仍須輸出：

```text
extreme aspect by candidate family
policy effect by candidate family
free_rect usage change
pin_pull usage change
boundary/placed_block usage change
```

如果 extreme aspects 主要來自 boundary/placed_block candidates，可能要修 candidate generator，而不是只 cap shape。

### Pareto Compare

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

Representatives：

```text
original
min_aspect_pathology
closest_to_ideal
min_disruption
best_hpwl
best_boundary
```

### Focus Cases

至少聚焦 Step7A worst cases：

```text
case 29
case 32
case 33
case 36
case 39
```

並明確標註：

```text
current Step7B smoke is 21..60 only
large/XL gap remains until Step7D
```

### Artifacts

```text
artifacts/research/step7b_shape_policy_alternatives.json
artifacts/research/step7b_posthoc_shape_probe.json
artifacts/research/step7b_construction_shape_replay.json
artifacts/research/step7b_pareto_shape_policy.json
artifacts/research/step7b_role_cap_reasons.json
artifacts/research/step7b_candidate_family_impact.json
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

### Step7D Observed Evidence

Step7D 已完成 representative coverage replay。

Decision：

```text
pivot_to_region_topology_before_step7c
```

原因：

```text
large/XL coverage exists.
large/XL is not a data gap.
large/XL Pareto winners are all no safe improvement.
construction shape policy does not generalize safely.
Step7C should not become a shape-policy iterative loop yet.
```

Representative suite：

```text
selected case ids: 2, 19, 24, 25, 79, 51, 76, 99, 91
has_large: true
has_xl: true
missing_category: none
```

By size bucket：

```text
small:
  original wins: 1
  construction shape policy wins: 1

medium:
  MIB/group policy wins: 2

large:
  no safe improvement: 3/3
  hard invalid rate: 0.5

XL:
  no safe improvement: 2/2
  hard invalid rate: 0.5
```

Interpretation：

```text
skinny blocks are a symptom, not the large/XL root cause.
shape policy can remain a local regularizer.
large/XL needs coarse region/topology diagnosis before iterative search.
```

## Step7E: Region / Topology Failure Diagnosis

Step7E 是 Step7D 的直接後續。

不要直接實作新的 region placer。  
先診斷 large/XL 為什麼 no safe improvement。

核心問題：

```text
large/XL failure starts at which layer?

1. cluster / net-community assignment is wrong
2. block-to-region assignment is wrong
3. candidate ordering places critical clusters too early/late
4. free-space fragmentation appears too early
5. MIB/group macros are not treated as coarse objects
6. local alternatives trigger long repair chains
7. legalizer is too weak for large/XL local perturbations
```

### Scope

```text
sidecar only
diagnostics first
no contest runtime integration
no new finalizer semantics
no hard global gates
no direct Step7C iterative solver yet
no full RL / NSGA-II
do not remove Step6/Step7A-D code
```

Step7E should produce structural explanations, not new penalty terms.

Bad pattern：

```text
if region_bad then add penalty
if hole_bad then add penalty
if cluster_bad then add penalty
```

Good pattern：

```text
diagnose where topology first diverges
identify whether failure is region, ordering, fragmentation, macro, or repair radius
recommend which architecture Step7F should implement
```

### Required Diagnostics

Region occupancy：

```text
partition frame / pin bbox into coarse grid or adaptive regions
measure block area occupancy per region
measure fixed/preplaced occupancy per region
measure unused capacity per region
measure overflow / underflow per region
```

Pin / terminal density：

```text
pin density map
terminal density map
external pin bbox
pin-density centroid per net-community if available
```

Net-community clustering：

```text
build block graph from net connectivity
run deterministic community / connected-component clustering
report cluster size, area, degree, external terminal pull
```

Block-to-region assignment：

```text
expected region from pin/community centroid
actual region from placed bbox center
assignment mismatch distance
assignment entropy
cluster spread across regions
```

Free-space topology：

```text
large open holes
fragmented small holes
narrow channels / slivers
hole area ratio
hole count
largest hole fraction
fragmentation score
```

Candidate ordering trace：

```text
construction order if available
first K placements per case
first placement that creates major region mismatch
first placement that causes large hole fragmentation
first MIB/group member placed away from its macro/cluster
```

Repair radius audit：

```text
for each rejected or hard-invalid alternative:
  moved block count
  max displacement
  mean displacement
  displacement chain length
  affected region count
  overlap/violation cause
```

If exact construction trace is not available, reconstruct the best possible proxy and label confidence:

```text
trace_confidence: exact | reconstructed | unavailable
```

### Required Case Coverage

Use Step7D representative suite first:

```text
2, 19, 24, 25, 79, 51, 76, 99, 91
```

At minimum, include:

```text
all large cases from Step7D
all XL cases from Step7D
one medium MIB/group-heavy winner
one small/medium case where shape policy worked
```

Reason：

```text
compare where shape policy works vs where it fails
```

### Failure Attribution

For each case, classify primary and secondary failure:

```text
region_capacity_mismatch
cluster_region_mismatch
candidate_ordering_failure
early_fragmentation
MIB_group_macro_failure
boundary_hull_ownership_failure
local_move_repair_radius_too_large
legalizer_capacity_failure
proxy_inconclusive
```

Each classification must include evidence fields, not only a label.

Example：

```text
primary_failure: cluster_region_mismatch
evidence:
  assignment_entropy: high
  cluster_spread_regions: 4
  pin_density_centroid_distance: large
  free_space_fragmentation: moderate
```

### Architecture Decision

Step7E must end with one of:

```text
promote_cluster_first_region_planner
pivot_to_macro_level_MIB_group_planner
pivot_to_candidate_ordering_policy
pivot_to_free_space_topology_generator
pivot_to_large_scale_legalizer_repair
inconclusive_due_to_trace_gap
```

Decision rules:

```text
If clusters are coherent but placed in wrong regions:
  promote_cluster_first_region_planner

If MIB/group macros dominate mismatch:
  pivot_to_macro_level_MIB_group_planner

If early placements create irreversible holes:
  pivot_to_candidate_ordering_policy
  or pivot_to_free_space_topology_generator

If alternatives are conceptually good but hard invalid after repair:
  pivot_to_large_scale_legalizer_repair

If trace is too weak:
  inconclusive_due_to_trace_gap
```

### Required Artifacts

```text
docs/step7e_region_topology_diagnosis.md
src/puzzleplace/diagnostics/region_topology.py
src/puzzleplace/diagnostics/placement_trace.py
scripts/step7e_run_region_topology_diagnosis.py
tests/test_step7e_region_topology.py

artifacts/research/step7e_region_occupancy.json
artifacts/research/step7e_pin_density_regions.json
artifacts/research/step7e_net_community_clusters.json
artifacts/research/step7e_block_region_assignment.json
artifacts/research/step7e_free_space_fragmentation.json
artifacts/research/step7e_candidate_ordering_trace.json
artifacts/research/step7e_repair_radius_audit.json
artifacts/research/step7e_failure_attribution.json
artifacts/research/step7e_decision.md
artifacts/research/step7e_visualizations/
```

Visualizations should include at least:

```text
region occupancy heatmap
pin-density overlay
cluster/community color overlay
free-space fragmentation overlay
assignment mismatch arrows
```

### Step7E Observed Evidence

Step7E 已完成 region / topology failure diagnosis。

Decision：

```text
pivot_to_large_scale_legalizer_repair
```

核心證據：

```text
Step7D suite covered: 2, 19, 24, 25, 79, 51, 76, 99, 91

primary failure counts:
  MIB_group_macro_failure: 4
  local_move_repair_radius_too_large: 5

large/XL no-safe-improvement cases:
  79, 51, 76, 99, 91

large/XL primary failure:
  local_move_repair_radius_too_large for all large/XL cases

hard_invalid_alternative_fraction:
  0.5

max_moved_block_fraction:
  1.0

max_affected_region_count:
  14..16
```

Interpretation：

```text
Large/XL alternatives are not merely bad ideas.
They become unsafe because local moves trigger global repair cascades.

The current repair/legalization path loses locality:
  local alternative -> broad displacement chain -> many affected regions -> no safe improvement

Shape policy is still useful in smaller/MIB/group-heavy cases,
but large/XL needs bounded repair before Step7C iterative search.
```

Important conclusion：

```text
Do not proceed to Step7C until repair radius is bounded or at least measured.
Iterative search without local repair control will create global oscillation.
```

## Step7F: Bounded Large-Scale Legalizer / Repair Radius Study

Step7F is the next architecture step after Step7E.

Goal：

```text
Test whether large/XL alternatives can become useful
when repair is bounded to a local / regional / graph neighborhood.
```

This is still a study, not runtime integration.

### Core Hypothesis

```text
If a move is local, repair must remain local.

If repair touches nearly every block,
then Pareto comparison is no longer evaluating the proposed move.
It is evaluating an accidental global reshuffle.
```

Step7F should answer：

```text
Can bounded repair convert some large/XL hard-invalid alternatives into safe alternatives?
Can it reduce moved_block_fraction and affected_region_count without destroying HPWL/bbox/boundary?
Which repair boundary is most stable: geometry window, graph neighborhood, region cells, or macro-aware component?
```

### Scope

```text
sidecar only
no contest runtime integration
no finalizer semantics change
no complete new global legalizer
no full Step7C iterative loop
no RL / NSGA-II
do not replace existing repair path
compare bounded repair against current repair as alternatives
```

Step7F may add a bounded repair module under `legalization/` or `experiments/`, but it must remain isolated from contest runtime.

### Repair Boundaries To Test

Generate repair alternatives for hard-invalid or no-safe-improvement cases:

```text
current_repair_baseline
geometry_window_repair
region_cell_repair
graph_hop_repair
macro_component_repair
cascade_capped_repair
rollback_to_original
```

Definitions：

```text
geometry_window_repair:
  repair blocks whose bboxes intersect an expanded bbox around moved/violating blocks

region_cell_repair:
  repair blocks inside the same coarse region cells as moved/violating blocks

graph_hop_repair:
  repair moved/violating blocks plus k-hop net neighbors

macro_component_repair:
  if moved block is MIB/group/boundary-critical, include the entire macro/component

cascade_capped_repair:
  allow repair expansion, but stop when moved fraction or affected region count exceeds cap

rollback_to_original:
  if bounded repair cannot restore hard feasibility, keep original
```

### Repair Seed Set

For each candidate alternative, start from:

```text
explicitly moved blocks
overlapping blocks
blocks causing boundary/frame violations
blocks causing MIB/group inconsistency
blocks in violated fixed/preplaced neighborhoods
```

Then expand by configurable policies:

```text
geometry radius
region-cell adjacency
net graph hops
MIB/group macro membership
boundary hull ownership
```

Every repair attempt must report its seed and expansion reason.

### Hard Legality Rule

Hard invalid alternatives do not enter Pareto front.

But Step7F should distinguish:

```text
hard_invalid_before_repair
hard_invalid_after_current_repair
hard_invalid_after_bounded_repair
bounded_repair_partial_success
bounded_repair_rejected_due_to_radius_cap
```

Do not silently fall back to global repair.

If bounded repair exceeds cap, mark it:

```text
repair_radius_exceeded
```

and either reject it or return `rollback_to_original`.

### Metrics

For every repair alternative:

```text
case_id
case_size_bucket
source_move_type
repair_mode
hard_feasible_before
hard_feasible_after
overlap_count_before
overlap_count_after
frame_protrusion_after
MIB_group_violation_after
moved_block_count
moved_block_fraction
max_displacement
mean_displacement
displacement_chain_length
affected_region_count
affected_region_fraction
repair_seed_count
repair_expanded_count
repair_radius_exceeded
hpwl_delta_norm
bbox_delta_norm
boundary_delta
aspect_pathology_delta
hole_fragmentation_delta
runtime_estimate
reject_reason
```

Primary success metrics：

```text
hard_feasible_after == true
moved_block_fraction substantially below current repair
affected_region_count substantially below current repair
no severe HPWL/bbox/boundary regression
```

### Required Cases

Use all Step7E large/XL failure cases:

```text
79, 51, 76, 99, 91
```

Also include comparison cases:

```text
one small/medium case where shape policy worked
one medium MIB/group-heavy case
one boundary-heavy case if available
```

Reason：

```text
Step7F must prove bounded repair helps large/XL
without breaking the smaller cases where local policies already work.
```

### Failure Attribution

For each failed bounded repair, classify:

```text
repair_window_too_small
repair_window_too_large
macro_component_missing
boundary_owner_missing
free_space_insufficient
legalizer_algorithm_insufficient
move_itself_incompatible
proxy_trace_insufficient
```

Each label needs evidence:

```text
remaining_overlap_count
unresolved_violation_type
required_extra_blocks
region_capacity_shortage
macro_members_excluded
cap_that_stopped_expansion
```

### Pareto / Selection

Bounded repair outputs should be compared as alternatives:

```text
original
current_repair_baseline
geometry_window_repair
region_cell_repair
graph_hop_repair
macro_component_repair
cascade_capped_repair
rollback_to_original
```

Pareto objectives should stay small:

```text
hard feasibility first
moved_block_fraction
affected_region_count
HPWL delta norm
bbox delta norm
boundary delta
```

Do not add every diagnostic as an objective.
Use extra diagnostics for explanation.

### Required Artifacts

```text
docs/step7f_bounded_repair_radius.md
src/puzzleplace/diagnostics/repair_radius.py
src/puzzleplace/legalization/__init__.py
src/puzzleplace/legalization/bounded_repair.py
scripts/step7f_run_bounded_repair_study.py
tests/test_step7f_bounded_repair.py

artifacts/research/step7f_repair_candidates.json
artifacts/research/step7f_bounded_repair_results.json
artifacts/research/step7f_repair_radius_metrics.json
artifacts/research/step7f_failure_attribution.json
artifacts/research/step7f_pareto_repair_selection.json
artifacts/research/step7f_decision.md
artifacts/research/step7f_visualizations/
```

Visualizations should include:

```text
before repair
after current repair
after bounded repair
repair seed blocks
expanded repair window
displacement arrows
affected region overlay
remaining violations if any
```

### Step7F Decision

Step7F must end with one of:

```text
promote_bounded_repair_to_step7c
pivot_to_macro_level_legalizer
pivot_to_region_replanner
pivot_to_move_generation_constraints
inconclusive_due_to_surrogate_or_trace_gap
```

Decision rules：

```text
If bounded repair makes large/XL alternatives feasible with low moved fraction:
  promote_bounded_repair_to_step7c

If failures concentrate around MIB/group components:
  pivot_to_macro_level_legalizer

If repair windows fail because region capacity is insufficient:
  pivot_to_region_replanner

If moves repeatedly require global repair:
  pivot_to_move_generation_constraints

If exact trace/data is too weak:
  inconclusive_due_to_surrogate_or_trace_gap
```

### Step7F Observed Evidence

Step7F 已完成 bounded repair study。

Decision：

```text
pivot_to_move_generation_constraints
```

核心證據：

```text
large/XL bounded success: 0 / 5
repair_window_too_large: 40 rows
bounded modes usually radius_exceeded or near-global
```

Interpretation：

```text
The current large/XL source moves are already too global.
Bounded repair can detect the problem but cannot rescue the moves.
Next step should constrain and route moves before repair.
```

Visualization caution：

```text
Some Step7F PNGs show very long displacement arrows.
This may be true global repair cascade, but may also be coordinate/arrow plotting error.
Step7G must verify arrow endpoints, block id matching, and before/after coordinate frames
before using displacement visuals as evidence.
```

## Step7G: Spatial Locality Map and Move Routing

Step7G replaces "generate move then hope repair can localize it" with:

```text
build locality maps
estimate move impact before repair
classify move as local / regional / macro / global
route it to the right repair/planner path
calibrate prediction against Step7F repair audit
```

This is not a hard gate.  
It is classification and routing.

### Multi-channel Locality Map

Build deterministic spatial maps first, not ML/RL yet:

```text
occupancy_mask
free_space_mask
fixed_preplaced_mask
pin_density_heatmap
net_community_demand_map
region_slack_map
hole_fragmentation_map
boundary_owner_map
MIB_group_closure_mask
repair_reachability_mask
```

Use at least two resolutions:

```text
coarse region grid
block-scale adaptive grid
```

Report sensitivity by grid resolution.

### Locality Estimator

For each candidate move, predict before repair:

```text
predicted_affected_blocks
predicted_affected_regions
predicted_macro_closure_size
predicted_region_slack
predicted_free_space_fit
predicted_repair_mode
predicted_locality_class
```

Classes：

```text
local -> bounded repair + Pareto
regional -> region repair / later region planner
macro -> MIB/group macro legalizer
global -> do not send to local selector
```

Do not simply reject regional/global moves.  
Label and route them.

### Visualization Sanity Precheck

Before calibration, audit Step7F displacement plots:

```text
arrow endpoint == after block center
before/after use same coordinate frame
block id matching is correct
after bbox / frame protrusion is measured
arrows do not autoscale the layout into unreadability
```

If suspicious, output:

```text
arrow_endpoint_debug.json
before_after_axis_check.json
topk_displacement_visualizations/
```

### Calibration Against Step7F

Use Step7F repair audit as weak labels:

```text
actual_moved_block_fraction
actual_affected_region_count
actual_radius_exceeded
actual_hard_feasible_after
actual_failure_attribution
```

Report:

```text
correct_local
correct_regional
correct_macro
correct_global
under_predicted_globality
over_predicted_globality
```

Step7G success requires invalid rate to fall without collapsing useful alternatives.

### Required Artifacts

```text
docs/step7g_spatial_locality_routing.md
src/puzzleplace/diagnostics/spatial_locality.py
src/puzzleplace/alternatives/locality_routing.py
scripts/step7g_run_spatial_locality_routing.py
tests/test_step7g_spatial_locality.py

artifacts/research/step7g_locality_maps.json
artifacts/research/step7g_move_locality_predictions.json
artifacts/research/step7g_routing_results.json
artifacts/research/step7g_calibration_report.json
artifacts/research/step7g_visualization_audit.json
artifacts/research/step7g_decision.md
artifacts/research/step7g_visualizations/
```

### Step7G Decision

Step7G must end with one of:

```text
promote_locality_routing_to_step7c
pivot_to_coarse_region_planner
pivot_to_macro_level_move_generator
pivot_to_visualization_or_trace_repair
inconclusive_due_to_prediction_quality
```

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

後續整理目錄時，維持 `src/puzzleplace/{diagnostics,alternatives,legalization,experiments}`、`scripts/step7*.py`、`docs/step7*.md`、`tests/test_step7_*.py` 的方向。

目前不要做大規模搬檔；先用新目錄新增 Step7 檔案，舊 Step6 sidecar 保留可追溯性。

## Immediate Next Task

如果使用者沒有另行指定，下一個 agent 應做：

```text
Step7G: Spatial Locality Map and Move Routing
```

Scope：

```text
sidecar only
no runtime integration
no RL yet
no full NSGA-II
no new region placer yet
no hard locality gate
no direct Step7C iterative loop
no large refactor beyond Step7 diagnostics / alternatives / experiment files
original layout must remain an alternative
```

Deliverables：

```text
docs/step7g_spatial_locality_routing.md
src/puzzleplace/diagnostics/spatial_locality.py
src/puzzleplace/alternatives/locality_routing.py
scripts/step7g_run_spatial_locality_routing.py
tests/test_step7g_spatial_locality.py
artifacts/research/step7g_locality_maps.json
artifacts/research/step7g_move_locality_predictions.json
artifacts/research/step7g_routing_results.json
artifacts/research/step7g_calibration_report.json
artifacts/research/step7g_visualization_audit.json
artifacts/research/step7g_decision.md
artifacts/research/step7g_visualizations/
```

Key question：

```text
Can multi-channel spatial masks predict whether a candidate move is
local, regional, macro, or global before repair?
```

Required focus cases:

```text
large/XL Step7F repair-window failures:
79, 51, 76, 99, 91

comparison cases:
small/medium shape-policy success
medium MIB/group-heavy case
boundary-heavy case if available
```

Required comparison:

```text
predicted locality class
actual Step7F repair behavior
routing decision
safe improvement retained
```

Required decision:

```text
promote_locality_routing_to_step7c
or pivot_to_coarse_region_planner
or pivot_to_macro_level_move_generator
or pivot_to_visualization_or_trace_repair
or inconclusive_due_to_prediction_quality
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
