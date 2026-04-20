可以。下面是一套可以直接丟給 coding agents / research agents 的 **分工式任務指令集**。我會把它設計成「多 agent 可並行、但有清楚依賴順序」的形式。

先提醒一點：你上傳的 contest PDF 是 v7，但官方 GitHub contest README 目前指向 v9，且 2026-04-19 changelog 已把 **fixed-shape / preplaced 改成 hard constraints**，soft violation 只保留 **boundary / grouping / MIB**；所以開發時應以官方 repo 的最新 evaluator 為準。官方 repo 也說 contest framework 內含 `iccad2026_evaluate.py`、`optimizer_template.py`、`training_example.py`，並提供 `get_training_dataloader()` / `get_validation_dataloader()` 自動從 Hugging Face 下載資料。([GitHub][1])
你上傳的 PDF 也確認 FloorSet-Lite 有 1M training samples、100 validation samples、hidden test、block 數 21–120，並且 evaluator 會檢查 hard feasibility 與 soft violations。

---

# 0. 全部 agent 共用的總指令

下面這段建議複製到每個 agent 任務的最前面。

```text
你正在參與一個研究原型系統開發，題目是 Instruction-Aware Puzzle Placement for Data-Driven Floorplanning。

目標不是直接回歸 final coordinates，而是建立一個 action-based、constraint-aware、instruction-aware 的 sequential placement policy。

你要遵守以下規則：

1. 以官方 IntelLabs/FloorSet repo 的 iccad2026contest evaluator 為準。
2. 不要 reverse-engineer dataset generator，也不要寫任何利用資料生成規則作弊的邏輯。
3. 所有 module 必須能在小樣本 smoke test 上跑通，再擴展到大資料。
4. 優先支援 FloorSet-Lite rectangular block setting。
5. 所有 code 必須可測試、可重現、可被後續 agents 接續。
6. 不要把 hard constraints 只寫成 loss；hard constraints 必須至少有 legality checker / action mask / verifier。
7. 研究主線是：
   data adapter → constraint/evaluator wrapper → action schema → pseudo trajectory → role labels → typed policy model → BC training → rollout → feedback refinement → contest optimizer integration。
8. 每完成一個任務，必須輸出：
   - 新增/修改的檔案列表
   - 執行命令
   - 通過的測試
   - 下一個 agent 可以依賴的 API contract
```

---

# 1. 建議 repo 結構

請第一個 agent 建立這個結構。官方 repo 放在 `external/FloorSet/`，我們自己的研究 code 放在 `src/puzzleplace/`。

```text
floorset-puzzle/
  external/
    FloorSet/                       # official repo, do not modify unless wrapping
  src/
    puzzleplace/
      __init__.py
      data/
        __init__.py
        floorset_adapter.py
        schema.py
      geometry/
        __init__.py
        boxes.py
        constraints.py
        legality.py
      actions/
        __init__.py
        schema.py
        executor.py
        candidates.py
        masks.py
      roles/
        __init__.py
        weak_labels.py
      trajectory/
        __init__.py
        pseudo.py
        replay.py
        negative_sampling.py
      models/
        __init__.py
        encoders.py
        policy.py
        losses.py
      train/
        __init__.py
        train_bc.py
        dataset_bc.py
      rollout/
        __init__.py
        greedy.py
        beam.py
      eval/
        __init__.py
        official.py
        metrics.py
        reports.py
      feedback/
        __init__.py
        rewards.py
        replay_buffer.py
        improve_bc.py
  scripts/
    setup_smoke.py
    download_smoke.py
    inspect_batch.py
    generate_pseudo_traces.py
    train_bc_small.py
    rollout_validate.py
  tests/
    test_data_adapter.py
    test_geometry.py
    test_action_schema.py
    test_pseudo_replay.py
    test_policy_shapes.py
  configs/
    bc_small.yaml
    rollout_small.yaml
  artifacts/
    data_cache/
    traces/
    checkpoints/
    reports/
  pyproject.toml
  README.md
```

---

# 2. Agent 0：建立開發環境與下載 smoke dataset

## 給 Agent 0 的指令

```text
任務名稱：Environment Bootstrap + FloorSet Smoke Download

你的目標：
建立可重現的 Python 開發環境，clone 官方 FloorSet repo，安裝 contest dependencies，確認 validation/training dataloader 可以啟動，並建立最小 smoke scripts。

背景：
官方 contest README 指出資料路徑預設是 FloorSet root，也就是 iccad2026contest/ 的上一層；training 在 LiteTensorData/，validation 在 LiteTensorDataTest/。get_training_dataloader() 和 get_validation_dataloader() 會自動從 Hugging Face 下載資料。

請做以下事情：

1. 建立專案根目錄：
   floorset-puzzle/

2. Clone 官方 repo 到：
   external/FloorSet/

3. 建立 Python venv：
   .venv/

4. 安裝官方依賴：
   external/FloorSet/iccad2026contest/requirements.txt

5. 額外安裝研究開發工具：
   pytest
   ruff
   black
   mypy
   pydantic
   networkx
   pandas
   matplotlib
   tqdm
   pyyaml
   rich

6. 建立 pyproject.toml，設定 pytest / ruff / black 基本規則。

7. 建立 scripts/download_smoke.py：
   - 切換或插入 external/FloorSet/iccad2026contest 到 sys.path
   - import get_validation_dataloader, get_training_dataloader
   - 先載 validation dataloader
   - 印出第一筆 batch 的欄位、tensor shape、block_count
   - 再用 get_training_dataloader(batch_size=1, num_samples=10) 載 10 筆 training sample
   - 不要一開始下載或 iterate 全部 1M samples

8. 建立 README.md 的 Environment section：
   - 如何建立 venv
   - 如何安裝依賴
   - 如何執行 smoke download
   - data 會放在哪裡

9. 建立 .gitignore：
   - .venv/
   - external/FloorSet/LiteTensorData/
   - external/FloorSet/LiteTensorDataTest/
   - artifacts/
   - __pycache__/
   - *.pt
   - *.ckpt

10. 執行 smoke test：
    python scripts/download_smoke.py

完成定義：
- `python scripts/download_smoke.py` 可以成功讀到 validation 第一筆資料。
- training dataloader 可以用 num_samples=10 讀取，不必完整掃 1M。
- README 有清楚 setup 指令。
- 不修改官方 repo 原始檔案。
```

## 建議命令

```bash
mkdir -p floorset-puzzle
cd floorset-puzzle

git clone https://github.com/IntelLabs/FloorSet.git external/FloorSet

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r external/FloorSet/iccad2026contest/requirements.txt
pip install pytest ruff black mypy pydantic networkx pandas matplotlib tqdm pyyaml rich
```

官方 README 的 quick start 也是先 clone repo、建立 venv、安裝 `iccad2026contest/requirements.txt`，再進入 `iccad2026contest` 執行 evaluator。([GitHub][1])

---

# 3. Agent 1：建立 FloorSet data adapter

## 給 Agent 1 的指令

```text
任務名稱：FloorSet Data Adapter

你的目標：
把官方 FloorSet dataloader 的 raw batch 包成我們自己的乾淨資料結構，讓後面 agents 不必直接依賴官方 batch 格式。

依賴：
Agent 0 已完成 external/FloorSet clone 和 dataloader smoke test。

請新增：

src/puzzleplace/data/schema.py
src/puzzleplace/data/floorset_adapter.py
tests/test_data_adapter.py
scripts/inspect_batch.py

請實作以下 dataclasses：

1. FloorSetCase
   fields:
   - case_id: str | int
   - block_count: int
   - area_targets: torch.Tensor        # [N]
   - b2b_edges: torch.Tensor           # [E_b2b, 3], columns = block_i, block_j, weight
   - p2b_edges: torch.Tensor           # [E_p2b, 3], columns = pin_i, block_i, weight
   - pins_pos: torch.Tensor            # [P, 2]
   - constraints: torch.Tensor         # [N, 5], columns = fixed, preplaced, mib, cluster, boundary
   - target_positions: torch.Tensor | None   # [N, 4], x,y,w,h if label available
   - metrics: torch.Tensor | None
   - raw: dict | None

2. ConstraintColumns enum 或常數：
   FIXED = 0
   PREPLACED = 1
   MIB = 2
   CLUSTER = 3
   BOUNDARY = 4

3. Boundary code constants:
   LEFT = 1
   RIGHT = 2
   TOP = 4
   BOTTOM = 8
   TOP_LEFT = 5
   TOP_RIGHT = 6
   BOTTOM_LEFT = 9
   BOTTOM_RIGHT = 10

4. 函數：
   - load_validation_cases(max_cases: int | None) -> list[FloorSetCase]
   - load_training_cases(num_samples: int, batch_size: int = 1) -> Iterator[FloorSetCase]
   - batch_to_case(batch, split: str, index: int) -> FloorSetCase

5. target_positions extraction：
   官方 label 可能是 polygon / fp_sol / tree_sol 結構，請用官方 evaluator 的 baseline extraction 邏輯作參考，但不要 copy 成不可維護的大段程式。
   對 FloorSet-Lite，target solution 應該可轉為 [x, y, w, h]。
   若解析失敗，target_positions 設為 None，但 scripts/inspect_batch.py 必須明確 print warning。

6. scripts/inspect_batch.py：
   - 讀 validation 第 0 筆
   - 印 block_count
   - 印 constraints 每欄的非零數量
   - 印 b2b/p2b edge 數量
   - 印 target_positions 是否成功解析
   - 印前 5 個 block 的 area/constraints/target position

7. tests/test_data_adapter.py：
   - validation 至少能讀一筆
   - block_count == number of non -1 area_targets
   - constraints shape second dim == 5
   - pins_pos shape second dim == 2
   - b2b_edges / p2b_edges 若非空，shape second dim == 3

完成定義：
- pytest tests/test_data_adapter.py 通過。
- scripts/inspect_batch.py 可以在 validation case 0 上印出合理資訊。
- 後續 agents 可以只 import FloorSetCase，不碰官方 raw batch。
```

---

# 4. Agent 2：建立 official evaluator wrapper 與 geometry kernel

## 給 Agent 2 的指令

```text
任務名稱：Official Evaluator Wrapper + Geometry Legality Kernel

你的目標：
建立我們自己的 evaluation / legality API，並與官方 iccad2026_evaluate.py 對齊。這一層會被 action executor、rollout、training reward 共同使用。

依賴：
Agent 1 的 FloorSetCase。

請新增：

src/puzzleplace/geometry/boxes.py
src/puzzleplace/geometry/legality.py
src/puzzleplace/geometry/constraints.py
src/puzzleplace/eval/official.py
tests/test_geometry.py

請實作：

A. Box utilities
1. box_area(box)
2. box_center(box)
3. bbox_area(positions)
4. pairwise_overlap_area(box_i, box_j)
5. count_overlaps(positions, eps=1e-6)
6. touches_edge(box, bbox, boundary_code, eps=1e-6)
7. abuts(box_i, box_j, eps=1e-6)
8. shared_edge_length(box_i, box_j, eps=1e-6)

Box format 統一為：
(x, y, w, h)

B. Hard constraint checker
函數：
check_hard_constraints(case: FloorSetCase, positions: Tensor | list[tuple]) -> HardConstraintReport

HardConstraintReport:
- feasible: bool
- overlap_violations: int
- area_violations: int
- dimension_violations: int
- messages: list[str]

注意：
- soft blocks: w*h 必須在 area target 1% tolerance 內。
- fixed-shape blocks: w,h 必須等於 target_positions 的 w,h。
- preplaced blocks: x,y,w,h 必須等於 target_positions。
- overlap > eps 才算 overlap；touching edge 可以。

C. Soft constraint checker
函數：
check_soft_constraints(case, positions) -> SoftConstraintReport

SoftConstraintReport:
- boundary_violations
- grouping_violations
- mib_violations
- total_soft_violations
- max_possible_violations
- violations_relative

注意：
以官方最新 contest README 為準，soft constraints 只計 boundary / grouping / MIB，不要把 fixed/preplaced 放進 soft violation。

D. Official evaluator wrapper
src/puzzleplace/eval/official.py 實作：
- evaluate_positions(case, positions, runtime=1.0) -> dict
  儘量呼叫或對齊官方 evaluate_solution。
- compare_with_official(case, positions) -> dict
  對同一組 positions 比較我們的 hard/soft report 和官方 metrics。

E. Tests
1. two non-overlap boxes -> count_overlaps == 0
2. touching boxes -> overlap == 0, abuts == True
3. overlapping boxes -> overlap > 0
4. square soft block with correct area -> no area violation
5. intentionally wrong area -> area violation
6. validation case target_positions 若可得，官方 target positions 應該 feasible 或接近 feasible

完成定義：
- pytest tests/test_geometry.py 通過。
- evaluate_positions 對 validation case 0 的 target_positions 可跑出可讀 metrics。
- README 補充 hard/soft constraint definitions。
```

官方最新 contest README 明確列出 hard constraints 包含 no overlap、soft-block area tolerance、fixed/preplaced dimension immutability；soft constraints 是 grouping、MIB、boundary，且 cost 公式包含 HPWL gap、area gap、soft violation 與 runtime factor。([GitHub][1])

---

# 5. Agent 3：設計 typed action schema 與 action executor

## 給 Agent 3 的指令

```text
任務名稱：Typed Puzzle Placement Action Schema

你的目標：
建立可解釋的 action API，讓後面 pseudo trajectory、policy decoder、rollout 都使用同一套 action schema。

依賴：
Agent 1 FloorSetCase
Agent 2 geometry/legality API

請新增：

src/puzzleplace/actions/schema.py
src/puzzleplace/actions/executor.py
tests/test_action_schema.py

請定義 enums：

Primitive:
- SELECT_BLOCK
- SET_SHAPE
- EQUALIZE_SHAPE
- SNAP_TO_BOUNDARY
- PULL_TO_PIN
- ATTACH
- ALIGN_CLUSTER
- PLACE_RELATIVE
- FREEZE
- NOOP

EntityType:
- BLOCK
- GROUP
- MIB_GROUP
- CLUSTER_GROUP
- TERMINAL
- BOUNDARY
- CANVAS
- NONE

RelationType:
- LEFT_OF
- RIGHT_OF
- ABOVE
- BELOW
- SAME_WIDTH
- SAME_HEIGHT
- SAME_SHAPE
- ALIGN_LEFT
- ALIGN_RIGHT
- ALIGN_TOP
- ALIGN_BOTTOM
- ALIGN_CENTER_X
- ALIGN_CENTER_Y
- TOUCH_LEFT
- TOUCH_RIGHT
- TOUCH_TOP
- TOUCH_BOTTOM
- NONE

請定義 dataclasses：

Action:
- primitive: Primitive
- arg1_type: EntityType
- arg1_id: int | tuple | None
- arg2_type: EntityType
- arg2_id: int | tuple | None
- relation: RelationType
- params: dict[str, float | int | str]
- source: str = "unknown"
- score: float | None = None

BoardState:
- case: FloorSetCase
- positions: list[tuple[float, float, float, float] | None]
- placed: list[bool]
- frozen: list[bool]
- step: int
- history: list[Action]

請在 executor.py 實作：

1. apply_action(state, action) -> BoardState
2. validate_action_preconditions(state, action) -> bool / report
3. explain_action(state, action) -> str
4. action_to_token(action) -> dict
5. token_to_action(token) -> Action

每個 primitive 的 MVP 行為：

SELECT_BLOCK:
- 不改 geometry，只記錄 selected block，可放在 params["selected_block"]

SET_SHAPE:
- 對 soft block 設定 w,h，使 w*h 接近 area target
- fixed/preplaced 必須用 target shape

EQUALIZE_SHAPE:
- 對 MIB group 使用 reference block shape，或用 params 中的 w,h

SNAP_TO_BOUNDARY:
- 將 block 放到目前 bbox 的某個 edge/corner，若 bbox 尚不存在，允許建立 provisional bbox

PULL_TO_PIN:
- 將 block center 放到 pin 附近，但必須經過 legality projection 或標記為 provisional

ATTACH:
- 將 arg1 block 貼到 arg2 block 的某一側，shared edge length > 0

ALIGN_CLUSTER:
- 對 cluster/group 中已放置 blocks 做軸向 alignment

PLACE_RELATIVE:
- 通用 relative placement fallback

FREEZE:
- 將 block/group 標記 frozen

NOOP:
- 只用於 padding，不應在 rollout 中主動選

完成定義：
- action 可以 serialize / deserialize。
- ATTACH touching case 通過測試。
- SNAP_TO_BOUNDARY 能產生 touches boundary 的位置。
- FREEZE 後該 block 不可被非 repair action 改動。
- 所有 action 都能輸出 human-readable explanation。
```

---

# 6. Agent 4：建立 weak role labeler

## 給 Agent 4 的指令

```text
任務名稱：Weak Role Labeling for Puzzle Placement

你的目標：
用 graph / constraint / final layout 統計建立 weak role labels，讓模型可以先學 block semantics。

依賴：
Agent 1 FloorSetCase
Agent 2 geometry
Agent 3 action schema

請新增：

src/puzzleplace/roles/weak_labels.py
tests/test_roles.py
scripts/inspect_roles.py

請定義 roles：

Role:
- ANCHOR
- HUB
- BOUNDARY_SEEKER
- SHAPE_LOCKED
- GROUP_LEADER
- FOLLOWER
- CHAIN_LINK
- FREE_SPACE_RESOLVER
- UNASSIGNED

請實作：

compute_block_features(case, target_positions=None) -> dict[str, Tensor]
features 至少包含：
- area
- normalized area
- b2b weighted degree
- p2b weighted degree
- graph degree
- approximate betweenness 或簡化 centrality
- fixed flag
- preplaced flag
- mib id
- cluster id
- boundary code
- target x,y,w,h if available
- target center if available
- bbox edge contact if available

assign_weak_roles(case, target_positions=None) -> list[set[Role]]

Role rules 初版：

ANCHOR:
- preplaced OR fixed OR top 10% area OR top 10% weighted degree

HUB:
- top 10% b2b weighted degree

BOUNDARY_SEEKER:
- boundary_code != 0 OR p2b weighted degree top 20%

SHAPE_LOCKED:
- fixed OR preplaced OR mib_id != 0

GROUP_LEADER:
- cluster_id != 0 且在同 cluster 中 degree 或 area 最大

FOLLOWER:
- cluster_id != 0 且非 group leader

CHAIN_LINK:
- high betweenness 或連接多個 clusters 的 block

FREE_SPACE_RESOLVER:
- target_positions available 時，若 block 位於 final bbox 內部且 degree low，可標記為 late-stage filler

scripts/inspect_roles.py：
- 對 validation case 0 印出每個 role 的 block count
- 印出 top hub / anchor block ids
- 若 target_positions 可得，畫出或輸出簡單 role table CSV 到 artifacts/reports/

測試：
- fixed/preplaced block 必須有 SHAPE_LOCKED
- boundary block 必須有 BOUNDARY_SEEKER
- cluster group 至少有一個 GROUP_LEADER

完成定義：
- role labels 可穩定產生。
- role labeler 不依賴 hidden test label。
- role labeler 可以在沒有 target_positions 時退化運作。
```

---

# 7. Agent 5：建立 pseudo-trajectory generator

## 給 Agent 5 的指令

```text
任務名稱：Pseudo-Trajectory Generation from Final Layouts

你的目標：
從 FloorSet final target_positions 反推出多條可 replay 的 semantic action trajectories，作為 behavior cloning 資料。

依賴：
Agent 1 FloorSetCase
Agent 2 geometry/evaluator
Agent 3 action schema/executor
Agent 4 roles

請新增：

src/puzzleplace/trajectory/pseudo.py
src/puzzleplace/trajectory/replay.py
src/puzzleplace/trajectory/negative_sampling.py
scripts/generate_pseudo_traces.py
tests/test_pseudo_replay.py

核心要求：
不要只產生一條 trajectory。每個 case 至少產生以下策略中的 3 種：

1. anchor_first
   preplaced/fixed → boundary → hub → group leader → followers → remaining

2. group_first
   MIB equalize → group leader → group attach → align/freeze cluster → remaining

3. wire_first
   high p2b external degree → high b2b hub → chain links → followers

4. reverse_peeling
   從 final layout 反向找 leaf/follower/低約束 block 移除，再 reverse 成 constructive order

5. constraint_priority
   fixed/preplaced shape → MIB → boundary → grouping → HPWL follower

請實作：

generate_pseudo_trajectories(case, max_traces=5) -> list[Trajectory]

Trajectory dataclass:
- case_id
- strategy_name
- actions: list[Action]
- metadata: dict

replay_trajectory(case, trajectory) -> ReplayReport:
- final_positions
- hard_feasible
- soft_report
- reconstruction_l1_error
- action_success_rate
- failed_steps

action extraction rules：
- fixed/preplaced/fixed-shape block:
  SET_SHAPE 或 PLACE_RELATIVE with target exact shape/position
- MIB group:
  EQUALIZE_SHAPE(group, reference)
- boundary block:
  SNAP_TO_BOUNDARY(block, boundary_code)
- group member that touches another group member in final layout:
  ATTACH(block, target_block, side/relation)
- block with strong p2b edge:
  PULL_TO_PIN(block, terminal)
- blocks in same row/column/edge:
  ALIGN_CLUSTER
- after stable group:
  FREEZE

重要：
pseudo trajectory 可以使用 target_positions 作為 teacher information，這是 training data label extraction；但最終 rollout / validation optimizer 不可以直接使用 validation target_positions 作為答案。

negative_sampling.py：
對每個 expert state/action 產生 negatives：
- same primitive wrong target
- right block wrong side
- legal but worse target
- illegal overlap action
- violates MIB shape
- violates boundary

輸出格式：
artifacts/traces/{split}_{case_id}_{strategy}.jsonl
每行包含：
- state summary 或 state id
- action token
- legal candidates if available
- negative candidates
- metadata

測試：
- validation case 0 可生成至少 3 條 trajectory
- 每條 trajectory 可 replay
- hard feasibility report 可輸出
- expert action serialization roundtrip works

完成定義：
- scripts/generate_pseudo_traces.py --split validation --max-cases 5 可以生成 traces。
- Candidate coverage 初版 report 可列出。
- replay report 存到 artifacts/reports/pseudo_replay_validation.json。
```

這一步是整個研究的核心之一：你要把 final answer dataset 轉成 behavior dataset。官方資料提供 final optimal layouts，但並沒有提供 semantic action trace，所以這裡的 trajectory extraction 是你的研究貢獻來源之一；官方 README 也提醒不應 reverse-engineer generator，而是發展 genuine algorithmic solutions。([GitHub][2])

---

# 8. Agent 6：建立 candidate generator 與 action masks

## 給 Agent 6 的指令

```text
任務名稱：Candidate Grounding and Typed Action Masks

你的目標：
為每個 BoardState 產生 typed legal action candidates，並建立 action mask。這是 imitation 和 rollout 的共同基礎。

依賴：
Agent 1 data
Agent 2 legality
Agent 3 action schema/executor
Agent 5 pseudo trajectories

請新增：

src/puzzleplace/actions/candidates.py
src/puzzleplace/actions/masks.py
tests/test_candidates.py
scripts/check_candidate_coverage.py

請實作：

generate_candidates(state, primitive=None, max_per_primitive=K) -> list[Action]

每個 primitive 的 candidates：

SET_SHAPE:
- unplaced block
- shape candidates:
  - square sqrt(area)
  - target shape if fixed/preplaced and target_positions exists
  - MIB reference shape if MIB group partially placed

EQUALIZE_SHAPE:
- each MIB group with inconsistent or unassigned shapes
- reference = first placed member or target/reference candidate

SNAP_TO_BOUNDARY:
- blocks with boundary_code != 0
- all required edge/corner candidates

PULL_TO_PIN:
- top weighted p2b edges for unplaced block
- candidate terminal ids sorted by weight

ATTACH:
- unplaced block to placed block
- prioritize same cluster, high b2b edge, geometric final adjacency if training mode
- candidate sides: LEFT/RIGHT/ABOVE/BELOW
- filter if immediately overlaps frozen blocks

ALIGN_CLUSTER:
- cluster_id groups with at least 2 placed blocks
- align left/right/top/bottom/center options

FREEZE:
- placed blocks/groups that satisfy local constraints

PLACE_RELATIVE:
- fallback relative placement candidates around placed anchors

請實作 masks：

1. primitive_mask(state)
2. block_mask(state, primitive)
3. target_mask(state, primitive, selected_block)
4. relation_mask(state, primitive, selected_block, target)
5. param_mask 或 param candidate list

請實作 coverage evaluation：

check_candidate_coverage.py：
- 讀 pseudo trajectories
- 對每個 state/action 產生 candidates
- 檢查 expert action 是否在 candidate set
- 輸出 coverage by primitive:
  - overall coverage
  - primitive-specific coverage
  - missing examples

完成定義：
- candidate coverage 初版 > 95%，理想 > 98%。
- illegal candidates 不應大量出現。
- masks 可被後面 model loss 使用。
```

---

# 9. Agent 7：建立 graph/state encoder 與 typed policy decoder

## 給 Agent 7 的指令

```text
任務名稱：Typed Graph Policy Model

你的目標：
實作第一版可訓練的 neural policy：
FloorSetCase + BoardState → typed action distribution。

依賴：
Agent 1 data schema
Agent 3 action schema
Agent 4 roles
Agent 6 candidate/masks

請新增：

src/puzzleplace/models/encoders.py
src/puzzleplace/models/policy.py
src/puzzleplace/models/losses.py
tests/test_policy_shapes.py

設計要求：

A. State features

Block features:
- area target
- normalized area
- placed flag
- frozen flag
- current x,y,w,h if placed else zeros
- fixed/preplaced/mib/cluster/boundary flags/codes
- b2b weighted degree
- p2b weighted degree
- role multi-hot vector

Terminal features:
- x,y
- total connected weight

Graph edges:
- b2b edges
- p2b edges
初版可以不使用 PyTorch Geometric，直接用 padded dense tensors 或 simple message passing。

B. Encoder

請先做 MVP：
- MLP block encoder
- MLP terminal encoder
- simple message passing:
  block receives weighted messages from connected blocks and terminals
- optional Transformer encoder over block tokens

C. Decoder

Factored action decoder：

1. primitive_head:
   p(primitive | state)

2. arg1_block_pointer:
   p(block_id | state, primitive)

3. arg2_target_pointer:
   p(target_id | state, primitive, arg1)
   target 可以是 block / terminal / group / boundary candidate

4. relation_head:
   p(relation | state, primitive, arg1, arg2)

5. param_head:
   discrete bins first，不要一開始直接 continuous regression

D. Mask support

所有 CE loss 都必須支援 masks：
- invalid primitive logits = -inf
- invalid block logits = -inf
- invalid target logits = -inf
- invalid relation logits = -inf

E. Losses

實作：
- primitive CE
- arg1 CE
- arg2 CE
- relation CE
- param CE / MSE
- role auxiliary BCE if role labels available
- validity auxiliary BCE optional
- candidate ranking loss optional

F. Tests

test_policy_shapes.py：
- 從 validation case 0 建立 initial BoardState
- generate candidates/masks
- forward model
- logits shape 正確
- masked loss 可 backward
- no NaN

完成定義：
- policy forward pass 可跑。
- 一筆 pseudo action 可計算 BC loss。
- loss.backward() 成功。
```

---

# 10. Agent 8：建立 behavior cloning 訓練 pipeline

## 給 Agent 8 的指令

```text
任務名稱：Behavior Cloning Training Pipeline

你的目標：
用 pseudo trajectories 訓練 typed action policy，先完成 small-scale BC baseline。

依賴：
Agent 5 pseudo traces
Agent 6 candidates/masks
Agent 7 policy model

請新增：

src/puzzleplace/train/dataset_bc.py
src/puzzleplace/train/train_bc.py
scripts/train_bc_small.py
configs/bc_small.yaml

Dataset 要求：

BCStepExample:
- case_id
- state representation 或 replay prefix
- expert action token
- candidates
- masks
- negative candidates optional
- role labels optional
- metadata:
  - primitive
  - strategy_name
  - step index
  - block_count

訓練流程：

1. 讀 artifacts/traces/*.jsonl
2. 對每個 example 重建 BoardState
3. generate masks/candidates
4. model forward
5. compute BC losses
6. log:
   - primitive accuracy
   - arg1 top-1/top-5
   - arg2 top-1/top-5
   - relation accuracy
   - legal top-1
   - total loss
7. 每 N steps 存 checkpoint 到 artifacts/checkpoints/

配置檔 bc_small.yaml：
- train_cases: 1000
- val_cases: 100
- max_traces_per_case: 3
- batch_size: 8
- lr: 1e-3
- epochs: 10
- hidden_dim: 128
- use_roles: true
- use_masks: true
- device: cuda_if_available

scripts/train_bc_small.py：
- 載 config
- 檢查 traces 是否存在；若不存在，提示先跑 generate_pseudo_traces.py
- 訓練 small model
- 儲存 best checkpoint
- 輸出 artifacts/reports/bc_small_metrics.json

Baselines：
請支援 config flags：
- use_roles=false
- use_masks=false
- use_multi_trace=false

完成定義：
- 可以在 100 cases / 3 traces 上完成一次 overfit sanity test。
- train loss 下降。
- primitive accuracy 高於 random。
- checkpoint 可 load。
```

---

# 11. Agent 9：建立 rollout engine：greedy + beam

## 給 Agent 9 的指令

```text
任務名稱：Policy Rollout Engine

你的目標：
把訓練好的 typed policy 變成可以從空盤面產生完整 floorplan 的 rollout system。

依賴：
Agent 2 evaluator
Agent 3 executor
Agent 6 candidates/masks
Agent 7 policy
Agent 8 checkpoint

請新增：

src/puzzleplace/rollout/greedy.py
src/puzzleplace/rollout/beam.py
scripts/rollout_validate.py
configs/rollout_small.yaml

請實作：

A. greedy_rollout(case, model, max_steps=None) -> RolloutReport

流程：
1. 初始化 BoardState，preplaced/fixed blocks 可先透過 required action 或 initial placement 處理。
2. 每步 generate candidates/masks。
3. model score candidates。
4. 選 top-1 legal action。
5. apply_action。
6. 每步跑 cheap hard check；每 K 步跑 full checker。
7. 結束條件：
   - all blocks placed
   - max_steps reached
   - no legal candidates
   - infeasible unrecoverable

B. beam_rollout(case, model, beam_width=4)

每步保留 top-k BoardState：
score = model logprob + lambda_reward * heuristic_progress_score - penalty * violations

C. RolloutReport:
- final_positions
- actions
- hard_report
- soft_report
- hpwl_gap if available
- area_gap if available
- cost if official evaluator available
- runtime
- failure_reason
- repair_count
- action_counts_by_primitive

D. scripts/rollout_validate.py：
- load checkpoint
- run validation cases 0..N
- compare greedy vs beam
- save:
  artifacts/reports/rollout_validation.json
  artifacts/reports/action_trace_case0.txt

E. MVP constraints：
- 若 model 還不穩，允許 fallback:
  - choose best legal heuristic action
  - but report fallback_count
- fallback_count 必須作為 metric，不可隱藏。

完成定義：
- validation case 0 可從空 state rollout 到 positions。
- evaluate_positions 可跑。
- greedy/beam 都有 report。
- action trace 可讀。
```

---

# 12. Agent 10：建立 metric dashboard 與 ablation runner

## 給 Agent 10 的指令

```text
任務名稱：Metrics, Reports, and Ablation Runner

你的目標：
建立標準化研究報告輸出，不只看 final cost，也看 behavior metrics。

依賴：
Agent 8 training
Agent 9 rollout

請新增：

src/puzzleplace/eval/metrics.py
src/puzzleplace/eval/reports.py
scripts/run_ablation_small.py

請實作以下 metrics：

A. Single-step imitation metrics
- primitive accuracy
- arg1 top-1/top-5
- arg2 top-1/top-5
- relation accuracy
- legal top-1 rate
- candidate coverage
- masked CE

B. Rollout metrics
- full feasible rate
- hard failure type count
- overlap violations
- area violations
- dimension violations
- boundary violation
- grouping violation
- MIB violation
- violations_relative
- HPWLgap
- Areagap_bbox
- cost
- runtime
- action count
- repair count
- fallback count
- greedy-vs-beam gap

C. Scaffold dependency metrics
- repair_dependency_ratio
- fallback_dependency_ratio
- rule_override_rate if available
- beam_gain

D. Ablation runner

run_ablation_small.py 支援：
1. no_role
2. no_mask
3. single_trace_only
4. greedy_only
5. beam_width_4
6. beam_width_8
7. no_feedback

輸出：
artifacts/reports/ablation_small.csv
artifacts/reports/ablation_small.md

完成定義：
- 可以對 small validation subset 產生比較表。
- 報表清楚分開 behavior metrics 和 final floorplanning metrics。
```

---

# 13. Agent 11：建立 feedback reward 與 offline improvement MVP

## 給 Agent 11 的指令

```text
任務名稱：Verifier Feedback and Offline Improvement MVP

你的目標：
在 BC policy 已能 rollout 後，建立第一版 feedback-driven improvement。先不要做複雜 PPO，先做 advantage-weighted behavior cloning 或 IQL-like in-sample improvement。

依賴：
Agent 9 rollout
Agent 10 metrics

請新增：

src/puzzleplace/feedback/rewards.py
src/puzzleplace/feedback/replay_buffer.py
src/puzzleplace/feedback/improve_bc.py
scripts/collect_rollout_buffer.py
scripts/train_awbc.py

請實作：

A. Dense reward components

compute_step_reward(prev_state, action, next_state, case) -> RewardBreakdown

RewardBreakdown:
- hard_feasible_delta
- overlap_margin_delta
- area_validity_delta
- boundary_progress_delta
- grouping_progress_delta
- mib_progress_delta
- hpwl_proxy_delta
- bbox_area_delta
- step_penalty
- total

注意：
hard violation 不應只靠 reward 懲罰；hard violation 要由 legality layer 阻擋或 terminal penalty。

B. Replay buffer schema

Transition:
- case_id
- state_token or serialized state summary
- action_token
- reward
- reward_breakdown
- next_state_token
- done
- final_cost
- hard_feasible
- behavior_logprob if available

C. collect_rollout_buffer.py
- load BC checkpoint
- rollout validation/train subset
- collect transitions
- label final cost and reward
- save artifacts/replay/buffer_small.jsonl

D. Advantage-weighted BC

train_awbc.py:
- 讀 expert pseudo transitions + rollout transitions
- 對高 reward / low final cost actions 加權
- loss = weight * BC_loss
- weight = exp(advantage / temperature)，但要 clip
- 加 KL-to-BC 或 behavior cloning anchor，避免 collapse

E. Evaluation
- compare BC checkpoint vs AWBC checkpoint
- feasibility 不可下降
- final cost / soft violations 應改善

完成定義：
- small buffer 可收集。
- AWBC 訓練可跑。
- 報告中比較 BC vs AWBC。
```

---

# 14. Agent 12：整合成 contest optimizer

## 給 Agent 12 的指令

```text
任務名稱：Contest Optimizer Integration

你的目標：
把 trained puzzle policy 包成官方 contest evaluator 可以呼叫的 optimizer file。

依賴：
Agent 9 rollout
Agent 8/11 checkpoint
Agent 2 official evaluator

請新增：

src/puzzleplace/eval/contest_optimizer.py
scripts/export_optimizer.py
artifacts/submission/my_puzzle_optimizer.py

請做：

1. 實作一個 class PuzzlePolicyOptimizer，繼承官方 FloorplanOptimizer 或符合官方 loader 需求。

2. solve signature 必須與官方相容：
   solve(
     block_count,
     area_targets,
     b2b_connectivity,
     p2b_connectivity,
     pins_pos,
     constraints,
     target_positions=None
   ) -> list[tuple[x, y, w, h]]

3. solve 裡不能使用 validation/test target_positions 作為答案。
   target_positions 只能用於 fixed/preplaced hard requirement，如果官方傳入它是為了 constraint checking / dimensions，必須只用 required hard fields。

4. 載入 checkpoint：
   - checkpoint path 可由環境變數 PUZZLEPLACE_CKPT 指定
   - 若沒有 checkpoint，fallback 到 simple legal heuristic，但必須 print warning

5. 使用 greedy 或 beam rollout：
   - default beam_width=4
   - runtime 控制：若 block_count 大，beam_width 自動降低

6. 輸出 positions list：
   - length == block_count
   - 每個元素是 float x,y,w,h
   - area tolerance satisfied
   - fixed/preplaced dimensions satisfied
   - no None

7. scripts/export_optimizer.py：
   - 把必要 import path 和 checkpoint config 打包
   - 生成 artifacts/submission/my_puzzle_optimizer.py

8. 驗收命令：
   cd external/FloorSet/iccad2026contest
   python iccad2026_evaluate.py --evaluate ../../../artifacts/submission/my_puzzle_optimizer.py --test-id 0
   python iccad2026_evaluate.py --validate ../../../artifacts/submission/my_puzzle_optimizer.py

完成定義：
- 官方 --test-id 0 可以跑。
- 官方 --validate 通過格式檢查。
- 若模型失敗，fallback 產生 feasible-ish result，但報告 fallback。
```

官方 contest README 指出提交實作是在 optimizer file 內實作 `solve()`，返回每個 block 的 `(x, y, width, height)`，並可用 `--evaluate`、`--test-id`、`--validate`、`--save-solutions` 等命令驗證。([GitHub][1])

---

# 15. Agent 13：建立 CI / regression tests

## 給 Agent 13 的指令

```text
任務名稱：Testing and Regression Guardrails

你的目標：
確保每個核心 module 都有 regression tests，避免後續 agents 改壞 action schema、geometry 或 evaluator。

依賴：
所有前面 modules。

請新增或整理：

tests/
  test_data_adapter.py
  test_geometry.py
  test_action_schema.py
  test_candidates.py
  test_pseudo_replay.py
  test_policy_shapes.py
  test_rollout_smoke.py
  test_official_wrapper.py

請建立 scripts/run_all_smoke_tests.sh：

內容：
1. pytest tests/test_geometry.py
2. pytest tests/test_action_schema.py
3. pytest tests/test_data_adapter.py
4. python scripts/inspect_batch.py --split validation --case-id 0
5. python scripts/generate_pseudo_traces.py --split validation --max-cases 2 --max-traces 3
6. pytest tests/test_pseudo_replay.py
7. python scripts/train_bc_small.py --config configs/bc_small.yaml --max-cases 20 --epochs 1
8. python scripts/rollout_validate.py --max-cases 2

請建立 artifacts/reports/smoke_summary.md，自動記錄：
- commit hash
- date
- number of cases
- passed tests
- key metrics

完成定義：
- 一條 command 可以跑完最小 smoke test。
- 任何重大錯誤會 fail fast。
```

---

# 16. Agent 14：建立研究實驗手冊

## 給 Agent 14 的指令

```text
任務名稱：Research Experiment Playbook

你的目標：
寫一份讓研究者可以照著跑的實驗手冊，包含 environment、資料下載、pseudo trace、BC training、rollout、ablation、contest integration。

請新增：

docs/
  00_environment.md
  01_dataset.md
  02_action_schema.md
  03_pseudo_trajectory.md
  04_behavior_cloning.md
  05_rollout_eval.md
  06_feedback_improvement.md
  07_ablation_plan.md
  08_known_failure_modes.md

每份文件要包含：
- purpose
- commands
- expected outputs
- metrics
- common failure modes
- next steps

特別要求：
08_known_failure_modes.md 必須列：
- pseudo trajectory replay 不可行
- candidate coverage 太低
- primitive accuracy 高但 rollout 差
- no-mask invalid action 爆炸
- role labels noisy
- repair dependency 太高
- large block count collapse
- feedback reward hacking
- official evaluator mismatch

完成定義：
- 新人只看 docs/00 到 docs/05 可以完整跑出第一版 BC + rollout。
```

---

# 17. 建議 agent 執行順序

## Phase A：環境與資料

| 順序 | Agent                 | 是否可並行            | 完成後產物                             |
| -: | --------------------- | ---------------- | --------------------------------- |
|  0 | Environment Bootstrap | 否                | venv、official repo、download smoke |
|  1 | Data Adapter          | 否                | `FloorSetCase`                    |
|  2 | Evaluator + Geometry  | 可與 3 部分並行，但最好等 1 | hard/soft checker                 |

## Phase B：行為資料

| 順序 | Agent             | 是否可並行   | 完成後產物                       |
| -: | ----------------- | ------- | --------------------------- |
|  3 | Action Schema     | 等 2     | typed action API            |
|  4 | Role Labeler      | 等 1     | weak roles                  |
|  5 | Pseudo Trajectory | 等 2,3,4 | pseudo traces               |
|  6 | Candidate + Masks | 等 3,5   | legal candidates / coverage |

## Phase C：模型與訓練

| 順序 | Agent            | 是否可並行   | 完成後產物               |
| -: | ---------------- | ------- | ------------------- |
|  7 | Policy Model     | 等 6     | typed policy        |
|  8 | BC Training      | 等 5,6,7 | BC checkpoint       |
|  9 | Rollout Engine   | 等 7,8   | greedy/beam rollout |
| 10 | Metrics/Ablation | 可與 9 並行 | reports             |

## Phase D：改進與落地

| 順序 | Agent             | 是否可並行  | 完成後產物               |
| -: | ----------------- | ------ | ------------------- |
| 11 | Feedback/AWBC     | 等 9,10 | improved checkpoint |
| 12 | Contest Optimizer | 等 9    | official optimizer  |
| 13 | CI Smoke Tests    | 持續     | regression suite    |
| 14 | Experiment Docs   | 持續     | research playbook   |

---

# 18. 第一週 MVP 版本：不要一次做太大

建議先讓 agents 只做這個 MVP：

```text
資料：
- validation 5 cases
- training 100–1000 cases
- block count 21–40 優先

actions：
- SET_SHAPE
- EQUALIZE_SHAPE
- SNAP_TO_BOUNDARY
- ATTACH
- FREEZE

暫緩：
- PULL_TO_PIN
- ALIGN_CLUSTER
- offline RL
- complex beam
- full 1M training
```

## MVP 完成標準

| 模組                | 最小成功條件                                         |
| ----------------- | ---------------------------------------------- |
| environment       | 可以讀 validation / small training                |
| data adapter      | FloorSetCase 正確                                |
| geometry          | overlap / area / boundary / grouping / MIB 可檢查 |
| action schema     | action 可 replay / serialize                    |
| pseudo trajectory | 5 cases 可生成 3 條 trace                          |
| candidate mask    | expert action coverage > 95%                   |
| BC model          | small set overfit 成功                           |
| rollout           | validation case 0 可產生完整 positions              |
| evaluator         | 官方 `--test-id 0` 可跑                            |
| report            | 有 behavior metrics，不只 final cost               |

---

# 19. 最重要的執行注意事項

1. **資料下載先 smoke，不要一開始 full scan 1M。**
   官方提供 auto-download dataloader，但 training data 很大；先用 `num_samples=10/100/1000`。官方 README 說 training/validation dataloaders 會自動下載，且資料應放在 `FloorSet/LiteTensorData/` 和 `FloorSet/LiteTensorDataTest/`。([GitHub][1])

2. **fixed/preplaced 版本差異要鎖定。**
   你上傳的 v7 PDF 與官方目前 v9 README 在 hard/soft definition 上有差異。開發 evaluator wrapper 時，以官方最新 `iccad2026_evaluate.py` 為準，避免最後 validation mismatch。

3. **不要讓 repair 掩蓋 policy 能力。**
   可以有 repair，但一定要記錄 `repair_count`、`fallback_count`、`repair_dependency_ratio`。

4. **pseudo trajectory 不要只有一種 order。**
   至少 anchor-first、group-first、wire-first、reverse-peeling 三到五種，否則 BC 會學到單一 heuristic artifact。

5. **每個 agent 都要交付 API contract。**
   例如 `FloorSetCase`、`Action`、`BoardState`、`Trajectory`、`RolloutReport` 一旦定義好，後續 agent 不應任意改欄位。

6. **最後要能回答研究問題：模型到底學到了什麼？**
   所以 metrics 必須包含 action accuracy、candidate coverage、legal rate、rollout survival、constraint-specific violations、repair ratio、scaffold fading，不只是 contest cost。

---

# 20. 可以直接貼給總控 agent 的總任務

最後這段可以直接給一個 orchestrator agent。

```text
你是本專案的 orchestrator。請按以下順序分派任務並檢查完成條件：

Phase A:
1. Agent 0 建立 environment，clone FloorSet，安裝 dependencies，跑 download smoke。
2. Agent 1 建立 FloorSetCase data adapter。
3. Agent 2 建立 geometry legality 與 official evaluator wrapper。

Phase B:
4. Agent 3 建立 typed action schema 與 executor。
5. Agent 4 建立 weak role labeler。
6. Agent 5 從 final layouts 生成 multi-strategy pseudo trajectories。
7. Agent 6 建立 candidate generator 與 typed action masks，並報告 expert candidate coverage。

Phase C:
8. Agent 7 實作 graph/state encoder + typed action decoder。
9. Agent 8 實作 behavior cloning training。
10. Agent 9 實作 greedy/beam rollout。
11. Agent 10 實作 metrics、reports、ablation runner。

Phase D:
12. Agent 11 實作 verifier feedback 和 advantage-weighted BC improvement。
13. Agent 12 整合成官方 contest optimizer。
14. Agent 13 建立 smoke tests 和 regression tests。
15. Agent 14 寫研究實驗手冊。

每個 agent 完成後，你要檢查：
- 是否有新增/修改檔案列表
- 是否有可執行命令
- 是否有測試結果
- 是否符合 API contract
- 是否有 artifacts/reports 中的輸出
- 是否避免使用 validation/test target positions 作為直接答案
- 是否沒有 reverse-engineer dataset generator

第一個 milestone 是：
在 validation case 0–4 上完成 pseudo trajectory generation、BC overfit、greedy rollout、official evaluator smoke run。

不要直接追求 full score。先證明：
1. action schema 可 replay；
2. candidate coverage 高；
3. model 可以學 single-step action；
4. rollout 不立即崩潰；
5. evaluator wrapper 和官方 evaluator 對齊。
```

這套分工能讓 agents 從環境、資料、約束、action、trajectory、模型、訓練、rollout 到官方 evaluator integration 一步步落地，同時保留研究驗證所需的可觀測指標。

[1]: https://github.com/IntelLabs/FloorSet/blob/main/iccad2026contest/README.md "FloorSet/iccad2026contest/README.md at main · IntelLabs/FloorSet · GitHub"
[2]: https://github.com/IntelLabs/FloorSet "GitHub - IntelLabs/FloorSet · GitHub"
