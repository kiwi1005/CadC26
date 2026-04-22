# HELP

## 目前狀態摘要（evidence-backed）
- **這個資料夾目前更像研究/協作工作區，不是已啟動的實作 repo。**
  - 證據：root 僅見 `.omx/`、`.skills/`、`AGENT.md`、`GUIDE.md`、`HELP.md`、`andrej-karpathy-skills-main/`。
  - 證據：未找到 `package.json`、`pyproject.toml`、`go.mod`、`Cargo.toml`、`Makefile`、`tsconfig.json`、`requirements.txt`。
  - 證據：未找到 `src/`、`tests/`、`scripts/`、`configs/`、`app/` 目錄。

- **`AGENT.md` 存在但目前是空檔。**
  - 證據：`AGENT.md` 為 `0 bytes`。

- **`GUIDE.md` 是完整的研究驗證計畫，不是可直接執行的 build/run 指南。**
  - 證據：`GUIDE.md` 共 `1057` 行。
  - 證據：章節明確覆蓋 `Dataset 與 evaluation setting`、`E0–E7`、`MVP scope`、`Milestone 1–6`、`Final recommendation`。

- **Karpathy-guidelines 在此任務主要意味著：先盤點、避免臆測、避免先蓋大框架。**
  - 證據：`~/.codex/skills/karpathy-guidelines/SKILL.md` 強調 `Think Before Coding`、`Simplicity First`、`Surgical Changes`、`Goal-Driven Execution`。
  - 推論：在未確認目標前，最安全的工作是文件整理、缺口記錄、與下一步選單，而不是 speculative implementation。

- **目前不是 git repository。**
  - 證據：`git status` 回報 `fatal: not a git repository`。
  - 影響：目前不適合要求 worker 以 commit-based workflow 交付程式變更；若要持續多人協作，之後最好明確決定是否初始化 git。

## 可安全先做的低風險工作
1. **把 `GUIDE.md` 轉成 operator-friendly 的執行入口。**
   - 例如整理成：
     - milestone checklist
     - experiment matrix
     - open questions / unblock list

2. **補齊 repo-level 導航文件。**
   - 最低風險的是補充：
     - 非空 `AGENT.md`
     - `README.md`
     - 更精煉的 `HELP.md`

3. **先定義最小可交付，而不是先寫框架。**
   - 建議先選一個最小目標：
     - 僅整理研究計畫
     - 建立最小實驗 scaffold
     - 建立 evaluator / pseudo-trajectory prototype

## 目前缺的關鍵資訊
1. **此 repo 的主要產出是什麼？**
   - A. 研究計畫整理
   - B. 實驗骨架/scaffold
   - C. 真正開始做 dataset/model/evaluator 實作
   - D. 單純整理 skill / agent workflow

2. **`GUIDE.md` 接下來應扮演哪種角色？**
   - 保持研究 memo
   - 拆成 PRD / milestones / task list
   - 轉成 implementation spec

3. **`AGENT.md` 空白是否刻意？**
   - 若不是刻意，應補 repo-level working agreement、build/test 入口、與禁止事項。

4. **是否要把這個資料夾正式變成可協作 repo？**
   - 若答案是 yes，至少需要：
     - git init / 遠端策略
     - README
     - 選定技術棧
     - 最小可執行命令

## 建議下一步選單（從低風險到較高風險）
### Option 1 — 研究整理模式（最低風險）
- 目標：把 `GUIDE.md` 拆成 `milestones + metrics + open questions`。
- 輸出：純文件；不碰 speculative code。
- 適合現在：**是**。

### Option 2 — 最小實驗啟動模式
- 先補這四個答案再動手：
  1. dataset 來源
  2. 預計語言/框架
  3. 第一個可執行命令
  4. 第一個驗證目標（建議 `E0 pseudo-trajectory sanity check`）
- 適合現在：**只有在你明確要開始實作時才適合**。

### Option 3 — 協作/倉庫整理模式
- 先補：`README.md`、非空 `AGENT.md`、git repo 初始化策略。
- 好處：後續 worker 才有穩定入口與提交流程。
- 風險：需要你確認這個資料夾是否真的要成為長期 repo。

## 目前建議（依 Karpathy-guidelines）
- **先不要做大型 implementation。**
- **先把目標類型講清楚。**
- 在未確認之前，最合理的持續工作是：
  1. evidence-backed inventory
  2. 文件導覽/決策清單
  3. 將 `GUIDE.md` 轉成更可執行的任務分解

## 本次 HELP 更新重點
- 新增 evidence-backed 現況摘要：說明此資料夾目前是研究/協作工作區，而非已啟動的實作 repo。
- 新增 operator 視角的低風險工作清單：只建議文件整理、導覽補強、與最小交付定義。
- 新增明確 open questions：主產出、GUIDE.md 角色、AGENT.md 是否刻意留白、是否需要正式 repo 化。
- 新增 next-step menu：研究整理 / 最小實驗啟動 / 協作整理 三種路徑。
- 新增環境限制：此目錄目前不是 git repository，且本輪依指示不初始化 git。

## 本輪盤點依據
- `AGENT.md` — 空檔
- `GUIDE.md` — 研究驗證計畫主文件
- `HELP.md` — operator help / open questions
- `.omx/context/team-intake-20260420T111200Z.md` — team intake snapshot
- `~/.codex/skills/karpathy-guidelines/SKILL.md` — 行為約束來源
- root directory listing / manifest search / source-dir search / `git status` 結果

## 2026-04-21 實作注意事項（新增）
1. **OMX team runtime 在這個專案上多次出現 lane drift / stalled。**
   - 現況：Agent 0–6 最終大多改為 leader 本地接手完成。
   - 建議：後續如果任務邊界仍容易漂移，優先用單 lane 或直接本地實作，不要過度依賴 long-running worker pane。

2. **目前 milestone 1 的最大風險不是 bootstrap，而是 candidate coverage。**
   - 現況：Agent 6 報告顯示 `heuristic coverage = 0.1667`，而 `offline teacher-hint augmented coverage = 1.0`。
   - 解讀：測量/報告管線已建立，但 inference-time candidate family 還不夠好。

3. **validation/test 的 target positions 只能作為 offline supervision / pseudo-label source。**
   - 現況：Agent 1/2/5/6/7 都必須維持這個邊界。
   - 建議：executor / rollout / candidate inference path 不應直接讀 target positions 作答案。

4. **目前 package 匯入依賴 `PYTHONPATH=src` 或 editable install。**
   - 現況：測試與腳本都以 `PYTHONPATH=src ...` 驗證。
   - 建議：後續 commands 若遇到 import 問題，先檢查是否帶了 `PYTHONPATH=src`。

5. **Agent 0 training smoke 會觸發大體積下載。**
   - 現況：已成功，但會花時間與磁碟空間。
   - 建議：後續 smoke / report 先重用已下載資料，不要反覆重抓。

6. **目前最適合先證明的是 small-scope learnability，不是 final score。**
   - 建議順序：Agent 7 encoder/decoder → Agent 8 BC overfit → Agent 9 greedy rollout → 再回頭看 candidate heuristic 是否要補強。

7. **Agent 11 的 verifier feedback 若只看 legality，容易對 shifted negative 不敏感。**
   - 現況：最初的 step feedback 對 toy case 的負樣本仍給出與正樣本相同分數，因為偏移後仍合法且面積不變。
   - 修正：目前 feedback score 額外加入對 target reference 的 alignment penalty，才把正負樣本 advantage 拉開。
   - 影響：AWBC 現在仍屬 weak-but-usable weighting，不代表已經得到高品質 value model。

8. **Contest evaluator 要求的是“檔案型 optimizer entrypoint”，不是單純 package class。**
   - 現況：官方 `validate_submission()` / `ContestEvaluator` 直接吃 Python 檔案路徑，並在模組層找 `FloorplanOptimizer` 子類。
   - 修正：目前以 repo root 的 `contest_optimizer.py` 作為提交入口，再委派到 `src/puzzleplace/optimizer/contest.py`。

9. **Contest solve 需要完整位置列表；但目前 rollout baseline 仍常停在 partial placements。**
   - 現況：beam rollout 雖優於 greedy，但仍會留下未放完的 blocks。
   - 修正：contest optimizer 目前帶有 deterministic fallback packing，確保 submission format 完整。
   - 風險：這只是格式與 smoke robustness 補丁，不代表 final quality 已足夠。

10. **所有 contest / training scripts 都同時依賴 `src/` 和 `external/FloorSet/iccad2026contest/` import path。**
    - 現況：若少任一個 path，`puzzleplace.*` 或 `iccad2026_evaluate` 都會失敗。
    - 建議：後續新增 script 時，先檢查 path injection 與 `PYTHONPATH=src` 是否一致。

11. **AWBC checkpoint 載入是 soft dependency。**
    - 現況：若 `artifacts/models/agent11_awbc_policy.pt` 不存在，contest optimizer 會回退到 heuristic policy。
    - 建議：評估成績時要註明是 checkpoint mode 還是 heuristic fallback，避免混淆結果。

12. **全 repo 的 `ruff` 目前仍有既有 style debt，不適合直接當作 agent13 gate。**
    - 現況：初版 regression matrix 用 `ruff check src/puzzleplace tests`，結果被多個舊測試檔的長行/import 排序問題擋住。
    - 判斷：那些問題不是 Agent 11-14 這一輪引入，也不影響本輪功能 correctness。
    - 修正：agent13 matrix 改為檢查本輪實際變更與關聯檔案，避免把 pre-existing formatting debt 當作功能回歸。

13. **preplaced anchors seeded into rollout state 之後，不能再把 `FREEZE` 當成一般候選動作。**
    - 現況：architect review 發現 candidate generator 會對所有 placed blocks 都產生 `FREEZE`，包含已在 initial state 中 frozen 的 preplaced anchors。
    - 風險：beam/greedy 可能浪費 rollout budget 在 no-op freeze，然後把完成責任推給 fallback packing。
    - 修正：candidate generator 現在會跳過已在 `state.frozen_blocks` 中的 block。

14. **regression matrix 不應硬編碼 `.venv/bin/python`。**
    - 現況：architect review 指出這會讓 readiness check 綁死 repo-local venv，而不是使用使用者當前啟動的 interpreter。
    - 修正：matrix script 改用 `sys.executable`，讓驗證環境與實際執行環境一致。

15. **Sprint 2 的 semantic coverage 不能再用 strict exact-action equality 來算。**
    - 現況：一開始 semantic / relaxed candidate 其實已經生成了高 recall 的 intent candidates，但 coverage script 還是拿 teacher pseudo trace 的 exact `(x, y, w, h)` action 做逐字比對，導致 coverage 假性偏低。
    - 修正：現在 semantic / relaxed coverage 改成 `same block + compatible shape/intent family` 的 matching；strict / teacher-hint 仍維持 exact match。

16. **semantic candidate 的 violation estimation 若每個 candidate 都跑完整 profile，會讓 coverage script 非常慢。**
    - 現況：最初每個 candidate 都呼叫整體 `summarize_violation_profile()`，形成 `O(candidates × blocks^2)`。
    - 修正：現在只估 local overlap / boundary / connectivity proxy，而不是整個 provisional layout profile。

17. **`slots=True` dataclass 不能直接用 `.__dict__` 出報表。**
    - 現況：`ViolationProfile` / `RepairReport` 在 script 輸出時都踩到這個問題。
    - 修正：改用 `dataclasses.asdict()`。

18. **semantic rollout smoke 的主要耗時不是 rollout 本身，而是腳本內先做的小型 BC 訓練。**
    - 現況：`scripts/rollout_validate.py` 在 rollout 前會先訓練 policy；若 epochs 太高，整體 smoke 時間會明顯變慢。
    - 建議：Sprint 2 smoke 預設應採用較小 epoch，例如 `ROLLOUT_EPOCHS=5`，把 heavy training 留給單獨實驗腳本。
