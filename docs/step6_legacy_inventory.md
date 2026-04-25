# Step6 Legacy Inventory

Step6G-P 已在 commit `17744b7` checkpoint。Step7 期間先保留 Step6 sidecar 可追溯性，不做大規模搬檔。

## Frozen Step6 sidecar groups

- Step6G/H: puzzle policy, virtual frame, PNG visualization.
- Step6I/J: boundary frame commitment and predicted-hull ownership.
- Step6K/L: boundary failure attribution and selective hull stabilization.
- Step6M/N/O: move library, metric pathology, guarded selector replay.
- Step6P: original-inclusive Pareto alternative selector and visualizations.

## Cleanup policy for Step7

- 不再擴張 `src/puzzleplace/research/*` 的 Step6 sidecar，除非是 bug fix 或 compatibility shim。
- 新 Step7 檔案優先放入 `src/puzzleplace/diagnostics`, `selection`, `alternatives` 等 AGENT.md 指定目錄。
- Step6 artifacts 仍在 `artifacts/research/step6*`，因 `.gitignore` 不進 commit。
- 若未來要搬到 `research/legacy_step6/`，必須先建立 import compatibility map 並跑 full pytest。

## Current Step7 entry

第一刀：Step7A Aspect Pathology and Role-Aware Shape Diagnosis。
