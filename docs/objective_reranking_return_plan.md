# Objective-Aware Reranking Return Plan

Date: 2026-04-25 Asia/Taipei

## Decision

Resume the Step4/Step5 objective-aware reranking/scorer path and pause the
Step6 transfer-policy widening branch. The current evidence says the candidate
pool already contains better solutions than the best-untrained baseline, while
Step6 is blocked on LOCO transfer and case-leakage risk.

## Implemented Gate 0 / Gate 1 Surface

Added a reproducible runner:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_cost_aware_reranker.py \
  --case-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --k 16 \
  --scorers hpwl_bbox_proxy hpwl_bbox_soft_proxy hpwl_bbox_soft_repair_proxy \
            displacement_proxy oracle_official_cost \
  --top-m 1 2 4 8 \
  --output artifacts/research/cost_aware_reranker_v0.json
```

The runner restores the v0 candidate-source contract:

- `heuristic`
- `untrained_seed0..10`
- `bc_seed0`, `bc_seed1`
- `awbc_seed0`, `awbc_seed1`

It writes both JSON and markdown reports and now includes top-M oracle recall,
which is the gate for deciding whether a two-stage learned reranker is worth
trying.

## Evidence Snapshot

Existing v0 artifact evidence:

| Item | Value |
| --- | ---: |
| Validation slice | 20 cases (`0..19`) |
| Candidate count | 16 |
| Best untrained mean official cost | 19.784 |
| Best trained mean official cost | 23.651 |
| Oracle best-of-K mean official cost | 18.001 |
| Best proxy (`hpwl_bbox_soft_repair_proxy`) mean official cost | 18.955 |
| Best proxy regret to oracle | 0.953 |
| Displacement proxy mean official cost | 36.988 |

Key interpretation:

- `oracle_official_cost` beats best-untrained on 19/20 cases, so the candidate
  family still has selectable upside.
- `hpwl_bbox_soft_repair_proxy` already beats best-untrained by 0.829 mean cost,
  so objective-aware selection is immediately useful.
- Displacement is not a valid primary selector for this phase.
- The old trained-policy loss was mostly HPWL/area quality (`77.2%` of the mean
  gap), not runtime (`0.9%`).

## Gate 1: Top-M Oracle Recall from Existing v0 Artifact

The restored scorer formulas exactly match the existing v0 selected candidates
for all five scorers on all 20 cases. Recomputing top-M recall from
`artifacts/research/cost_aware_reranker_v0.json` gives:

| Scorer | top-1 | top-2 | top-4 | top-8 | Mean oracle rank by scorer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hpwl_bbox_proxy` | 0.250 | 0.600 | 0.750 | 0.900 | 3.30 |
| `hpwl_bbox_soft_proxy` | 0.350 | 0.550 | 0.750 | 0.950 | 3.15 |
| `hpwl_bbox_soft_repair_proxy` | 0.400 | 0.600 | 0.750 | 0.950 | 3.05 |
| `displacement_proxy` | 0.100 | 0.100 | 0.150 | 0.550 | 7.80 |
| `oracle_official_cost` | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 |

Gate interpretation: best proxy top-4 oracle recall is `0.750`, so the next
architecture can reasonably be a two-stage reranker over proxy top-M, instead of
jumping directly to a larger candidate portfolio.

## Next Work Package

1. Re-run the restored runner on validation `0..19` when a fresh full artifact is
   needed. The historical source run took about `2065s`, so use it deliberately.
2. Add a guarded offline/contest selection hook that chooses among K finalized
   candidates using the objective-aware proxy.
3. Prototype a top-M pairwise/listwise reranker over the proxy shortlist only.
4. Verify selected-vs-oracle-vs-best-untrained with HPWL/area/soft/runtime
   decomposition before widening K.

## Do Not Do Now

- Do not widen Step6 transfer-policy experiments to neutral10.
- Do not tune adversarial/fusion/loss weights to rescue Step6.
- Do not use repair displacement as the primary selection metric.
- Do not add case-specific or manual shortcut features.

## Implemented Guarded Selection Hook

Added a guarded contest/offline selection seam:

- `src/puzzleplace/scoring/proxy_scorer.py`
  - extracts target-independent proxy features from finalized candidate layouts:
    connectivity proxy, bbox area, soft/hard repair proxies, changed-block and
    fallback penalties.
  - selects the lowest `hpwl_bbox_soft_repair_proxy` score by default.
- `src/puzzleplace/optimizer/contest.py`
  - keeps legacy behavior when `objective_selection_k=1`.
  - enables portfolio selection when `objective_selection_k>1`.
  - records selection evidence in `last_report`: candidate count, selector name,
    selected source, selected index, and selected score.

Default behavior remains conservative: `ContestOptimizer()` still runs one
primary candidate. Objective-aware portfolio selection is opt-in, for example:

```python
ContestOptimizer(objective_selection_k=4)
```

This is the guarded hook needed before training a top-M reranker.

## Implemented Two-Stage LOCO Reranker

Added an offline top-M pairwise reranker audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_two_stage_reranker.py \
  --input artifacts/research/cost_aware_reranker_v0.json \
  --output artifacts/research/two_stage_reranker_loco_v0.json \
  --top-m 4
```

Result on the existing v0 artifact:

| Selector | Mean cost | Mean regret to oracle | Oracle hits | Notes |
| --- | ---: | ---: | ---: | --- |
| Proxy only (`hpwl_bbox_soft_repair_proxy`) | 18.955 | 0.953 | 8 | Stage-1 baseline |
| Two-stage pairwise LOCO, top-4 | 18.578 | 0.577 | 8 | Improves proxy by 0.377 mean cost |
| Oracle | 18.001 | 0.000 | 20 | Candidate-pool ceiling |

Interpretation: the top-M hypothesis passed a first LOCO audit. The pairwise
reranker improved 4 cases and hurt 2 cases versus the proxy-only selector, so it
should remain guarded rather than replacing the proxy by default. The next safe
step is a guarded top-M integration path that can fall back to proxy when the
learned ranker is low-confidence.
