# Step6N Metric Semantics + Case Pathology + Selector Decision Trace

## Scope

Step6N is diagnostic only. It explains Step6M move-selection behavior and
proposes guard candidates, but does not apply new selector thresholds or runtime
integration.

## Metric semantics

The Step6M deltas are directional:

- `b_d > 0`: boundary satisfaction improved.
- `hpwl_d > 0`: HPWL proxy worsened.
- `bbox_d > 0`: bbox area increased.
- `soft_d < 0`: fewer soft boundary violations.

Step6N adds normalized deltas:

```text
hpwl_delta_norm = hpwl_delta / baseline_hpwl
bbox_delta_norm = bbox_delta / baseline_bbox_area
```

## Artifacts

- `artifacts/research/step6n_metric_semantics.json`
- `artifacts/research/step6n_case_pathology_report.json`
- `artifacts/research/step6n_selector_decision_trace.json`
- `artifacts/research/step6n_guard_calibration_candidates.json`
- `artifacts/research/step6n_profile_summary.md`
- `artifacts/research/step6n_suspicious_visualizations/`

## Current suspicious cases

Step6N flags 12 suspicious simple-compaction selections:

```text
8, 10, 14, 15, 21, 23, 28, 31, 32, 33, 37, 38
```

Important interpretation:

- Many high raw HPWL deltas are less severe after normalization.
- The strongest post-hoc guard is not raw HPWL alone.
- Spatial imbalance worsening is a recurring signal.
- HPWL regression per boundary gain flags cases `21, 23, 28, 37`.

## Scale coverage

The Step6M 40-case prefix covers:

```text
small: 20 cases
medium: 20 cases
large: 0 cases
xl: 0 cases
```

Large/XL reporting paths are emitted, but the current data has a coverage gap:

```text
large/xl absent from current Step6M 40-case prefix
```

## Guard candidates

Step6N proposes, but does not apply:

1. reject `simple_compaction` if normalized HPWL delta is too high;
2. reject if HPWL regression per boundary gain is too high;
3. reject if spatial balance worsens beyond threshold;
4. reject if normalized bbox regression is too high unless soft improvement dominates.

Current diagnostic recommendation:

```text
do not tune selector from raw hpwl_d only;
calibrate with normalized HPWL, boundary gain, and spatial imbalance together.
```
