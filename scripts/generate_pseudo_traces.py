#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.data import adapt_validation_batch
from puzzleplace.trajectory import compare_positions, generate_pseudo_traces, replay_trace
from scripts.download_smoke import FLOORSET_ROOT, _auto_approve_downloads, _ensure_import_paths


def main() -> None:
    _ensure_import_paths()
    _auto_approve_downloads()
    from iccad2026_evaluate import get_validation_dataloader

    batch = next(iter(get_validation_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1)))
    case = adapt_validation_batch(batch, case_id="validation-0")
    traces = generate_pseudo_traces(case, max_traces=4)
    payload = []
    for trace in traces:
        replayed = replay_trace(case, trace)
        payload.append(
            {
                "trace": trace.name,
                "ordered_blocks": trace.ordered_blocks,
                "action_count": len(trace.actions),
                "comparison": compare_positions(case, replayed),
            }
        )
    print(json.dumps({"case_id": case.case_id, "trace_count": len(payload), "traces": payload}, indent=2))


if __name__ == "__main__":
    main()
