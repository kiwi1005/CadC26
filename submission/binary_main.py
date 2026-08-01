"""JSON stdin/stdout wrapper for official binary execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.optimizer import solve  # noqa: E402


FIELD_ORDER = (
    "block_count",
    "area_targets",
    "b2b_connectivity",
    "p2b_connectivity",
    "pins_pos",
    "constraints",
    "target_positions",
)


def main() -> int:
    payload = json.load(sys.stdin)
    args, kwargs = _decode_payload(payload)
    placements = solve(*args, **kwargs)
    json.dump({"placements": _jsonable_placements(placements)}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


def _decode_payload(payload: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise TypeError("JSON payload must be an object or argument list")
    if "args" in payload:
        return list(payload["args"]), dict(payload.get("kwargs", {}))
    if "kwargs" in payload:
        return [], dict(payload["kwargs"])
    args = [payload[name] for name in FIELD_ORDER if name in payload]
    return args, {}


def _jsonable_placements(placements: Any) -> list[list[float]]:
    return [[float(x), float(y), float(w), float(h)] for x, y, w, h in placements]


if __name__ == "__main__":
    raise SystemExit(main())
