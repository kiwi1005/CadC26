from __future__ import annotations

import json

from hcfp.cli import main


def test_demo_runs_bounded_cpu_lane(capsys) -> None:
    code = main(
        [
            "demo",
            "--device",
            "cpu",
            "--candidates",
            "2",
            "--steps",
            "1",
            "--projection-steps",
            "4",
            "--beam",
            "1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["feasible"] is True
    assert payload["block_count"] == 6
    assert len(payload["placements"]) == 6
