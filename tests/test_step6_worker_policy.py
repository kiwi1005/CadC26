from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEP6_WORKER_SCRIPTS = [
    ROOT / "scripts" / "run_step6_hierarchical_rollout_control_audit.py",
    ROOT / "scripts" / "run_step6c_hierarchical_action_q_audit.py",
    ROOT / "scripts" / "run_step6d_rollout_return_label_diagnostic.py",
    ROOT / "scripts" / "run_step6e_rollout_label_stability.py",
    ROOT / "scripts" / "run_step6e_majority_advantage_ranker.py",
]


def _literal_keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.literal_eval(keyword.value)
    raise AssertionError(f"missing keyword {name!r} in {ast.unparse(call)}")


def _worker_arg_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        try:
            first_arg = ast.literal_eval(node.args[0])
        except (ValueError, SyntaxError):
            continue
        if first_arg == "--workers":
            calls.append(node)
    return calls


def test_step6_parallel_runners_default_to_serial_smoke_mode() -> None:
    for script in STEP6_WORKER_SCRIPTS:
        calls = _worker_arg_calls(script)
        assert len(calls) == 1, script
        call = calls[0]
        assert _literal_keyword(call, "default") == 1, script
        help_text = _literal_keyword(call, "help")
        assert "independent" in help_text, script
        assert "smoke" in help_text, script
        assert "48" in help_text, script


def test_step6_parallel_runners_cap_workers_by_independent_job_count() -> None:
    for script in STEP6_WORKER_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert "min(int(args.workers), len(jobs))" in text, script
        assert "default=48" not in text, script
