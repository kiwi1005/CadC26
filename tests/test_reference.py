from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "src" / "hcfp" / "reference.py"
SCRIPT_PATH = ROOT / "scripts" / "audit_floorset_v10.py"


def load_reference():
    spec = importlib.util.spec_from_file_location("reference_under_test", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reference = load_reference()


def run(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def make_fake_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    checkout = tmp_path / "FloorSet"
    evaluator = checkout / "iccad2026contest" / "iccad2026_evaluate.py"
    evaluator.parent.mkdir(parents=True)
    (checkout / "helpers").mkdir()
    (checkout / "helpers" / "__init__.py").write_text("from .math_ops import area\n", encoding="utf-8")
    (checkout / "helpers" / "math_ops.py").write_text("def area():\n    return 1\n", encoding="utf-8")
    evaluator.write_text(
        "import helpers\nfrom helpers import math_ops\nfrom . import local_util\n",
        encoding="utf-8",
    )
    (evaluator.parent / "__init__.py").write_text("", encoding="utf-8")
    (evaluator.parent / "local_util.py").write_text("VALUE = 1\n", encoding="utf-8")
    run(["git", "init"], checkout)
    run(["git", "config", "user.email", "test@example.com"], checkout)
    run(["git", "config", "user.name", "Test User"], checkout)
    run(["git", "add", "."], checkout)
    run(["git", "commit", "-m", "fake floorset"], checkout)
    head = run(["git", "rev-parse", "HEAD"], checkout)
    digest = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    return checkout, head, digest


def fake_spec(head: str, digest: str):
    return reference.ReferenceSpec(
        repo_url="https://example.invalid/FloorSet.git",
        commit=head,
        evaluator_path="iccad2026contest/iccad2026_evaluate.py",
        evaluator_sha256=digest,
    )


def test_official_spec_is_pinned():
    spec = reference.OFFICIAL_FLOORSET_V10
    assert spec.repo_url == "https://github.com/IntelLabs/FloorSet.git"
    assert spec.commit == "aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa"
    assert spec.evaluator_sha256 == "64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498"
    assert spec.evaluator_path == "iccad2026contest/iccad2026_evaluate.py"


def test_audit_reference_checkout_discovers_local_import_closure(tmp_path: Path):
    checkout, head, digest = make_fake_checkout(tmp_path)
    audit = reference.audit_reference_checkout(checkout, fake_spec(head, digest))

    assert audit.head == head
    assert audit.clean
    assert audit.evaluator_sha256 == digest
    assert audit.import_paths == (
        "helpers/__init__.py",
        "helpers/math_ops.py",
        "iccad2026contest/__init__.py",
        "iccad2026contest/iccad2026_evaluate.py",
        "iccad2026contest/local_util.py",
    )


def test_audit_rejects_wrong_head(tmp_path: Path):
    checkout, _head, digest = make_fake_checkout(tmp_path)
    spec = fake_spec("0" * 40, digest)

    try:
        reference.audit_reference_checkout(checkout, spec)
    except reference.ReferenceAuditError as exc:
        assert "does not match pinned commit" in str(exc)
    else:
        raise AssertionError("wrong HEAD was accepted")


def test_audit_rejects_dirty_checkout(tmp_path: Path):
    checkout, head, digest = make_fake_checkout(tmp_path)
    (checkout / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    try:
        reference.audit_reference_checkout(checkout, fake_spec(head, digest))
    except reference.ReferenceAuditError as exc:
        assert "checkout is not clean" in str(exc)
    else:
        raise AssertionError("dirty checkout was accepted")


def test_audit_rejects_wrong_evaluator_hash(tmp_path: Path):
    checkout, head, _digest = make_fake_checkout(tmp_path)

    try:
        reference.audit_reference_checkout(checkout, fake_spec(head, "0" * 64))
    except reference.ReferenceAuditError as exc:
        assert "does not match pinned hash" in str(exc)
    else:
        raise AssertionError("wrong evaluator hash was accepted")


def test_cli_reports_failure_for_mismatched_official_pin(tmp_path: Path):
    checkout, _head, _digest = make_fake_checkout(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--checkout", str(checkout), "--json"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "does not match pinned commit" in payload["error"]


def test_cli_rejects_fetch_cache_outside_repo(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--fetch-cache",
            "--cache-dir",
            str(tmp_path / "outside-cache"),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "cache dir must stay inside this repo" in payload["error"]
