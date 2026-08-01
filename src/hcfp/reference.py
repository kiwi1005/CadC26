"""Pinned FloorSet v10 reference checkout auditing for HCFP.

This module intentionally does not execute the official scorer.  It verifies the
provenance of a checkout and statically maps the local Python import closure for
the pinned evaluator.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import subprocess
from collections import deque
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class ReferenceSpec:
    repo_url: str
    commit: str
    evaluator_path: str
    evaluator_sha256: str


@dataclasses.dataclass(frozen=True)
class CheckoutAudit:
    root: Path
    head: str
    status_lines: tuple[str, ...]
    evaluator: Path
    evaluator_sha256: str
    import_paths: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.status_lines


OFFICIAL_FLOORSET_V10 = ReferenceSpec(
    repo_url="https://github.com/IntelLabs/FloorSet.git",
    commit="aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa",
    evaluator_path="iccad2026contest/iccad2026_evaluate.py",
    evaluator_sha256="64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498",
)


class ReferenceAuditError(RuntimeError):
    """Raised when a reference checkout fails provenance or hash validation."""


def run_git(checkout: Path, args: Iterable[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReferenceAuditError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def checkout_root(checkout: Path) -> Path:
    root = run_git(checkout, ["rev-parse", "--show-toplevel"])
    return Path(root).resolve()


def verify_checkout(checkout: Path, spec: ReferenceSpec = OFFICIAL_FLOORSET_V10) -> tuple[Path, str, tuple[str, ...]]:
    root = checkout_root(checkout)
    head = run_git(root, ["rev-parse", "HEAD"])
    if head != spec.commit:
        raise ReferenceAuditError(f"checkout HEAD {head} does not match pinned commit {spec.commit}")
    status = run_git(root, ["status", "--porcelain=v1"]).splitlines()
    if status:
        raise ReferenceAuditError("checkout is not clean:\n" + "\n".join(status))
    return root, head, tuple(status)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evaluator_hash(root: Path, spec: ReferenceSpec = OFFICIAL_FLOORSET_V10) -> tuple[Path, str]:
    evaluator = (root / spec.evaluator_path).resolve()
    try:
        evaluator.relative_to(root)
    except ValueError as exc:
        raise ReferenceAuditError(f"evaluator path escapes checkout: {spec.evaluator_path}") from exc
    if not evaluator.is_file():
        raise ReferenceAuditError(f"evaluator not found: {spec.evaluator_path}")
    actual = sha256_file(evaluator)
    if actual != spec.evaluator_sha256:
        raise ReferenceAuditError(
            f"evaluator SHA256 {actual} does not match pinned hash {spec.evaluator_sha256}"
        )
    return evaluator, actual


def discover_import_closure(root: Path, entry_path: str | Path) -> tuple[str, ...]:
    entry = (root / entry_path).resolve() if not Path(entry_path).is_absolute() else Path(entry_path).resolve()
    root = root.resolve()
    try:
        entry.relative_to(root)
    except ValueError as exc:
        raise ReferenceAuditError(f"entry path escapes checkout: {entry}") from exc
    if not entry.is_file():
        raise ReferenceAuditError(f"entry module not found: {entry}")

    seen: set[Path] = set()
    pending: deque[Path] = deque([entry])
    while pending:
        path = pending.popleft()
        if path in seen:
            continue
        seen.add(path)
        for module in _local_imports(root, path):
            if module not in seen:
                pending.append(module)

    return tuple(sorted(_relative_posix(root, path) for path in seen))


def audit_reference_checkout(
    checkout: Path,
    spec: ReferenceSpec = OFFICIAL_FLOORSET_V10,
) -> CheckoutAudit:
    root, head, status = verify_checkout(checkout, spec)
    evaluator, evaluator_hash = verify_evaluator_hash(root, spec)
    import_paths = discover_import_closure(root, evaluator)
    return CheckoutAudit(
        root=root,
        head=head,
        status_lines=status,
        evaluator=evaluator,
        evaluator_sha256=evaluator_hash,
        import_paths=import_paths,
    )


def fetch_reference_checkout(
    cache_dir: Path,
    spec: ReferenceSpec = OFFICIAL_FLOORSET_V10,
) -> Path:
    cache_dir = cache_dir.resolve()
    if cache_dir.exists():
        if not (cache_dir / ".git").exists():
            raise ReferenceAuditError(f"cache path exists but is not a git checkout: {cache_dir}")
        run_git(cache_dir, ["fetch", "--tags", "origin"])
    else:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["git", "clone", spec.repo_url, str(cache_dir)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ReferenceAuditError(f"git clone failed: {detail}")
    run_git(cache_dir, ["checkout", "--detach", spec.commit])
    return cache_dir


def _local_imports(root: Path, path: Path) -> tuple[Path, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise ReferenceAuditError(f"cannot parse Python module {path}: {exc}") from exc

    found: set[Path] = set()
    current_package = _package_name(root, path)
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import_base(current_package, node.module, node.level)
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
        for module_name in candidates:
            local_path = _module_to_path(root, module_name)
            if local_path is not None:
                found.add(local_path)
    return tuple(sorted(found))


def _package_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts.pop()
    return ".".join(parts)


def _resolve_from_import_base(current_package: str, module: str | None, level: int) -> str | None:
    if level == 0:
        return module
    package_parts = current_package.split(".") if current_package else []
    keep = len(package_parts) - level + 1
    if keep < 0:
        return module
    parts = package_parts[:keep]
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)


def _module_to_path(root: Path, module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    module_file = root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file.resolve()
    package_file = root / relative / "__init__.py"
    if package_file.is_file():
        return package_file.resolve()
    return None


def _relative_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
