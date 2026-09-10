#!/usr/bin/env python
"""Backport FlashInfer's fd_exchange annotation fix into the active GPU venv."""
from __future__ import annotations

import argparse
import array
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

UPSTREAM = "https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/fd_exchange.py"


def patched_source(source: bytes) -> bytes:
    tree = ast.parse(source)
    if any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
           and any(a.name == "annotations" for a in n.names) for n in tree.body):
        return source
    # Refuse unrelated versions: this backport addresses only the reported signature.
    known = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "_fd_ancillary" and n.returns is not None
             and ast.unparse(n.returns) == "tuple[tuple[int, int, array.array[int]]]"]
    if len(known) != 1:
        raise RuntimeError("Unrecognized fd_exchange source; no files changed. Inspect the installed FlashInfer version.")
    lines = source.splitlines(keepends=True)
    first = tree.body[0]
    docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    index = first.end_lineno if docstring else first.lineno - 1
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    lines.insert(index, b"from __future__ import annotations" + newline)
    result = b"".join(lines)
    compile(result, "fd_exchange.py", "exec", dont_inherit=True)
    return result


def installed_target() -> tuple[Path, str] | None:
    try:
        dist = importlib.metadata.distribution("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return None
    target = Path(dist.locate_file("flashinfer/comm/fd_exchange.py")).resolve()
    return (target, dist.version) if target.is_file() else None


def check_compatibility() -> dict:
    installed = installed_target()
    if installed is None:
        return {"fd_exchange": "absent"}
    target, version = installed
    source = target.read_bytes()
    report = {"version": version, "fd_exchange_sha256": hashlib.sha256(source).hexdigest()}
    try:
        array.array[int]
    except TypeError:
        if patched_source(source) != source:
            raise RuntimeError("FlashInfer fd_exchange annotations are incompatible with this Python. "
                               "Run: python repair_flashinfer_annotations.py") from None
    return report


def repair_file(target: Path) -> dict:
    original = target.read_bytes()
    updated = patched_source(original)
    before = hashlib.sha256(original).hexdigest()
    if updated == original:
        return {"status": "already_fixed", "path": str(target), "sha256": before}
    backup = target.with_name(target.name + ".rmcq-" + before[:12] + ".bak")
    if backup.exists():
        if backup.read_bytes() != original:
            raise RuntimeError(f"Backup differs from original: {backup}; no files changed")
    else:
        backup.write_bytes(original)
    target.write_bytes(updated)
    return {"status": "patched", "path": str(target), "backup": str(backup),
            "original_sha256": before, "sha256": hashlib.sha256(updated).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without changing installed files")
    args = parser.parse_args()
    if args.check:
        # --check answers a question, so a file that needs repair is a finding to
        # report, not a crash. Exit 1 keeps it usable from a script.
        try:
            print(json.dumps({"status": "ok", **check_compatibility()}, indent=2))
        except RuntimeError as exc:
            print(json.dumps({"status": "needs_repair", "reason": str(exc),
                              "fix": "python repair_flashinfer_annotations.py"}, indent=2))
            raise SystemExit(1) from None
        return
    installed = installed_target()
    if installed is None:
        raise RuntimeError("flashinfer-python fd_exchange.py was not found in this Python environment")
    target, version = installed
    if sys.prefix == sys.base_prefix or not target.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Activate the GPU virtual environment; refusing to modify files outside that venv")
    report = repair_file(target)
    report.update(version=version, python=sys.version, upstream=UPSTREAM)
    state = Path(__file__).resolve().parent / ".run_state/flashinfer_annotation_repair.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
