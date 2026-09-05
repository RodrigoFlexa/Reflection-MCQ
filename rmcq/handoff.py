"""Portable, hash-verified handoffs split into Git-friendly compressed parts."""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

CHUNK_BYTES = 20 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack(root: Path, files: list[Path], destination: Path, metadata: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(set(files))
    if not files:
        raise ValueError("No files to share")
    manifest = {**metadata, "format": 1, "files": [], "parts": []}
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "payload.tar.gz"
        with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as tar:
                for path in files:
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f"Not a regular file: {path}")
                    relative = path.resolve().relative_to(root.resolve()).as_posix()
                    manifest["files"].append({"path": relative, "sha256": sha256(path)})
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        tar.addfile(info, source)
        with archive.open("rb") as source:
            index = 0
            while chunk := source.read(CHUNK_BYTES):
                path = destination / f"payload.part{index:04d}"
                path.write_bytes(chunk)
                manifest["parts"].append({"name": path.name, "sha256": sha256(path), "bytes": len(chunk)})
                index += 1
    # Publish the index last: a partial package is not a completed handoff.
    (destination / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def unpack(root: Path, source: Path, experiment_id: str, stage: str) -> dict:
    manifest = json.loads((source / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or manifest.get("experiment_id") != experiment_id or manifest.get("stage") != stage:
        raise ValueError("Handoff identity does not match the requested run/stage")
    allowed = (f"experiment_exchange/{experiment_id}/", "data/processed/race/",
               f"data/results/reflection_top1/{experiment_id}/")
    expected = {}
    for entry in manifest["files"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name or not name.startswith(allowed):
            raise ValueError(f"Unsafe archive path: {name}")
        if name in expected:
            raise ValueError(f"Duplicate archive path: {name}")
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Archive target escapes repository: {name}")
        expected[name] = entry["sha256"]
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        archive = temporary / "payload.tar.gz"
        with archive.open("wb") as output:
            for index, part in enumerate(manifest["parts"]):
                if part["name"] != f"payload.part{index:04d}":
                    raise ValueError("Unexpected handoff part name")
                path = source / part["name"]
                if sha256(path) != part["sha256"]:
                    raise ValueError(f"Corrupt handoff part: {path.name}")
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for entry in tar:
                if not entry.isfile() or entry.name not in expected or entry.name in seen:
                    raise ValueError(f"Unexpected archive entry: {entry.name}")
                staged = temporary / "checked" / entry.name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as stream, staged.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                if sha256(staged) != expected[entry.name]:
                    raise ValueError(f"Corrupt handoff file: {entry.name}")
                seen.add(entry.name)
        if seen != set(expected):
            raise ValueError("Archive is missing files")
        # All checks finish before any repository files are changed.
        for name in sorted(seen):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = temporary / "checked" / name
            replacement = target.with_name(target.name + ".importing")
            shutil.copyfile(staged, replacement)
            replacement.replace(target)
    return manifest
