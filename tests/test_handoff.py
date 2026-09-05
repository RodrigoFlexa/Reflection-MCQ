import json
from pathlib import Path

import pytest

from rmcq import handoff
import experiment_ops as ops


def test_handoff_roundtrip_includes_race_and_detects_corruption(tmp_path, monkeypatch):
    root = tmp_path / "gpu"
    target = tmp_path / "petrobras"
    experiment_id = "abcdef012345"
    files = [root / "data/processed/race/test.jsonl",
             root / f"experiment_exchange/{experiment_id}/manifest.json"]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{"question":"Leitura: ação?"}\n', encoding="utf-8")
    package = tmp_path / "package"
    monkeypatch.setattr(handoff, "CHUNK_BYTES", 64)
    metadata = handoff.pack(root, files, package, {"experiment_id": experiment_id, "stage": "prepare"})
    assert len(metadata["parts"]) > 1
    handoff.unpack(target, package, experiment_id, "prepare")
    for file in files:
        assert (target / file.relative_to(root)).read_bytes() == file.read_bytes()
    part = package / metadata["parts"][0]["name"]
    part.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Corrupt handoff part"):
        handoff.unpack(tmp_path / "damaged_target", package, experiment_id, "prepare")
    assert not (tmp_path / "damaged_target").exists()


def test_handoff_rejects_wrong_run_and_path_traversal(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    manifest = {"format": 1, "experiment_id": "abcdef012345", "stage": "prepare",
                "files": [{"path": "data/processed/race/../../../.env", "sha256": "wrong"}], "parts": []}
    (package / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        handoff.unpack(tmp_path / "target", package, "111111111111", "prepare")
    with pytest.raises(ValueError, match="Unsafe"):
        handoff.unpack(tmp_path / "target", package, "abcdef012345", "prepare")


def test_stage_environment_replays_limits_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "local-test-value")
    limits = dict(max_model_len=None, generation_seed=42, dtype="bfloat16", vllm_deterministic=True,
                  vllm_max_num_seqs=32, azure_max_tokens=1024, azure_reasoning_min_tokens=4000,
                  azure_reasoning_effort="low")
    env = ops.frozen_environment({"runtime_limits": limits})
    assert env["RMCQ_MAX_MODEL_LEN"] == "none"
    assert env["RMCQ_AZURE_REASONING_MIN_TOKENS"] == "4000"
    assert env["RMCQ_VLLM_DETERMINISTIC"] == "1"
    assert env["AZURE_OPENAI_API_KEY"] == "local-test-value"
    assert "AZURE_OPENAI_API_KEY" not in limits


def test_share_refuses_incomplete_stage_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "ROOT", tmp_path)
    monkeypatch.setattr(ops, "STATE", tmp_path / ".run_state")
    monkeypatch.setattr(ops, "SHARED", tmp_path / "experiment_handoff")
    monkeypatch.setattr(ops, "git", lambda *args, **kwargs: pytest.fail("Git must not be called"))
    with pytest.raises(RuntimeError, match="not complete"):
        ops.share("abcdef012345", "prepare")


def test_final_azure_empty_is_audited_even_when_legacy_fail_fast_enabled():
    from threading import Lock
    from types import SimpleNamespace
    from rmcq.backends.azure import AzureBackend
    from rmcq.backends.base import Generation
    backend = AzureBackend.__new__(AzureBackend)
    backend._lock = Lock()
    backend._calls = backend._empties = backend._content_filters = 0
    backend.audit_empty_outputs = True
    backend._check_empty(Generation(text="", finish_reason="stop"), SimpleNamespace(usage=None))
    assert backend._calls == backend._empties == 1
