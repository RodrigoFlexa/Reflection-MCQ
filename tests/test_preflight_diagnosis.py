"""The preflight must say what broke and what to do, not only that it broke.

The FlashInfer case is the one that actually stopped a run: an installed
FlashInfer whose `_fd_ancillary` annotation subscripts `array.array`, which only
Python 3.12+ allows. Upstream fixed it by importing PEP 563 annotations; the
repair script backports exactly that one line into the active venv.
"""
import json
from pathlib import Path
import textwrap

import pytest

import validation_preflight as preflight
import run_experiment as run
from repair_flashinfer_annotations import patched_source

TRACEBACK = textwrap.dedent("""\
    Traceback (most recent call last):
      File "/home/rodrigo.flexa/Reflection-MCQ/venv/lib/python3.10/site-packages/vllm/__init__.py", line 9, in <module>
        from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
      File "/home/rodrigo.flexa/Reflection-MCQ/venv/lib/python3.10/site-packages/flashinfer/comm/fd_exchange.py", line 59, in <module>
        def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
    TypeError: 'type' object is not subscriptable
    """)

UNPATCHED = textwrap.dedent('''\
    """File descriptor exchange."""

    import array
    import socket


    def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
        return ((socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd])),)
    ''').encode()


def test_flashinfer_failure_is_named_with_its_fix():
    advice = preflight.diagnose(TRACEBACK)
    assert len(advice) == 1
    assert "repair_flashinfer_annotations.py" in advice[0]


def test_the_guard_message_is_diagnosed_too():
    """The preflight guard stops it earlier than vLLM does; both must be named."""
    guarded = ("RuntimeError: FlashInfer fd_exchange annotations are incompatible with this "
               "Python. Run: python tools/repair_flashinfer_annotations.py")
    advice = preflight.diagnose(guarded)
    assert len(advice) == 1 and "repair_flashinfer_annotations.py" in advice[0]


def test_unrelated_failures_are_not_misdiagnosed():
    assert preflight.diagnose("TypeError: 'type' object is not subscriptable") == []
    assert preflight.diagnose("torch.OutOfMemoryError: CUDA out of memory") != []


def test_a_failure_before_the_model_loop_is_written_to_the_error_file(tmp_path, monkeypatch, capsys):
    """This is the gap the FlashInfer stop exposed: it reached only the log."""
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    def explode(args):
        raise RuntimeError("FlashInfer fd_exchange annotations are incompatible with this Python.")

    monkeypatch.setattr(preflight, "run", explode)
    monkeypatch.setattr("sys.argv", ["validation_preflight.py", "--gpu", "4", "--part", "p1"])
    with pytest.raises(RuntimeError):
        preflight.main()
    recorded = json.loads((tmp_path / ".run_state/validation_preflight_error.p1.json").read_text())
    assert recorded["stage"] == "preflight" and recorded["part"] == "p1" and recorded["gpu"] == "4"
    assert "Traceback" in recorded["output"]
    assert "repair_flashinfer_annotations.py" in recorded["advice"][0]
    assert "DIAGNOSIS:" in capsys.readouterr().out


def test_a_recorded_model_failure_is_not_overwritten_by_the_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)

    def already_recorded(args):
        preflight.record_failure(preflight.error_log_path(None), "CUDA out of memory",
                                 stage="model_check", model="phi2")
        raise preflight.PreflightFailure("phi2 failed")

    monkeypatch.setattr(preflight, "run", already_recorded)
    monkeypatch.setattr("sys.argv", ["validation_preflight.py"])
    with pytest.raises(preflight.PreflightFailure):
        preflight.main()
    recorded = json.loads((tmp_path / ".run_state/validation_preflight_error.json").read_text())
    assert recorded["stage"] == "model_check" and recorded["model"] == "phi2"


def test_repair_adds_the_upstream_future_import_once():
    once = patched_source(UNPATCHED)
    assert b"from __future__ import annotations" in once
    assert once.splitlines()[1] == b"from __future__ import annotations"
    assert patched_source(once) == once  # idempotent: reruns change nothing


def test_repair_refuses_a_file_it_does_not_recognize():
    with pytest.raises(RuntimeError, match="no files changed"):
        patched_source(b'"""Doc."""\n\n\ndef something_else():\n    return 1\n')


class FakeHub:
    """Stands in for HfApi: names a few repos this token may not read."""

    def __init__(self, blocked=(), broken=()):
        self.blocked, self.broken, self.seen = set(blocked), set(broken), []

    def model_info(self, repo):
        self.seen.append(repo)
        if repo in self.blocked:
            from huggingface_hub.errors import GatedRepoError
            # The constructor's signature varies across huggingface_hub versions
            # and wants a real httpx response; the class is what we match on.
            error = GatedRepoError.__new__(GatedRepoError)
            error.args = ("403 Client Error. Cannot access gated repo",)
            raise error
        if repo in self.broken:
            raise ConnectionError("proxy unreachable")
        return {"id": repo}


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub
    holder = {}

    def install(fake):
        holder["fake"] = fake
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake)
        return fake

    return install


def test_gated_repos_are_all_named_before_any_weights_load(hub):
    from rmcq.config import MODELS
    fake = hub(FakeHub(blocked={MODELS["llama3.2-3b"].repo_id}))
    with pytest.raises(RuntimeError) as failure:
        preflight.check_repo_access(["phi2", "llama3.1-8b", "llama3.2-3b"])
    message = str(failure.value)
    # Every repo is asked about, so one run names every missing grant.
    assert len(fake.seen) == 3
    assert "huggingface.co/meta-llama/Llama-3.2-3B-Instruct" in message
    assert "per repository" in message
    assert preflight.diagnose(message), "the gated case must carry its advice"


def test_a_network_problem_does_not_block_a_cached_server(hub, capsys):
    from rmcq.config import MODELS
    hub(FakeHub(broken={MODELS["phi2"].repo_id}))
    report = preflight.check_repo_access(["phi2", "llama3.1-8b"])
    assert report["blocked"] == [] and "phi2" in report["undetermined"]
    assert "could not confirm Hub access for phi2" in capsys.readouterr().out


def test_only_the_two_meta_llama_repos_are_gated_in_the_grid():
    """Documents which checkpoints need a grant, so the runbook stays honest."""
    from rmcq.config import MODELS
    needs_grant = {key for key in run.VALIDATION_MODELS
                   if MODELS[key].repo_id.startswith("meta-llama/")}
    assert needs_grant == {"llama3.1-8b", "llama3.2-3b"}


def test_each_partition_checks_its_models_and_the_judge():
    for part, models in run.VALIDATION_PARTITIONS.items():
        checked = preflight.models_to_check(part)
        assert set(models) <= set(checked)
        assert run.DEFAULT_JUDGE in checked
        assert not set(checked) - set(models) - {run.DEFAULT_JUDGE}
    assert preflight.models_to_check(None) == list(run.VALIDATION_MODELS)


def test_unknown_partition_is_rejected():
    with pytest.raises(ValueError, match="Unknown partition"):
        preflight.models_to_check("p9")


def test_preflight_reports_are_written_per_partition():
    assert run.part_suffix(None) == ""
    assert Path(f"validation_preflight{run.part_suffix('p1')}.json").name == "validation_preflight.p1.json"
