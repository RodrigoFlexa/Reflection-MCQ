"""The preflight must say what broke and what to do, not only that it broke.

The FlashInfer case is the one that actually stopped a run: an installed
FlashInfer whose `_fd_ancillary` annotation subscripts `array.array`, which only
Python 3.12+ allows. Upstream fixed it by importing PEP 563 annotations; the
repair script backports exactly that one line into the active venv.
"""
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


def test_unrelated_failures_are_not_misdiagnosed():
    assert preflight.diagnose("TypeError: 'type' object is not subscriptable") == []
    assert preflight.diagnose("torch.OutOfMemoryError: CUDA out of memory") != []


def test_repair_adds_the_upstream_future_import_once():
    once = patched_source(UNPATCHED)
    assert b"from __future__ import annotations" in once
    assert once.splitlines()[1] == b"from __future__ import annotations"
    assert patched_source(once) == once  # idempotent: reruns change nothing


def test_repair_refuses_a_file_it_does_not_recognize():
    with pytest.raises(RuntimeError, match="no files changed"):
        patched_source(b'"""Doc."""\n\n\ndef something_else():\n    return 1\n')


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
