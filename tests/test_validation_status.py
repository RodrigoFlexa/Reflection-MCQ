import json
import os
import sys

import pytest

import experiment_ops as ops
import validation_ops as validation


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "ROOT", tmp_path)
    monkeypatch.setattr(ops, "STATE", tmp_path / ".run_state")
    monkeypatch.setattr(ops, "SHARED", tmp_path / "experiment_handoff")
    monkeypatch.setattr(sys, "argv", ["validation_ops.py", "status"])
    return ops.STATE


@pytest.mark.parametrize("status", ["starting", "running", "failed"])
def test_status_before_id_shows_job_without_using_old_run(state, capsys, status):
    ops.write(state / "job.json", {"stage": "validation-local", "status": status,
                                   "pid": os.getpid(), "log": ".run_state/validation-local.log"})
    (state / "validation_experiment_id").write_text("abcdef012345\n")
    ops.write(ops.SHARED / "current_validation.json", {"experiment_id": "111111111111"})
    validation.main()
    output = capsys.readouterr().out
    assert status in output and ".run_state/validation-local.log" in output
    assert "ID not assigned yet" in output
    assert "abcdef012345" not in output and "111111111111" not in output


def test_status_calls_out_a_worker_that_died_without_saying_so(state, capsys):
    """A job file keeps claiming `running` long after its worker was killed."""
    ops.write(ops.job_path("p1"), {"stage": "validation-local", "part": "p1", "gpu": "4",
                                   "status": "running", "pid": 2 ** 22,
                                   "log": ".run_state/validation-local.p1.log"})
    validation.main()
    output = capsys.readouterr().out
    assert "IS GONE" in output and "p1" in output and "GPU 4" in output


def test_status_reports_a_live_worker_as_live(state, capsys):
    ops.write(ops.job_path("p2"), {"stage": "validation-local", "part": "p2", "gpu": "5",
                                   "status": "running", "pid": os.getpid(),
                                   "log": ".run_state/validation-local.p2.log"})
    validation.main()
    output = capsys.readouterr().out
    assert "alive" in output and "IS GONE" not in output


def test_status_with_no_job_or_id_is_informative(state, capsys):
    validation.main()
    assert "No validation run selected" in capsys.readouterr().out


def test_status_after_prepare_starts_reports_pending_stages(state, capsys):
    ops.write(state / "job.json", {"stage": "validation-local", "status": "running"})
    (state / "experiment_id").write_text("abcdef012345\n")
    validation.main()
    output = capsys.readouterr().out
    assert "Validation: abcdef012345" in output
    assert "prepare: pending" in output and "self-eval: pending" in output


def test_status_explicit_id_works_during_another_preflight(state, monkeypatch, capsys):
    ops.write(state / "job.json", {"stage": "validation-local", "status": "running"})
    monkeypatch.setattr(sys, "argv", ["validation_ops.py", "status", "--experiment-id", "abcdef012345"])
    validation.main()
    assert "Validation: abcdef012345" in capsys.readouterr().out


def test_status_does_not_hide_corrupt_job_state(state):
    state.mkdir(parents=True)
    (state / "job.json").write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        validation.main()
