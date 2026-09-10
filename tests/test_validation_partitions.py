"""Two GPUs must produce exactly the run one GPU would have produced.

The stub backend is deterministic, so the partitioned run and the whole run are
compared row by row: if splitting the grid changed any outcome, these fail.
"""
import json
import sys

import pytest

import experiment_ops as ops
import run_experiment as run


def item(dataset, split, index):
    return dict(uid=f"{dataset}-{split}-{index}", dataset=dataset, split=split,
                problem_type="process", context="A passage about " + split,
                question=f"Which option {index}?", choices=[{"label": c, "text": c} for c in "ABCD"],
                answerKey="C", num_choices=4, rationale=None,
                article_uid=f"article-{split}",
                race_subset="high" if dataset == "race" else None)


def seed_datasets(root):
    for dataset in run.DEFAULT_DATASETS:
        folder = root / "data/processed" / dataset
        run.save_jsonl(folder / "train.jsonl", [item(dataset, "train", 0)])
        run.save_jsonl(folder / "validation.jsonl", [item(dataset, "validation", i) for i in (1, 2)])


def fake_retrieval(state, model, device):
    return [{"dataset": d, "val_uid": v["uid"], "eval_uid": v["uid"], "eval_split": "validation",
             "source_uid": data["train"][0]["uid"], "similarity": .9,
             "validation_item": v, "source_item": data["train"][0]}
            for d, data in state.items() for v in data["validation"]]


def drive(root, monkeypatch, *stages_with_parts, skip_gated=False):
    """Run the given (stage, part) steps in one tmp root; return the run id."""
    monkeypatch.setattr(run, "find_root", lambda: root)
    monkeypatch.setattr(run, "retrieve_top1", fake_retrieval)
    monkeypatch.chdir(root)
    rid = None
    for stage, part in stages_with_parts:
        argv = ["run_experiment.py", stage, "--backend", "stub", "--teacher-backend", "stub"]
        if stage == "prepare":
            argv += ["--preset", "validation-threshold"]
        else:
            argv += ["--experiment-id", rid]
        if part:
            argv += [f"--{part}"]
        if skip_gated:
            argv += ["--skip-gated"]
        monkeypatch.setattr(sys, "argv", argv)
        run.main()
        if rid is None:
            rid = json.loads(next((root / "experiment_exchange").glob("*/manifest.json"))
                             .read_text(encoding="utf-8"))["experiment_id"]
    return rid


def set_skips(root, part, skipped):
    """Stand in for the preflight's Hub verdict, which is what the stages read."""
    path = root / f".run_state/validation_gated_skips.{part}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"part": part, "skipped": skipped}), encoding="utf-8")


def sorted_outcomes(root, rid):
    rows = run.load_jsonl(root / "data/results/reflection_top1" / rid
                          / "self_eval/analysis/all_outcomes.jsonl")
    return sorted(rows, key=lambda r: (r["model"], r["val_uid"], r["condition"]))


def test_partitions_cover_the_grid_exactly():
    run.assert_partitions_cover_validation()
    assigned = [m for models in run.VALIDATION_PARTITIONS.values() for m in models]
    assert len(assigned) == len(set(assigned)) == len(run.VALIDATION_MODELS)


def test_partition_selects_only_its_models():
    models = list(run.VALIDATION_MODELS)
    assert run.partition_models(None, models) == models
    p1, p2 = (run.partition_models(p, models) for p in ("p1", "p2"))
    assert set(p1) | set(p2) == set(models) and not set(p1) & set(p2)
    # Order follows the frozen manifest, not the partition literal.
    assert p1 == [m for m in models if m in p1]


def test_partition_refuses_a_grid_it_does_not_partition():
    with pytest.raises(ValueError, match="nine-model validation grid"):
        run.partition_models("p1", ["phi2", "llama3.1-8b"])


def test_two_partitions_reproduce_the_single_gpu_run(tmp_path, monkeypatch):
    whole, split = tmp_path / "whole", tmp_path / "split"
    for root in (whole, split):
        root.mkdir()
        seed_datasets(root)

    reference_id = drive(whole, monkeypatch, ("prepare", None), ("self-eval", None))
    partitioned_id = drive(split, monkeypatch,
                           ("prepare", "p1"), ("self-eval", "p1"),
                           ("prepare", "p2"), ("self-eval", "p2"))
    # Same frozen configuration, so the same run id: both GPUs write one run.
    assert partitioned_id == reference_id

    exchange, results = (split / "experiment_exchange" / partitioned_id,
                         split / "data/results/reflection_top1" / partitioned_id)
    # Before merge, nothing claims the run is finished.
    assert not (exchange / "prepare_receipt.json").exists()
    assert not (results / "self_eval/self_eval_receipt.json").exists()
    assert (exchange / "prepare_receipt.p1.json").exists()

    monkeypatch.setattr(sys, "argv",
                        ["run_experiment.py", "merge", "--experiment-id", partitioned_id])
    run.main()

    assert json.loads((exchange / "prepare_receipt.json").read_text())["complete"] is True
    receipt = json.loads((results / "self_eval/self_eval_receipt.json").read_text())
    assert receipt["complete"] is True and receipt["merged_from"] == ["p1", "p2"]
    assert sorted_outcomes(split, partitioned_id) == sorted_outcomes(whole, reference_id)


def test_a_gated_model_is_skipped_then_filled_in_by_the_same_command(tmp_path, monkeypatch):
    """The whole point: start now without llama3.2-3b, add it once access lands.

    The second pass must not regenerate the models that already ran, and the
    finished run must be indistinguishable from one that never had a gap.
    """
    whole, gap = tmp_path / "whole", tmp_path / "gap"
    for root in (whole, gap):
        root.mkdir()
        seed_datasets(root)
    reference_id = drive(whole, monkeypatch, ("prepare", None), ("self-eval", None))

    # Pass 1: llama3.2-3b is unreadable, so p1 runs its other four students.
    set_skips(gap, "p1", ["llama3.2-3b"])
    set_skips(gap, "p2", [])
    rid = drive(gap, monkeypatch, ("prepare", "p1"), ("self-eval", "p1"),
                ("prepare", "p2"), ("self-eval", "p2"), skip_gated=True)
    assert rid == reference_id
    exchange = gap / "experiment_exchange" / rid
    results = gap / "data/results/reflection_top1" / rid
    assert not (exchange / "students/llama3.2-3b").exists()
    assert (exchange / "students/qwen2.5-3b/train.jsonl").exists()
    receipt = json.loads((exchange / "prepare_receipt.p1.json").read_text())
    assert receipt["skipped_models"] == ["llama3.2-3b"] and "llama3.2-3b" not in receipt["student_models"]

    # Merge refuses to close a run with a hole in it, and names the hole.
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "merge", "--experiment-id", rid])
    with pytest.raises(RuntimeError, match="Nothing to merge yet"):
        run.main()
    phases = {e["phase"]: e for e in json.loads((results / "merge_report.json").read_text())["phases"]}
    assert phases["prepare"]["missing_models"] == ["llama3.2-3b"]

    # Pass 2: access granted. The same command, and only the gap is generated.
    set_skips(gap, "p1", [])
    generated = []
    original = run.cached_generate
    monkeypatch.setattr(run, "cached_generate",
                        lambda *a, **k: (generated.append(a[1].parent.name), original(*a, **k))[1])
    drive(gap, monkeypatch, ("prepare", "p1"), ("self-eval", "p1"), skip_gated=True)
    assert set(generated) == {"llama3.2-3b"}, f"regenerated finished models: {sorted(set(generated))}"

    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "merge", "--experiment-id", rid])
    run.main()
    assert json.loads((exchange / "prepare_receipt.json").read_text())["complete"] is True
    # The filled-in run equals the run that never had a gap, row for row.
    assert sorted_outcomes(gap, rid) == sorted_outcomes(whole, reference_id)


def test_skipping_is_opt_in(tmp_path, monkeypatch):
    """Without --skip-gated the skip file is ignored, so a run cannot lose a model quietly."""
    root = tmp_path / "strict"
    root.mkdir()
    seed_datasets(root)
    set_skips(root, "p1", ["llama3.2-3b"])
    assert run.gated_skips(root, "p1", enabled=False) == []
    assert run.gated_skips(root, "p1", enabled=True) == ["llama3.2-3b"]
    rid = drive(root, monkeypatch, ("prepare", "p1"))
    assert (root / "experiment_exchange" / rid / "students/llama3.2-3b/train.jsonl").exists()


def test_merge_waits_for_the_missing_partition(tmp_path, monkeypatch):
    root = tmp_path / "half"
    root.mkdir()
    seed_datasets(root)
    rid = drive(root, monkeypatch, ("prepare", "p1"), ("self-eval", "p1"))
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "merge", "--experiment-id", rid])
    with pytest.raises(RuntimeError, match="Nothing to merge yet"):
        run.main()
    exchange = root / "experiment_exchange" / rid
    results = root / "data/results/reflection_top1" / rid
    assert not (exchange / "prepare_receipt.json").exists()
    assert not (results / "self_eval/self_eval_receipt.json").exists()
    report = json.loads((results / "merge_report.json").read_text())
    incomplete = {entry["phase"]: entry for entry in report["phases"]}
    assert incomplete["prepare"]["status"] == "incomplete"
    assert set(incomplete["prepare"]["missing_models"]) == set(run.VALIDATION_PARTITIONS["p2"])


def test_a_partition_cannot_start_the_stages_it_does_not_own(monkeypatch):
    for stage in ("teacher", "merge", "status"):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", stage, "--p1"])
        with pytest.raises(SystemExit):
            run.parse_args()


def test_partitions_get_separate_job_files_and_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "STATE", tmp_path / ".run_state")
    assert ops.job_path(None).name == "job.json"
    assert ops.job_path("p1").name == "job.p1.json"
    ops.write(ops.job_path("p1"), {"stage": "validation-local", "status": "running", "pid": 1})
    ops.write(ops.job_path("p2"), {"stage": "validation-local", "status": "complete", "pid": 2})
    assert [part for part, _ in ops.job_states()] == ["p1", "p2"]


def test_shared_work_is_serialized_between_partitions(tmp_path):
    """Retrieval must happen once: the second partition waits, it does not redo it."""
    import threading
    lock, order, inside = tmp_path / "pairs.lock", [], threading.Event()

    def worker(name, hold):
        with run.exclusive(lock):
            order.append(f"enter-{name}")
            if hold:
                inside.set()
                threading.Event().wait(0.2)
            order.append(f"leave-{name}")

    first = threading.Thread(target=worker, args=("a", True))
    first.start()
    assert inside.wait(2), "first holder never entered the lock"
    second = threading.Thread(target=worker, args=("b", False))
    second.start()
    for thread in (first, second):
        thread.join(5)
    assert order == ["enter-a", "leave-a", "enter-b", "leave-b"]


def test_sibling_run_protects_the_shared_experiment_id(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(ops, "STATE", tmp_path / ".run_state")
    ops.write(ops.job_path("p1"), {"stage": "validation-local", "status": "running",
                                   "pid": os.getpid()})
    assert ops.sibling_running("p2") is True
    assert ops.sibling_running("p1") is False
    ops.write(ops.job_path("p1"), {"stage": "validation-local", "status": "complete",
                                   "pid": os.getpid()})
    assert ops.sibling_running("p2") is False
