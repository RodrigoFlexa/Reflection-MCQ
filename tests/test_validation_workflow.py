import json
import sys

import pandas as pd
import pytest

import run_experiment as run
import experiment_ops as ops
from rmcq.config import MODELS
from rmcq.test_analysis import load_run
from rmcq.validation_analysis import study


def item(dataset, split, index):
    return dict(uid=f"{dataset}-{split}-{index}", dataset=dataset, split=split,
                problem_type="process", context="A passage about " + split,
                question=f"Which option {index}?", choices=[{"label": c, "text": c} for c in "ABCD"],
                answerKey="C", num_choices=4, rationale=None,
                article_uid=f"article-{split}",
                race_subset="high" if dataset == "race" else None)


def test_nine_model_local_then_teacher_completion_preserves_self(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "find_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    for dataset in run.DEFAULT_DATASETS:
        folder = tmp_path / "data/processed" / dataset
        run.save_jsonl(folder / "train.jsonl", [item(dataset, "train", 0)])
        run.save_jsonl(folder / "validation.jsonl", [item(dataset, "validation", i) for i in (1, 2)])

    def retrieval(state, model, device):
        return [{"dataset": d, "val_uid": v["uid"], "eval_uid": v["uid"], "eval_split": "validation",
                 "source_uid": data["train"][0]["uid"], "similarity": .9,
                 "validation_item": v, "source_item": data["train"][0]}
                for d, data in state.items() for v in data["validation"]]
    monkeypatch.setattr(run, "retrieve_top1", retrieval)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "prepare", "--preset", "validation-threshold", "--backend", "stub"])
    run.main()
    path = next((tmp_path / "experiment_exchange").glob("*/manifest.json"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rid = manifest["experiment_id"]
    assert manifest["models"] == list(run.VALIDATION_MODELS)
    assert manifest["teacher_role"] == "teacher-only"
    assert manifest["eval_split"] == "validation"
    assert all(MODELS[key].provider == "hf" for key in manifest["models"])

    def stage(name):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", name, "--experiment-id", rid, "--teacher-backend", "stub"])
        run.main()

    stage("self-eval")
    result = tmp_path / "data/results/reflection_top1" / rid
    partial = run.load_jsonl(result / "self_eval/analysis/all_outcomes.jsonl")
    assert len(partial) == 9 * 5 * 2 * 3
    assert {r["condition"] for r in partial} == set(run.SELF_CONDITIONS)
    assert all("audit_flags" in r and "eval_split" in r for r in partial)
    frame, sources = load_run(tmp_path, rid, "validation")
    assert len(frame) == len(partial) and "self_eval" in sources[0]

    original_generate = run.cached_generate
    calls = []
    def record(*args, **kwargs):
        calls.append((args[1], list(args[2])))
        return original_generate(*args, **kwargs)
    monkeypatch.setattr(run, "cached_generate", record)
    stage("teacher")
    assert len(calls) == 9 * 2
    assert all("teacher_" in p.name for p, _ in calls)
    assert not (path.parent / "teacher/validation.jsonl").exists()
    assert not (path.parent / "teacher/train.jsonl").exists()
    assert json.loads((path.parent / "teacher_receipt.json").read_text())["validation_generations"] == 0
    calls.clear()
    stage("finish")
    final = run.load_jsonl(result / "analysis/all_outcomes.jsonl")
    assert len(final) == 9 * 5 * 2 * 5
    assert all(r["model"] != run.DEFAULT_TEACHER for r in final)
    assert [r for r in final if r["condition"] in run.SELF_CONDITIONS] == partial
    assert all("baseline" not in key and "self_" not in key for _, keys in calls for key in keys)
    assert json.loads((result / "finish_receipt.json").read_text())["self_snapshot_reused"] is True
    assert len(load_run(tmp_path, rid, "validation")[0]) == len(final)
    assert len(load_run(tmp_path, rid, "validation", phase="self")[0]) == len(partial)
    views = study(frame, thresholds=[0, .9, 1], min_n=1, min_coverage=0)
    assert set(views) == {"completo", "filtrado"}
    assert not views["completo"]["selected"].condition.str.startswith("teacher").any()
    assert views["completo"]["thresholds"].query("threshold == 1 and condition != 'baseline'").n.eq(0).all()


def test_preset_rejects_hf_and_test(monkeypatch):
    for extra in (["--backend", "hf"], ["--eval-split", "test"], ["--teacher-role", "reference"]):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", "prepare", "--preset", "validation-threshold", *extra])
        with pytest.raises(SystemExit):
            run.parse_args()


def test_model_mapping_and_budgets_are_explicit():
    assert MODELS["deepseek-r1-0528-qwen3-8b"].repo_id.endswith("DeepSeek-R1-0528-Qwen3-8B")
    assert MODELS["deepseek-r1-distill-qwen-1.5b"].repo_id.endswith("DeepSeek-R1-Distill-Qwen-1.5B")
    assert run.training_answer_budget("deepseek-r1-0528-qwen3-8b", "final") == 4096
    assert run.reflection_budget("deepseek-r1-distill-qwen-1.5b", "simple", "final") == 4096
    for key in ["ministral-3-3b", "ministral-3-8b"]:
        assert MODELS[key].extra_kwargs["vllm_kwargs"]["tokenizer_mode"] == "mistral"


def test_partial_handoff_is_not_a_finished_experiment(tmp_path, monkeypatch):
    rid = "abcdef012345"
    monkeypatch.setattr(ops, "ROOT", tmp_path)
    monkeypatch.setattr(ops, "STATE", tmp_path / ".run_state")
    monkeypatch.setattr(ops, "SHARED", tmp_path / "experiment_handoff")
    monkeypatch.setattr(ops, "git", lambda *args, **kwargs: "revision")
    exchange, result = ops.paths(rid)
    run.save_json(exchange / "manifest.json", {"eval_split": "validation"})
    run.save_json(result / "self_eval/self_eval_receipt.json", {"complete": True})
    run.save_jsonl(result / "self_eval/analysis/all_outcomes.jsonl", [{"condition": "baseline"}])
    ops.share(rid, "self-eval", publish=False)
    assert json.loads((ops.SHARED / "current_validation.json").read_text())["stage"] == "self-eval"
    assert not ops.receipt(rid, "finish").exists()
    from rmcq.handoff import unpack
    target = tmp_path / "restored"
    unpack(target, ops.SHARED / rid / "self-eval", rid, "self-eval")
    assert (target / f"data/results/reflection_top1/{rid}/self_eval/analysis/all_outcomes.jsonl").exists()


def test_threshold_study_refuses_test_split():
    with pytest.raises(ValueError, match="validation only"):
        study(pd.DataFrame({"eval_split": ["test"]}))
