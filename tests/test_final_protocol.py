import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

import run_experiment as run
from prepare_datasets import normalize_race
from rmcq.analysis import annotate_outcomes, filter_outcomes
from rmcq.backends.base import Generation
from rmcq.backends.stub import StubBackend


def race(split, index, article="A reading passage.", question="What happened?"):
    return normalize_race({"example_id": "high19088.txt", "article": article,
                           "question": question, "options": ["First", "Second", "Third", "Fourth"],
                           "answer": "C"}, split, index)


def test_race_preserves_shared_articles_and_question_identity(tmp_path):
    first, second = race("train", 0), race("train", 1, question="Why did it happen?")
    assert first["uid"] != second["uid"]
    assert first["article_uid"] == second["article_uid"]
    assert first["context"] == "A reading passage."
    assert first["answerKey"] == "C"
    assert first["race_subset"] == "high"
    train = [first, second, race("train", 2, article="Overlapping test passage.")]
    test = [race("test", 0, article="Overlapping test passage.", question="Another question?")]
    folder = tmp_path / "data" / "processed" / "race"
    run.save_jsonl(folder / "train.jsonl", train)
    run.save_jsonl(folder / "test.jsonl", test)
    state, audit = run.load_splits(tmp_path, ["race"], None, eval_split="test")
    assert len(state["race"]["train"]) == 2
    assert state["race"]["validation"] == test
    assert audit[0]["article_overlap_removed"] == 1


class RecordingBackend(StubBackend):
    def __init__(self, model="qwen3-8b"):
        super().__init__(model)
        self.calls = []
        self.max_len = 2048
        self.tokenizer = object()

    def render_token_ids(self, tokenizer, prompt):
        return list(range(int(prompt.split(":")[0])))

    def generate(self, prompts, params, desc=""):
        self.calls.append((prompts, params.max_new_tokens))
        return [Generation(text="<think>unfinished", finish_reason="length", completion_tokens=params.max_new_tokens)
                if p.endswith("partial") else Generation(text="FINAL ANSWER: C", finish_reason="stop")
                for p in prompts]


def test_final_single_attempt_keeps_raw_and_skips_only_long_item(tmp_path):
    backend = RecordingBackend()
    prompts = {"ok": "10:ok", "long": "1800:ok", "partial": "10:partial"}
    path = tmp_path / "answers.jsonl"
    result = run.cached_generate(backend, path, prompts, 1024, 8, False, "test", profile="final")
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 2
    assert result["long"]["finish_reason"] == "prompt_context_exceeded"
    assert result["partial"]["text"] == ""
    assert result["partial"]["raw_text"] == "<think>unfinished"
    assert result["partial"]["partial_think"]
    assert result["partial"]["completion_tokens"] == 1024
    assert result["partial"]["finish_reason"] == "length_exhausted"
    run.cached_generate(backend, path, prompts, 1024, 8, False, "test", profile="final")
    assert len(backend.calls) == 1  # completed failures are checkpointed too
    verdict = run.resolve_answers(backend, tmp_path, "test", result,
                                  {key: race("test", i) for i, key in enumerate(prompts)}, 8, False, profile="final")
    assert verdict["ok"]["correct"] is True
    assert verdict["long"]["correct"] is None
    assert len(backend.calls) == 1  # no judging an incomplete answer


def test_phi_reflection_budget_adapts_without_truncating_prompt(tmp_path):
    backend = RecordingBackend("phi2")
    output = run.cached_generate(backend, tmp_path / "reflection.jsonl", {"q": "1500:ok"},
                                 768, 8, False, "reflection", profile="final", adapt_to_context=True)
    assert backend.calls == [(["1500:ok"], 512)]
    assert output["q"]["budget_reduced_for_context"] is True


def test_filters_preserve_denominators_and_compare_same_items():
    rows = [{"model": "qwen", "dataset": "race", "eval_split": "test", "val_uid": uid,
             "condition": condition, "correct": correct, "audit_flags": flags}
            for uid, condition, correct, flags in [
                ("a", "baseline", True, []), ("a", "self_simple", None, ["length_exhausted"]),
                ("b", "baseline", False, []), ("b", "self_simple", True, [])]]
    original = deepcopy(rows)
    kept, audit = filter_outcomes(rows, exclude_flags=["length_exhausted"], paired=True)
    assert {r["val_uid"] for r in kept} == {"b"}
    assert all(r["n_original"] == 2 and r["n_selected"] == 1 for r in audit)
    assert rows == original
    with pytest.raises(ValueError, match="Duplicate"):
        filter_outcomes(rows + [rows[0]])


def test_test_pairs_cannot_reuse_validation_cache(tmp_path):
    old = tmp_path / "old"
    run.save_json(old / "manifest.json", {"datasets": ["race"], "eval_split": "validation",
                  "validation_cap": None, "train_cap": None, "embedding_model": "bge", "seed": 42})
    run.save_jsonl(old / "pairs" / "race.jsonl", [{}])
    args = SimpleNamespace(eval_split="test", validation_cap=None, train_cap=None, embedding_model="bge")
    assert run.find_compatible_pair_exchange(tmp_path / "new", ["race"], args) is None


def test_real_retrieval_loop_allows_repeated_sources_and_marks_embedding_truncation(monkeypatch):
    import numpy as np

    class Embedder:
        max_seq_length = 8

        def __init__(self, *args, **kwargs):
            pass

        def tokenizer(self, texts, **kwargs):
            return {"input_ids": [list(range(len(text.split()))) for text in texts]}

        def encode(self, texts, **kwargs):
            if texts[0].startswith("Represent"):
                return np.array([[1., 0.], [1., 0.]])
            return np.array([[1., 0.], [0., 1.]])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Embedder))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    train = [race("train", i, article=f"Training article {i}") for i in range(2)]
    evaluation = [race("test", i, article="An evaluation article with many tokens to exceed the embedding limit.") for i in range(2)]
    pairs = run.retrieve_top1({"race": {"train": train, "validation": evaluation, "eval_split": "test"}}, "bge", "cpu")
    assert len(pairs) == 2
    assert pairs[0]["source_uid"] == pairs[1]["source_uid"] == train[0]["uid"]
    assert all(p["embedding_truncated"] and p["eval_split"] == "test" for p in pairs)
    assert run.embedding_text(train[0]) == run.embedding_text({**train[0], "answerKey": "A", "rationale": "A secret answer"})


def test_qwen_template_disables_thinking():
    backend = StubBackend("qwen3-8b")
    received = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            received.update(kwargs)
            return [1, 2]

    backend.render_token_ids(Tokenizer(), "Question?")
    assert received["enable_thinking"] is False


def test_three_stage_test_run_uses_frozen_models_and_split(tmp_path, monkeypatch):
    folder = tmp_path / "data" / "processed" / "race"
    run.save_jsonl(folder / "train.jsonl", [race("train", 0)])
    run.save_jsonl(folder / "test.jsonl", [race("test", i, article="Different article.", question=f"Question {i}?") for i in range(2)])
    monkeypatch.setattr(run, "find_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    def retrieval(state, model, device):
        # Both questions deliberately select the same training source.
        source = state["race"]["train"][0]
        return [{"dataset": "race", "val_uid": item["uid"], "eval_uid": item["uid"],
                 "eval_split": "test", "source_uid": source["uid"], "similarity": .9,
                 "validation_item": item, "source_item": source} for item in state["race"]["validation"]]
    monkeypatch.setattr(run, "retrieve_top1", retrieval)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "prepare", "--models", "phi2,qwen3-8b",
                                     "--datasets", "race", "--backend", "stub", "--eval-split", "test"])
    run.main()
    manifest_path = next((tmp_path / "experiment_exchange").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_id = manifest["experiment_id"]
    assert manifest["reflection_max_tokens"]["phi2"]["simple"] == 768
    for stage in ("teacher", "finish"):
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", stage, "--experiment-id", experiment_id,
                                         "--teacher-backend", "stub", "--eval-split", "validation"])
        run.main()  # deliberately wrong CLI split must not override the manifest
    outcomes = run.load_jsonl(tmp_path / "data" / "results" / "reflection_top1" / experiment_id / "analysis" / "all_outcomes.jsonl")
    assert len(outcomes) == 26  # 2 items * (2 students * 5 conditions + GPT * 3)
    assert all(r["eval_split"] == "test" and r["race_subset"] == "high" for r in outcomes)
    assert all("audit_flags" in r and "evaluation_generation" in r for r in outcomes)
    assert all("no_reflection" not in r["condition"] for r in outcomes)
    assert len(run.load_jsonl(manifest_path.parent / "students" / "phi2" / "train.jsonl")) == 1


def test_notebook_code_cells_compile():
    from pathlib import Path
    root = Path(run.__file__).parent
    for path in root.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell{i}", "exec")
