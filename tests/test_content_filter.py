from types import SimpleNamespace

from rmcq.backends.azure import is_content_filter_error
from rmcq.backends.base import Generation
from run_experiment import (
    cached_generate,
    resolve_answers,
    summarize,
    unavailable_memory_method,
)


class FakeAzureError(RuntimeError):
    def __init__(self, message, body=None, status_code=400):
        super().__init__(message)
        self.body = body
        self.status_code = status_code


def test_azure_content_policy_errors_are_recognized_precisely():
    assert is_content_filter_error(
        FakeAzureError(
            "request rejected",
            {"code": "ResponsibleAIPolicyViolation", "category": "jailbreak"},
        )
    )
    assert is_content_filter_error(FakeAzureError("conteúdo malicioso no prompt"))
    assert not is_content_filter_error(FakeAzureError("401 invalid API key", status_code=401))
    assert not is_content_filter_error(FakeAzureError("certificate verify failed", status_code=None))


def test_filtered_source_propagates_an_auditable_unavailable_reason():
    attempt = {"eval_method": "content_filter", "reflection_status": {"simple": "not_generated"}}
    assert unavailable_memory_method(attempt, attempt, "simple") == "source_answer_content_filter"

    valid_attempt = {"eval_method": "parser"}
    reflection = {"reflection_status": {"simple": "content_filter"}}
    assert unavailable_memory_method(valid_attempt, reflection, "simple") == "source_reflection_content_filter"


def test_accuracy_summary_reports_coverage_instead_of_hiding_filtered_rows():
    rows = [
        {"model": "gpt", "dataset": "arc", "condition": "baseline", "correct": True},
        {"model": "gpt", "dataset": "arc", "condition": "baseline", "correct": None},
    ]
    arc = next(row for row in summarize(rows) if row["dataset"] == "arc")
    assert arc["n"] == 2
    assert arc["resolved"] == 1
    assert arc["coverage"] == 0.5
    assert arc["accuracy"] == 1.0


def test_filtered_generation_is_checkpointed_and_not_sent_to_judge(tmp_path):
    class FilterBackend:
        key = "gpt-filtered"
        spec = SimpleNamespace(provider="azure")

        def generate(self, prompts, params, desc=""):
            return [Generation(text="", finish_reason="content_filter") for _ in prompts]

    item = {
        "question": "Q?", "context": None,
        "choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
        "answerKey": "A",
    }
    backend = FilterBackend()
    generated = cached_generate(
        backend, tmp_path / "filtered.jsonl", {"q1": "prompt"},
        max_tokens=100, batch_size=8, fresh=False, description="filtered test",
    )
    verdicts = resolve_answers(
        backend, tmp_path, "filtered", generated, {"q1": item}, 8, False
    )
    assert generated["q1"]["finish_reason"] == "content_filter"
    assert verdicts["q1"] == {
        "selected_answer": None, "correct": None, "eval_method": "content_filter"
    }
    assert not (tmp_path / "judge_filtered.jsonl").exists()
