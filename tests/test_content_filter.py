from threading import Lock
from types import SimpleNamespace

from rmcq.backends.azure import AzureBackend, is_content_filter_error
from rmcq.backends.base import Generation, GenParams
from run_experiment import (
    cached_generate,
    find_compatible_pair_exchange,
    reflection_budget,
    reflection_retry_budget,
    resolve_answers,
    summarize,
    unavailable_memory_method,
    validation_prompt_issue,
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


def test_reflection_budgets_respect_each_models_context_size():
    assert reflection_budget("phi2", "simple") == 256
    assert reflection_retry_budget("phi2", "simple") == 384
    assert reflection_budget("phi2", "complex") == 384
    assert reflection_retry_budget("phi2", "complex") == 512
    assert reflection_budget("llama3.1-8b", "complex") == 1024
    assert reflection_retry_budget("llama3.1-8b", "complex") == 1536


def test_azure_uses_temperature_when_supported_and_omits_it_for_reasoning_models():
    backend = AzureBackend.__new__(AzureBackend)
    backend.deployment = "gpt-4o"
    backend.reasoning = False
    backend.max_tokens = 1024
    backend._unsupported = set()
    kwargs = backend.build_kwargs("prompt", GenParams(temperature=0.7))
    assert kwargs["temperature"] == 0.7

    backend.reasoning = True
    kwargs = backend.build_kwargs("prompt", GenParams(temperature=0.7))
    assert "temperature" not in kwargs


def test_azure_drops_rejected_temperature_case_insensitively():
    backend = AzureBackend.__new__(AzureBackend)
    backend._unsupported = set()
    backend._lock = Lock()
    kwargs = {"model": "deployment", "messages": [], "temperature": 0.7}
    error = FakeAzureError("Unsupported parameter: Temperature")
    assert backend._maybe_drop_parameter(error, kwargs)
    assert "temperature" not in kwargs
    assert "temperature" in backend._unsupported


def test_azure_empty_length_is_returned_for_orchestrator_to_discard():
    backend = AzureBackend.__new__(AzureBackend)
    backend._lock = Lock()
    backend._calls = 0
    backend._empties = 0
    backend._content_filters = 0
    backend.key = "gpt-5-4-petrobras"
    backend.deployment = "gpt-5-4-petrobras"
    backend.reasoning = True
    response = SimpleNamespace(usage=None)
    backend._check_empty(Generation(text="", finish_reason="length"), response)
    assert backend._calls == 1
    assert backend._empties == 1


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


def test_local_truncation_retries_only_failed_items_with_larger_budget(tmp_path):
    class RetryBackend:
        key = "phi2"
        spec = SimpleNamespace(provider="hf")

        def __init__(self):
            self.budgets = []
            self.temperatures = []

        def generate(self, prompts, params, desc=""):
            self.budgets.append(params.max_new_tokens)
            self.temperatures.append(params.temperature)
            if len(self.budgets) == 1:
                return [Generation(text="partial", finish_reason="length") for _ in prompts]
            return [
                Generation(text="Reasoning: short.\nFINAL ANSWER: A", finish_reason="stop")
                for _ in prompts
            ]

    backend = RetryBackend()
    generated = cached_generate(
        backend, tmp_path / "retry.jsonl", {"q1": "prompt"},
        max_tokens=256, batch_size=8, fresh=False, description="retry test",
        temperature=0.7,
    )
    assert backend.budgets == [256, 384]
    assert backend.temperatures == [0.7, 0.7]
    assert generated["q1"]["finish_reason"] == "stop"
    assert generated["q1"]["max_new_tokens_used"] == 384


def test_second_truncation_is_discarded_and_becomes_unresolved(tmp_path):
    class AlwaysTruncatedBackend:
        key = "phi2"
        spec = SimpleNamespace(provider="hf")

        def __init__(self):
            self.budgets = []

        def generate(self, prompts, params, desc=""):
            self.budgets.append(params.max_new_tokens)
            return [Generation(text="partial", finish_reason="length") for _ in prompts]

    backend = AlwaysTruncatedBackend()
    generated = cached_generate(
        backend, tmp_path / "truncated.jsonl", {"q1": "prompt"},
        max_tokens=512, batch_size=2, fresh=False, description="truncated test",
    )
    assert backend.budgets == [512, 768]
    assert generated["q1"]["text"] == ""
    assert generated["q1"]["finish_reason"] == "length_exhausted"

    item = {
        "question": "Q?", "context": None,
        "choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
        "answerKey": "A",
    }
    verdicts = resolve_answers(
        backend, tmp_path, "truncated", generated, {"q1": item}, 2, False
    )
    assert verdicts["q1"] == {
        "selected_answer": None, "correct": None, "eval_method": "length_exhausted"
    }


def test_retry_that_exceeds_context_discards_only_affected_item(tmp_path):
    class ContextLimitedBackend:
        key = "phi2"
        spec = SimpleNamespace(provider="hf")
        tokenizer = object()
        max_len = 2048

        def __init__(self):
            self.calls = []

        def render_token_ids(self, tokenizer, prompt):
            return list(range(int(prompt)))

        def generate(self, prompts, params, desc=""):
            self.calls.append((list(prompts), params.max_new_tokens))
            if len(self.calls) == 1:
                return [Generation(text="partial", finish_reason="length") for _ in prompts]
            return [Generation(text="complete", finish_reason="stop") for _ in prompts]

    backend = ContextLimitedBackend()
    generated = cached_generate(
        backend, tmp_path / "context_retry.jsonl",
        {"overflow": "1019", "fits": "800"},
        max_tokens=768, batch_size=2, fresh=False, description="context retry test",
    )
    assert backend.calls == [(["1019", "800"], 768), (["800"], 1152)]
    assert generated["overflow"]["finish_reason"] == "length_exhausted"
    assert generated["overflow"]["discard_reason"] == "retry_exceeds_context"
    assert generated["fits"]["finish_reason"] == "stop"


def test_phi2_rejects_oversized_memory_and_full_transfer_prompt():
    class PhiBackend:
        max_len = 2048
        tokenizer = object()

        def count_tokens(self, text):
            return int(text)

        def render_token_ids(self, tokenizer, prompt):
            return list(range(int(prompt)))

    backend = PhiBackend()
    memory_issue = validation_prompt_issue(
        backend, "phi2", "1000", "teacher_complex", "513", 384
    )
    assert memory_issue == {
        "eval_method": "reflection_token_limit_exceeded",
        "reflection_tokens": 513,
        "reflection_token_limit": 512,
    }
    context_issue = validation_prompt_issue(
        backend, "phi2", "1700", "self_complex", "400", 384
    )
    assert context_issue["eval_method"] == "transfer_context_exceeded"
    assert validation_prompt_issue(
        backend, "phi2", "1600", "self_complex", "400", 384
    ) is None


def test_larger_students_keep_their_reflections_when_prompt_fits():
    class LlamaBackend:
        max_len = 8192
        tokenizer = object()

        def count_tokens(self, text):
            return int(text)

        def render_token_ids(self, tokenizer, prompt):
            return list(range(int(prompt)))

    assert validation_prompt_issue(
        LlamaBackend(), "llama3.1-8b", "3000", "teacher_complex", "1200", 512
    ) is None


def test_empty_judge_is_unresolved_without_aborting_stage(tmp_path):
    class EmptyJudgeBackend:
        key = "phi2"
        spec = SimpleNamespace(provider="hf")

        def generate(self, prompts, params, desc=""):
            return [Generation(text="", finish_reason="stop") for _ in prompts]

    item = {
        "question": "Q?", "context": None,
        "choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
        "answerKey": "A",
    }
    generated = {"q1": {"text": "I cannot decide.", "finish_reason": "stop"}}
    verdicts = resolve_answers(
        EmptyJudgeBackend(), tmp_path, "train", generated, {"q1": item}, 8, False
    )
    assert verdicts["q1"] == {
        "selected_answer": None,
        "correct": None,
        "eval_method": "judge_empty_exhausted",
    }


def test_compatible_retrieval_can_be_reused_across_pipeline_versions(tmp_path):
    old_exchange = tmp_path / "old-run"
    (old_exchange / "pairs").mkdir(parents=True)
    (old_exchange / "pairs" / "arc.jsonl").write_text("{}\n", encoding="utf-8")
    (old_exchange / "manifest.json").write_text(
        '{"pipeline_version":"v1","datasets":["arc"],"validation_cap":null,'
        '"train_cap":null,"embedding_model":"embedding","seed":42}',
        encoding="utf-8",
    )
    args = SimpleNamespace(
        validation_cap=None, train_cap=None, embedding_model="embedding",
    )
    assert find_compatible_pair_exchange(tmp_path / "new-run", ["arc"], args) == old_exchange
