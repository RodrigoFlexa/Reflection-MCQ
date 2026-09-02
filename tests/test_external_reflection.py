from pathlib import Path

from run_external_reflection import (
    ANALYSIS_DEPTHS,
    EXTERNAL_BY_BASE,
    exchange_paths,
    is_content_filter_error,
)


def test_external_formats_are_paired_and_only_simple_complex():
    assert EXTERNAL_BY_BASE == {
        "simple": "external_simple",
        "complex": "external_complex",
    }
    assert ANALYSIS_DEPTHS == (
        "simple", "external_simple", "complex", "external_complex",
    )


def test_exchange_is_sharded_by_student_dataset_and_format():
    request, response = exchange_paths(
        Path("exchange"), "phi4-mini", "aqua", "external_simple"
    )
    assert request == Path("exchange/requests/phi4-mini/aqua/external_simple.jsonl")
    assert response == Path("exchange/responses/phi4-mini/aqua/external_simple.jsonl")


def test_only_content_policy_errors_are_skippable():
    assert is_content_filter_error(RuntimeError("ResponsibleAIPolicyViolation: jailbreak"))
    assert is_content_filter_error(RuntimeError("conteúdo malicioso no prompt"))
    assert not is_content_filter_error(RuntimeError("401 invalid API key"))
    assert not is_content_filter_error(RuntimeError("certificate verify failed"))
