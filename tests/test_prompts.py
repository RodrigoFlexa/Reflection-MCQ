from rmcq.backends.base import Generation, strip_thinking
from rmcq.prompts import (
    STUDENT_REFLECTION_PROMPTS,
    TEACHER_REFLECTION_PROMPTS,
    build_reflection_prompt,
    build_transfer_prompt,
)


ITEM = {
    "uid": "q1",
    "question": "What follows?",
    "context": "A premise.",
    "choices": [{"label": "A", "text": "First"}, {"label": "B", "text": "Second"}],
    "answerKey": "B",
}


def test_student_prompt_receives_correct_answer_as_private_feedback():
    prompt = build_reflection_prompt(ITEM, "FINAL ANSWER: A", False, "simple", "student")
    assert prompt.startswith(STUDENT_REFLECTION_PROMPTS["simple"])
    assert "Correct answer (private feedback):\nB) Second" in prompt
    assert "previous answer was INCORRECT" in prompt


def test_teacher_prompt_does_not_receive_correct_answer():
    prompt = build_reflection_prompt(ITEM, "FINAL ANSWER: A", False, "complex", "teacher")
    assert prompt.startswith(TEACHER_REFLECTION_PROMPTS["complex"])
    assert "Correct answer" not in prompt
    assert "B) Second" in prompt  # still present as an ordinary answer choice


def test_transfer_includes_compact_source_case_without_source_options():
    target = ITEM | {"uid": "q2", "question": "New question?"}
    prompt = build_transfer_prompt(
        target, ITEM, "I chose A after reading the premise.", False,
        "Check the premise before eliminating choices.",
    )
    assert "<training_case>" in prompt
    assert "A premise.\n\nWhat follows?" in prompt
    source_section = prompt.split("</training_case>")[0]
    assert "A) First\nB) Second" not in source_section
    assert "Correct answer:\nB) Second" in source_section
    assert "Agent's previous response:\nI chose A" in source_section
    assert "Outcome:\nINCORRECT" in source_section
    assert "Reflection:\nCheck the premise" in source_section
    assert "New question?" in prompt


def test_thinking_is_removed_from_every_generation():
    raw = "<think>very long hidden trace</think>\nFINAL ANSWER: B"
    assert strip_thinking(raw) == "FINAL ANSWER: B"
    assert Generation(raw).text == "FINAL ANSWER: B"
    assert strip_thinking("<think>truncated trace") == ""
