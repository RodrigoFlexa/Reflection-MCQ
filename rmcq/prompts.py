"""Prompts canonicos do experimento top-1 em duas etapas.

O mesmo arquivo e importado nos servidores GPU e Petrobras. Assim, o texto
registrado no manifesto e exatamente o texto usado na geracao.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


ANSWER_PROMPT = """You are answering a multiple-choice question.

Question: {question}

Options:
{options}

Respond in exactly two parts. Keep the reasoning to at most 60 words.
Reasoning: <brief reasoning>
FINAL ANSWER: <letter>"""


STUDENT_REFLECTION_PROMPTS = {
    "simple": """You are given:

1. The question and answer choices.
2. Your previous answer.
3. Whether your answer was correct or incorrect.
4. The correct answer.

Reflect on your reasoning in 3–5 sentences (about 60–100 words).

Analyze:

- Approach: What reasoning approach did you use?
- Key factor: What fact, relationship, constraint, or clue affected your decision?
- Error/Success: If incorrect, what specific reasoning mistake caused the error? If correct, what reasoning step helped?
- Lesson: What one general reasoning rule should you use for similar problems?

The lesson must describe what to do differently, not just “be careful” or “think harder.”

Focus on reasoning, not the answer.

Do not solve the question again or state the correct answer.""",
    "complex": """You are given:

1. The question and answer choices.
2. Your previous answer.
3. Whether your answer was correct or incorrect.
4. The correct answer.

Analyze your reasoning in detail.

Reflect on your reasoning in 8–12 sentences (about 160–240 words).

Consider:

- Interpretation: What was the question asking? Did you interpret it correctly?
- Strategy: What reasoning strategy did you use? Was it appropriate?
- Evidence: Which facts, relationships, or constraints did you use or miss?
- Assumptions: What assumptions or shortcuts affected your reasoning?
- Alternatives: Which other possibilities should you have considered?
- Diagnosis: What specifically caused the error, or what made the reasoning successful?
- Improvement: What should you change when solving similar problems?
- Lesson: State a precise, general reasoning rule that can transfer to other problems.

Do not give vague advice such as “be more careful.” Focus on how to reason, not on the specific answer.

Do not solve the question again or state the correct answer.""",
}


TEACHER_REFLECTION_PROMPTS = {
    "simple": """You are given:
1. The original multiple-choice question.
2. The student's previous answer.
3. Feedback indicating whether the student's answer was correct or incorrect.

Write a brief reflection (3–6 sentences, about 60–120 words) on the student's previous response.

Discuss:

- The main factors that influenced their answer.
- Any assumptions or uncertainties they had.
- How the feedback supports or challenges their approach.
- One lesson they would apply when answering similar questions in the future.

If the answer was correct, explain why their approach was effective and note any remaining uncertainty.

If the answer was incorrect, identify the most likely source of the error without simply restating that the answer was wrong.

**Do not answer the question again or identify which option is correct.**""",
    "complex": """You are given:
1. The original multiple-choice question.
2. The student's previous answer.
3. Feedback indicating whether the student's answer was correct or incorrect.

Write a detailed reflection (8–12 sentences, about 160–240 words) on the student's previous response.

Analyze:

- The reasoning strategy they used to reach their answer.
- The evidence or cues from the question that influenced their decision.
- Any assumptions, heuristics, or uncertainties that affected their judgment.
- How the feedback confirms or contradicts their reasoning.
- Whether their conclusion depended on missing knowledge, incorrect interpretation, overconfidence, or insufficient evaluation of alternatives.
- How they would improve their reasoning process for similar problems in the future.

If the answer was correct, explain which parts of their reasoning were reliable and whether their confidence was appropriately calibrated.

If the answer was incorrect, explain what aspect of their reasoning should change rather than merely noting the correct outcome.

**Do not answer the question again, identify the correct option, or speculate about what the correct answer is.**""",
}

REFLECTION_DEPTHS = ("simple", "complex")


TRANSFER_PROMPT = """You are answering a new multiple-choice question.

First, review one related training case. It is context for how to reason, not a demonstration whose conclusion should be copied. Its answer labels belong only to that earlier case.

<training_case>
Question:
{source_question}

Correct answer:
{source_correct_answer}

Agent's previous response:
{source_response}

Outcome:
{source_outcome}

Reflection:
{reflection}
</training_case>

Now answer the validation question independently.

Question: {question}

Options:
{options}

Respond in exactly two parts. Keep the reasoning to at most 60 words.
Reasoning: <brief reasoning>
FINAL ANSWER: <letter>"""


JUDGE_PROMPT = """You are grading a multiple-choice response.

Question: {question}

Options:
{options}

Correct option: {correct_label}) {correct_text}

Candidate response:
{response}

Decide only whether the option ultimately selected by the candidate matches the correct option.
Your entire response must be exactly one word, with no explanation:
CORRECT
or
INCORRECT"""


def format_options(choices: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{choice['label']}) {choice['text']}" for choice in choices)


def format_question(item: dict[str, Any]) -> str:
    context = (item.get("context") or "").strip()
    question = item["question"].strip()
    return f"{context}\n\n{question}" if context else question


def correct_answer_text(item: dict[str, Any]) -> str:
    label = item["answerKey"]
    text = next(choice["text"] for choice in item["choices"] if choice["label"] == label)
    return f"{label}) {text}"


def build_answer_prompt(item: dict[str, Any]) -> str:
    return ANSWER_PROMPT.format(
        question=format_question(item), options=format_options(item["choices"])
    )


def build_reflection_prompt(
    item: dict[str, Any],
    previous_answer: str,
    was_correct: bool,
    depth: str = "simple",
    perspective: str = "student",
) -> str:
    """Build an exact student or teacher reflection task."""
    if depth not in REFLECTION_DEPTHS:
        raise ValueError(f"invalid reflection depth: {depth!r}")
    if perspective not in {"student", "teacher"}:
        raise ValueError(f"invalid reflection perspective: {perspective!r}")

    if perspective == "student":
        instruction = STUDENT_REFLECTION_PROMPTS[depth]
        answer_header = "Your previous answer:"
        private_feedback = f"\n\nCorrect answer (private feedback):\n{correct_answer_text(item)}"
    else:
        instruction = TEACHER_REFLECTION_PROMPTS[depth]
        answer_header = "Student's previous answer:"
        private_feedback = ""
    outcome = "CORRECT" if was_correct else "INCORRECT"

    return (
        f"{instruction}\n\n"
        f"Original multiple-choice question:\n{format_question(item)}\n\n"
        f"Answer choices:\n{format_options(item['choices'])}\n\n"
        f"{answer_header}\n{previous_answer.strip()}\n\n"
        f"Feedback: The previous answer was {outcome}."
        f"{private_feedback}"
    )


def build_transfer_prompt(
    item: dict[str, Any],
    source_item: dict[str, Any],
    source_response: str,
    source_was_correct: bool,
    reflection: str,
) -> str:
    if not (reflection or "").strip():
        raise ValueError("reflection cannot be empty")
    if not (source_response or "").strip():
        raise ValueError("source response cannot be empty")
    return TRANSFER_PROMPT.format(
        source_question=format_question(source_item),
        source_correct_answer=correct_answer_text(source_item),
        source_response=source_response.strip(),
        source_outcome="CORRECT" if source_was_correct else "INCORRECT",
        reflection=reflection.strip(),
        question=format_question(item),
        options=format_options(item["choices"]),
    )


def build_eval_prompt(
    item: dict[str, Any],
    reflections: Sequence[str],
    source_items: Sequence[dict[str, Any]] | None = None,
    source_answers: Sequence[str] | None = None,
    source_was_correct: Sequence[bool] | None = None,
    **_: Any,
) -> str:
    """Compatibility wrapper; the new protocol accepts exactly one top-1 pair."""
    if not reflections:
        return build_answer_prompt(item)
    if (
        len(reflections) != 1 or not source_items or len(source_items) != 1
        or not source_answers or len(source_answers) != 1
        or not source_was_correct or len(source_was_correct) != 1
    ):
        raise ValueError("the top-1 protocol requires one aligned source case")
    return build_transfer_prompt(
        item, source_items[0], source_answers[0], source_was_correct[0], reflections[0]
    )


def build_judge_prompt(item: dict[str, Any], response: str) -> str:
    label = item["answerKey"]
    text = next(choice["text"] for choice in item["choices"] if choice["label"] == label)
    return JUDGE_PROMPT.format(
        question=format_question(item), options=format_options(item["choices"]),
        correct_label=label, correct_text=text, response=(response or "").strip(),
    )


_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*\(?([A-H])\)?", re.IGNORECASE)
_VERDICT_RE = re.compile(
    r"^\s*(?:Verdict:\s*)?(CORRECT|INCORRECT)\s*[.!]?\s*$", re.IGNORECASE
)


def extract_final_answer(text: str) -> str | None:
    matches = _FINAL_ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def parse_judge_verdict(text: str) -> bool | None:
    match = _VERDICT_RE.search(text or "")
    return None if match is None else match.group(1).upper() == "CORRECT"
