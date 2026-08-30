"""
Os três prompts do notebook 07: baseline, reflexão e avaliação com reflexão.

Regra do projeto: existe UM prompt de baseline, UM conjunto de prompts de
reflexão e UM prompt de avaliação, todos aqui. Tratamento específico de modelo
vira parâmetro, nunca um caminho de código separado.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

# ===========================================================================
# 1. BASELINE — responder sem reflexão nenhuma
# ===========================================================================

ANSWER_PROMPT = """You are answering a multiple-choice question.

Question: {question}

Options:
{options}

Instructions:
- Think step by step before answering.
- Choose exactly one option.
- End your response with this exact line, and nothing after it:
FINAL ANSWER: <letter>"""


def format_options(choices: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{c['label']}) {c['text']}" for c in choices)


def format_question(item: dict[str, Any]) -> str:
    """Enunciado, com a premissa/contexto prefixado quando o dataset tem um."""
    context = item.get("context")
    if context:
        return f"{context.strip()}\n\n{item['question'].strip()}"
    return item["question"].strip()


def build_answer_prompt(item: dict[str, Any]) -> str:
    """O prompt de baseline. Vale para todo modelo e dataset."""
    return ANSWER_PROMPT.format(
        question=format_question(item),
        options=format_options(item["choices"]),
    )


# ===========================================================================
# 2. REFLEXÃO — o professor (ou o próprio aluno) comenta uma resposta anterior
# ===========================================================================
#
# Duas profundidades (simple/complex) x duas perspectivas (student/teacher).
# Perspectiva "student" é autorreflexão (aluno == professor); "teacher" é
# reflexão externa, escrita na terceira pessoa sobre a resposta do aluno.

REFLECTION_PROMPTS = {
    ("simple", "student"): """You are given:
1 - The original multiple-choice question.
2 - Your previous answer.
3 - Feedback indicating whether your answer was correct or incorrect.

Write a brief reflection (3-6 sentences) on your previous response. Discuss:
- The main factors that influenced your answer.
- Any assumptions or uncertainties you had.
- How the feedback supports or challenges your approach.
- One lesson you would apply when answering similar questions in the future.

If the answer was correct, explain why your approach was effective and note any remaining uncertainty.
If the answer was incorrect, identify the most likely source of the error without simply restating that the answer was wrong.
Do not answer the question again or identify which option is correct.""",
    ("complex", "student"): """You are given:
1 - The original multiple-choice question.
2 - Your previous answer.
3 - Feedback indicating whether your answer was correct or incorrect.

Write a detailed reflection on your previous response. Analyze:
- The reasoning strategy you used to reach your answer.
- The evidence or cues from the question that influenced your decision.
- Any assumptions, heuristics, or uncertainties that affected your judgment.
- How the feedback confirms or contradicts your reasoning.
- Whether your conclusion depended on missing knowledge, incorrect interpretation, overconfidence, or insufficient evaluation of alternatives.
- How you would improve your reasoning process for similar problems in the future.

If the answer was correct, explain which parts of your reasoning were reliable and whether your confidence was appropriately calibrated.
If the answer was incorrect, explain what aspect of your reasoning should change rather than merely noting the correct outcome.
Do not answer the question again, identify the correct option, or speculate about what the correct answer is.""",
    ("simple", "teacher"): """You are given:
1. The original multiple-choice question.
2. The student's previous answer.
3. Feedback indicating whether the student's answer was correct or incorrect.

Write a brief reflection (3–6 sentences) on the student's previous response.

Discuss:

- The main factors that influenced their answer.

- Any assumptions or uncertainties they had.

- How the feedback supports or challenges their approach.

- One lesson they would apply when answering similar questions in the future.

If the answer was correct, explain why their approach was effective and note any remaining uncertainty.

If the answer was incorrect, identify the most likely source of the error without simply restating that the answer was wrong.

**Do not answer the question again or identify which option is correct.**""",
    ("complex", "teacher"): """You are given:
1. The original multiple-choice question.
2. The student's previous answer.
3. Feedback indicating whether the student's answer was correct or incorrect.

Write a detailed reflection on the student's previous response.

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

FEEDBACK_CORRECT = "Feedback: Your answer was CORRECT."
FEEDBACK_INCORRECT = "Feedback: Your answer was INCORRECT."
FEEDBACK_CORRECT_TEACHER = "Feedback: The student's answer was CORRECT."
FEEDBACK_INCORRECT_TEACHER = "Feedback: The student's answer was INCORRECT."


def build_reflection_prompt(
    item: dict[str, Any],
    previous_answer: str,
    was_correct: bool,
    depth: str = "simple",
    perspective: str = "teacher",
) -> str:
    """Um dos quatro prompts da grade profundidade x perspectiva."""
    if (depth, perspective) not in REFLECTION_PROMPTS:
        raise ValueError(
            f"combinação inválida: depth={depth!r}, perspective={perspective!r}. "
            f"Válidas: {sorted(REFLECTION_PROMPTS)}"
        )
    instruction = REFLECTION_PROMPTS[(depth, perspective)]
    if perspective == "teacher":
        feedback = FEEDBACK_CORRECT_TEACHER if was_correct else FEEDBACK_INCORRECT_TEACHER
        answer_header = "Student's previous answer:"
    else:
        feedback = FEEDBACK_CORRECT if was_correct else FEEDBACK_INCORRECT
        answer_header = "Your previous answer:"

    return (
        f"{instruction}\n\n"
        f"Question: {format_question(item)}\n\n"
        f"Options:\n{format_options(item['choices'])}\n\n"
        f"{answer_header}\n{previous_answer.strip()}\n\n"
        f"{feedback}"
    )


# ===========================================================================
# 3. AVALIAÇÃO COM REFLEXÃO — injeta as k reflexões recuperadas no prompt
# ===========================================================================
#
# Duas versões, trocáveis por RMCQ_EVAL_PROMPT:
#
# v1 — as reflexões vêm ANTES do enquadramento da tarefa, como uma lista
#      numerada ("[Lesson 1]"). É o formato original.
#
# v2 (padrão) — layout revisto para modelos pequenos (phi4-mini, llama3-8b):
#   1. Enquadramento primeiro, questão por último — um decoder causal atende
#      mais ao que está perto do fim do prompt.
#   2. Delimitador duro (<notes>/<note id="i">) em vez de "[Lesson 1]" + "---",
#      para o enunciado da nota não competir com o enunciado a responder.
#   3. Questão de origem ligada por padrão, com contexto — sem ela a nota vira
#      conselho sem referente.
#   4. Orçamento por nota (compact_reflection corta pela CABEÇA: é lá que
#      mora a narração da questão de origem, e a cauda tem a parte que
#      generaliza — "Lesson: ...").
#   5. Neutralização de letra (neutralize_option_letters troca "you selected
#      D" por "you selected [letter]"): a letra da nota é de OUTRA questão, e
#      lida antes de responder ancoraria o aluno na letra D da questão nova.
#
# Sem nenhuma reflexão recuperada, build_eval_prompt() devolve
# build_answer_prompt() byte a byte — o fallback é idêntico ao baseline.

RETRIEVED_HEADER = """Below are lessons you recorded after answering other, different multiple-choice questions in the past. They are ordered from least to most relevant to the question you are about to answer.

They are not about the question below and they do not contain its answer. Use them only as guidance on how to reason.

{reflections}
---
"""

RETRIEVED_ITEM = """[Lesson {i}]
{reflection}
"""

RETRIEVED_ITEM_WITH_SOURCE = """[Lesson {i} — recorded on a different question: "{source_question}"]
{reflection}
"""

INJECT_SOURCE_QUESTION = os.environ.get("RMCQ_INJECT_SOURCE_QUESTION", "0") in (
    "1", "true", "True",
)
SOURCE_QUESTION_MAX_CHARS = 220

EVAL_PROMPT_VERSION = os.environ.get("RMCQ_EVAL_PROMPT", "v2").strip().lower()

NOTES_HEADER_V2 = """You are answering a multiple-choice question.

First, some notes from earlier attempts at OTHER questions. They are reference material about how to reason. None of them is about the question below, and none of them contains its answer.

<notes>
{notes}
</notes>

"""

NOTE_V2 = """<note id="{i}">
{body}
</note>"""

NOTE_SOURCE_V2 = 'This note was written about a different question: "{source_question}"'
NOTE_OUTCOME_V2 = {
    True: "The earlier answer to that question was correct.",
    False: "The earlier answer to that question was incorrect.",
}

EVAL_TAIL_V2 = """Question: {question}

Options:
{options}

Instructions:
- The notes are advice on how to reason, nothing more. The correct option here may be a different letter than any letter mentioned in a note.
- Think step by step before answering.
- Choose exactly one option.
- End your response with this exact line, and nothing after it:
FINAL ANSWER: <letter>"""

# 120 palavras ~ o tamanho de uma reflexão `simple` (mediana medida: 128), o
# corte é quase inócuo nelas e vale 5x nas `complex`. 0 desliga.
NOTE_MAX_WORDS = int(os.environ.get("RMCQ_NOTE_MAX_WORDS", "120"))

# 600 caracteres cobrem a premissa completa do LogiQA2; 220 (v1) cortavam a
# premissa no meio, o que é pior que omiti-la.
SOURCE_QUESTION_MAX_CHARS_V2 = int(os.environ.get("RMCQ_SOURCE_QUESTION_MAX_CHARS", "600"))

NEUTRALIZE_OPTION_LETTERS = os.environ.get("RMCQ_NEUTRALIZE_LETTERS", "1") in (
    "1", "true", "True",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s")

# "option D", "answer (B)", "chose C" — sempre letra MAIÚSCULA isolada, então
# o artigo "a" e palavras comuns não são atingidos. O separador aceita vírgula
# e dois-pontos, não só espaço ("the correct answer, B, highlights...").
_LETTER_SEP = r"(?:\s*[,:]\s*|\s+)"
_LETTER_COPULA = r"\s+(?:is|was|are|were|being|would\s+be|should\s+be|must\s+be)\s+"
_LETTER_MENTIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b((?:option|choice|answer|alternative|response)s?" + _LETTER_SEP + r")\(?([A-H])\)?(?![\w-])"),
    re.compile(r"\b((?:option|choice|answer|alternative|response)s?" + _LETTER_COPULA + r")\(?([A-H])\)?(?![\w-])"),
    re.compile(r"\b((?:chose|choose|chosen|selected|select|picked|pick|answered)" + _LETTER_SEP + r")\(?([A-H])\)?(?![\w-])"),
    re.compile(r"(\s)\(([A-H])\)(?![\w-])"),
)

# Continuação de enumeração: "options [letter], C, and D".
_LETTER_ENUM = re.compile(r"(\[letter\])(,?\s+(?:and|or)\s+|,\s*)([A-H])(?![\w-])")

# Enumeração SEM palavra-gatilho: "while A, B, and D are essential". Exige
# duas ou mais letras encadeadas, o que descarta o artigo "A" isolado.
_LETTER_TOKEN = r"(?:\[letter\]|[A-H])"
_LETTER_RUN = re.compile(
    r"(?<![\w\[°º])(" + _LETTER_TOKEN + r"(?:(?:\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)" + _LETTER_TOKEN + r")+)(?![\w-])"
)
_LETTER_QUOTED = re.compile(r"([\"“'])([A-H])([\"”'])")


def _neutralize_runs(text: str) -> str:
    def sub(match: re.Match[str]) -> str:
        run = match.group(1)
        if len(re.findall(r"(?<!\[letter)\b[A-H]\b", run)) + run.count("[letter]") < 2:
            return run
        return re.sub(r"(?<![\w\[])[A-H](?![\w-])", "[letter]", run)

    return _LETTER_RUN.sub(sub, text)


def neutralize_option_letters(text: str) -> str:
    """
    Troca menções a letras de alternativa por "[letter]".

    A reflexão foi escrita sobre OUTRA questão, onde "D" era outra coisa. Ler
    "you selected D" antes de responder ancora o aluno na letra D da questão
    nova. NÃO cobre letra solta sem nenhuma pista ("arguing that A could be
    possible"): o preço de um falso positivo no artigo "A" é maior que o do
    resíduo.
    """
    for pattern in _LETTER_MENTIONS:
        text = pattern.sub(r"\1[letter]", text)
    text = _LETTER_QUOTED.sub(r"\1[letter]\3", text)
    text = re.sub(r"(?<![\w\[°º(])([A-H])(?=\)(?![\w-]))", "[letter]", text)  # "B) was the most aligned"
    while True:
        text, n = _LETTER_ENUM.subn(r"\1\2[letter]", text)
        if not n:
            break
    return _neutralize_runs(text)


def _text_units(text: str) -> list[str]:
    """Linhas e frases da reflexão, na ordem, sem vazios — a unidade de corte."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in _SENTENCE_END.split(line) if p.strip()]
        units.extend(parts or [line])
    return units


def compact_reflection(text: str, max_words: int | None = None) -> str:
    """
    Limita a reflexão a `max_words`, cortando pela CABEÇA.

    A cabeça narra a questão de origem; a cauda traz o que transfere ("Use the
    negation test for necessary assumptions..."). Cortar pela cauda, que seria
    o reflexo natural, joga fora exatamente a parte reaproveitável.
    """
    max_words = NOTE_MAX_WORDS if max_words is None else max_words
    text = (text or "").strip()
    if max_words <= 0 or len(text.split()) <= max_words:
        return text

    kept: list[str] = []
    total = 0
    for unit in reversed(_text_units(text)):
        n = len(unit.split())
        if kept and total + n > max_words:
            break
        kept.append(unit)
        total += n
    kept.reverse()
    if not kept:
        return " ".join(text.split()[:max_words])
    out = ""
    for unit in kept:
        sep = "\n" if out and _BULLET.match(unit) else (" " if out else "")
        out += sep + unit
    return "(...) " + out


def format_source_question(text: str, max_chars: int | None = None) -> str:
    """Enunciado de origem em uma linha, cortado em fronteira de frase."""
    max_chars = SOURCE_QUESTION_MAX_CHARS_V2 if max_chars is None else max_chars
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[: stop + 1] if stop > max_chars // 2 else cut.rstrip()) + " (...)"


def build_retrieval_prefix(
    reflections: Sequence[str],
    source_questions: Sequence[str] | None = None,
    include_source: bool | None = None,
    source_was_correct: Sequence[bool | None] | None = None,
    version: str | None = None,
) -> str:
    """
    Bloco com as k reflexões recuperadas, em similaridade crescente.

    `reflections` deve chegar JÁ ordenado do menos para o mais similar — a
    mais parecida com a questão nova fica por último, adjacente a ela, onde um
    decoder causal atende mais.

    No v2 este bloco é só a PRIMEIRA metade do prompt (enquadramento + notas):
    a questão vem depois, montada por build_eval_prompt.
    """
    if not reflections:
        return ""

    version = (version or EVAL_PROMPT_VERSION).lower()

    if version == "v1":
        include = INJECT_SOURCE_QUESTION if include_source is None else include_source
        blocks = []
        for i, reflection in enumerate(reflections, start=1):
            text = reflection.strip()
            if include and source_questions:
                src = source_questions[i - 1].strip().replace("\n", " ")
                if len(src) > SOURCE_QUESTION_MAX_CHARS:
                    src = src[: SOURCE_QUESTION_MAX_CHARS - 3] + "..."
                blocks.append(RETRIEVED_ITEM_WITH_SOURCE.format(
                    i=i, reflection=text, source_question=src))
            else:
                blocks.append(RETRIEVED_ITEM.format(i=i, reflection=text))
        return RETRIEVED_HEADER.format(reflections="\n".join(blocks))

    include = True if include_source is None else include_source
    notes = []
    for i, reflection in enumerate(reflections, start=1):
        lines = []
        if include and source_questions:
            lines.append(NOTE_SOURCE_V2.format(
                source_question=format_source_question(source_questions[i - 1])))
        if source_was_correct:
            outcome = NOTE_OUTCOME_V2.get(source_was_correct[i - 1])
            if outcome:
                lines.append(outcome)

        text = compact_reflection(reflection)
        if NEUTRALIZE_OPTION_LETTERS:
            text = neutralize_option_letters(text)
        lines.append(text)

        notes.append(NOTE_V2.format(i=i, body="\n".join(lines)))

    return NOTES_HEADER_V2.format(notes="\n\n".join(notes))


def build_eval_prompt(
    item: dict[str, Any],
    reflections: Sequence[str],
    source_questions: Sequence[str] | None = None,
    source_was_correct: Sequence[bool | None] | None = None,
    include_source: bool | None = None,
    version: str | None = None,
) -> str:
    """
    Prompt de avaliação: notas recuperadas + a questão nova.

    Sem nenhuma reflexão recuperada, devolve build_answer_prompt() byte a
    byte — o fallback é comparável ao baseline em vez de ser uma condição à parte.
    """
    if not reflections:
        return build_answer_prompt(item)

    version = (version or EVAL_PROMPT_VERSION).lower()
    prefix = build_retrieval_prefix(
        reflections, source_questions, include_source, source_was_correct, version
    )

    if version == "v1":
        return prefix + build_answer_prompt(item)

    return prefix + EVAL_TAIL_V2.format(
        question=format_question(item),
        options=format_options(item["choices"]),
    )
