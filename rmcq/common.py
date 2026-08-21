"""
Prompt, extrator e formato de saída — os três congelados do projeto.

Regra do Caderno (seção 3, "Detalhes de Output"): existe UM prompt de resposta,
UM extrator e UM formato de saída, todos aqui. Tratamento específico de modelo
vira coluna no JSONL, nunca um caminho de código separado.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Iterator, Sequence

from rmcq.config import CHOICE_LABELS

# ===========================================================================
# 1. PROMPTS  (Caderno, seção 4 — não editar sem versionar)
# ===========================================================================
#
# REVISÃO 2026-08-21 — os dois prompts de perspectiva "teacher" foram
# reescritos para a rodada com professor GPT-5. Mudou o enquadramento (saiu o
# parágrafo "You are reviewing the response of another language model"); a
# lista de tópicos pedidos é a mesma.
#
# CONSEQUÊNCIA METODOLÓGICA: as reflexões já gravadas em
# results/reflections/{aluno}__{professor local}__{depth}/ foram geradas com o
# texto ANTIGO. Comparar "GPT-5 como professor" com elas mistura duas mudanças
# — o professor e o prompt. Para uma comparação limpa, regere as reflexões dos
# professores locais com este texto (apague os diretórios e rode `reflect` de
# novo); para uma comparação apenas indicativa, registre a ressalva.
#
# Os prompts de perspectiva "student" (autorreflexão) NÃO mudaram: eles só
# entram quando aluno == professor, o que não acontece com professor de API.

ANSWER_PROMPT = """You are answering a multiple-choice question.

Question: {question}

Options:
{options}

Instructions:
- Think step by step before answering.
- Choose exactly one option.
- End your response with this exact line, and nothing after it:
FINAL ANSWER: <letter>"""


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


# --- Injeção das k reflexões recuperadas (era [A PREENCHER] no Caderno) -----
#
# Duas decisões de formato, com a razão de cada uma:
#
# 1. ORDEM: as reflexões aparecem em similaridade CRESCENTE, então a mais
#    parecida com a questão nova fica por último, imediatamente antes dela.
#    Modelos de decodificação causal atendem mais ao que está perto do fim do
#    prompt, então essa ordem coloca o conselho mais relevante na posição de
#    maior peso. Também mantém a posição relativa estável entre k = 1, 3 e 5.
#
# 2. A QUESTÃO DE ORIGEM NÃO É INCLUÍDA por padrão. Incluir transformaria a
#    reflexão em exemplo few-shot de uma questão quase idêntica, que é
#    exatamente o confundidor de memorização que a extensão existe para
#    eliminar. Também inflaria o prompt (as premissas do LogiQA2 têm ~100
#    tokens cada) em 460 mil gerações. Para ablação, ligue
#    RMCQ_INJECT_SOURCE_QUESTION=1.

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


# --- Retry com feedback e sem reflexão (condição de controle) ---------------
#
# O Caderno já registra o problema desta condição: "sua resposta estava
# incorreta" elimina uma alternativa sozinho. Por isso o prompt diz
# explicitamente qual letra foi descartada — assim o ganho de informação fica
# medido e declarado, em vez de escondido no fraseado.

RETRY_PROMPT = """You previously answered this multiple-choice question and your answer was INCORRECT.

Question: {question}

Options:
{options}

Your previous answer was {previous_letter}, which is not correct. Rule it out.

Instructions:
- Think step by step before answering.
- Choose exactly one option, different from {previous_letter}.
- End your response with this exact line, and nothing after it:
FINAL ANSWER: <letter>"""


# ===========================================================================
# 2. MONTAGEM DO PROMPT
# ===========================================================================


def format_options(choices: Sequence[dict[str, str]]) -> str:
    return "\n".join(f"{c['label']}) {c['text']}" for c in choices)


def format_question(item: dict[str, Any]) -> str:
    """Enunciado, com a premissa/contexto prefixado quando o dataset tem um."""
    context = item.get("context")
    if context:
        return f"{context.strip()}\n\n{item['question'].strip()}"
    return item["question"].strip()


def build_answer_prompt(item: dict[str, Any]) -> str:
    """O prompt congelado de resposta. Vale para todo modelo, dataset e etapa."""
    return ANSWER_PROMPT.format(
        question=format_question(item),
        options=format_options(item["choices"]),
    )


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


def build_retrieval_prefix(
    reflections: Sequence[str],
    source_questions: Sequence[str] | None = None,
    include_source: bool | None = None,
) -> str:
    """
    Prefixo com as k reflexões recuperadas, em similaridade crescente.

    `reflections` deve chegar JÁ ordenado do menos para o mais similar. Quem
    ordena é rmcq.retrieval; aqui só formatamos, para que a ordem seja uma
    decisão de um só lugar.
    """
    if not reflections:
        return ""

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


def build_eval_prompt(
    item: dict[str, Any],
    reflections: Sequence[str],
    source_questions: Sequence[str] | None = None,
) -> str:
    """Prompt da etapa de avaliação: reflexões recuperadas + prompt congelado."""
    prefix = build_retrieval_prefix(reflections, source_questions)
    return prefix + build_answer_prompt(item) if prefix else build_answer_prompt(item)


def build_retry_prompt(item: dict[str, Any], previous_letter: str) -> str:
    """Condição de controle: feedback de erro, sem nenhuma reflexão."""
    return RETRY_PROMPT.format(
        question=format_question(item),
        options=format_options(item["choices"]),
        previous_letter=previous_letter,
    )


# ===========================================================================
# 3. EXTRATOR ÚNICO
# ===========================================================================

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

# Regras que buscam a LETRA, na ordem em que são tentadas.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("strict", re.compile(r"FINAL\s+ANSWER\s*:\s*([A-H])\b")),
    (
        "loose_final",
        re.compile(r"[*_#\s]*FINAL[\s_*]*ANSWER[*_\s]*[:\-–]?[*_\s]*\(?\[?([A-H])\)?\]?"),
    ),
    (
        "answer_is",
        re.compile(
            r"(?:answer|option|choice|alternativa|resposta)\b[^A-Za-z0-9]{0,20}"
            r"(?:is|é|:)?[^A-Za-z0-9]{0,10}\(?\[?([A-H])\)?\]?\b",
            re.IGNORECASE,
        ),
    ),
)

# Uma letra sozinha numa linha. Só é considerada nas últimas linhas da resposta:
# no meio do texto, "B)" é quase sempre parte de uma enumeração das alternativas,
# não a conclusão.
_BARE_LETTER = re.compile(r"^[*_\s(\[]*([A-H])[)\].*_\s]*$")
BARE_LETTER_TAIL_LINES = 3

# Slots em que um VALOR (não uma letra) pode aparecer como resposta.
_VALUE_SLOTS = (
    re.compile(r"FINAL\s+ANSWER\s*:?\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)\s*(?:[.\n]|$)", re.IGNORECASE),
)


@dataclass
class Extraction:
    """Resultado da extração. `letter is None` significa abstenção, não erro."""

    letter: str | None
    method: str
    followed_format: bool
    n_candidates: int
    matched_value: str | None = None  # preenchido quando o método foi value_match


def _normalize_value(text: str) -> str:
    """Tira ruído de formatação para comparar um valor com o texto de uma opção."""
    out = str(text).strip()
    out = re.sub(r"^[*_`\s(\[]+|[*_`\s)\].,;:!]+$", "", out)
    out = out.replace(",", "").replace("$", "").replace("%", "")
    return re.sub(r"\s+", " ", out).strip().lower()


def _as_number(text: str) -> float | None:
    try:
        return float(_normalize_value(text))
    except (ValueError, TypeError):
        return None


def _resolve_value(content: str, choices: Sequence[dict[str, str]]) -> tuple[str | None, str | None]:
    """
    Casa um valor emitido pelo modelo com o rótulo da alternativa correspondente.

    Existe por causa do GSM8K: lá as alternativas SÃO números, e um modelo que
    acabou de fazer a aritmética termina naturalmente com `FINAL ANSWER: 72` em
    vez da letra. Sem esta regra isso conta como abstenção, e a acurácia do
    GSM8K cai por um motivo que não tem nada a ver com raciocínio.

    Só resolve quando o valor casa com UMA alternativa. Se casar com duas ou
    mais, devolve None: um empate aqui é ambiguidade real, e chutar entre as
    duas inventaria acurácia.
    """
    norm = _normalize_value(content)
    if not norm:
        return None, None

    num = _as_number(norm)
    matches: list[str] = []

    for choice in choices:
        opt = _normalize_value(choice.get("text", ""))
        if not opt:
            continue
        opt_num = _as_number(opt)
        # Comparação numérica quando os dois lados são números: assim "0.20"
        # casa com "0.2" e "1,200" com "1200".
        if num is not None and opt_num is not None:
            if abs(num - opt_num) < 1e-9:
                matches.append(choice["label"])
            continue
        if norm == opt:
            matches.append(choice["label"])

    if len(matches) == 1:
        return matches[0], norm

    # Sem casamento exato, tenta conter/estar contido — útil quando o modelo
    # reescreve a alternativa com pequenas variações. Exige unicidade.
    if not matches and num is None and len(norm) >= 4:
        loose = [
            c["label"] for c in choices
            if (o := _normalize_value(c.get("text", ""))) and len(o) >= 4
            and (o in norm or norm in o)
        ]
        if len(loose) == 1:
            return loose[0], norm

    return None, None


def extract_final_answer(
    text: str,
    choices: Sequence[dict[str, str]] | Sequence[str] | None = None,
) -> Extraction:
    """
    Extrai a resposta escolhida do texto gerado.

    Cascata do formato exigido para os degradados, em cinco níveis:

    1. `strict`      — `FINAL ANSWER: B`, exatamente como o prompt pede
    2. `loose_final` — a mesma linha com ruído de markdown, parênteses, negrito
    3. `answer_is`   — resposta em prosa: "the correct answer is D"
    4. `value_match` — o modelo emitiu o VALOR da alternativa, não a letra
    5. `bare_letter` — uma letra sozinha nas últimas linhas

    O método usado é registrado no JSONL. Se `strict` cair muito para um modelo,
    isso é um achado sobre aderência a instrução, não um bug a esconder no
    extrator; e as métricas trazem `format_adherence` justamente para isso.

    Em todas as regras a ÚLTIMA ocorrência vale, porque o modelo pode enumerar
    alternativas antes de concluir. Blocos `<think>` são removidos primeiro, para
    que a letra não venha do rascunho interno de modelos de raciocínio.

    `choices` aceita a lista de alternativas do item (`[{"label", "text"}]`) ou
    só os rótulos (`["A", "B", ...]`). Com apenas os rótulos, a regra
    `value_match` não pode ser aplicada, porque não há texto para comparar.
    """
    if not text:
        return Extraction(None, "none", False, 0)

    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _UNCLOSED_THINK.sub(" ", cleaned)

    # Normaliza `choices` para as duas formas de que precisamos.
    choice_dicts: list[dict[str, str]] = []
    if choices:
        first = choices[0]
        if isinstance(first, dict):
            choice_dicts = [dict(c) for c in choices]  # type: ignore[arg-type]
            allowed = {c["label"] for c in choice_dicts}
        else:
            allowed = {str(c) for c in choices}
    else:
        allowed = set(CHOICE_LABELS)

    # Níveis 1 a 3: buscam a letra.
    for method, pattern in _PATTERNS:
        hits = [m.group(1).upper() for m in pattern.finditer(cleaned)]
        hits = [h for h in hits if h in allowed]
        if hits:
            return Extraction(
                letter=hits[-1],
                method=method,
                followed_format=(method == "strict"),
                n_candidates=len(set(hits)),
            )

    # Nível 4: o valor da alternativa em vez da letra.
    if choice_dicts:
        for pattern in _VALUE_SLOTS:
            captures = [m.group(1) for m in pattern.finditer(cleaned)]
            for content in reversed(captures):  # a última ocorrência vale
                label, value = _resolve_value(content, choice_dicts)
                if label and label in allowed:
                    return Extraction(
                        letter=label, method="value_match",
                        followed_format=False, n_candidates=1, matched_value=value,
                    )

    # Nível 5: letra solta, só perto do fim.
    tail = [ln for ln in cleaned.splitlines() if ln.strip()][-BARE_LETTER_TAIL_LINES:]
    bare = [
        m.group(1).upper()
        for ln in tail
        if (m := _BARE_LETTER.match(ln.strip())) and m.group(1).upper() in allowed
    ]
    if bare:
        return Extraction(
            letter=bare[-1], method="bare_letter",
            followed_format=False, n_candidates=len(set(bare)),
        )

    return Extraction(None, "none", False, 0)


def strip_think(text: str) -> str:
    """Remove blocos de raciocínio interno. Usado nas métricas de comprimento."""
    return _UNCLOSED_THINK.sub("", _THINK_BLOCK.sub("", text)).strip()


# ===========================================================================
# 4. FORMATO DE SAÍDA ÚNICO
# ===========================================================================


@dataclass
class Record:
    """
    Uma linha do JSONL. Vale para baseline, reflexão, avaliação e controles.

    Campos novos entram aqui como coluna opcional. Nunca criar um segundo
    formato de saída.
    """

    uid: str
    dataset: str
    split: str
    problem_type: str
    stage: str        # baseline | reflect | eval | retry | selfcons
    condition: str    # no_reflection | self_reflection | external_reflection | ...

    student_model: str
    teacher_model: str | None = None

    prompt: str = ""
    raw_output: str = ""
    predicted: str | None = None
    gold: str = ""
    is_correct: bool | None = None
    extraction_method: str = "none"
    followed_format: bool = False

    reflection_depth: str | None = None
    reflection_perspective: str | None = None
    reflection_text: str | None = None
    retrieved_uids: list[str] = field(default_factory=list)
    retrieved_similarities: list[float] = field(default_factory=list)
    k: int | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None

    seed: int | None = None
    temperature: float | None = None
    attempt: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def make_record(
    item: dict[str, Any],
    *,
    stage: str,
    condition: str,
    student_model: str,
    prompt: str,
    output: str,
    **kwargs: Any,
) -> Record:
    """Atalho que já roda o extrator e resolve is_correct de forma consistente."""
    # Passa as alternativas inteiras, não só os rótulos: sem o texto, a regra
    # value_match não funciona, e o GSM8K (alternativas numéricas) perde
    # respostas válidas como abstenção.
    ext = extract_final_answer(output, item["choices"])
    return Record(
        uid=item["uid"],
        dataset=item["dataset"],
        split=item["split"],
        problem_type=item.get("problem_type", ""),
        stage=stage,
        condition=condition,
        student_model=student_model,
        prompt=prompt,
        raw_output=output,
        predicted=ext.letter,
        gold=item["answerKey"],
        # None quando não houve letra: abstenção é registrada, não convertida em erro.
        is_correct=None if ext.letter is None else ext.letter == item["answerKey"],
        extraction_method=ext.method,
        followed_format=ext.followed_format,
        **_with_extraction_meta(kwargs, ext),
    )


def _with_extraction_meta(kwargs: dict[str, Any], ext: Extraction) -> dict[str, Any]:
    """Guarda em `extra` o que a extração viu, para auditoria posterior."""
    extra = dict(kwargs.pop("extra", None) or {})
    extra["extraction_candidates"] = ext.n_candidates
    if ext.matched_value is not None:
        extra["extraction_matched_value"] = ext.matched_value
    return {**kwargs, "extra": extra}


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_as_json(row))
            fh.write("\n")
            n += 1
    return n


def _as_json(row: Any) -> str:
    if isinstance(row, Record):
        return row.to_json()
    if hasattr(row, "__dataclass_fields__"):
        return json.dumps(asdict(row), ensure_ascii=False)
    return json.dumps(row, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ===========================================================================
# 5. SELEÇÃO DE AMOSTRAS: COCHRAN
# ===========================================================================


def cochran_sample_size(
    population: int,
    confidence: float = 0.95,
    margin: float = 0.05,
    proportion: float = 0.5,
    finite_correction: bool = True,
) -> dict[str, Any]:
    """
    Tamanho de amostra pela fórmula de Cochran (1977).

        n0 = z² · p · (1 - p) / e²
        n  = n0 / (1 + (n0 - 1) / N)

    Com p = 0.5 a variância é máxima, então n0 é o caso conservador: para 95% de
    confiança e margem de 5 pontos, n0 ≈ 385 independentemente de N. A correção
    finita só reduz, e reduz mais quanto menor a população.
    """
    if population <= 0:
        raise ValueError("population deve ser positivo")
    if not 0 < confidence < 1:
        raise ValueError("confidence deve estar em (0, 1)")
    if not 0 < margin < 1:
        raise ValueError("margin deve estar em (0, 1)")

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    n0_raw = (z**2) * proportion * (1 - proportion) / (margin**2)
    n_raw = n0_raw / (1 + (n0_raw - 1) / population) if finite_correction else n0_raw
    n = min(population, math.ceil(n_raw))

    return {
        "population": population,
        "confidence": confidence,
        "margin": margin,
        "proportion": proportion,
        "z": round(z, 6),
        "n0_raw": round(n0_raw, 4),
        "n0": math.ceil(n0_raw),
        "finite_correction": finite_correction,
        "n_raw": round(n_raw, 4),
        "n": n,
        "fraction_of_population": round(n / population, 6),
    }


def stratified_sample(
    items: Sequence[dict[str, Any]],
    n: int,
    stratify_by: str = "answerKey",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Amostra n itens preservando a distribuição de `stratify_by`.

    Alocação proporcional com maiores restos: cada estrato recebe
    floor(n · share) e as vagas restantes vão aos maiores restos. Garante soma
    exata igual a n e evita que o gabarito da amostra fique enviesado — se a
    proporção de "A" muda, a acurácia do chute muda com ela.
    """
    if n >= len(items):
        return list(items)

    rng = random.Random(seed)

    strata: dict[Any, list[int]] = {}
    for idx, item in enumerate(items):
        strata.setdefault(item.get(stratify_by), []).append(idx)

    total = len(items)
    quotas: dict[Any, int] = {}
    remainders: list[tuple[float, Any]] = []
    for key, idxs in strata.items():
        exact = n * len(idxs) / total
        quotas[key] = int(math.floor(exact))
        remainders.append((exact - math.floor(exact), key))

    leftover = n - sum(quotas.values())
    remainders.sort(key=lambda t: (-t[0], -len(strata[t[1]]), str(t[1])))
    for _, key in remainders[:leftover]:
        quotas[key] += 1

    chosen: list[int] = []
    for key, idxs in strata.items():
        chosen.extend(rng.sample(idxs, min(quotas[key], len(idxs))))

    return [items[i] for i in sorted(chosen)]


def label_distribution(items: Sequence[dict[str, Any]], key: str = "answerKey") -> dict[str, float]:
    counts = Counter(item.get(key) for item in items)
    total = sum(counts.values()) or 1
    return {str(k): round(v / total, 4) for k, v in sorted(counts.items(), key=lambda t: str(t[0]))}


# ===========================================================================
# 6. VALIDAÇÃO DO SCHEMA
# ===========================================================================


def label_invariant_report(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Verifica e resume a padronização de rótulos de um conjunto de itens.

    O invariante do projeto: **todo item, em todo dataset, usa rótulos de letra
    contíguos começando em A** — `A, B, C, ...`. O número de alternativas pode
    variar entre itens e entre datasets (o AQuA tem 5, uma fração do ARC tem 3
    ou 5), e isso não é problema: o que não pode variar é a forma do rótulo.

    A razão de o invariante ser sobre a forma e não sobre a quantidade: o
    extrator monta o conjunto de letras aceitas a partir das alternativas de
    CADA item, então um item de 3 alternativas nunca aceita "D". E a análise
    pondera a acurácia do chute por `num_choices`, item a item. O que quebraria
    as duas coisas é um rótulo numérico ou fora de ordem, porque aí qualquer
    número solto no raciocínio se tornaria candidato a resposta.

    `violations` vazio significa invariante satisfeito.
    """
    patterns = Counter("".join(c["label"] for c in i.get("choices") or []) for i in items)
    sizes = Counter(len(i.get("choices") or []) for i in items)

    violations: list[dict[str, Any]] = []
    for item in items:
        labels = [c.get("label") for c in item.get("choices") or []]
        expected = list(CHOICE_LABELS[: len(labels)])
        if labels != expected:
            violations.append({"uid": item.get("uid"), "labels": labels, "expected": expected})
        elif item.get("answerKey") not in labels:
            violations.append({
                "uid": item.get("uid"), "answerKey": item.get("answerKey"), "labels": labels,
            })

    n = len(items)
    chance = sum(1 / s for s in (len(i.get("choices") or []) for i in items) if s) / n if n else 0.0

    return {
        "n": n,
        "label_patterns": dict(patterns),
        "num_choices": dict(sorted(sizes.items())),
        "uniform": len(sizes) == 1,
        "violations": violations,
        "ok": not violations,
        "chance_accuracy": round(chance, 4),
    }


def validate_mcq_item(item: dict[str, Any]) -> list[str]:
    """Lista de problemas no item. Vazia significa item válido."""
    problems: list[str] = []

    for name in ("uid", "dataset", "split", "question", "choices", "answerKey"):
        if not item.get(name):
            problems.append(f"campo obrigatório ausente ou vazio: {name}")

    choices = item.get("choices") or []
    if len(choices) < 2:
        problems.append(f"precisa de ao menos 2 alternativas, tem {len(choices)}")

    labels = [c.get("label") for c in choices]
    if len(set(labels)) != len(labels):
        problems.append(f"rótulos duplicados: {labels}")
    if labels and labels != list(CHOICE_LABELS[: len(labels)]):
        problems.append(f"rótulos não canônicos: {labels}")
    if any(not str(c.get("text", "")).strip() for c in choices):
        problems.append("alternativa com texto vazio")

    texts = [str(c.get("text", "")).strip().lower() for c in choices]
    if len(set(texts)) != len(texts):
        problems.append("alternativas duplicadas")

    if item.get("answerKey") not in labels:
        problems.append(f"answerKey {item.get('answerKey')!r} não está em {labels}")

    return problems
