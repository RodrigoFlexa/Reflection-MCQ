"""
Ciclo completo de reflexão num item só: baseline -> reflexão -> avaliação.

    python example_reflection.py <chave-do-modelo>

Mostra os três prompts de rmcq/prompts.py em ação:
  1. build_answer_prompt   — responde sem reflexão (baseline)
  2. build_reflection_prompt — o próprio modelo comenta a resposta errada
  3. build_eval_prompt     — responde de novo, agora com a reflexão no prompt

Não depende de dataset nem de índice de similaridade: o "item anterior" e o
"item novo" são dois exemplos fixos, só para exercitar os três prompts de
ponta a ponta contra um backend de verdade (ou o stub, sem GPU).
"""

from __future__ import annotations

import sys

import rmcq  # noqa: F401 — carrega o .env antes de qualquer import de torch
from rmcq.backends import get_backend
from rmcq.backends.base import GenParams
from rmcq.prompts import build_answer_prompt, build_eval_prompt, build_reflection_prompt

PREVIOUS_ITEM = {
    "question": "If 3 apples cost $6, how much do 5 apples cost?",
    "context": None,
    "choices": [
        {"label": "A", "text": "$8"},
        {"label": "B", "text": "$10"},
        {"label": "C", "text": "$12"},
        {"label": "D", "text": "$15"},
    ],
    "answerKey": "B",
}

NEW_ITEM = {
    "question": "If 4 pens cost $12, how much do 7 pens cost?",
    "context": None,
    "choices": [
        {"label": "A", "text": "$18"},
        {"label": "B", "text": "$21"},
        {"label": "C", "text": "$24"},
        {"label": "D", "text": "$28"},
    ],
    "answerKey": "B",
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    model_key = sys.argv[1]
    params = GenParams(max_new_tokens=300)

    with get_backend(model_key) as backend:
        # 1. baseline: responde o item anterior sem nenhuma reflexão
        baseline_prompt = build_answer_prompt(PREVIOUS_ITEM)
        [baseline] = backend.generate([baseline_prompt], params)
        print("=== baseline ===")
        print(baseline.text, "\n")

        was_correct = PREVIOUS_ITEM["answerKey"] in baseline.text
        print(f"(considerado {'correto' if was_correct else 'incorreto'} para o exemplo)\n")

        # 2. reflexão: o modelo comenta a própria resposta
        reflection_prompt = build_reflection_prompt(
            PREVIOUS_ITEM, baseline.text, was_correct, depth="simple", perspective="student",
        )
        [reflection] = backend.generate([reflection_prompt], params)
        print("=== reflexão ===")
        print(reflection.text, "\n")

        # 3. avaliação: responde um item NOVO com a reflexão injetada no prompt
        eval_prompt = build_eval_prompt(
            NEW_ITEM,
            reflections=[reflection.text],
            source_questions=[PREVIOUS_ITEM["question"]],
            source_was_correct=[was_correct],
        )
        [evaluation] = backend.generate([eval_prompt], params)
        print("=== avaliação (com reflexão) ===")
        print(evaluation.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
