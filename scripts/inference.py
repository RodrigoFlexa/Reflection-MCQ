"""
DEPRECADO. Substituído por `rmcq/backends/` e `rmcq/stages/`.

`ModelRunner` foi dividido em duas partes com responsabilidades separadas:

- `rmcq.backends.get_backend(model_key)` cuida de carregar e gerar, com
  implementações para vLLM, transformers e um stub sem GPU;
- as etapas em `rmcq.stages` cuidam de montar prompt, extrair resposta e gravar.

Equivalências:

    ModelRunner(key)                 -> get_backend(key)
    runner.answer(item)              -> ver rmcq.stages.baseline.run
    runner.reflect(item, ...)        -> ver rmcq.stages.reflect.run
    runner.unload()                  -> backend.unload(), ou use `with`
    accuracy(records)                -> rmcq.stages.analyze.accuracy_block

Este shim mantém `ModelRunner` e `accuracy` funcionando para o notebook 02.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rmcq.backends import GenParams, get_backend  # noqa: E402
from rmcq.backends.base import Generation  # noqa: F401,E402
from rmcq.common import (  # noqa: E402
    Record,
    build_answer_prompt,
    build_reflection_prompt,
    make_record,
)
from rmcq.config import SEED, STUDENT_GEN, TEACHER_GEN  # noqa: E402
from rmcq.stages.analyze import accuracy_block  # noqa: E402


class ModelRunner:
    """Envelope de compatibilidade em volta de um Backend."""

    def __init__(self, model_key: str, backend: str | None = None, **kwargs: Any) -> None:
        self.key = model_key
        self.backend = get_backend(model_key, backend, **kwargs)
        self.seed = SEED

    @property
    def model(self):
        return getattr(self.backend, "model", None) or getattr(self.backend, "llm", None)

    @property
    def tokenizer(self):
        return getattr(self.backend, "tokenizer", None)

    def generate(self, prompt: str, system: str | None = None, **gen_kwargs: Any) -> Generation:
        params = GenParams.from_config({**STUDENT_GEN, **gen_kwargs}, seed=self.seed)
        return self.backend.generate([prompt], params, system=system)[0]

    def answer(
        self,
        item: dict[str, Any],
        stage: str = "baseline",
        condition: str = "no_reflection",
        prefix: str = "",
        **overrides: Any,
    ) -> Record:
        params = GenParams.from_config({**STUDENT_GEN, **overrides}, seed=self.seed)
        prompt = (prefix + "\n\n" if prefix else "") + build_answer_prompt(item)
        gen = self.backend.generate([prompt], params)[0]
        return make_record(
            item,
            stage=stage,
            condition=condition,
            student_model=self.key,
            prompt=prompt,
            output=gen.text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            latency_s=gen.latency_s,
            seed=self.seed,
            temperature=params.temperature,
        )

    def reflect(
        self,
        item: dict[str, Any],
        previous_answer: str,
        was_correct: bool,
        depth: str = "simple",
        perspective: str = "teacher",
        **overrides: Any,
    ) -> Generation:
        params = GenParams.from_config({**TEACHER_GEN, **overrides}, seed=self.seed)
        prompt = build_reflection_prompt(item, previous_answer, was_correct, depth, perspective)
        return self.backend.generate([prompt], params)[0]

    def unload(self) -> None:
        self.backend.unload()


def accuracy(records: Sequence[Record | dict[str, Any]]) -> dict[str, Any]:
    rows = [r if isinstance(r, dict) else r.__dict__ for r in records]
    block = accuracy_block(rows)
    # Nomes antigos, mantidos para o notebook 02.
    block["accuracy_strict"] = block.get("accuracy")
    return block
