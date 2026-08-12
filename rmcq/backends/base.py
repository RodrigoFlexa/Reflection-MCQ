"""
Interface de geração. Uma abstração, três implementações.

Todo o pipeline fala só com `Backend.generate`. A diferença entre vLLM,
transformers e o stub de teste fica confinada aqui, e a escolha é uma variável
de ambiente. Isso permite desenvolver e depurar com transformers, rodar a grade
com vLLM, e testar o pipeline inteiro sem GPU com o stub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class GenParams:
    """
    Parâmetros de decodificação, independentes de backend.

    `temperature = 0` significa greedy, e cada backend traduz isso do seu jeito
    (transformers quer do_sample=False, vLLM quer temperature=0.0).
    """

    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    seed: int | None = None
    stop: tuple[str, ...] = ()

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> "GenParams":
        """Converte os dicts STUDENT_GEN / TEACHER_GEN / SELFCONS_GEN do config."""
        merged = {**cfg, **overrides}
        do_sample = merged.get("do_sample", False)
        temperature = merged.get("temperature") or 0.0
        return cls(
            max_new_tokens=int(merged.get("max_new_tokens", 1024)),
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(merged.get("top_p") or 1.0) if do_sample else 1.0,
            n=int(merged.get("n", 1)),
            seed=merged.get("seed"),
            stop=tuple(merged.get("stop", ())),
        )


@dataclass
class Generation:
    """Uma geração e seu custo, para alimentar as colunas do Record."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    samples: list[str] = field(default_factory=list)  # preenchido quando n > 1


class Backend(ABC):
    """
    Contrato mínimo de um motor de geração.

    Implementações devem garantir três coisas:

    1. `generate` devolve resultados NA MESMA ORDEM dos prompts recebidos,
       mesmo que internamente reordene por tamanho para eficiência.
    2. O chat template do modelo é aplicado dentro do backend. Quem chama passa
       o conteúdo da mensagem do usuário, não o texto já formatado.
    3. `unload` libera VRAM de verdade, para permitir carregar o próximo modelo
       no mesmo processo.
    """

    def __init__(self, model_key: str) -> None:
        from rmcq.config import MODELS

        if model_key not in MODELS:
            raise KeyError(f"modelo {model_key!r} não está em config.MODELS: {sorted(MODELS)}")
        self.key = model_key
        self.spec = MODELS[model_key]

    @abstractmethod
    def generate(
        self,
        prompts: Sequence[str],
        params: GenParams,
        system: str | None = None,
        desc: str = "",
    ) -> list[Generation]:
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    def unload(self) -> None:
        return None

    def template_kwargs(self) -> dict[str, Any]:
        """Qwen3 é de pensamento híbrido; os outros ignoram esta flag."""
        from rmcq.config import QWEN_ENABLE_THINKING

        if "qwen3" in self.key:
            return {"enable_thinking": QWEN_ENABLE_THINKING}
        return {}

    def render(self, tokenizer: Any, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs(),
        )

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key})"
