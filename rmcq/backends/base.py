"""
Interface de geração. Uma abstração, várias implementações.

Quem chama fala só com `Backend.generate`. A diferença entre vLLM,
transformers, Azure OpenAI, Ollama e o stub de teste fica confinada a cada
implementação em rmcq/backends/, e a escolha é uma variável de ambiente
(RMCQ_BACKEND) ou o `provider` do ModelSpec — ver rmcq.backends.get_backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping
import re
from typing import Any, Sequence


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_START = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove embedded reasoning traces and keep only visible model output."""
    value = text or ""
    value = _THINK_BLOCK.sub("", value)
    if "</think>" in value.lower():
        value = re.split(r"</think\s*>", value, flags=re.IGNORECASE)[-1]
    # A generation truncated inside a leading think block has no usable answer.
    if _THINK_START.match(value):
        return ""
    return value.strip()


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
        """Converte um dict de parâmetros de decodificação (estilo transformers/vLLM)."""
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

    def __post_init__(self) -> None:
        self.text = strip_thinking(self.text)
        if self.samples:
            self.samples = [strip_thinking(sample) for sample in self.samples]


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
        """Optional chat-template controls for locally loaded models."""
        return {}

    def render(self, tokenizer: Any, prompt: str, system: str | None = None) -> str:
        """Render a chat prompt as text for APIs that explicitly need text."""
        wrapper = self.spec.extra_kwargs.get("prompt_wrapper")
        if wrapper:
            body = f"{system}\n\n{prompt}" if system else prompt
            return str(wrapper).format(prompt=body)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs(),
        )

    def render_token_ids(
        self, tokenizer: Any, prompt: str, system: str | None = None
    ) -> list[int]:
        """Apply the chat template directly to tokens, without a text round-trip.

        Some tokenizers (notably Transformers' MistralCommonBackend) warn that
        ``apply_chat_template(tokenize=False)`` followed by a separate encode
        can change or duplicate special tokens. Local backends consume token
        ids, so they should use this lossless path instead.
        """
        wrapper = self.spec.extra_kwargs.get("prompt_wrapper")
        if wrapper:
            rendered = self.render(tokenizer, prompt, system)
            encoded = tokenizer(rendered, add_special_tokens=True)["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            return [int(token_id) for token_id in encoded]

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **self.template_kwargs(),
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("chat template returned an unexpected batched token structure")
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key})"
