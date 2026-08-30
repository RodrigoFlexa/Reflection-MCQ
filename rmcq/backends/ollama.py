"""
Backend Ollama: fala com um servidor Ollama local (ou remoto) via HTTP.

É o caminho mais curto para testar um modelo novo: baixe com
`ollama pull <tag>`, declare a tag em MODELS (ou em RMCQ_OLLAMA_MODELS, ver
rmcq/config.py) e chame `get_backend(<tag>)` — nenhum código novo é
necessário.

Diferença dos backends hf/vllm: aqui não existe tokenizer nem chat template
locais. O servidor já aplica o template do modelo, então enviamos mensagens
(system/user) em vez de texto pré-formatado — o mesmo raciocínio do backend
Azure.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX, OLLAMA_TIMEOUT
from rmcq.store import get_logger, progress

log = get_logger(__name__)


class OllamaBackend(Backend):
    """
    Cliente HTTP para `ollama serve` (padrão: http://localhost:11434).

    `unload()` pede ao servidor para descarregar o modelo (keep_alive=0), para
    liberar VRAM antes de carregar o próximo — mesmo contrato dos outros
    backends, mesmo que aqui o processo do Python não segure nada na GPU.
    """

    def __init__(
        self,
        model_key: str,
        base_url: str | None = None,
        num_ctx: int | None = None,
        keep_alive: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(model_key)

        try:
            import requests
        except ImportError as exc:  # noqa: TRY003
            raise ImportError(
                "o pacote 'requests' não está instalado. Rode: pip install requests"
            ) from exc

        self._requests = requests
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = self.spec.extra_kwargs.get("tag", self.spec.repo_id)
        self.num_ctx = num_ctx or OLLAMA_NUM_CTX
        self.keep_alive = keep_alive if keep_alive is not None else OLLAMA_KEEP_ALIVE
        self.timeout = timeout or OLLAMA_TIMEOUT

        log.info("OllamaBackend(%s) model=%s url=%s", model_key, self.model, self.base_url)

    # -- montagem da chamada ------------------------------------------------

    def _options(self, params: GenParams) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_predict": params.max_new_tokens,
        }
        if params.seed is not None:
            options["seed"] = params.seed
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if params.stop:
            options["stop"] = list(params.stop)
        return options

    def _complete_one(self, prompt: str, params: GenParams, system: str | None) -> Generation:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(params),
            "keep_alive": self.keep_alive,
        }

        started = time.perf_counter()
        response = self._requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        latency = time.perf_counter() - started

        text = (data.get("message") or {}).get("content", "")
        return Generation(
            text=text,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
            latency_s=round(latency, 4),
            finish_reason="stop" if data.get("done", True) else "length",
        )

    # -- interface Backend ---------------------------------------------------

    def generate(
        self,
        prompts: Sequence[str],
        params: GenParams,
        system: str | None = None,
        desc: str = "",
    ) -> list[Generation]:
        if not prompts:
            return []

        results: list[Generation] = []
        for prompt in progress(prompts, desc=desc or f"{self.key} (ollama)"):
            first = self._complete_one(prompt, params, system)
            samples = [first.text]
            for _ in range(max(1, params.n) - 1):
                samples.append(self._complete_one(prompt, params, system).text)
            results.append(
                Generation(
                    text=first.text,
                    prompt_tokens=first.prompt_tokens,
                    completion_tokens=first.completion_tokens,
                    latency_s=first.latency_s,
                    finish_reason=first.finish_reason,
                    samples=samples if params.n > 1 else [],
                )
            )
        return results

    def count_tokens(self, text: str) -> int:
        # Ollama não expõe um tokenizer por HTTP; aproximação por caracteres.
        return max(1, len(text) // 4)

    def unload(self) -> None:
        try:
            self._requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - unload é best-effort
            log.warning("não consegui pedir unload ao Ollama: %s", exc)

    def render(self, tokenizer: Any, prompt: str, system: str | None = None) -> str:
        # O chat template é do servidor. Nunca chamado por este backend, mas
        # sobrescrito para não explodir se alguém chamar sem tokenizer.
        return f"{system}\n\n{prompt}" if system else prompt
