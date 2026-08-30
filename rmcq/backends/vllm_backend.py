"""
Backend vLLM. É o que se usa para rodar a grade.

O ganho vem do continuous batching: em vez de esperar o lote inteiro terminar,
o vLLM coloca uma nova sequência na vaga assim que outra acaba, e a GPU fica
saturada. Como as respostas deste experimento variam muito de tamanho (uma
questão de ARC resolve em 80 tokens, uma de GSM8K pode usar 600), a diferença
contra batching estático é grande.

Nota sobre a semente: o vLLM aceita seed na criação do engine e por requisição.
Com temperatura 0 a saída é determinística de qualquer forma; com amostragem, a
semente por requisição é o que garante reprodutibilidade da etapa de reflexão.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import (
    HF_HOME,
    MAX_MODEL_LEN,
    SEED,
    TORCH_DTYPE,
    VLLM_DETERMINISTIC,
    VLLM_GPU_UTIL,
    VLLM_MAX_NUM_SEQS,
    hf_token,
    n_visible_gpus,
)
from rmcq.store import get_logger

log = get_logger(__name__)


class VLLMBackend(Backend):
    def __init__(
        self,
        model_key: str,
        dtype: str = TORCH_DTYPE,
        gpu_util: float = VLLM_GPU_UTIL,
        max_model_len: int | None = MAX_MODEL_LEN,
        deterministic: bool = VLLM_DETERMINISTIC,
        device: int | None = None,
    ) -> None:
        super().__init__(model_key)

        from vllm import LLM

        if hf_token():
            os.environ.setdefault("HF_TOKEN", hf_token())
        os.environ.setdefault("HF_HOME", str(HF_HOME))

        # device fixa este engine numa GPU específica (ex.: aluno na 4, juiz na 5),
        # sobrescrevendo CUDA_VISIBLE_DEVICES só para o subprocesso do engine core que o
        # LLM(...) abaixo dispara. Isto funciona apesar de o resto do projeto depender de
        # CUDA_VISIBLE_DEVICES ser lido uma vez só, antes de qualquer import de torch (ver
        # .env): quem inicializa CUDA de verdade é o subprocesso filho do vLLM, criado por
        # multiprocessing DEPOIS desta atribuição, e ele herda o ambiente do momento em que
        # nasce -- o processo pai (este) nunca toca CUDA diretamente.
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
            n_gpus = 1
        else:
            # Uma GPU visível = sem paralelismo. Duas ou mais = tensor parallel.
            # Quem controla isso é CUDA_VISIBLE_DEVICES no .env.
            n_gpus = n_visible_gpus()
        tp_size = max(1, n_gpus) if n_gpus > 0 else 1

        log.info(
            "carregando %s (%s) via vLLM  tp=%d  gpu_util=%.2f  deterministic=%s",
            model_key, self.spec.repo_id, tp_size, gpu_util, deterministic,
        )

        kwargs: dict[str, Any] = dict(
            model=self.spec.repo_id,
            tokenizer=self.spec.repo_id,
            dtype=dtype,
            trust_remote_code=self.spec.trust_remote_code,
            gpu_memory_utilization=gpu_util,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            seed=SEED,
            download_dir=str(HF_HOME),
        )
        self.deterministic = deterministic
        if deterministic:
            # Ver o comentário de VLLM_DETERMINISTIC em rmcq/config.py. Estas
            # quatro flags tiram as fontes de variação numérica que dependem da
            # composição do lote; sem elas a mesma pergunta, com temperatura 0,
            # responde diferente em ~5,6% dos casos entre execuções.
            kwargs.update(
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                enforce_eager=True,
                max_num_seqs=VLLM_MAX_NUM_SEQS,
            )
        # O conjunto de kwargs do LLM muda entre versões do vLLM (enable_chunked_prefill
        # saiu do construtor em algumas builds). Filtrar pela assinatura evita quebrar
        # um run de horas por um kwarg que aquela versão não conhece -- e avisa qual
        # caiu, para não haver silêncio sobre determinismo que não está valendo.
        kwargs = self._supported_kwargs(LLM, kwargs)
        try:
            self.llm = LLM(**kwargs)
        finally:
            if device is not None:
                if prev_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        self.tokenizer = self.llm.get_tokenizer()
        self.max_len = max_model_len or getattr(
            self.llm.llm_engine.model_config, "max_model_len", None
        )
        log.info("  max_model_len=%s", self.max_len)

    # -- utilidades ---------------------------------------------------------

    @staticmethod
    def _supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Descarta kwargs que esta versão do vLLM não aceita, avisando quais."""
        import inspect

        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        known = set(sig.parameters)
        dropped = sorted(k for k in kwargs if k not in known)
        if dropped:
            log.warning(
                "vLLM desta versão não aceita %s -- ignorado. Se estiver em modo "
                "determinístico, confira se o efeito equivalente está desligado por "
                "variável de ambiente.", ", ".join(dropped),
            )
        return {k: v for k, v in kwargs.items() if k in known}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _sampling(self, params: GenParams) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            n=params.n,
            temperature=params.temperature,  # 0.0 já é greedy no vLLM
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            seed=params.seed if params.seed is not None else SEED,
            stop=list(params.stop) or None,
        )

    # -- geração ------------------------------------------------------------

    def generate(
        self,
        prompts: Sequence[str],
        params: GenParams,
        system: str | None = None,
        desc: str = "",
    ) -> list[Generation]:
        if not prompts:
            return []

        from vllm import TokensPrompt

        rendered = [self.render(self.tokenizer, p, system) for p in prompts]
        # Tokenizamos aqui e entregamos ids ao engine, em vez de texto. Cortar em
        # tokens, decodificar e deixar o vLLM re-tokenizar não preserva a
        # contagem: o corte cai no meio de um caractere multibyte ou de um token
        # especial e o texto volta com mais tokens do que o orçamento. Passando
        # TokensPrompt, o que cortamos é exatamente o que o engine recebe.
        token_ids = [self.tokenizer(r, add_special_tokens=False)["input_ids"] for r in rendered]

        if self.max_len:
            budget = self.max_len - params.max_new_tokens
            if budget <= 0:
                raise ValueError(
                    f"{self.key}: max_new_tokens={params.max_new_tokens} não deixa espaço "
                    f"para o prompt em um contexto de {self.max_len} tokens. Baixe "
                    f"RMCQ_MAX_NEW_TOKENS ou suba RMCQ_MAX_MODEL_LEN."
                )
            long_ones = 0
            for i, ids in enumerate(token_ids):
                if len(ids) > budget:
                    # Truncamento à esquerda: o final do prompt (a questão e as
                    # instruções de formato) é o que não pode ser perdido.
                    token_ids[i] = ids[-budget:]
                    long_ones += 1
            if long_ones:
                log.warning("%d prompts truncados à esquerda para caber em %d tokens",
                            long_ones, budget)

        started = time.perf_counter()
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in token_ids],
            self._sampling(params),
            use_tqdm=True,
        )
        total_latency = time.perf_counter() - started

        # O vLLM devolve na ordem dos prompts, mas confirmamos por request_id
        # em vez de confiar: um resultado deslocado corromperia todo o JSONL.
        by_id = {out.request_id: out for out in outputs}
        ordered = (
            [by_id[str(i)] for i in range(len(rendered))]
            if all(str(i) in by_id for i in range(len(rendered)))
            else list(outputs)
        )

        per_item = total_latency / max(len(ordered), 1)
        results: list[Generation] = []
        for out in ordered:
            samples = [c.text for c in out.outputs]
            results.append(
                Generation(
                    text=samples[0],
                    prompt_tokens=len(out.prompt_token_ids or []),
                    completion_tokens=sum(len(c.token_ids) for c in out.outputs),
                    latency_s=round(per_item, 4),
                    finish_reason=out.outputs[0].finish_reason or "",
                    samples=samples if params.n > 1 else [],
                )
            )

        gen_tokens = sum(r.completion_tokens for r in results)
        log.info(
            "  %d prompts, %d tokens gerados em %.1fs (%.0f tok/s)",
            len(results), gen_tokens, total_latency, gen_tokens / max(total_latency, 1e-6),
        )
        return results

    def unload(self) -> None:
        log.info("liberando %s", self.key)
        import contextlib
        import gc

        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()

        with contextlib.suppress(AttributeError):
            del self.llm

        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
