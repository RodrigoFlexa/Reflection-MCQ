"""
Backend transformers, com batching ordenado por tamanho.

Use para desenvolver, depurar e inspecionar geração item a item. Para rodar a
grade completa, prefira o vLLM: aqui a GPU fica ociosa esperando a sequência
mais longa de cada lote.

Dois detalhes que costumam passar batido e produzir lixo silencioso:

1. **Padding à esquerda.** Modelos decoder-only geram a partir do último token.
   Com padding à direita, os tokens de pad ficam entre o prompt e a geração e o
   modelo continua a partir de um pad. O resultado não quebra, só fica ruim.
2. **Ordenação por tamanho.** Agrupar prompts de tamanho parecido reduz muito o
   padding desperdiçado. A ordem original é restaurada no fim, porque quem
   chama depende dela para casar resultado com item.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import (
    HF_BATCH_SIZE,
    HF_HOME,
    LOAD_IN_4BIT,
    MAX_MODEL_LEN,
    SEED,
    TORCH_DTYPE,
    hf_token,
)
from rmcq.store import get_logger, progress

log = get_logger(__name__)


class HFBackend(Backend):
    def __init__(
        self,
        model_key: str,
        batch_size: int = HF_BATCH_SIZE,
        dtype: str = TORCH_DTYPE,
        load_in_4bit: bool = LOAD_IN_4BIT,
    ) -> None:
        super().__init__(model_key)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.batch_size = max(1, batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.repo_id,
            cache_dir=str(HF_HOME),
            token=hf_token(),
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ver docstring do módulo: obrigatório para geração em lote.
        self.tokenizer.padding_side = "left"
        # O padrão do HF é truncar à direita, o que comeria justamente o fim do
        # prompt (a questão e as instruções de formato). O backend vLLM trunca à
        # esquerda; sem isto os dois cortariam lados opostos do mesmo prompt.
        self.tokenizer.truncation_side = "left"

        load_kwargs: dict[str, Any] = {
            "cache_dir": str(HF_HOME),
            "token": hf_token(),
            "trust_remote_code": self.spec.trust_remote_code,
            "device_map": "auto",
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["dtype"] = getattr(torch, dtype)

        log.info("carregando %s (%s) via transformers", model_key, self.spec.repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.spec.repo_id, **load_kwargs)
        self.model.eval()

        self.max_len = MAX_MODEL_LEN or getattr(self.model.config, "max_position_embeddings", None)
        log.info("  device=%s  max_len=%s", self.model.device, self.max_len)

    # -- utilidades ---------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _gen_kwargs(self, params: GenParams) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": params.n,
        }
        if params.greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                do_sample=True, temperature=params.temperature, top_p=params.top_p
            )
        return kwargs

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

        torch = self._torch
        torch.manual_seed(params.seed if params.seed is not None else SEED)

        token_ids = [self.render_token_ids(self.tokenizer, p, system) for p in prompts]
        lengths = [len(ids) for ids in token_ids]

        budget = (self.max_len - params.max_new_tokens) if self.max_len else None
        if budget is not None:
            if budget <= 0:
                raise ValueError(
                    f"{self.key}: max_new_tokens={params.max_new_tokens} não deixa espaço "
                    f"para o prompt em um contexto de {self.max_len} tokens. Baixe "
                    f"RMCQ_MAX_NEW_TOKENS ou suba RMCQ_MAX_MODEL_LEN."
                )
            over = [i for i, n in enumerate(lengths) if n > budget]
            if over:
                log.warning(
                    "%d prompts excedem o orçamento (%d tokens); serão truncados à esquerda",
                    len(over), budget,
                )

        # Decrescente por tamanho: o primeiro lote é o mais caro, então um OOM
        # aparece imediatamente em vez de depois de meia hora de execução.
        order = sorted(range(len(token_ids)), key=lambda i: -lengths[i])

        results: list[Generation | None] = [None] * len(token_ids)
        batches = [order[i: i + self.batch_size] for i in range(0, len(order), self.batch_size)]

        for batch in progress(batches, desc=desc or f"{self.key} (hf)"):
            batch_ids = [token_ids[i][-budget:] if budget is not None else token_ids[i] for i in batch]
            enc = self.tokenizer.pad(
                {"input_ids": batch_ids},
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            n_prompt = int(enc["input_ids"].shape[-1])
            started = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(**enc, **self._gen_kwargs(params))
            latency = time.perf_counter() - started

            completions = out[:, n_prompt:]
            per_item = latency / max(len(batch), 1)

            for pos, idx in enumerate(batch):
                # Com n > 1 as amostras vêm intercaladas: item0_s0, item0_s1, ...
                lo = pos * params.n
                sample_ids = completions[lo: lo + params.n]
                samples = self.tokenizer.batch_decode(sample_ids, skip_special_tokens=True)

                # Tokens reais gerados, descontando o padding do fim.
                tok_counts = [
                    int((row != self.tokenizer.pad_token_id).sum().item()) for row in sample_ids
                ]
                # Comprimento sem padding do prompt deste item específico.
                attn = enc["attention_mask"][pos]
                real_prompt_tokens = int(attn.sum().item())

                results[idx] = Generation(
                    text=samples[0],
                    prompt_tokens=real_prompt_tokens,
                    completion_tokens=sum(tok_counts),
                    latency_s=round(per_item, 4),
                    finish_reason="length" if max(tok_counts) >= params.max_new_tokens else "stop",
                    samples=samples if params.n > 1 else [],
                )

            del out, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert all(r is not None for r in results), "algum prompt não recebeu resultado"
        return results  # type: ignore[return-value]

    def unload(self) -> None:
        log.info("liberando %s", self.key)
        try:
            del self.model
        except AttributeError:
            pass
        import gc

        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
