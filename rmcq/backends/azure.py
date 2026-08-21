"""
Backend de geração via Azure OpenAI, para os professores grandes (GPT-5, GPT-4).

Implementa guia-azure-openai-fgl.md. Um cliente "de livro-texto" quebraria de
três formas nesta grade, e cada uma delas tem uma defesa aqui:

1. **Modelos de reasoning gastam orçamento pensando.** gpt-5, o1, o3 e o4-mini
   usam `max_completion_tokens`, rejeitam `temperature` e `seed`, e consomem
   parte do orçamento em raciocínio interno ANTES de escrever. Um teto pensado
   para uma reflexão curta volta como `content=""` com `finish_reason="length"`.
   Daí o piso `AZURE_REASONING_MIN_TOKENS`.
2. **Gateways rejeitam parâmetros opcionais** que o modelo suporta. Um 400 que
   cita o nome do parâmetro derruba aquele parâmetro e repete a chamada — não a
   execução inteira.
3. **Resposta vazia não é abstenção.** Se um deployment errado devolvesse "" e
   isso virasse `reflection_text=""`, a grade inteira terminaria com 14 mil
   linhas plausíveis e vazias, e a análise não acusaria nada. Aqui a primeira
   vazia aborta com o diagnóstico.

O cache em disco não é só economia: `reflect` só grava o JSONL depois que o lote
inteiro (até ~380 prompts) volta, então uma queda no fim do lote jogaria fora
tudo que já foi pago. Com cache, o rerun replica o lote de graça.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import ssl
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import (
    AZURE_API_KEY_VAR,
    AZURE_BASE_URL_VAR,
    AZURE_CA_BUNDLE,
    AZURE_API_VERSION,
    AZURE_BACKOFF_BASE,
    AZURE_BACKOFF_MAX,
    AZURE_CACHE,
    AZURE_CONCURRENCY,
    AZURE_ENDPOINT_VAR,
    AZURE_FAIL_ON_EMPTY,
    AZURE_HEALTH_CHECK_CALLS,
    AZURE_MAX_EMPTY_RATE,
    AZURE_MAX_RETRIES,
    AZURE_MAX_TOKENS,
    AZURE_REASONING_EFFORT,
    AZURE_REASONING_MIN_TOKENS,
    CACHE_DIR,
    azure_deployment,
)
from rmcq.store import get_logger, progress

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Detecção de modelo de reasoning
# ---------------------------------------------------------------------------
# A checagem é sobre o nome do DEPLOYMENT, não sobre o nome oficial do modelo:
# deployments corporativos costumam vir com prefixo ou sufixo próprio, e é o
# nome do deployment que o código tem em mãos.

REASONING_MARKERS = ("gpt-5", "gpt5", "o1-", "o3-", "o4-", "-o1", "-o3", "-o4")


def is_reasoning_deployment(name: str) -> bool:
    n = (name or "").lower()
    return any(m in n for m in REASONING_MARKERS)


def budget(max_tokens: int, reasoning: bool, reasoning_min_tokens: int) -> int:
    """Teto de tokens de resposta. 0 significa "não mande cap nenhum"."""
    if not reasoning:
        return max_tokens
    if reasoning_min_tokens <= 0:
        return 0
    return max(max_tokens, reasoning_min_tokens)


# Parâmetros que dá para sacrificar se o gateway reclamar deles. A lista é
# fechada de propósito: `model` e `messages` nunca saem.
OPTIONAL_PARAMS = (
    "seed", "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "response_format", "max_tokens", "max_completion_tokens", "reasoning_effort",
)

# Erros que valem uma nova tentativa: o problema é do momento, não da chamada.
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)
RETRY_NAMES = ("ratelimit", "timeout", "apiconnection", "serviceunavailable", "internalserver")


def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _is_transient(exc: Exception) -> bool:
    if _status_of(exc) in RETRY_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(marker in name for marker in RETRY_NAMES)


def _retry_after_seconds(exc: Exception) -> float:
    """Respeita o Retry-After do servidor quando ele diz quanto esperar."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000.0
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        pass
    return 0.0


class AzureEmptyResponse(RuntimeError):
    """Resposta vazia do Azure. É falha de configuração, não abstenção."""


class AzureBackend(Backend):
    """
    Professor de API. Sem VRAM, sem download, sem chat template local.

    `unload()` é barato porque não há o que descarregar — mas o método existe
    para que o `with get_backend(...)` de reflect.py funcione igual ao dos
    modelos locais.
    """

    def __init__(
        self,
        model_key: str,
        deployment: str | None = None,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        use_cache: bool | None = None,
    ) -> None:
        super().__init__(model_key)

        self.deployment = deployment or azure_deployment(model_key)
        self.reasoning = is_reasoning_deployment(self.deployment)
        self.concurrency = max(1, int(concurrency or AZURE_CONCURRENCY))
        self.max_tokens = int(max_tokens or AZURE_MAX_TOKENS)
        self.use_cache = AZURE_CACHE if use_cache is None else use_cache

        # Parâmetros que este gateway já rejeitou. Memorizados para não
        # reenviar em toda chamada seguinte.
        self._unsupported: set[str] = set()
        self._lock = threading.Lock()
        self._calls = 0
        self._empties = 0

        self._client = self._make_client()

        log.info(
            "AzureBackend(%s) deployment=%s modo=%s concorrência=%d teto=%d tokens%s",
            model_key, self.deployment,
            "reasoning" if self.reasoning else "chat",
            self.concurrency,
            budget(self.max_tokens, self.reasoning, AZURE_REASONING_MIN_TOKENS),
            " cache=on" if self.use_cache else "",
        )

    # -- cliente ------------------------------------------------------------

    def _make_client(self) -> Any:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # noqa: TRY003
            raise ImportError(
                "o SDK da OpenAI não está instalado. "
                "Rode: pip install -r requirements-azure.txt"
            ) from exc

        api_key = os.environ.get(AZURE_API_KEY_VAR, "").strip()
        base_url = os.environ.get(AZURE_BASE_URL_VAR, "").strip()
        endpoint = os.environ.get(AZURE_ENDPOINT_VAR, "").strip()

        if not api_key:
            raise RuntimeError(
                f"{AZURE_API_KEY_VAR} ausente. Defina no .env desta máquina "
                f"(veja .env.example) — nunca commite o .env."
            )
        if not base_url and not endpoint:
            raise RuntimeError(
                f"defina {AZURE_BASE_URL_VAR} ou {AZURE_ENDPOINT_VAR} no .env.\n"
                f"  {AZURE_BASE_URL_VAR}: a URL é usada como está — é o que gateway "
                f"corporativo costuma exigir.\n"
                f"  {AZURE_ENDPOINT_VAR}: o SDK monta /openai/deployments/<modelo>/... "
                f"em cima dela (padrão <recurso>.openai.azure.com)."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_version": AZURE_API_VERSION,
            # O retry é nosso: precisamos intercalar a queda de parâmetro
            # rejeitado com o backoff, e o SDK não sabe fazer isso.
            "max_retries": 0,
        }
        # Os dois são mutuamente exclusivos no SDK; base_url ganha porque é a
        # forma explícita — quem a define está dizendo "a URL é exatamente esta".
        if base_url:
            kwargs["base_url"] = base_url
            if endpoint:
                log.warning(
                    "%s e %s definidos; usando %s (são mutuamente exclusivos no SDK)",
                    AZURE_BASE_URL_VAR, AZURE_ENDPOINT_VAR, AZURE_BASE_URL_VAR,
                )
        else:
            kwargs["azure_endpoint"] = endpoint

        http_client = self._make_http_client()
        if http_client is not None:
            kwargs["http_client"] = http_client

        return AzureOpenAI(**kwargs)

    def _make_http_client(self) -> Any:
        """
        Cliente HTTP com o certificado raiz corporativo, quando houver.

        Rede com inspeção TLS apresenta um certificado assinado pela CA da
        empresa. Sem esse PEM, a verificação falha e a conexão nem chega ao
        Azure — erro de SSL que não se parece nada com um problema de API.
        """
        if not AZURE_CA_BUNDLE:
            return None

        from rmcq import ROOT

        caminho = Path(AZURE_CA_BUNDLE)
        if not caminho.is_absolute():
            caminho = ROOT / caminho
        if not caminho.exists():
            raise RuntimeError(
                f"AZURE_OPENAI_CA_BUNDLE aponta para {caminho}, que não existe.\n"
                f"Copie o PEM da CA raiz para essa máquina, ou deixe a variável "
                f"vazia se a rede não fizer inspeção TLS."
            )

        import httpx

        log.info("usando certificado raiz corporativo: %s", caminho)
        try:
            return httpx.Client(verify=str(caminho), timeout=httpx.Timeout(600.0, connect=30.0))
        except ssl.SSLError as exc:
            # "[X509] PEM lib" não diz nada a quem só copiou um arquivo errado.
            raise RuntimeError(
                f"{caminho} não é um certificado PEM válido ({exc}).\n"
                f"O arquivo precisa começar com '-----BEGIN CERTIFICATE-----'. "
                f"Confira se a cópia veio inteira e não em formato DER/PKCS#12."
            ) from exc

    # -- montagem da chamada ------------------------------------------------

    def build_kwargs(
        self,
        prompt: str,
        params: GenParams,
        system: str | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Traduz GenParams para o dialeto certo (chat ou reasoning)."""
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {"model": self.deployment, "messages": messages}

        b = budget(self.max_tokens, self.reasoning, AZURE_REASONING_MIN_TOKENS)
        if b > 0:
            kwargs["max_completion_tokens" if self.reasoning else "max_tokens"] = b

        if not self.reasoning:
            # temperature e seed são do ramo chat apenas. Um modelo de reasoning
            # recusa os dois.
            kwargs["temperature"] = params.temperature
            if params.top_p and params.top_p < 1.0:
                kwargs["top_p"] = params.top_p
            if params.seed is not None:
                kwargs["seed"] = params.seed
        elif AZURE_REASONING_EFFORT:
            kwargs["reasoning_effort"] = AZURE_REASONING_EFFORT

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for name in self._unsupported:
            kwargs.pop(name, None)
        return kwargs

    def _maybe_drop_parameter(self, exc: Exception, kwargs: dict[str, Any]) -> bool:
        """
        400/422 citando o nome de um parâmetro opcional: derruba e repete.

        Devolve True se algo foi removido — aí a chamada vale uma nova tentativa
        sem gastar orçamento de retry transitório.
        """
        if _status_of(exc) not in (400, 422):
            return False
        message = str(exc)
        for name in OPTIONAL_PARAMS:
            if name in self._unsupported or name not in kwargs:
                continue
            if re.search(rf"\b{re.escape(name)}\b", message):
                with self._lock:
                    self._unsupported.add(name)
                kwargs.pop(name, None)
                log.warning(
                    "gateway rejeitou %r; removido e não será mais enviado (%s)",
                    name, message.splitlines()[0][:160],
                )
                return True
        return False

    # -- cache --------------------------------------------------------------

    def _cache_path(self, kwargs: dict[str, Any]):
        key = json.dumps(
            {
                "deployment": self.deployment,
                "messages": kwargs.get("messages"),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "max_tokens": kwargs.get("max_tokens"),
                "max_completion_tokens": kwargs.get("max_completion_tokens"),
                "seed": kwargs.get("seed"),
                "reasoning_effort": kwargs.get("reasoning_effort"),
                "response_format": kwargs.get("response_format"),
            },
            sort_keys=True, ensure_ascii=False,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # Dois níveis: 28 mil arquivos num diretório só é desagradável de listar.
        return CACHE_DIR / "azure" / digest[:2] / f"{digest}.json"

    def _cache_read(self, path) -> Generation | None:
        if not self.use_cache or not path.exists():
            return None
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # cache corrompido é como cache ausente
        return Generation(
            text=row["text"],
            prompt_tokens=row.get("prompt_tokens", 0),
            completion_tokens=row.get("completion_tokens", 0),
            latency_s=row.get("latency_s", 0.0),
            finish_reason=row.get("finish_reason", ""),
        )

    def _cache_write(self, path, gen: Generation) -> None:
        if not self.use_cache:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "text": gen.text,
                        "prompt_tokens": gen.prompt_tokens,
                        "completion_tokens": gen.completion_tokens,
                        "latency_s": gen.latency_s,
                        "finish_reason": gen.finish_reason,
                        "deployment": self.deployment,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("não consegui gravar o cache em %s: %s", path, exc)

    # -- uma chamada --------------------------------------------------------

    def _complete_one(
        self,
        prompt: str,
        params: GenParams,
        system: str | None,
        json_mode: bool,
    ) -> Generation:
        kwargs = self.build_kwargs(prompt, params, system=system, json_mode=json_mode)

        cache_path = self._cache_path(kwargs)
        cached = self._cache_read(cache_path)
        if cached is not None:
            return cached

        attempt = 0
        last_exc: Exception | None = None

        while attempt <= AZURE_MAX_RETRIES:
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - classificado logo abaixo
                last_exc = exc
                if self._maybe_drop_parameter(exc, kwargs):
                    continue  # não conta como tentativa: a chamada mudou
                if not _is_transient(exc) or attempt == AZURE_MAX_RETRIES:
                    self._explain_fatal(exc)
                    raise
                delay = min(AZURE_BACKOFF_MAX, AZURE_BACKOFF_BASE**attempt) * (0.5 + random.random())
                time.sleep(max(delay, _retry_after_seconds(exc)))
                attempt += 1
                continue

            gen = self._to_generation(response, time.perf_counter() - started)
            self._check_empty(gen, response)
            self._cache_write(cache_path, gen)
            return gen

        raise RuntimeError(f"chamada ao Azure falhou após {AZURE_MAX_RETRIES} tentativas") from last_exc

    def _explain_fatal(self, exc: Exception) -> None:
        """
        Loga a tradução de um erro que não vai se resolver sozinho.

        O 404 do Azure é o caso que mais custa tempo: a mensagem do servidor é
        "Resource Not Found" para três causas bem diferentes, e sem contexto
        não dá para saber qual é.
        """
        status = _status_of(exc)
        if status == 404:
            log.error(
                "404 do Azure para deployment=%r em %s.\n"
                "  'Resource Not Found' aqui tem três causas possíveis:\n"
                "   1. o nome do deployment não existe nesse recurso (mais comum);\n"
                "   2. AZURE_OPENAI_ENDPOINT aponta para outro recurso, ou veio com\n"
                "      caminho sobrando (deve ser só https://<recurso>.openai.azure.com/);\n"
                "   3. AZURE_OPENAI_API_VERSION é antiga demais para este modelo —\n"
                "      api-version=%s; a família gpt-5 precisa de uma de 2025.\n"
                "  Rode `python diag_azure.py` para descobrir qual das três é.",
                self.deployment, os.environ.get(AZURE_ENDPOINT_VAR, "?"), AZURE_API_VERSION,
            )
        elif status == 401:
            log.error(
                "401 do Azure: AZURE_OPENAI_API_KEY inválida, ou é a chave de um "
                "recurso diferente do endpoint configurado."
            )
        elif status == 403:
            log.error(
                "403 do Azure: a chave é válida mas não tem permissão neste "
                "deployment, ou há restrição de rede/IP no recurso."
            )

    def _to_generation(self, response: Any, latency_s: float) -> Generation:
        choice = response.choices[0] if response.choices else None
        text = (getattr(getattr(choice, "message", None), "content", None) or "").strip()
        usage = getattr(response, "usage", None)
        return Generation(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency_s,
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )

    def _check_empty(self, gen: Generation, response: Any) -> None:
        """
        Vazio é falha, e falha alto.

        Deixar passar significaria terminar a grade com milhares de reflexões
        em branco e uma análise que não acusa nada de errado.
        """
        with self._lock:
            self._calls += 1
            if gen.text:
                return
            self._empties += 1
            calls, empties = self._calls, self._empties

        diagnosis = self._diagnose(gen, response)
        if AZURE_FAIL_ON_EMPTY:
            raise AzureEmptyResponse(
                f"{self.key} ({self.deployment}) devolveu resposta vazia. {diagnosis}"
            )

        rate = empties / calls
        log.warning("resposta vazia (%d/%d). %s", empties, calls, diagnosis)
        if calls >= AZURE_HEALTH_CHECK_CALLS and rate > AZURE_MAX_EMPTY_RATE:
            raise AzureEmptyResponse(
                f"{self.key} ({self.deployment}): {empties} de {calls} respostas vazias "
                f"({rate:.0%} > {AZURE_MAX_EMPTY_RATE:.0%}). {diagnosis}"
            )

    def _diagnose(self, gen: Generation, response: Any) -> str:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        if gen.finish_reason == "length" and self.reasoning:
            return (
                f"O modelo gastou o orçamento inteiro raciocinando "
                f"(finish_reason='length', reasoning_tokens={reasoning_tokens}) e não sobrou "
                f"nada para a resposta. Suba RMCQ_AZURE_REASONING_MIN_TOKENS (tente 6000-8000) "
                f"ou RMCQ_AZURE_MAX_TOKENS. Orçamento atual: "
                f"{budget(self.max_tokens, True, AZURE_REASONING_MIN_TOKENS)}."
            )
        if gen.finish_reason == "length":
            return f"finish_reason='length': suba RMCQ_AZURE_MAX_TOKENS (atual: {self.max_tokens})."
        if gen.finish_reason == "content_filter":
            return "O filtro de conteúdo do Azure bloqueou a resposta."
        return (
            f"finish_reason={gen.finish_reason!r}. Confira se RMCQ_AZURE_DEPLOYMENT_* aponta "
            f"para um deployment que existe neste recurso."
        )

    # -- interface Backend --------------------------------------------------

    def generate(
        self,
        prompts: Sequence[str],
        params: GenParams,
        system: str | None = None,
        desc: str = "",
    ) -> list[Generation]:
        if not prompts:
            return []

        results: list[Generation | None] = [None] * len(prompts)
        # Falhar rápido tem que ser rápido de verdade: sem esta bandeira, um
        # deployment mal configurado detectado no primeiro prompt ainda pagaria
        # os outros 380 do lote, porque o executor espera todo mundo terminar
        # antes de propagar a exceção.
        aborted = threading.Event()

        def work(i: int) -> int:
            if aborted.is_set():
                raise RuntimeError("lote abortado por falha anterior")
            try:
                results[i] = self._complete_one(prompts[i], params, system, json_mode=False)
            except BaseException:
                aborted.set()
                raise
            return i

        # Uma thread só não vale o overhead do executor, e facilita depurar.
        if self.concurrency == 1:
            for i in progress(range(len(prompts)), desc=desc or f"{self.key} (azure)"):
                work(i)
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [pool.submit(work, i) for i in range(len(prompts))]
                try:
                    for future in progress(futures, desc=desc or f"{self.key} (azure)"):
                        future.result()  # propaga a primeira exceção
                except BaseException:
                    aborted.set()
                    for f in futures:
                        f.cancel()  # os que ainda não começaram nem chegam a sair
                    raise

        # A ordem é garantia do contrato do Backend: o índice de escrita é o
        # índice do prompt, então concorrência não embaralha nada.
        missing = [i for i, g in enumerate(results) if g is None]
        if missing:
            raise RuntimeError(f"{len(missing)} gerações não voltaram (índices {missing[:5]}...)")
        return [g for g in results if g is not None]

    def count_tokens(self, text: str) -> int:
        encoder = self._encoder()
        if encoder is not None:
            return len(encoder.encode(text))
        return max(1, len(text) // 4)

    _ENCODER: Any = None
    _ENCODER_TRIED = False

    def _encoder(self) -> Any:
        cls = type(self)
        if cls._ENCODER_TRIED:
            return cls._ENCODER
        cls._ENCODER_TRIED = True
        try:
            import tiktoken

            cls._ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 - tiktoken é opcional
            log.debug("tiktoken indisponível; contagem de tokens fica aproximada")
            cls._ENCODER = None
        return cls._ENCODER

    def unload(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def render(self, tokenizer: Any, prompt: str, system: str | None = None) -> str:
        # O chat template é do servidor. Nunca chamado por este backend, mas
        # sobrescrito para não explodir se alguém chamar sem tokenizer.
        return f"{system}\n\n{prompt}" if system else prompt
