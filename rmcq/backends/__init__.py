"""Seleção de backend. O tipo vem do .env (RMCQ_BACKEND) e pode ser sobreposto."""

from __future__ import annotations

import os

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import (
    AZURE_API_KEY_VAR,
    AZURE_BASE_URL_VAR,
    AZURE_ENDPOINT_VAR,
    BACKEND,
    MODELS,
    OLLAMA_BASE_URL,
)
from rmcq.store import get_logger

log = get_logger(__name__)

__all__ = ["Backend", "Generation", "GenParams", "get_backend", "available_backends"]

_BACKENDS = ("vllm", "hf", "stub", "azure", "ollama")


def available_backends() -> dict[str, str]:
    """Quais backends dão para usar nesta máquina, e por que os outros não."""
    status: dict[str, str] = {"stub": "ok (sem GPU)"}

    try:
        import openai  # noqa: F401

        # A URL pode vir de qualquer uma das duas variáveis; a chave é obrigatória.
        tem_url = os.environ.get(AZURE_BASE_URL_VAR) or os.environ.get(AZURE_ENDPOINT_VAR)
        missing = []
        if not tem_url:
            missing.append(f"{AZURE_BASE_URL_VAR} (ou {AZURE_ENDPOINT_VAR})")
        if not os.environ.get(AZURE_API_KEY_VAR):
            missing.append(AZURE_API_KEY_VAR)
        status["azure"] = (
            f"indisponível: falta {', '.join(missing)} no .env" if missing
            else f"ok (openai {openai.__version__})"
        )
    except ImportError:
        status["azure"] = "indisponível: pip install -r requirements-azure.txt"

    try:
        import requests

        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5)
            n = len((r.json() or {}).get("models", [])) if r.ok else None
            status["ollama"] = (
                f"ok ({n} modelo(s) no servidor {OLLAMA_BASE_URL})" if r.ok
                else f"servidor respondeu {r.status_code} em {OLLAMA_BASE_URL}"
            )
        except requests.RequestException as exc:
            status["ollama"] = f"servidor inacessível em {OLLAMA_BASE_URL}: {type(exc).__name__}"
    except ImportError:
        status["ollama"] = "indisponível: pip install requests"

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        status["hf"] = "ok" if torch.cuda.is_available() else "ok, mas sem CUDA (muito lento)"
    except ImportError as exc:
        status["hf"] = f"indisponível: {exc.name} não instalado"

    try:
        import vllm  # noqa: F401

        status["vllm"] = f"ok (vllm {vllm.__version__})"
    except ImportError:
        status["vllm"] = "indisponível: pip install vllm"
    except Exception as exc:  # noqa: BLE001 - vLLM às vezes falha no import por CUDA
        status["vllm"] = f"indisponível: {type(exc).__name__}: {exc}"

    return status


def get_backend(model_key: str, kind: str | None = None, **kwargs) -> Backend:
    """
    Instancia o backend pedido.

    Modelos com provider fixo (azure, ollama) ignoram `kind` e vão sempre para
    o backend do seu provider — não faria sentido pedir um deployment Azure
    via --backend vllm. A exceção deliberada é --backend stub: troca QUALQUER
    modelo pelo stub, para testar a integração sem gastar GPU nem API.

    Se o vLLM foi pedido para um modelo local (provider="hf") mas não importa,
    caímos para transformers com um aviso em vez de abortar.
    """
    kind = (kind or BACKEND).lower()
    if kind not in _BACKENDS:
        raise ValueError(f"backend {kind!r} desconhecido. Válidos: {_BACKENDS}")

    spec = MODELS.get(model_key)
    if spec is None:
        raise KeyError(f"modelo {model_key!r} não está em config.MODELS: {sorted(MODELS)}")

    if kind == "stub":
        from rmcq.backends.stub import StubBackend

        return StubBackend(model_key, **kwargs)

    if spec.provider == "azure":
        from rmcq.backends.azure import AzureBackend

        return AzureBackend(model_key, **kwargs)

    if spec.provider == "ollama":
        from rmcq.backends.ollama import OllamaBackend

        return OllamaBackend(model_key, **kwargs)

    # A partir daqui, spec.provider == "hf": pesos locais, engine escolhido por `kind`.
    if kind in ("azure", "ollama"):
        raise ValueError(
            f"--backend {kind} vale só para modelos provider={kind!r}; "
            f"{model_key!r} é provider={spec.provider!r}."
        )

    if kind == "vllm":
        try:
            from rmcq.backends.vllm_backend import VLLMBackend

            return VLLMBackend(model_key, **kwargs)
        except ImportError:
            log.warning("vLLM não importou; caindo para o backend transformers")
            kind = "hf"

    from rmcq.backends.hf import HFBackend

    return HFBackend(model_key, **kwargs)
