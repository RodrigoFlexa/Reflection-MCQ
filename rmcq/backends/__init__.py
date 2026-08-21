"""Seleção de backend. O tipo vem do .env (RMCQ_BACKEND) e pode ser sobreposto."""

from __future__ import annotations

import os

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import (
    API_ONLY,
    AZURE_API_KEY_VAR,
    AZURE_ENDPOINT_VAR,
    BACKEND,
    MODELS,
)
from rmcq.store import get_logger

log = get_logger(__name__)

__all__ = ["Backend", "Generation", "GenParams", "get_backend", "available_backends"]

_BACKENDS = ("vllm", "hf", "stub", "azure")


def available_backends() -> dict[str, str]:
    """Quais backends dão para usar nesta máquina, e por que os outros não."""
    status: dict[str, str] = {"stub": "ok (sem GPU)"}

    try:
        import openai  # noqa: F401

        missing = [v for v in (AZURE_ENDPOINT_VAR, AZURE_API_KEY_VAR) if not os.environ.get(v)]
        status["azure"] = (
            f"indisponível: falta {', '.join(missing)} no .env" if missing
            else f"ok (openai {openai.__version__})"
        )
    except ImportError:
        status["azure"] = "indisponível: pip install -r requirements-azure.txt"

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

    Se o vLLM foi pedido mas não importa, caímos para transformers com um aviso
    em vez de abortar: o pipeline continua correto, só mais lento. Um run de
    dezenas de horas não deve morrer por uma dependência opcional.

    Modelos de API (provider="azure") ignoram `kind`: numa mesma grade o aluno
    roda em vLLM local e o professor roda no Azure, e `--backend vllm` não pode
    desviar o professor para uma GPU onde não existem pesos dele. O `--backend
    stub` é a exceção deliberada — é assim que se testa a plumbaria sem gastar
    API.
    """
    kind = (kind or BACKEND).lower()
    if kind not in _BACKENDS:
        raise ValueError(f"backend {kind!r} desconhecido. Válidos: {_BACKENDS}")

    spec = MODELS.get(model_key)
    if spec is not None and spec.is_api and kind != "stub":
        from rmcq.backends.azure import AzureBackend

        return AzureBackend(model_key, **kwargs)

    if API_ONLY and spec is not None and not spec.is_api and kind != "stub":
        raise RuntimeError(
            f"RMCQ_API_ONLY=1: esta máquina só roda modelos de API, e {model_key!r} "
            f"exige pesos locais e GPU.\n"
            f"Se a intenção era usar um professor de API, passe-o explicitamente "
            f"(ex.: --teachers gpt5) ou fixe RMCQ_TEACHERS no .env.\n"
            f"Modelos de API registrados: {[k for k, s in MODELS.items() if s.is_api]}"
        )

    if kind == "azure":
        raise ValueError(
            f"--backend azure vale só para modelos de API; {model_key!r} tem "
            f"provider={getattr(spec, 'provider', '?')!r}. "
            f"Modelos de API disponíveis: {[k for k, s in MODELS.items() if s.is_api]}"
        )

    if kind == "stub":
        from rmcq.backends.stub import StubBackend

        return StubBackend(model_key, **kwargs)

    if kind == "vllm":
        try:
            from rmcq.backends.vllm_backend import VLLMBackend

            return VLLMBackend(model_key, **kwargs)
        except ImportError:
            log.warning("vLLM não importou; caindo para o backend transformers")
            kind = "hf"

    from rmcq.backends.hf import HFBackend

    return HFBackend(model_key, **kwargs)
