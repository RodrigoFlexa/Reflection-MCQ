"""Seleção de backend. O tipo vem do .env (RMCQ_BACKEND) e pode ser sobreposto."""

from __future__ import annotations

from rmcq.backends.base import Backend, Generation, GenParams
from rmcq.config import BACKEND
from rmcq.store import get_logger

log = get_logger(__name__)

__all__ = ["Backend", "Generation", "GenParams", "get_backend", "available_backends"]

_BACKENDS = ("vllm", "hf", "stub")


def available_backends() -> dict[str, str]:
    """Quais backends dão para usar nesta máquina, e por que os outros não."""
    status: dict[str, str] = {"stub": "ok (sem GPU)"}

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
    """
    kind = (kind or BACKEND).lower()
    if kind not in _BACKENDS:
        raise ValueError(f"backend {kind!r} desconhecido. Válidos: {_BACKENDS}")

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
