"""
Uso mínimo do pacote: pega um modelo por chave e gera texto.

    python example.py <chave-do-modelo> ["pergunta"]

Exemplos:
    python example.py phi2                       # backend do .env (RMCQ_BACKEND)
    RMCQ_BACKEND=stub python example.py phi2     # sem GPU, só para testar a integração
    python example.py llama3.1:8b                 # se registrado via RMCQ_OLLAMA_MODELS
    python example.py gpt-5-mini-petrobras         # se registrado via RMCQ_AZURE_DEPLOYMENTS
"""

from __future__ import annotations

import sys

import rmcq  # noqa: F401 — carrega o .env antes de qualquer import de torch
from rmcq.backends import get_backend
from rmcq.backends.base import GenParams


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    model_key = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of France?"

    with get_backend(model_key) as backend:
        [generation] = backend.generate([question], GenParams(max_new_tokens=200))

    print(generation.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
