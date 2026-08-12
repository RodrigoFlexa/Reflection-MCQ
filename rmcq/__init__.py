"""
rmcq — framework experimental da extensão do paper de Reflection.

IMPORTANTE: este módulo carrega o .env no momento do import, antes que qualquer
outra coisa importe torch. É a única janela em que CUDA_VISIBLE_DEVICES ainda
surte efeito: depois que o runtime CUDA inicializa, mudar a variável não faz
nada. Por isso todo entrypoint deve importar `rmcq` (ou algo dentro de `rmcq`)
antes de importar torch, transformers ou vllm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parser mínimo de .env. Sem dependência externa, sem interpolação."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Remove aspas envolventes, preservando string vazia entre aspas
        # (CUDA_VISIBLE_DEVICES="" é a forma de forçar CPU).
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:
            val = val.split(" #")[0].rstrip()
        if key:
            values[key] = val
    return values


def load_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """
    Carrega o .env para os.environ.

    Por padrão o ambiente já existente ganha: `CUDA_VISIBLE_DEVICES=3 python -m
    rmcq ...` na linha de comando sobrepõe o .env, que é o comportamento
    esperado de quem passa a variável explicitamente.
    """
    env_path = path or (ROOT / ".env")
    values = _parse_env_file(env_path)

    applied = {}
    for key, val in values.items():
        if override or key not in os.environ:
            os.environ[key] = val
            applied[key] = val

    return applied


_APPLIED_ENV = load_env()
_ENV_FILE_FOUND = (ROOT / ".env").exists()


def env_summary() -> str:
    """Uma linha descrevendo de onde vieram as variáveis relevantes."""
    src = ".env" if _ENV_FILE_FOUND else "sem .env (usando defaults e ambiente)"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_desc = "todas as GPUs visíveis" if gpu is None else (f"CUDA_VISIBLE_DEVICES={gpu!r}" if gpu != "" else "CPU (CUDA_VISIBLE_DEVICES vazio)")
    return f"{src} | {gpu_desc} | backend={os.environ.get('RMCQ_BACKEND', 'vllm')}"


def warn_if_torch_already_imported() -> None:
    """Avisa se torch entrou antes de nós; nesse caso a escolha de GPU não pegou."""
    if "torch" in sys.modules and _APPLIED_ENV.get("CUDA_VISIBLE_DEVICES") is not None:
        print(
            "AVISO: torch foi importado antes de rmcq, então CUDA_VISIBLE_DEVICES "
            "do .env pode não ter surtido efeito. Importe rmcq primeiro.",
            file=sys.stderr,
        )


warn_if_torch_already_imported()
