"""
Etapa 0b: download dos quatro modelos para models/hf_cache/.

snapshot_download é retomável e deduplicado: rodar de novo não rebaixa o que já
está completo. Checkpoints em formatos que nem transformers nem vLLM usam
(original/*.pth, consolidated.safetensors, .gguf) são ignorados — economiza
cerca de 30 GB.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Sequence

from rmcq.config import (
    ACTIVE_MODELS,
    ALL_MODELS,
    HF_HOME,
    INACTIVE_MODELS,
    MODEL_MANIFEST,
    MODELS,
    MODELS_DIR,
    ensure_dirs,
    hf_token,
)
from rmcq.data import _resolve
from rmcq.store import get_logger

log = get_logger(__name__)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

TOKENIZER_PATTERNS = [
    "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json",
    "*.py",  # necessário para trust_remote_code (Phi-4 mini)
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_access(key: str) -> dict[str, Any]:
    """Confere existência e permissão sem baixar pesos."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    spec = MODELS[key]
    api = HfApi(token=hf_token())
    result: dict[str, Any] = {
        "repo_id": spec.repo_id, "params": spec.params, "roles": list(spec.roles),
    }

    try:
        info = api.model_info(spec.repo_id, files_metadata=True, revision=spec.revision)
    except GatedRepoError:
        result.update(status="gated_denied", hint=(
            f"aceite a licença em https://huggingface.co/{spec.repo_id} "
            "e defina HF_TOKEN no .env"
        ))
        return result
    except RepositoryNotFoundError:
        result.update(status="not_found", hint="repo_id inexistente ou privado; confira config.py")
        return result
    except Exception as exc:  # noqa: BLE001
        result.update(status="error", hint=f"{type(exc).__name__}: {exc}")
        return result

    def ignored(name: str) -> bool:
        return any(fnmatch(name, pat) for pat in spec.ignore_patterns)

    keep = [f for f in info.siblings if f.size and not ignored(f.rfilename)]
    skip = [f for f in info.siblings if f.size and ignored(f.rfilename)]

    result.update(
        status="ok", gated=bool(info.gated), sha=info.sha,
        architectures=(info.config or {}).get("architectures"),
        download_bytes=sum(f.size for f in keep),
        skipped_bytes=sum(f.size for f in skip),
        n_files=len(keep),
    )
    return result


def download_one(key: str, tokenizers_only: bool = False) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    spec = MODELS[key]
    log.info("%s (%s) <- %s", key, spec.params, spec.repo_id)

    info = check_access(key)
    if info["status"] != "ok":
        log.error("  INDISPONÍVEL [%s]: %s", info["status"], info.get("hint", ""))
        return info

    if tokenizers_only:
        log.info("  modo tokenizers-only: só configs e tokenizer")
    else:
        log.info(
            "  %d arquivos, %s (ignorando %s de checkpoints duplicados)",
            info["n_files"], human(info["download_bytes"]), human(info["skipped_bytes"]),
        )

    started = time.time()
    local = snapshot_download(
        repo_id=spec.repo_id, revision=spec.revision, cache_dir=str(HF_HOME),
        token=hf_token(),
        allow_patterns=TOKENIZER_PATTERNS if tokenizers_only else None,
        ignore_patterns=None if tokenizers_only else list(spec.ignore_patterns),
        max_workers=4,
    )
    elapsed = time.time() - started
    on_disk = sum(f.stat().st_size for f in Path(local).rglob("*") if f.is_file())
    log.info("  pronto em %.0fs, %s em disco", elapsed, human(on_disk))

    info.update(
        local_path=local, tokenizers_only=tokenizers_only,
        bytes_on_disk=on_disk, elapsed_s=round(elapsed, 1),
    )
    return info


def report_environment() -> None:
    from rmcq import env_summary
    from rmcq.backends import available_backends
    from rmcq.config import runtime_summary

    print("Ambiente")
    print("-" * 72)
    print(f"  {env_summary()}")
    for name, status in available_backends().items():
        print(f"  backend {name:<6} {status}")

    try:
        import torch

        print(f"  torch        {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"    cuda:{i}      {p.name}, {human(p.total_memory)}")
            print(f"  bf16         {torch.cuda.is_bf16_supported()}")
        else:
            print("  cuda         INDISPONÍVEL — inferência 8B na CPU é impraticável")
    except ImportError:
        print("  torch        não instalado")

    try:
        import transformers

        print(f"  transformers {transformers.__version__}")
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        if (major, minor) < (4, 51):
            print("  AVISO: Qwen3 exige transformers>=4.51")
    except ImportError:
        print("  transformers não instalado")

    print(f"  HF_TOKEN     {'presente' if hf_token() else 'AUSENTE (Llama 3 vai falhar)'}")
    print(f"  cache        {HF_HOME}")
    total, _, free = shutil.disk_usage(HF_HOME if HF_HOME.exists() else Path.cwd())
    print(f"  disco livre  {human(free)} de {human(total)}")
    print(f"  runtime      {json.dumps(runtime_summary(), ensure_ascii=False)}")
    print("-" * 72)


def _keys(models: Sequence[str] | None) -> tuple[str, ...]:
    """
    Sem argumento, baixa só os modelos ativos.

    O registro tem quatro modelos, mas a rodada atual usa dois. Baixar os quatro
    por padrão custaria ~28 GB de download que ninguém pediu. `--models all`
    baixa todos, e nomear um inativo explicitamente também funciona.
    """
    if not models:
        if INACTIVE_MODELS:
            log.info(
                "baixando só os modelos ativos: %s (fora da rodada: %s; "
                "use --models all para baixar todos)",
                ", ".join(ACTIVE_MODELS), ", ".join(INACTIVE_MODELS),
            )
        return ACTIVE_MODELS
    return _resolve(models, ALL_MODELS, "modelo")


def run(
    models: Sequence[str] | None = None,
    tokenizers_only: bool = False,
    check_only: bool = False,
) -> int:
    ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    keys = _keys(models)
    report_environment()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log.error("huggingface_hub não encontrado. Rode: pip install -r requirements.txt")
        return 1

    if check_only:
        print(f"\n{'modelo':<13} {'params':>7} {'gated':>6} {'a baixar':>11} {'ignorado':>11}  status")
        print("-" * 72)
        total = problems = 0
        for key in keys:
            info = check_access(key)
            if info["status"] == "ok":
                total += info["download_bytes"]
                print(
                    f"{key:<13} {info['params']:>7} {str(info['gated']):>6} "
                    f"{human(info['download_bytes']):>11} {human(info['skipped_bytes']):>11}  ok"
                )
            else:
                problems += 1
                print(f"{key:<13} {info['params']:>7} {'?':>6} {'-':>11} {'-':>11}  {info['status']}")
                print(f"  -> {info.get('hint', '')}")
        print("-" * 72)
        print(f"Total a baixar: {human(total)}")
        return 1 if problems else 0

    manifest: dict[str, Any] = {}
    if MODEL_MANIFEST.exists():
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8")).get("models", {})

    failures = []
    started = time.time()
    for key in keys:
        try:
            info = download_one(key, tokenizers_only=tokenizers_only)
            manifest[key] = info
            if info["status"] != "ok":
                failures.append(key)
        except Exception as exc:  # noqa: BLE001
            log.error("  FALHOU %s: %s: %s", key, type(exc).__name__, exc)
            failures.append(key)
            manifest[key] = {"status": "error", "hint": f"{type(exc).__name__}: {exc}"}

    MODEL_MANIFEST.write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.time() - started, 1),
            "cache_dir": str(HF_HOME), "models": manifest,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("manifesto: %s", MODEL_MANIFEST)

    if failures:
        log.error("modelos com falha: %s", ", ".join(failures))
        return 1

    print("\nTudo baixado. Valide com: python -m rmcq smoke")
    return 0


def smoke(models: Sequence[str] | None = None, backend_kind: str | None = None) -> int:
    """Carrega cada modelo e responde uma questão conhecida com o prompt congelado."""
    from rmcq.backends import GenParams, get_backend
    from rmcq.common import build_answer_prompt, extract_final_answer

    item = {
        "uid": "smoke-0", "dataset": "smoke", "split": "test", "problem_type": "knowledge",
        "context": None,
        "question": "Which of the following is the largest planet in the Solar System?",
        "choices": [
            {"label": "A", "text": "Mars"}, {"label": "B", "text": "Jupiter"},
            {"label": "C", "text": "Earth"}, {"label": "D", "text": "Venus"},
        ],
        "answerKey": "B", "num_choices": 4,
    }
    prompt = build_answer_prompt(item)
    params = GenParams(max_new_tokens=512, temperature=0.0)

    results = {}
    for key in _keys(models):
        try:
            with get_backend(key, backend_kind) as backend:
                gen = backend.generate([prompt], params, desc=f"smoke {key}")[0]
            ext = extract_final_answer(gen.text, list("ABCD"))
            ok = ext.letter == "B"
            results[key] = ok
            print(f"\n--- {key} ---")
            print(f"  extraído {ext.letter} (esperado B) via '{ext.method}' -> {'OK' if ok else 'FALHOU'}")
            print(f"  {gen.completion_tokens} tokens em {gen.latency_s}s")
            print(f"  final: ...{gen.text.strip()[-160:]!r}")
        except Exception as exc:  # noqa: BLE001
            results[key] = False
            print(f"\n--- {key} --- FALHOU: {type(exc).__name__}: {exc}")

    print("\nResumo:")
    for key, ok in results.items():
        print(f"  {key:<13} {'OK' if ok else 'FALHOU'}")
    return 0 if all(results.values()) else 1
