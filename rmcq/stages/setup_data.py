"""
Etapa 0a: download dos cinco benchmarks para data/raw/.

Não transforma nada. Só materializa os dados puros em parquet, um arquivo por
split, e escreve um manifesto com contagens e hashes. A formatação para o schema
unificado acontece em notebooks/01.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from rmcq.config import DATASET_MANIFEST, DATASETS, RAW_DIR, ensure_dirs, hf_token
from rmcq.data import resolve_datasets
from rmcq.store import get_logger

log = get_logger(__name__)


def raw_path(dataset: str, split: str) -> Path:
    return RAW_DIR / dataset / f"{split}.parquet"


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def inspect_parquet(path: Path) -> tuple[int, list[str]]:
    """Contagem de linhas e colunas sem carregar os dados."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, list(pf.schema_arrow.names)


def download_one(key: str, force: bool = False) -> dict[str, Any]:
    from datasets import load_dataset

    spec = DATASETS[key]
    (RAW_DIR / key).mkdir(parents=True, exist_ok=True)

    log.info("%s <- %s%s", key, spec.repo_id, f" [{spec.config}]" if spec.config else "")
    if spec.notes:
        log.debug("  nota: %s", spec.notes)

    entry: dict[str, Any] = {
        "repo_id": spec.repo_id, "config": spec.config, "revision": spec.revision,
        "problem_type": spec.problem_type, "native_mcq": spec.native_mcq, "splits": {},
    }

    pending = {
        canonical: hub
        for canonical, hub in spec.splits.items()
        if force or not raw_path(key, canonical).exists()
    }

    if not pending:
        log.info("  já presente (use --force para rebaixar)")
    else:
        load_kwargs: dict[str, Any] = {}
        if hf_token():
            load_kwargs["token"] = hf_token()
        if spec.revision:
            load_kwargs["revision"] = spec.revision

        dsd = load_dataset(spec.repo_id, spec.config, **load_kwargs)
        for canonical, hub in pending.items():
            if hub not in dsd:
                raise KeyError(
                    f"split {hub!r} não existe em {spec.repo_id}. Disponíveis: {list(dsd)}"
                )
            dest = raw_path(key, canonical)
            dsd[hub].to_parquet(dest)
            log.info("  %-11s %7d linhas -> %s", canonical, len(dsd[hub]), dest.name)

    # Manifesto lido do disco: reflete o que existe, não o que passou na memória.
    for canonical in spec.splits:
        path = raw_path(key, canonical)
        if not path.exists():
            entry["splits"][canonical] = {"status": "missing"}
            continue
        n_rows, columns = inspect_parquet(path)
        expected = spec.expected_rows.get(canonical)
        entry["splits"][canonical] = {
            "status": "ok" if (expected is None or n_rows == expected) else "row_mismatch",
            "rows": n_rows, "expected_rows": expected, "columns": columns,
            "bytes": path.stat().st_size, "sha256_16": file_digest(path),
        }

    return entry


def verify(keys: Sequence[str]) -> bool:
    print(f"\n{'dataset':<12} {'split':<11} {'linhas':>8} {'esperado':>9}  status")
    print("-" * 60)

    all_ok = True
    for key in keys:
        spec = DATASETS[key]
        for canonical in spec.splits:
            path = raw_path(key, canonical)
            expected = spec.expected_rows.get(canonical)
            if not path.exists():
                print(f"{key:<12} {canonical:<11} {'-':>8} {expected or '?':>9}  AUSENTE")
                all_ok = False
                continue
            n_rows, _ = inspect_parquet(path)
            if expected is None:
                status = "ok (sem referência)"
            elif n_rows == expected:
                status = "ok"
            else:
                status = f"DIVERGE ({n_rows - expected:+d})"
                all_ok = False
            print(f"{key:<12} {canonical:<11} {n_rows:>8,} {expected or '?':>9}  {status}")

    print("-" * 60)
    print("Tudo conforme o esperado." if all_ok else "Há divergências acima.")
    return all_ok


def run(
    datasets: Sequence[str] | None = None,
    force: bool = False,
    verify_only: bool = False,
) -> int:
    ensure_dirs()
    keys = resolve_datasets(datasets)

    if verify_only:
        return 0 if verify(keys) else 1

    try:
        import datasets as _  # noqa: F401
    except ImportError:
        log.error("pacote 'datasets' não encontrado. Rode: pip install -r requirements.txt")
        return 1

    started = time.time()
    manifest: dict[str, Any] = {}
    if DATASET_MANIFEST.exists():
        manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8")).get("datasets", {})

    failures = []
    for key in keys:
        try:
            manifest[key] = download_one(key, force=force)
        except Exception as exc:  # noqa: BLE001 - um dataset ruim não para o resto
            log.error("  FALHOU %s: %s: %s", key, type(exc).__name__, exc)
            failures.append(key)

    DATASET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DATASET_MANIFEST.write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.time() - started, 1),
            "datasets": manifest,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("manifesto: %s", DATASET_MANIFEST)
    verify([k for k in keys if k not in failures])

    if failures:
        log.error("datasets com falha: %s", ", ".join(failures))
        return 1

    print("\nPróximo passo: notebooks/01_formatacao_e_selecao.ipynb")
    return 0
