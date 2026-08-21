"""
Pacote de troca entre esta máquina e o ambiente dos professores de API.

O problema: a etapa `reflect` precisa das perguntas (`data/splits/*/train.jsonl`)
e das respostas que o aluno já deu (`results/baseline/*/*_train.jsonl`), mas os
professores GPT-5/GPT-4 só existem atrás do Azure OpenAI, dentro de uma rede que
não tem nem GPU nem os pesos dos modelos pequenos. E `results/` e `data/*` são
gitignored — o git sozinho não leva nada disso.

A solução é um diretório versionado, `exchange/`, com duas direções:

    exchange/to-azure/     perguntas + respostas dos alunos   (daqui para lá)
    exchange/from-azure/   reflexões geradas pelos GPT        (de lá para cá)

**Por que há checksum em tudo.** Se os uids das duas máquinas divergirem — um
split regerado, um baseline pela metade, um arquivo truncado no meio do push —
as reflexões passam a casar com a pergunta errada. Nada no pipeline detectaria
isso depois: a análise rodaria até o fim e produziria números plausíveis sobre
pares errados. Por isso o import confere sha256 e contagem de linhas antes de
materializar qualquer coisa, e aborta em vez de avisar.

Importar é idempotente: o default junta por uid e só acrescenta o que falta,
então rodar duas vezes não duplica linha nem desfaz trabalho local.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from rmcq.common import read_jsonl
from rmcq.config import REFLECTIONS_DIR, ensure_dirs
from rmcq.data import (
    EXCHANGE_DIRECTIONS,
    baseline_path,
    exchange_dir,
    exchange_manifest_path,
    reflections_path,
    resolve_datasets,
    resolve_depths,
    resolve_students,
    resolve_teachers,
    split_path,
)
from rmcq.store import JsonlStore, get_logger

log = get_logger(__name__)

SPLIT = "train"  # a reflexão só acontece sobre o treino (ver stages/reflect.py)

TO_AZURE, FROM_AZURE = EXCHANGE_DIRECTIONS


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _n_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _git_sha() -> str:
    """Commit de onde o pacote saiu. Só informativo, mas resolve muita dúvida."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "desconhecido"
    except (OSError, subprocess.SubprocessError):
        return "desconhecido"


def _copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"rows": _n_lines(dst), "sha256": _sha256(dst), "bytes": dst.stat().st_size}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_to_azure(
    students: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Empacota o que o ambiente Azure precisa para gerar reflexões."""
    ensure_dirs()
    students = resolve_students(students)
    datasets = resolve_datasets(datasets)

    root = exchange_dir(TO_AZURE)
    files: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for dataset in datasets:
        src = split_path(dataset, SPLIT)
        if not src.exists():
            missing.append(str(src))
            continue
        rel = f"splits/{dataset}/{SPLIT}.jsonl"
        files[rel] = _copy(src, root / rel)

    for student in students:
        for dataset in datasets:
            src = baseline_path(student, dataset, SPLIT)
            if not src.exists():
                missing.append(str(src))
                continue
            rel = f"baseline/{student}/{dataset}_{SPLIT}.jsonl"
            files[rel] = _copy(src, root / rel)

    if missing:
        # Falta baseline de treino é erro de sequência, não de digitação: sem
        # ele o professor não tem o que refletir.
        raise FileNotFoundError(
            "faltam arquivos para o pacote:\n  " + "\n  ".join(missing)
            + "\n\nGere o que falta antes: python -m rmcq baseline --splits train"
        )

    manifest = {
        "direction": TO_AZURE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "split": SPLIT,
        "students": list(students),
        "datasets": list(datasets),
        "files": files,
    }
    _write_manifest(TO_AZURE, manifest)
    return _summarize(manifest, root)


def export_from_azure(
    students: Sequence[str] | None = None,
    teachers: Sequence[str] | None = None,
    depths: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Empacota as reflexões geradas, para voltarem por git."""
    ensure_dirs()
    students = resolve_students(students)
    teachers = resolve_teachers(teachers)
    depths = resolve_depths(depths)
    datasets = resolve_datasets(datasets)

    root = exchange_dir(FROM_AZURE)
    files: dict[str, dict[str, Any]] = {}
    empty: list[str] = []

    for student in students:
        for teacher in teachers:
            for depth in depths:
                for dataset in datasets:
                    src = reflections_path(student, teacher, depth, dataset)
                    if not src.exists() or _n_lines(src) == 0:
                        empty.append(f"{student}__{teacher}__{depth}/{dataset}")
                        continue
                    rel = f"reflections/{src.parent.name}/{src.name}"
                    files[rel] = _copy(src, root / rel)

    if not files:
        raise FileNotFoundError(
            "nenhuma reflexão encontrada para empacotar. Rode `python -m rmcq reflect` antes."
        )
    if empty:
        log.warning(
            "%d combinação(ões) sem reflexão, fora do pacote: %s%s",
            len(empty), ", ".join(empty[:5]), " ..." if len(empty) > 5 else "",
        )

    manifest = {
        "direction": FROM_AZURE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "split": SPLIT,
        "students": list(students),
        "teachers": list(teachers),
        "depths": list(depths),
        "datasets": list(datasets),
        "missing": empty,
        "files": files,
    }
    _write_manifest(FROM_AZURE, manifest)
    return _summarize(manifest, root)


def _write_manifest(direction: str, manifest: dict[str, Any]) -> None:
    path = exchange_manifest_path(direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    files = manifest["files"]
    total_bytes = sum(f["bytes"] for f in files.values())
    total_rows = sum(f["rows"] for f in files.values())
    log.info(
        "pacote %s: %d arquivos, %d linhas, %s em %s",
        manifest["direction"], len(files), total_rows, _human(total_bytes), root,
    )
    return {
        "direction": manifest["direction"],
        "root": root,
        "n_files": len(files),
        "rows": total_rows,
        "bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _verify(direction: str) -> dict[str, Any]:
    """Lê o manifesto e confere cada arquivo. Aborta na primeira divergência."""
    manifest_path = exchange_manifest_path(direction)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifesto ausente: {manifest_path}\n"
            f"Você fez `git pull` da branch de troca? O pacote vem versionado no git."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("direction") != direction:
        raise ValueError(
            f"manifesto é da direção {manifest.get('direction')!r}, "
            f"mas o import pediu {direction!r}"
        )

    root = exchange_dir(direction)
    problems: list[str] = []
    for rel, expected in manifest["files"].items():
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: ausente")
            continue
        rows = _n_lines(path)
        if rows != expected["rows"]:
            problems.append(f"{rel}: {rows} linhas, esperado {expected['rows']}")
            continue
        digest = _sha256(path)
        if digest != expected["sha256"]:
            problems.append(f"{rel}: sha256 {digest[:12]}…, esperado {expected['sha256'][:12]}…")

    if problems:
        # Não seguimos com "só os arquivos bons": um pacote parcialmente
        # corrompido é um pacote em que não dá para confiar em nada.
        raise ValueError(
            f"pacote {direction} não confere ({len(problems)} problema(s)):\n  "
            + "\n  ".join(problems)
            + "\n\nRefaça o export na máquina de origem e transfira de novo."
        )

    log.info("pacote %s conferido: %d arquivos íntegros", direction, len(manifest["files"]))
    return manifest


def _merge_jsonl(src: Path, dst: Path, overwrite: bool) -> dict[str, int]:
    """
    Junta por uid, sem duplicar.

    Idempotência importa aqui porque `git pull` + `import` é um par que a gente
    vai repetir várias vezes enquanto a grade roda em lotes.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not dst.exists():
        shutil.copy2(src, dst)
        n = _n_lines(dst)
        return {"added": n, "kept": 0}

    have = JsonlStore(dst).done_keys()
    incoming = [row for row in read_jsonl(src) if str(row.get("uid")) not in have]
    if incoming:
        JsonlStore(dst).append(incoming)
    return {"added": len(incoming), "kept": len(have)}


def import_to_azure(overwrite: bool = False) -> dict[str, Any]:
    """No ambiente Azure: materializa perguntas e baselines nos caminhos do pipeline."""
    ensure_dirs()
    manifest = _verify(TO_AZURE)
    root = exchange_dir(TO_AZURE)

    stats = {"files": 0, "added": 0, "kept": 0}
    for rel in manifest["files"]:
        src = root / rel
        parts = Path(rel).parts
        if parts[0] == "splits":
            dst = split_path(parts[1], SPLIT)
        elif parts[0] == "baseline":
            dataset = Path(parts[2]).stem.rsplit(f"_{SPLIT}", 1)[0]
            dst = baseline_path(parts[1], dataset, SPLIT)
        else:
            raise ValueError(f"entrada inesperada no pacote: {rel}")

        result = _merge_jsonl(src, dst, overwrite)
        stats["files"] += 1
        stats["added"] += result["added"]
        stats["kept"] += result["kept"]
        log.info("  %s -> %s (+%d)", rel, dst, result["added"])

    log.info(
        "importado de %s: %d arquivos, %d linhas novas, %d já existentes",
        TO_AZURE, stats["files"], stats["added"], stats["kept"],
    )
    return stats


def import_from_azure(overwrite: bool = False) -> dict[str, Any]:
    """De volta na máquina local: coloca as reflexões onde `eval` vai procurar."""
    ensure_dirs()
    manifest = _verify(FROM_AZURE)
    root = exchange_dir(FROM_AZURE)

    stats = {"files": 0, "added": 0, "kept": 0}
    for rel in manifest["files"]:
        parts = Path(rel).parts  # reflections/{tag}/{dataset}.jsonl
        if parts[0] != "reflections":
            raise ValueError(f"entrada inesperada no pacote: {rel}")

        dst = REFLECTIONS_DIR / parts[1] / parts[2]
        result = _merge_jsonl(root / rel, dst, overwrite)
        stats["files"] += 1
        stats["added"] += result["added"]
        stats["kept"] += result["kept"]
        log.info("  %s -> %s (+%d)", rel, dst, result["added"])

    log.info(
        "importado de %s: %d arquivos, %d reflexões novas, %d já existentes",
        FROM_AZURE, stats["files"], stats["added"], stats["kept"],
    )
    return stats


# ---------------------------------------------------------------------------
# Fachada para o CLI
# ---------------------------------------------------------------------------


def export(direction: str, **kwargs: Any) -> dict[str, Any]:
    if direction == TO_AZURE:
        return export_to_azure(
            students=kwargs.get("students"), datasets=kwargs.get("datasets"),
        )
    if direction == FROM_AZURE:
        return export_from_azure(
            students=kwargs.get("students"), teachers=kwargs.get("teachers"),
            depths=kwargs.get("depths"), datasets=kwargs.get("datasets"),
        )
    raise ValueError(f"direção {direction!r} desconhecida. Válidas: {EXCHANGE_DIRECTIONS}")


def load(direction: str, overwrite: bool = False) -> dict[str, Any]:
    if direction == TO_AZURE:
        return import_to_azure(overwrite=overwrite)
    if direction == FROM_AZURE:
        return import_from_azure(overwrite=overwrite)
    raise ValueError(f"direção {direction!r} desconhecida. Válidas: {EXCHANGE_DIRECTIONS}")


def plan(direction: str) -> list[dict[str, Any]]:
    """O que o pacote desta direção contém, sem materializar nada."""
    path = exchange_manifest_path(direction)
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "stage": f"exchange:{direction}",
            "file": rel,
            "rows": info["rows"],
            "bytes": info["bytes"],
        }
        for rel, info in manifest["files"].items()
    ]
