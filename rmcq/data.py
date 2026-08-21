"""
Leitura dos splits e resolução de seletores da linha de comando.

Um só lugar que sabe onde os dados moram e o que "all" significa em cada
dimensão da grade. Isso evita que cada etapa reimplemente a expansão de
seletores de forma ligeiramente diferente.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from rmcq.common import read_jsonl
from rmcq.config import (
    ALL_DATASETS,
    BASELINE_DIR,
    DATASETS,
    DEPTHS,
    EVAL_DIR,
    K_VALUES,
    REFLECTIONS_DIR,
    RETRY_DIR,
    SELFCONS_DIR,
    SPLITS_DIR,
    STUDENTS,
    TEACHERS,
    config_tag,
)

_ALL = ("all", "todos", "*")


def _resolve(values: Sequence[str] | None, universe: Sequence[str], name: str) -> tuple[str, ...]:
    if not values or any(v.lower() in _ALL for v in values):
        return tuple(universe)
    unknown = [v for v in values if v not in universe]
    if unknown:
        raise ValueError(f"{name} desconhecido(s): {unknown}. Válidos: {list(universe)}")
    # Preserva a ordem do universo, para que os caminhos de saída sejam estáveis
    # independentemente da ordem em que o usuário digitou.
    return tuple(u for u in universe if u in set(values))


def resolve_datasets(values: Sequence[str] | None = None) -> tuple[str, ...]:
    return _resolve(values, ALL_DATASETS, "dataset")


def resolve_students(values: Sequence[str] | None = None) -> tuple[str, ...]:
    return _resolve(values, STUDENTS, "aluno")


def resolve_teachers(values: Sequence[str] | None = None) -> tuple[str, ...]:
    return _resolve(values, TEACHERS, "professor")


def resolve_depths(values: Sequence[str] | None = None) -> tuple[str, ...]:
    return _resolve(values, DEPTHS, "profundidade")


def resolve_ks(values: Sequence[int] | None = None) -> tuple[int, ...]:
    if not values:
        return tuple(K_VALUES)
    return tuple(sorted(set(int(v) for v in values)))

# Splits que o experimento realmente consome. O laço é treino -> reflexão ->
# teste; validação não entra em nenhuma etapa. Ver DEFAULT_LOOP_SPLITS abaixo.
DEFAULT_LOOP_SPLITS = ("train", "test")

# Direções do pacote de troca com o ambiente dos professores de API.
# "to-azure" leva as perguntas e as respostas dos alunos; "from-azure" traz as
# reflexões de volta. Ver rmcq/stages/exchange.py.
EXCHANGE_DIRECTIONS = ("to-azure", "from-azure")


def resolve_splits(
    values: Sequence[str] | None,
    dataset: str,
    default: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """
    Splits a processar.

    Sem argumento, usa `default` (que as etapas passam como treino + teste), não
    todos os splits: gerar baseline em validação custaria 28% da etapa sem que
    nada a jusante consumisse o resultado. `--splits all` força os três.
    """
    available = tuple(DATASETS[dataset].splits)
    if not values:
        wanted = set(default) if default else set(available)
        return tuple(s for s in available if s in wanted)
    if any(v.lower() in _ALL for v in values):
        return available
    # Silenciosamente ignora splits que o dataset não tem: o GSM8K não tem
    # validation, e pedir `--splits train validation test` não deve dar erro.
    return tuple(s for s in available if s in set(values))


# ---------------------------------------------------------------------------
# Leitura dos itens
# ---------------------------------------------------------------------------


def split_path(dataset: str, split: str) -> Path:
    return SPLITS_DIR / dataset / f"{split}.jsonl"


@lru_cache(maxsize=64)
def _load_cached(dataset: str, split: str) -> tuple[dict[str, Any], ...]:
    path = split_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe.\n"
            f"Rode: python -m rmcq setup-data --datasets {dataset}\n"
            f"e execute notebooks/01_formatacao_e_selecao.ipynb"
        )
    return tuple(read_jsonl(path))


def load_split(dataset: str, split: str) -> list[dict[str, Any]]:
    """Itens de um split, no schema MCQ unificado. Cacheado em memória."""
    return list(_load_cached(dataset, split))


def load_index(dataset: str, split: str) -> dict[str, dict[str, Any]]:
    """Mesmo conteúdo de load_split, indexado por uid."""
    return {item["uid"]: item for item in load_split(dataset, split)}


def dataset_summary() -> list[dict[str, Any]]:
    rows = []
    for key, spec in DATASETS.items():
        row: dict[str, Any] = {"dataset": key, "tipo": spec.problem_type}
        for split in ("train", "validation", "test"):
            path = split_path(key, split)
            row[split] = len(read_jsonl(path)) if path.exists() else 0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Caminhos de saída, um lugar só
# ---------------------------------------------------------------------------


def baseline_path(model: str, dataset: str, split: str) -> Path:
    return BASELINE_DIR / model / f"{dataset}_{split}.jsonl"


def reflections_path(student: str, teacher: str, depth: str, dataset: str) -> Path:
    return REFLECTIONS_DIR / config_tag(student, teacher, depth) / f"{dataset}.jsonl"


def eval_path(student: str, teacher: str, depth: str, k: int, dataset: str) -> Path:
    return EVAL_DIR / config_tag(student, teacher, depth, k) / f"{dataset}.jsonl"


def retry_path(model: str, dataset: str) -> Path:
    return RETRY_DIR / model / f"{dataset}.jsonl"


def selfcons_path(model: str, dataset: str, n: int) -> Path:
    return SELFCONS_DIR / f"{model}__n{n}" / f"{dataset}.jsonl"


def exchange_dir(direction: str) -> Path:
    """
    Raiz de um dos dois lados do pacote de troca.

    Os nomes são por DIREÇÃO e não por caixa de entrada/saída de propósito:
    "to-azure" e "from-azure" querem dizer a mesma coisa nas duas máquinas, e
    "outbox" não.
    """
    from rmcq.config import EXCHANGE_DIR

    if direction not in EXCHANGE_DIRECTIONS:
        raise ValueError(f"direção {direction!r} desconhecida. Válidas: {EXCHANGE_DIRECTIONS}")
    return EXCHANGE_DIR / direction


def exchange_manifest_path(direction: str) -> Path:
    return exchange_dir(direction) / "manifest.json"


def index_paths(dataset: str, embedder: str) -> dict[str, Path]:
    """Arquivos do índice de similaridade. O nome carrega o embedder usado."""
    from rmcq.config import INDEX_DIR

    slug = embedder.replace("/", "_")
    base = INDEX_DIR / slug / dataset
    return {
        "dir": base,
        "train_emb": base / "train.npy",
        "test_emb": base / "test.npy",
        "neighbors": base / "neighbors.npz",
        "meta": base / "meta.json",
    }
