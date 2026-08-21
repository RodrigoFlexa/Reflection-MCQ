"""
Etapa 1: baseline sem reflexão.

Cada modelo responde a todas as questões, de todos os splits, sem nenhuma
reflexão. É a base de comparação de tudo, e é reaproveitada duas vezes:

- o baseline de **treino** é a resposta sobre a qual o professor escreve a
  reflexão na etapa 2. Sem esse reaproveitamento, cada par aluno-professor
  regeraria as mesmas 1.763 respostas do aluno, 4 vezes por aluno.
- o baseline de **teste** é o ponto de comparação da etapa 4 e o ponto de
  partida das condições de retry e self-consistency.

A ordem dos laços não é arbitrária: modelo por fora, dataset por dentro. Carregar
um modelo de 8B custa de 30 s a alguns minutos; fazer isso por dataset
multiplicaria esse custo por cinco sem nenhum ganho.

**Splits: treino e teste, não validação.** O laço experimental é treino ->
reflexão -> teste, e nenhuma etapa a jusante lê validação. Gerar baseline nela
custaria 28% desta etapa para produzir arquivos que ninguém abre. A validação
fica reservada para escolher hiperparâmetros — k, embedder — sem tocar no teste;
quando isso for necessário, `--splits validation` gera sob demanda.
"""

from __future__ import annotations

from typing import Any, Sequence

from rmcq.backends import GenParams, get_backend
from rmcq.common import make_record
from rmcq.common import build_answer_prompt
from rmcq.config import COND_BASELINE, SEED, STUDENT_GEN, ensure_dirs
from rmcq.data import (
    DEFAULT_LOOP_SPLITS,
    baseline_path,
    load_split,
    resolve_datasets,
    resolve_splits,
    resolve_students,
)
from rmcq.store import JsonlStore, Timer, get_logger

log = get_logger(__name__)


def plan(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    O que falta fazer, sem carregar nenhum modelo.

    Alimenta o --dry-run e o comando status. Consulta o store para descontar o
    que já está pronto, então o custo estimado é o custo restante, não o total.
    """
    rows = []
    for model in resolve_students(models):
        for dataset in resolve_datasets(datasets):
            for split in resolve_splits(splits, dataset, DEFAULT_LOOP_SPLITS):
                items = load_split(dataset, split)
                store = JsonlStore(baseline_path(model, dataset, split))
                done = store.done_keys()
                pending = [i for i in items if i["uid"] not in done]
                rows.append({
                    "stage": "baseline",
                    "model": model,
                    "dataset": dataset,
                    "split": split,
                    "total": len(items),
                    "done": len(items) - len(pending),
                    "pending": len(pending),
                    "path": store.path,
                })
    return rows


def run(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    backend_kind: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_dirs()

    models = resolve_students(models)
    datasets = resolve_datasets(datasets)
    params = GenParams.from_config(STUDENT_GEN, seed=SEED)

    stats = {"generated": 0, "skipped": 0, "elapsed_s": 0.0, "tokens_out": 0}

    for model in models:
        # Descobre se há trabalho para este modelo ANTES de carregá-lo.
        work = [
            (d, s)
            for d in datasets
            for s in resolve_splits(splits, d, DEFAULT_LOOP_SPLITS)
            if _pending(model, d, s, limit)
        ]
        if not work:
            log.info("[%s] baseline completo, nada a fazer", model)
            continue

        with Timer() as timer, get_backend(model, backend_kind) as backend:
            for dataset, split in work:
                items = _pending(model, dataset, split, limit)
                if not items:
                    continue

                store = JsonlStore(baseline_path(model, dataset, split))
                log.info(
                    "[%s] %s/%s: %d pendentes (%d já feitos)",
                    model, dataset, split, len(items), len(store.done_keys()),
                )

                prompts = [build_answer_prompt(i) for i in items]
                gens = backend.generate(
                    prompts, params, desc=f"{model} {dataset}/{split}"
                )

                records = [
                    make_record(
                        item,
                        stage="baseline",
                        condition=COND_BASELINE,
                        student_model=model,
                        prompt=prompt,
                        output=gen.text,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        latency_s=gen.latency_s,
                        seed=SEED,
                        temperature=params.temperature,
                    )
                    for item, prompt, gen in zip(items, prompts, gens)
                ]
                store.append(records)

                stats["generated"] += len(records)
                stats["tokens_out"] += sum(g.completion_tokens for g in gens)
                n_ok = sum(1 for r in records if r.is_correct)
                log.info(
                    "  gravado %d, acerto %.1f%%, abstenção %d",
                    len(records),
                    100 * n_ok / max(len(records), 1),
                    sum(1 for r in records if r.predicted is None),
                )

        stats["elapsed_s"] += timer.elapsed

    log.info(
        "baseline: %d gerações, %d tokens, %.1f min",
        stats["generated"], stats["tokens_out"], stats["elapsed_s"] / 60,
    )
    return stats


def _pending(model: str, dataset: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    items = load_split(dataset, split)
    if limit:
        items = items[:limit]
    done = JsonlStore(baseline_path(model, dataset, split)).done_keys()
    return [i for i in items if i["uid"] not in done]


def load_baseline(model: str, dataset: str, split: str) -> dict[str, dict[str, Any]]:
    """
    Baseline indexado por uid. É a interface de reaproveitamento.

    Falha explicitamente se o arquivo não existe, porque uma etapa posterior que
    silenciosamente rodasse sem o baseline produziria resultados incomparáveis.
    """
    store = JsonlStore(baseline_path(model, dataset, split))
    if not store.exists():
        raise FileNotFoundError(
            f"baseline ausente: {store.path}\n"
            f"Rode: python -m rmcq baseline --models {model} --datasets {dataset} --splits {split}"
        )
    return {row["uid"]: row for row in store.read_all()}
