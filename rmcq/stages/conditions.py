"""
Condições de controle: retry com feedback e self-consistency.

Ambas existem para impedir que o ganho da reflexão seja superestimado. As duas
reaproveitam o baseline de teste em vez de gerar de novo.

---------------------------------------------------------------------------
RETRY COM FEEDBACK, SEM REFLEXÃO
---------------------------------------------------------------------------

O Caderno já registra o problema: "sua resposta estava incorreta" elimina uma
alternativa sozinho. Em quatro alternativas, isso sobe o chute de 25% para 33%.
Sem essa condição, esse ganho puramente informacional seria atribuído à reflexão.

Uma ressalva de interpretação que precisa entrar no paper: **esta condição não é
diretamente comparável à etapa de avaliação.** Aqui o modelo revisita a MESMA
questão sabendo que errou. Na avaliação ele responde uma questão NOVA com
reflexões vindas de outras questões, e não recebe feedback sobre a própria
resposta. O retry controla o protocolo do paper anterior (reflexão na mesma
questão), não o da extensão.

Só itens que o baseline errou entram, porque só neles existe feedback de erro a
dar. A acurácia da condição é reportada sobre o conjunto completo de teste,
combinando os acertos do baseline com os itens recuperados no retry.

---------------------------------------------------------------------------
SELF-CONSISTENCY
---------------------------------------------------------------------------

N amostras com temperatura, voto majoritário. A literatura recente mostra
self-consistency batendo debate multi-agente (88,2 contra 83,0 no GSM8K), então
sem esta condição não há como afirmar que a reflexão faz algo que amostragem
barata não faria.

Sobre "orçamento equivalente": não é possível igualar exatamente, porque a
reflexão gasta tokens de ENTRADA (o prompt fica maior) e self-consistency gasta
tokens de SAÍDA (N gerações). Registramos os dois lados por linha, e a
comparação de custo é feita na etapa de análise, onde os números reais estão
disponíveis. Fixar N a priori pretendendo igualar orçamento daria uma falsa
precisão.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from rmcq.backends import GenParams, get_backend
from rmcq.common import build_retry_prompt, extract_final_answer, make_record
from rmcq.config import (
    COND_RETRY,
    COND_SELFCONS,
    SEED,
    SELFCONS_GEN,
    SELFCONS_N,
    STUDENT_GEN,
    ensure_dirs,
)
from rmcq.data import load_index, resolve_datasets, resolve_students, retry_path, selfcons_path
from rmcq.stages.baseline import load_baseline
from rmcq.store import JsonlStore, Timer, get_logger, sample_key

log = get_logger(__name__)

SPLIT = "test"


# ===========================================================================
# Retry com feedback
# ===========================================================================


def plan_retry(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for model in resolve_students(models):
        for dataset in resolve_datasets(datasets):
            try:
                wrong = _wrong_uids(model, dataset)
            except FileNotFoundError:
                rows.append({
                    "stage": "retry", "model": model, "dataset": dataset,
                    "total": 0, "done": 0, "pending": 0, "note": "baseline ausente",
                })
                continue
            done = JsonlStore(retry_path(model, dataset)).done_keys()
            pending = [u for u in wrong if u not in done]
            rows.append({
                "stage": "retry", "model": model, "dataset": dataset,
                "total": len(wrong), "done": len(wrong) - len(pending),
                "pending": len(pending), "path": retry_path(model, dataset),
            })
    return rows


def run_retry(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    backend_kind: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    models = resolve_students(models)
    datasets = resolve_datasets(datasets)
    params = GenParams.from_config(STUDENT_GEN, seed=SEED)
    stats = {"generated": 0, "recovered": 0, "elapsed_s": 0.0}

    for model in models:
        work = []
        for dataset in datasets:
            try:
                wrong = _wrong_uids(model, dataset)
            except FileNotFoundError as exc:
                log.warning("pulando %s/%s: %s", model, dataset, exc.args[0].splitlines()[0])
                continue
            done = JsonlStore(retry_path(model, dataset)).done_keys()
            pending = [u for u in wrong if u not in done]
            if limit:
                pending = pending[:limit]
            if pending:
                work.append((dataset, pending))

        if not work:
            log.info("[%s] retry completo", model)
            continue

        with Timer() as timer, get_backend(model, backend_kind) as backend:
            for dataset, pending in work:
                items = load_index(dataset, SPLIT)
                base = load_baseline(model, dataset, SPLIT)

                prompts, metas = [], []
                for uid in pending:
                    prev = base[uid].get("predicted")
                    if prev is None:
                        # Abstenção: não há letra para descartar, então o retry
                        # vira só o prompt normal com aviso de erro. Registramos
                        # a diferença em extra para não misturar os dois casos.
                        prev_letter = "(no valid answer)"
                    else:
                        prev_letter = prev
                    prompts.append(build_retry_prompt(items[uid], prev_letter))
                    metas.append((items[uid], prev))

                gens = backend.generate(prompts, params, desc=f"{model} retry {dataset}")

                records = []
                for (item, prev), prompt, gen in zip(metas, prompts, gens):
                    rec = make_record(
                        item,
                        stage="retry",
                        condition=COND_RETRY,
                        student_model=model,
                        prompt=prompt,
                        output=gen.text,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        latency_s=gen.latency_s,
                        seed=SEED,
                        temperature=params.temperature,
                        attempt=2,
                        extra={
                            "baseline_predicted": prev,
                            "baseline_abstained": prev is None,
                            # Se o modelo repetir a letra que acabou de ser
                            # descartada, isso é error loop: a métrica central
                            # do paper anterior.
                            "repeated_ruled_out_letter": None,
                        },
                    )
                    rec.extra["repeated_ruled_out_letter"] = (
                        rec.predicted is not None and rec.predicted == prev
                    )
                    records.append(rec)

                JsonlStore(retry_path(model, dataset)).append(records)
                n_rec = sum(1 for r in records if r.is_correct)
                n_loop = sum(1 for r in records if r.extra["repeated_ruled_out_letter"])
                stats["generated"] += len(records)
                stats["recovered"] += n_rec

                log.info(
                    "  %s: %d retries, %d recuperados (%.1f%%), %d error loops",
                    dataset, len(records), n_rec,
                    100 * n_rec / max(len(records), 1), n_loop,
                )

        stats["elapsed_s"] += timer.elapsed

    return stats


def _wrong_uids(model: str, dataset: str) -> list[str]:
    """uids que o baseline errou ou nos quais se absteve, na ordem do split."""
    base = load_baseline(model, dataset, SPLIT)
    return [
        uid for uid in load_index(dataset, SPLIT)
        if uid in base and not base[uid].get("is_correct")
    ]


# ===========================================================================
# Self-consistency
# ===========================================================================


def plan_selfcons(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    n: int = SELFCONS_N,
) -> list[dict[str, Any]]:
    rows = []
    for model in resolve_students(models):
        for dataset in resolve_datasets(datasets):
            items = load_index(dataset, SPLIT)
            done = JsonlStore(selfcons_path(model, dataset, n)).done_keys()
            pending = [u for u in items if u not in done]
            rows.append({
                "stage": "selfcons", "model": model, "dataset": dataset, "n": n,
                "total": len(items), "done": len(items) - len(pending),
                "pending": len(pending),
                # n amostras por item, mas numa só chamada de geração.
                "generations": len(pending) * n,
                "path": selfcons_path(model, dataset, n),
            })
    return rows


def run_selfcons(
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    n: int = SELFCONS_N,
    backend_kind: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    from rmcq.common import build_answer_prompt

    ensure_dirs()
    models = resolve_students(models)
    datasets = resolve_datasets(datasets)
    params = GenParams.from_config(SELFCONS_GEN, n=n, seed=SEED)
    stats = {"generated": 0, "elapsed_s": 0.0, "tokens_out": 0}

    for model in models:
        work = []
        for dataset in datasets:
            uids = list(load_index(dataset, SPLIT))
            if limit:
                uids = uids[:limit]
            done = JsonlStore(selfcons_path(model, dataset, n)).done_keys()
            pending = [u for u in uids if u not in done]
            if pending:
                work.append((dataset, pending))

        if not work:
            log.info("[%s] self-consistency completo", model)
            continue

        with Timer() as timer, get_backend(model, backend_kind) as backend:
            for dataset, pending in work:
                items = load_index(dataset, SPLIT)
                prompts = [build_answer_prompt(items[u]) for u in pending]

                gens = backend.generate(
                    prompts, params, desc=f"{model} selfcons n={n} {dataset}"
                )

                records = []
                for uid, prompt, gen in zip(pending, prompts, gens):
                    item = items[uid]
                    votes = [
                        extract_final_answer(s, item["choices"]).letter
                        for s in (gen.samples or [gen.text])
                    ]
                    valid = [v for v in votes if v is not None]

                    if valid:
                        # Empate resolvido pela ordem alfabética do rótulo, para
                        # ser determinístico. Registramos a margem para que
                        # empates possam ser filtrados na análise.
                        counts = Counter(valid)
                        top = max(counts.values())
                        winner = sorted(l for l, c in counts.items() if c == top)[0]
                        margin = top / len(valid)
                        tied = sum(1 for c in counts.values() if c == top) > 1
                    else:
                        winner, margin, tied = None, 0.0, False

                    rec = make_record(
                        item,
                        stage="selfcons",
                        condition=COND_SELFCONS,
                        student_model=model,
                        prompt=prompt,
                        # A saída canônica é a primeira amostra; as demais ficam
                        # em extra. O voto sobrescreve predicted abaixo.
                        output=gen.samples[0] if gen.samples else gen.text,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        latency_s=gen.latency_s,
                        seed=SEED,
                        temperature=params.temperature,
                        extra={
                            "n_samples": n,
                            "votes": votes,
                            "n_valid_votes": len(valid),
                            "vote_margin": round(margin, 4),
                            "tie": tied,
                            "samples": gen.samples,
                        },
                    )
                    # O resultado da condição é o voto majoritário, não a
                    # primeira amostra. Sobrescrevemos explicitamente.
                    rec.predicted = winner
                    rec.is_correct = None if winner is None else winner == item["answerKey"]
                    records.append(rec)

                JsonlStore(selfcons_path(model, dataset, n)).append(records)
                stats["generated"] += len(records) * n
                stats["tokens_out"] += sum(g.completion_tokens for g in gens)

                n_ok = sum(1 for r in records if r.is_correct)
                n_tie = sum(1 for r in records if r.extra["tie"])
                log.info(
                    "  %s: %d itens x %d amostras, acerto %.1f%%, %d empates",
                    dataset, len(records), n, 100 * n_ok / max(len(records), 1), n_tie,
                )

        stats["elapsed_s"] += timer.elapsed

    return stats
