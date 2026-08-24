"""
Etapa 4: avaliação no teste com reflexões recuperadas.

Para cada questão de teste, buscamos as k questões de treino semanticamente mais
próximas, pegamos as reflexões que o professor escreveu sobre elas, injetamos no
prompt e pedimos ao aluno que responda.

Esta etapa domina o custo — cerca de 87% do total. Duas otimizações:

1. **ALUNO no laço externo.** Quem gera aqui é o aluno, então agrupamos por
   aluno: 4 cargas de modelo, e cada aluno carregado roda todas as suas
   combinações de professor, profundidade, k e dataset. Se a combinação fosse o
   laço externo, seriam 96 cargas.

2. **Índice compartilhado.** Os vizinhos de cada questão de teste são os mesmos
   para todas as 96 configurações, porque a similaridade depende só do texto das
   questões. Carregamos o índice uma vez por dataset e reusamos. Note também que
   os vizinhos de k=1 são um prefixo dos de k=5, então nada é recalculado.

O que NÃO fazemos: reaproveitar a resposta entre valores de k. Poderia parecer
que k=1 e k=3 compartilham trabalho, mas o prompt é diferente, então a geração é
diferente. São 3 execuções mesmo.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from rmcq.backends import GenParams, get_backend
from rmcq.common import EVAL_SYSTEM, build_eval_prompt, make_record
from rmcq.config import EMBEDDER, SEED, STUDENT_GEN, condition_for, ensure_dirs
from rmcq.data import (
    eval_path,
    load_index,
    resolve_datasets,
    resolve_depths,
    resolve_ks,
    resolve_students,
    resolve_teachers,
)
from rmcq.retrieval import Neighbors
from rmcq.stages.reflect import load_reflections
from rmcq.store import JsonlStore, Timer, get_logger

log = get_logger(__name__)

SPLIT = "test"


def plan(
    students: Sequence[str] | None = None,
    teachers: Sequence[str] | None = None,
    depths: Sequence[str] | None = None,
    ks: Sequence[int] | None = None,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for student in resolve_students(students):
        for teacher in resolve_teachers(teachers):
            for depth in resolve_depths(depths):
                for k in resolve_ks(ks):
                    for dataset in resolve_datasets(datasets):
                        items = load_index(dataset, SPLIT)
                        store = JsonlStore(eval_path(student, teacher, depth, k, dataset))
                        done = store.done_keys()
                        pending = [u for u in items if u not in done]
                        rows.append({
                            "stage": "eval",
                            "student": student,
                            "teacher": teacher,
                            "depth": depth,
                            "k": k,
                            "dataset": dataset,
                            "condition": condition_for(student, teacher),
                            "total": len(items),
                            "done": len(items) - len(pending),
                            "pending": len(pending),
                            "path": store.path,
                        })
    return rows


def run(
    students: Sequence[str] | None = None,
    teachers: Sequence[str] | None = None,
    depths: Sequence[str] | None = None,
    ks: Sequence[int] | None = None,
    datasets: Sequence[str] | None = None,
    backend_kind: str | None = None,
    embedder: str = EMBEDDER,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_dirs()

    students = resolve_students(students)
    teachers = resolve_teachers(teachers)
    depths = resolve_depths(depths)
    ks = resolve_ks(ks)
    datasets = resolve_datasets(datasets)

    params = GenParams.from_config(STUDENT_GEN, seed=SEED)
    stats: dict[str, Any] = {
        "generated": 0, "elapsed_s": 0.0, "tokens_out": 0,
        "tokens_in": 0, "missing_reflections": [],
    }

    # Índice por dataset, carregado uma vez e compartilhado por tudo.
    neighbors: dict[str, Neighbors] = {}
    for dataset in datasets:
        neighbors[dataset] = Neighbors(dataset, embedder)
        log.info("índice: %s", neighbors[dataset])

    combos: dict[str, list[tuple[str, str, int, str]]] = defaultdict(list)
    for student in students:
        for teacher in teachers:
            for depth in depths:
                for k in ks:
                    for dataset in datasets:
                        combos[student].append((teacher, depth, k, dataset))

    for student, work_all in combos.items():
        # Resolve pendências e reflexões faltantes antes de carregar o aluno.
        work = []
        for teacher, depth, k, dataset in work_all:
            try:
                refl = load_reflections(student, teacher, depth, dataset)
            except FileNotFoundError as exc:
                log.warning("pulando %s: %s", (teacher, depth, k, dataset), exc.args[0].splitlines()[0])
                stats["missing_reflections"].append(f"{student}/{teacher}/{depth}/{dataset}")
                continue
            pending = _pending(student, teacher, depth, k, dataset, limit)
            if pending:
                work.append((teacher, depth, k, dataset, pending, refl))

        if not work:
            log.info("[aluno=%s] nada a fazer", student)
            continue

        log.info(
            "[aluno=%s] %d configurações, %d respostas a gerar",
            student, len(work), sum(len(w[4]) for w in work),
        )

        with Timer() as timer, get_backend(student, backend_kind) as backend:
            for teacher, depth, k, dataset, pending, refl in work:
                items = load_index(dataset, SPLIT)
                nn = neighbors[dataset]
                condition = condition_for(student, teacher)

                # Candidatos restritos ao que realmente tem reflexão: assim o k
                # pedido é o k entregue, mesmo em piloto ou run parcial.
                available = {u for u, r in refl.items() if r.get("reflection_text")}
                ranked = nn.top_k_for_all(k, allowed_uids=available)
                train_items = load_index(dataset, "train")

                prompts, metas = [], []
                for uid in pending:
                    texts, src_questions, src_correct, used_uids, sims = [], [], [], [], []
                    for train_uid, sim in ranked.get(uid, []):
                        row = refl.get(train_uid)
                        if not row or not row.get("reflection_text"):
                            continue
                        texts.append(row["reflection_text"])
                        src_questions.append(train_items[train_uid]["question"])
                        # Se a lição veio de um acerto ou de um erro na
                        # questão de origem (não da questão atual — ver
                        # rmcq.common, seção "Injeção das k reflexões").
                        src_correct.append(row.get("extra", {}).get("source_was_correct"))
                        used_uids.append(train_uid)
                        sims.append(sim)

                    prompts.append(
                        build_eval_prompt(items[uid], texts, src_questions, src_correct)
                    )
                    metas.append((items[uid], used_uids, sims))

                gens = backend.generate(
                    prompts, params, system=EVAL_SYSTEM,
                    desc=f"{student} eval {teacher}/{depth}/k{k}/{dataset}",
                )

                records = [
                    make_record(
                        item,
                        stage="eval",
                        condition=condition,
                        student_model=student,
                        prompt=prompt,
                        output=gen.text,
                        teacher_model=teacher,
                        reflection_depth=depth,
                        reflection_perspective=(
                            "student" if student == teacher else "teacher"
                        ),
                        retrieved_uids=used_uids,
                        retrieved_similarities=[round(s, 6) for s in sims],
                        k=k,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        latency_s=gen.latency_s,
                        seed=SEED,
                        temperature=params.temperature,
                        extra={
                            # Guardado para a figura de transferability: a
                            # similaridade da reflexão mais próxima é o eixo x.
                            "top1_similarity": max(sims) if sims else None,
                            "mean_similarity": (sum(sims) / len(sims)) if sims else None,
                            "n_reflections_injected": len(used_uids),
                            # `prompt` guarda só o turno `user`. O `system` é
                            # fixo em toda linha desta etapa, mas gravamos aqui
                            # mesmo assim — sem isso o JSONL sozinho não
                            # reproduz o input real enviado ao backend.
                            "system_prompt": EVAL_SYSTEM,
                        },
                    )
                    for (item, used_uids, sims), prompt, gen in zip(metas, prompts, gens)
                ]

                JsonlStore(eval_path(student, teacher, depth, k, dataset)).append(records)
                stats["generated"] += len(records)
                stats["tokens_out"] += sum(g.completion_tokens for g in gens)
                stats["tokens_in"] += sum(g.prompt_tokens for g in gens)

                n_ok = sum(1 for r in records if r.is_correct)
                log.info(
                    "  %s/%s/k%d/%s: %d respostas, acerto %.1f%%",
                    teacher, depth, k, dataset, len(records),
                    100 * n_ok / max(len(records), 1),
                )

                # Um arquivo rotulado k=3 em que a maioria das linhas recebeu 1
                # reflexão não é um resultado de k=3. Isso acontece quando a
                # etapa de reflexão está incompleta para este dataset, e é fácil
                # de não notar porque nada quebra.
                short = sum(1 for r in records if len(r.retrieved_uids) < k)
                if short:
                    stats["underfilled"] = stats.get("underfilled", 0) + short
                    actual = sum(len(r.retrieved_uids) for r in records) / max(len(records), 1)
                    log.warning(
                        "    ATENÇÃO: %d de %d linhas receberam menos de k=%d reflexões "
                        "(média real %.2f). Reflexões incompletas para %s/%s/%s/%s.",
                        short, len(records), k, actual, student, teacher, depth, dataset,
                    )

        stats["elapsed_s"] += timer.elapsed

    log.info(
        "avaliação: %d gerações, %d tokens de entrada, %d de saída, %.1f min",
        stats["generated"], stats["tokens_in"], stats["tokens_out"], stats["elapsed_s"] / 60,
    )
    return stats


def _pending(
    student: str, teacher: str, depth: str, k: int, dataset: str, limit: int | None
) -> list[str]:
    uids = list(load_index(dataset, SPLIT))
    if limit:
        uids = uids[:limit]
    done = JsonlStore(eval_path(student, teacher, depth, k, dataset)).done_keys()
    return [u for u in uids if u not in done]
