"""
Etapa 2: geração das reflexões sobre o split de treino.

O professor lê a questão, a resposta que o aluno deu no baseline e o feedback de
acerto ou erro, e escreve uma reflexão. Reflexão é escrita para TODA resposta,
certa ou errada, como manda a seção 3 do Caderno.

**A otimização que importa aqui é a ordem dos laços.** A grade tem 4 alunos × 4
professores × 2 profundidades = 32 combinações. Se o laço externo fosse a
combinação, seriam 32 cargas de modelo. Como o modelo carregado é sempre o
PROFESSOR, agrupamos por professor: 4 cargas, e cada professor carregado escreve
as reflexões de todos os 4 alunos nas 2 profundidades antes de sair da memória.

Além disso, a resposta do aluno vem do baseline de treino, que já existe. Sem
esse reaproveitamento, cada uma das 32 combinações regeraria as respostas do
aluno: 32 × 1.763 = 56 mil gerações a mais, para obter exatamente os mesmos
textos.

Diagonal e fora da diagonal usam prompts diferentes: aluno == professor é
autorreflexão e usa a perspectiva do aluno; aluno != professor é reflexão externa
e usa a perspectiva do professor. É a distinção da seção 4 do Caderno.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from rmcq.backends import GenParams, get_backend
from rmcq.common import Record, build_reflection_prompt, strip_think
from rmcq.config import SEED, TEACHER_GEN, condition_for, ensure_dirs, perspective_for
from rmcq.data import (
    load_index,
    reflections_path,
    resolve_datasets,
    resolve_depths,
    resolve_students,
    resolve_teachers,
)
from rmcq.stages.baseline import load_baseline
from rmcq.store import JsonlStore, Timer, get_logger

log = get_logger(__name__)

SPLIT = "train"  # reflexões só são geradas sobre o treino


def plan(
    students: Sequence[str] | None = None,
    teachers: Sequence[str] | None = None,
    depths: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for teacher in resolve_teachers(teachers):
        for student in resolve_students(students):
            for depth in resolve_depths(depths):
                for dataset in resolve_datasets(datasets):
                    items = load_index(dataset, SPLIT)
                    store = JsonlStore(reflections_path(student, teacher, depth, dataset))
                    done = store.done_keys()
                    pending = [u for u in items if u not in done]
                    rows.append({
                        "stage": "reflect",
                        "student": student,
                        "teacher": teacher,
                        "depth": depth,
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
    datasets: Sequence[str] | None = None,
    backend_kind: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_dirs()

    students = resolve_students(students)
    teachers = resolve_teachers(teachers)
    depths = resolve_depths(depths)
    datasets = resolve_datasets(datasets)

    params = GenParams.from_config(TEACHER_GEN, seed=SEED)
    stats = {"generated": 0, "elapsed_s": 0.0, "tokens_out": 0, "missing_baseline": []}

    # Agrupa por professor: uma carga de modelo por professor.
    todo: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for teacher in teachers:
        for student in students:
            for depth in depths:
                for dataset in datasets:
                    todo[teacher].append((student, depth, dataset))

    for teacher, combos in todo.items():
        # Filtra o que realmente falta antes de carregar o professor.
        work = []
        for student, depth, dataset in combos:
            try:
                pending = _pending(student, teacher, depth, dataset, limit)
            except FileNotFoundError as exc:
                log.warning("pulando %s/%s/%s: %s", student, depth, dataset, exc.args[0].splitlines()[0])
                stats["missing_baseline"].append(f"{student}/{dataset}")
                continue
            if pending:
                work.append((student, depth, dataset, pending))

        if not work:
            log.info("[professor=%s] nada a fazer", teacher)
            continue

        total_pending = sum(len(w[3]) for w in work)
        log.info(
            "[professor=%s] %d combinações, %d reflexões a gerar",
            teacher, len(work), total_pending,
        )

        with Timer() as timer, get_backend(teacher, backend_kind) as backend:
            for student, depth, dataset, pending in work:
                items = load_index(dataset, SPLIT)
                base = load_baseline(student, dataset, SPLIT)
                perspective = perspective_for(student, teacher)
                condition = condition_for(student, teacher)

                prompts, contexts = [], []
                for uid in pending:
                    item = items[uid]
                    answer = base[uid]
                    prompts.append(
                        build_reflection_prompt(
                            item,
                            previous_answer=answer["raw_output"],
                            # Abstenção conta como incorreta para o feedback: o
                            # aluno não entregou uma resposta válida.
                            was_correct=bool(answer.get("is_correct")),
                            depth=depth,
                            perspective=perspective,
                        )
                    )
                    contexts.append((item, answer))

                gens = backend.generate(
                    prompts, params,
                    desc=f"{teacher} refl {student}/{depth}/{dataset}",
                )

                records = []
                for (item, answer), prompt, gen in zip(contexts, prompts, gens):
                    records.append(
                        Record(
                            uid=item["uid"],
                            dataset=dataset,
                            split=SPLIT,
                            problem_type=item.get("problem_type", ""),
                            stage="reflect",
                            condition=condition,
                            student_model=student,
                            teacher_model=teacher,
                            prompt=prompt,
                            raw_output=gen.text,
                            # A reflexão não responde a questão; predicted e
                            # is_correct ficam nulos de propósito. O que vale
                            # aqui é reflection_text.
                            predicted=None,
                            gold=item["answerKey"],
                            is_correct=None,
                            reflection_depth=depth,
                            reflection_perspective=perspective,
                            reflection_text=strip_think(gen.text),
                            prompt_tokens=gen.prompt_tokens,
                            completion_tokens=gen.completion_tokens,
                            latency_s=gen.latency_s,
                            seed=SEED,
                            temperature=params.temperature,
                            extra={
                                # Guarda a correção da resposta refletida: a
                                # métrica de utility precisa saber se a reflexão
                                # nasceu de um acerto ou de um erro.
                                "source_was_correct": answer.get("is_correct"),
                                "source_predicted": answer.get("predicted"),
                            },
                        )
                    )

                JsonlStore(reflections_path(student, teacher, depth, dataset)).append(records)
                stats["generated"] += len(records)
                stats["tokens_out"] += sum(g.completion_tokens for g in gens)

                mean_words = sum(len(r.reflection_text.split()) for r in records) / max(len(records), 1)
                log.info(
                    "  %s/%s/%s: %d reflexões, %.0f palavras em média",
                    student, depth, dataset, len(records), mean_words,
                )

        stats["elapsed_s"] += timer.elapsed

    log.info(
        "reflexões: %d gerações, %d tokens, %.1f min",
        stats["generated"], stats["tokens_out"], stats["elapsed_s"] / 60,
    )
    return stats


def _pending(student: str, teacher: str, depth: str, dataset: str, limit: int | None) -> list[str]:
    """uids que ainda precisam de reflexão. Levanta se o baseline não existe."""
    items = load_index(dataset, SPLIT)
    base = load_baseline(student, dataset, SPLIT)

    uids = [u for u in items if u in base]
    if len(uids) < len(items):
        log.warning(
            "%s/%s: baseline cobre %d de %d itens de treino",
            student, dataset, len(uids), len(items),
        )
    if limit:
        uids = uids[:limit]

    done = JsonlStore(reflections_path(student, teacher, depth, dataset)).done_keys()
    return [u for u in uids if u not in done]


def load_reflections(student: str, teacher: str, depth: str, dataset: str) -> dict[str, dict[str, Any]]:
    """Reflexões indexadas por uid da questão de treino que as gerou."""
    store = JsonlStore(reflections_path(student, teacher, depth, dataset))
    if not store.exists():
        raise FileNotFoundError(
            f"reflexões ausentes: {store.path}\n"
            f"Rode: python -m rmcq reflect --students {student} "
            f"--teachers {teacher} --depths {depth} --datasets {dataset}"
        )
    return {row["uid"]: row for row in store.read_all()}
