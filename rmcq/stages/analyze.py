"""
Etapa 5: métricas (Caderno, seção 5).

Produz CSVs em results/analysis/ mais um resumo em markdown. Nada aqui chama
modelo: só lê os JSONL das etapas anteriores.

As definições que importam, e por que são assim:

**Acurácia.** Reportada em duas versões. `accuracy` conta abstenção como erro (é
o número do paper). `accuracy_answered` só considera itens em que uma letra foi
extraída. A diferença separa "não sabe resolver" de "não sabe seguir o formato",
que são achados distintos.

**Reflection utility.** Taxa de virada errado→certo MENOS certo→errado, do
baseline para a condição com reflexão, no mesmo item. Subtrair o segundo termo é
o ponto: uma reflexão que conserta cinco itens e estraga cinco tem utility zero,
não ganho de cinco.

**Transferability.** A mesma utility, condicionada à faixa de similaridade entre
a questão de teste e a de treino que gerou a reflexão. É a figura central do
paper: se a utility não decai com a distância semântica, a reflexão não está
transferindo, está fazendo outra coisa.

**Persistência de erro.** Fração dos erros de treino do aluno que reaparecem no
teste apesar da reflexão em memória. É a versão transferível do error loop do
paper anterior.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Sequence

from rmcq.common import strip_think
from rmcq.config import (
    ANALYSIS_DIR,
    EMBEDDER,
    SELFCONS_N,
    condition_for,
    ensure_dirs,
)
from rmcq.data import (
    baseline_path,
    eval_path,
    reflections_path,
    resolve_datasets,
    resolve_depths,
    resolve_ks,
    resolve_students,
    resolve_teachers,
    retry_path,
    selfcons_path,
)
from rmcq.store import JsonlStore, get_logger

log = get_logger(__name__)

# Faixas de similaridade da figura de transferability. Escolhidas para que as
# duas primeiras isolem recuperação genuinamente distante e a última capture
# quase-duplicatas que sobraram da deduplicação.
SIM_BINS = ((0.0, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))


# ---------------------------------------------------------------------------
# Métricas elementares
# ---------------------------------------------------------------------------


def accuracy_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    answered = [r for r in rows if r.get("predicted") is not None]
    n_correct = sum(1 for r in answered if r.get("is_correct"))

    # Distribuição dos métodos de extração. Auditável de propósito: a acurácia
    # de um modelo que raramente usa o formato exigido depende das regras de
    # resgate, e isso precisa estar visível em vez de embutido no número final.
    methods = Counter(str(r.get("extraction_method") or "none") for r in rows)

    return {
        "n": n,
        "n_answered": len(answered),
        "n_abstained": n - len(answered),
        "n_correct": n_correct,
        "accuracy": round(n_correct / n, 4),
        "accuracy_answered": round(n_correct / len(answered), 4) if answered else None,
        "format_adherence": round(sum(1 for r in rows if r.get("followed_format")) / n, 4),
        **{f"extr_{m}": methods.get(m, 0) for m in (
            "strict", "loose_final", "answer_is", "value_match", "bare_letter", "none",
        )},
        # Acurácia restrita às linhas em que o formato exigido foi seguido. Se
        # divergir muito da acurácia geral, as regras de resgate estão fazendo
        # trabalho pesado e isso merece uma linha no paper.
        "accuracy_strict_format_only": (
            round(
                sum(1 for r in rows if r.get("followed_format") and r.get("is_correct"))
                / max(sum(1 for r in rows if r.get("followed_format")), 1), 4,
            )
            if any(r.get("followed_format") for r in rows) else None
        ),
        "mean_prompt_tokens": round(sum(r.get("prompt_tokens") or 0 for r in rows) / n, 1),
        "mean_completion_tokens": round(sum(r.get("completion_tokens") or 0 for r in rows) / n, 1),
        "total_tokens": sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rows),
        "mean_latency_s": round(sum(r.get("latency_s") or 0 for r in rows) / n, 4),
    }


def chance_accuracy(items: Iterable[dict[str, Any]]) -> float:
    """Acurácia esperada de um chute, ponderada pelo nº de alternativas."""
    vals = [1 / i["num_choices"] for i in items if i.get("num_choices")]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def text_stats(texts: Sequence[str]) -> dict[str, Any]:
    """Comprimento e diversidade lexical (TTR) de um conjunto de reflexões."""
    if not texts:
        return {"n": 0}

    words_per = [len(t.split()) for t in texts]
    chars_per = [len(t) for t in texts]
    # TTR por documento e agregado. O agregado é sensível ao tamanho da amostra,
    # então o número que compara condições é a média do TTR por documento.
    ttr_per = [
        len(set(w.lower() for w in t.split())) / max(len(t.split()), 1) for t in texts
    ]
    all_words = [w.lower() for t in texts for w in t.split()]

    return {
        "n": len(texts),
        "mean_words": round(sum(words_per) / len(texts), 1),
        "median_words": sorted(words_per)[len(words_per) // 2],
        "mean_chars": round(sum(chars_per) / len(texts), 1),
        "mean_ttr": round(sum(ttr_per) / len(ttr_per), 4),
        "corpus_ttr": round(len(set(all_words)) / max(len(all_words), 1), 4),
    }


def utility(
    baseline: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Reflection utility: (errado→certo) − (certo→errado), sobre os itens comuns.

    Abstenção é tratada como incorreta aqui, porque a pergunta é "a condição
    melhorou o resultado?", e uma abstenção não é um resultado melhor.
    """
    shared = [u for u in condition if u in baseline]
    if not shared:
        return {"n_shared": 0}

    w2r = r2w = kept_r = kept_w = 0
    for uid in shared:
        was = bool(baseline[uid].get("is_correct"))
        now = bool(condition[uid].get("is_correct"))
        if not was and now:
            w2r += 1
        elif was and not now:
            r2w += 1
        elif was and now:
            kept_r += 1
        else:
            kept_w += 1

    n = len(shared)
    return {
        "n_shared": n,
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "kept_right": kept_r,
        "kept_wrong": kept_w,
        "rate_wrong_to_right": round(w2r / n, 4),
        "rate_right_to_wrong": round(r2w / n, 4),
        "utility": round((w2r - r2w) / n, 4),
        "baseline_accuracy": round((kept_r + r2w) / n, 4),
        "condition_accuracy": round((kept_r + w2r) / n, 4),
        "delta_accuracy": round((w2r - r2w) / n, 4),
    }


def transferability(
    baseline: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
    bins: Sequence[tuple[float, float]] = SIM_BINS,
) -> list[dict[str, Any]]:
    """Utility por faixa de similaridade da reflexão mais próxima."""
    buckets: dict[tuple[float, float], dict[str, dict[str, Any]]] = {b: {} for b in bins}

    for uid, row in condition.items():
        if uid not in baseline:
            continue
        sim = (row.get("extra") or {}).get("top1_similarity")
        if sim is None:
            sims = row.get("retrieved_similarities") or []
            sim = max(sims) if sims else None
        if sim is None:
            continue
        for lo, hi in bins:
            if lo <= sim < hi:
                buckets[(lo, hi)][uid] = row
                break

    out = []
    for (lo, hi), rows in buckets.items():
        block = utility(baseline, rows)
        out.append({
            "sim_lo": lo,
            "sim_hi": hi,
            "sim_bin": f"[{lo:.2f}, {hi:.2f})",
            **block,
        })
    return out


def error_persistence(
    train_baseline: dict[str, dict[str, Any]],
    test_condition: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Versão transferível do error loop.

    Não é o mesmo item, então "persistência" aqui é: entre as questões de teste
    cuja reflexão recuperada veio de um item que o aluno ERROU no treino, quantas
    o aluno também erra no teste? Se a reflexão sobre um erro não evita o erro
    seguinte parecido, ela não está transferindo nada.
    """
    from_error = [
        r for r in test_condition.values()
        if (r.get("extra") or {}).get("source_errors", 0) > 0
    ]
    if not from_error:
        # Sem a anotação, cai no cálculo direto pelos uids recuperados.
        from_error = []
        for row in test_condition.values():
            srcs = row.get("retrieved_uids") or []
            if any(
                src in train_baseline and not train_baseline[src].get("is_correct")
                for src in srcs
            ):
                from_error.append(row)

    if not from_error:
        return {"n": 0}

    still_wrong = sum(1 for r in from_error if not r.get("is_correct"))
    return {
        "n_with_reflection_from_error": len(from_error),
        "still_wrong": still_wrong,
        "error_persistence": round(still_wrong / len(from_error), 4),
    }


# ---------------------------------------------------------------------------
# Coleta sobre todos os arquivos
# ---------------------------------------------------------------------------


def _load(path) -> dict[str, dict[str, Any]]:
    store = JsonlStore(path)
    return {r["uid"]: r for r in store.read_all()} if store.exists() else {}


def collect(
    students: Sequence[str] | None = None,
    teachers: Sequence[str] | None = None,
    depths: Sequence[str] | None = None,
    ks: Sequence[int] | None = None,
    datasets: Sequence[str] | None = None,
    selfcons_n: int = SELFCONS_N,
) -> dict[str, list[dict[str, Any]]]:
    from rmcq.data import load_split

    students = resolve_students(students)
    teachers = resolve_teachers(teachers)
    depths = resolve_depths(depths)
    ks = resolve_ks(ks)
    datasets = resolve_datasets(datasets)

    tables: dict[str, list[dict[str, Any]]] = {
        "accuracy": [], "reflections": [], "utility": [],
        "transferability": [], "cost": [], "persistence": [],
    }

    # --- baseline -----------------------------------------------------------
    baselines: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for student in students:
        for dataset in datasets:
            for split in ("train", "test"):
                rows = _load(baseline_path(student, dataset, split))
                baselines[(student, dataset, split)] = rows
                if rows:
                    tables["accuracy"].append({
                        "condition": "no_reflection", "student": student,
                        "teacher": None, "depth": None, "k": None,
                        "dataset": dataset, "split": split,
                        "chance": chance_accuracy(load_split(dataset, split)),
                        **accuracy_block(list(rows.values())),
                    })

    # --- reflexões: comprimento, TTR ---------------------------------------
    for student in students:
        for teacher in teachers:
            for depth in depths:
                for dataset in datasets:
                    rows = _load(reflections_path(student, teacher, depth, dataset))
                    if not rows:
                        continue
                    texts = [
                        strip_think(r.get("reflection_text") or "") for r in rows.values()
                    ]
                    tables["reflections"].append({
                        "student": student, "teacher": teacher, "depth": depth,
                        "dataset": dataset,
                        "condition": condition_for(student, teacher),
                        **text_stats([t for t in texts if t]),
                        "mean_completion_tokens": round(
                            sum(r.get("completion_tokens") or 0 for r in rows.values())
                            / max(len(rows), 1), 1,
                        ),
                        "from_correct": sum(
                            1 for r in rows.values()
                            if (r.get("extra") or {}).get("source_was_correct")
                        ),
                        "from_incorrect": sum(
                            1 for r in rows.values()
                            if not (r.get("extra") or {}).get("source_was_correct")
                        ),
                    })

    # --- avaliação: acurácia, utility, transferability ----------------------
    for student in students:
        for teacher in teachers:
            for depth in depths:
                for k in ks:
                    for dataset in datasets:
                        rows = _load(eval_path(student, teacher, depth, k, dataset))
                        if not rows:
                            continue
                        base = baselines.get((student, dataset, "test"), {})
                        cond = condition_for(student, teacher)
                        tag = {
                            "condition": cond, "student": student, "teacher": teacher,
                            "depth": depth, "k": k, "dataset": dataset, "split": "test",
                        }

                        tables["accuracy"].append({
                            **tag,
                            "chance": chance_accuracy(load_split(dataset, "test")),
                            **accuracy_block(list(rows.values())),
                        })
                        if base:
                            tables["utility"].append({**tag, **utility(base, rows)})
                            for band in transferability(base, rows):
                                tables["transferability"].append({**tag, **band})
                            tables["persistence"].append({
                                **tag,
                                **error_persistence(
                                    baselines.get((student, dataset, "train"), {}), rows
                                ),
                            })

    # --- controles ----------------------------------------------------------
    for student in students:
        for dataset in datasets:
            base = baselines.get((student, dataset, "test"), {})

            retry_rows = _load(retry_path(student, dataset))
            if retry_rows and base:
                # A acurácia da condição é sobre TODO o teste: acertos do
                # baseline mais itens recuperados no retry.
                combined = dict(base)
                combined.update(retry_rows)
                loops = sum(
                    1 for r in retry_rows.values()
                    if (r.get("extra") or {}).get("repeated_ruled_out_letter")
                )
                tables["accuracy"].append({
                    "condition": "retry_feedback", "student": student, "teacher": None,
                    "depth": None, "k": None, "dataset": dataset, "split": "test",
                    "chance": chance_accuracy(load_split(dataset, "test")),
                    **accuracy_block(list(combined.values())),
                })
                tables["utility"].append({
                    "condition": "retry_feedback", "student": student, "teacher": None,
                    "depth": None, "k": None, "dataset": dataset, "split": "test",
                    **utility(base, combined),
                    "error_loops": loops,
                    "error_loop_rate": round(loops / max(len(retry_rows), 1), 4),
                })

            sc_rows = _load(selfcons_path(student, dataset, selfcons_n))
            if sc_rows:
                tables["accuracy"].append({
                    "condition": "self_consistency", "student": student, "teacher": None,
                    "depth": None, "k": selfcons_n, "dataset": dataset, "split": "test",
                    "chance": chance_accuracy(load_split(dataset, "test")),
                    **accuracy_block(list(sc_rows.values())),
                })
                if base:
                    tables["utility"].append({
                        "condition": "self_consistency", "student": student,
                        "teacher": None, "depth": None, "k": selfcons_n,
                        "dataset": dataset, "split": "test",
                        **utility(base, sc_rows),
                        "mean_vote_margin": round(
                            sum((r.get("extra") or {}).get("vote_margin", 0)
                                for r in sc_rows.values()) / max(len(sc_rows), 1), 4,
                        ),
                        "ties": sum(1 for r in sc_rows.values()
                                    if (r.get("extra") or {}).get("tie")),
                    })

    # --- custo por condição -------------------------------------------------
    for row in tables["accuracy"]:
        tables["cost"].append({
            k: row.get(k) for k in (
                "condition", "student", "teacher", "depth", "k", "dataset", "split",
                "n", "mean_prompt_tokens", "mean_completion_tokens", "total_tokens",
                "mean_latency_s",
            )
        })

    return tables


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], path) -> int:
    import csv

    if not rows:
        return 0
    # União das chaves, preservando a ordem de primeira aparição.
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def embedders_used(datasets: Sequence[str]) -> list[str]:
    """
    Quais embedders realmente produziram os índices em disco.

    Lido do meta.json, não do config: rotular o resumo com o embedder do .env
    enquanto o índice foi construído com outro produziria uma tabela que mente
    sobre a própria procedência.
    """
    from rmcq.config import INDEX_DIR

    found = []
    for meta in sorted(INDEX_DIR.rglob("meta.json")):
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("dataset") in datasets and info.get("embedder") not in found:
            found.append(info["embedder"])
    return found


def run(
    students: Sequence[str] | None = None,
    teachers: Sequence[str] | None = None,
    depths: Sequence[str] | None = None,
    ks: Sequence[int] | None = None,
    datasets: Sequence[str] | None = None,
    selfcons_n: int = SELFCONS_N,
) -> dict[str, Any]:
    ensure_dirs()
    tables = collect(students, teachers, depths, ks, datasets, selfcons_n)
    used = embedders_used(resolve_datasets(datasets))

    written = {}
    for name, rows in tables.items():
        n = _write_csv(rows, ANALYSIS_DIR / f"{name}.csv")
        written[name] = n
        log.info("%-16s %5d linhas -> analysis/%s.csv", name, n, name)

    summary = _markdown_summary(tables, used)
    (ANALYSIS_DIR / "summary.md").write_text(summary, encoding="utf-8")
    (ANALYSIS_DIR / "tables.json").write_text(
        json.dumps(tables, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    log.info("resumo -> analysis/summary.md")
    return {"written": written, "summary_path": ANALYSIS_DIR / "summary.md"}


def _markdown_summary(
    tables: dict[str, list[dict[str, Any]]],
    embedders: Sequence[str] | None = None,
) -> str:
    import time

    if not embedders:
        label = "nenhum índice encontrado"
    else:
        label = ", ".join(f"`{e}`" for e in embedders)

    lines = [
        "# Resumo da análise",
        "",
        f"Gerado em {time.strftime('%Y-%m-%d %H:%M')} | embedder do índice: {label}",
        "",
    ]
    if embedders and any(e == "hashing" for e in embedders):
        lines += [
            "> **Atenção:** o índice foi construído com o embedder `hashing`, que é "
            "lexical e existe apenas para testar o pipeline. Os números de "
            "transferability abaixo não têm significado semântico. Reconstrua com "
            f"`python -m rmcq index --force --embedder {EMBEDDER}`.",
            "",
        ]

    acc = tables["accuracy"]
    if not acc:
        lines += ["Nenhum resultado encontrado. Rode as etapas antes de analisar.", ""]
        return "\n".join(lines)

    # --- acurácia por condição --------------------------------------------
    lines += ["## Acurácia por condição (agregada sobre datasets de teste)", ""]
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for row in acc:
        if row.get("split") == "test":
            by_cond.setdefault(str(row["condition"]), []).append(row)

    lines += ["| condição | configs | n | acurácia | s/ abstenção | aderência ao formato |",
              "|---|---|---|---|---|---|"]
    for cond, rows in sorted(by_cond.items()):
        n = sum(r["n"] for r in rows)
        correct = sum(r["n_correct"] for r in rows)
        answered = sum(r["n_answered"] for r in rows)
        fmt = sum(r["format_adherence"] * r["n"] for r in rows) / max(n, 1)
        lines.append(
            f"| {cond} | {len(rows)} | {n:,} | {correct / max(n, 1):.3f} | "
            f"{correct / max(answered, 1):.3f} | {fmt:.3f} |"
        )
    lines.append("")

    # --- utility ----------------------------------------------------------
    util = [r for r in tables["utility"] if r.get("n_shared")]
    if util:
        lines += ["## Reflection utility, melhores e piores configurações", ""]
        ranked = sorted(util, key=lambda r: -r["utility"])
        lines += ["| condição | aluno | professor | prof. | k | dataset | utility | E→C | C→E |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in ranked[:5] + (["..."] if len(ranked) > 10 else []) + ranked[-5:]:
            if row == "...":
                lines.append("| ... | | | | | | | | |")
                continue
            lines.append(
                f"| {row['condition']} | {row['student']} | {row.get('teacher') or '—'} | "
                f"{row.get('depth') or '—'} | {row.get('k') or '—'} | {row['dataset']} | "
                f"{row['utility']:+.4f} | {row['wrong_to_right']} | {row['right_to_wrong']} |"
            )
        lines.append("")

    # --- transferability ---------------------------------------------------
    trans = [r for r in tables["transferability"] if r.get("n_shared")]
    if trans:
        lines += [
            "## Transferability: utility por faixa de similaridade",
            "",
            "É a figura central. Se a utility não cai com a distância semântica, "
            "o ganho não vem de transferência.",
            "",
            "| faixa de cosseno | n | utility | E→C | C→E |",
            "|---|---|---|---|---|",
        ]
        agg: dict[str, dict[str, int]] = {}
        for row in trans:
            b = agg.setdefault(row["sim_bin"], {"n": 0, "w2r": 0, "r2w": 0})
            b["n"] += row["n_shared"]
            b["w2r"] += row["wrong_to_right"]
            b["r2w"] += row["right_to_wrong"]
        for band in sorted(agg):
            b = agg[band]
            lines.append(
                f"| {band} | {b['n']:,} | {(b['w2r'] - b['r2w']) / max(b['n'], 1):+.4f} | "
                f"{b['w2r']} | {b['r2w']} |"
            )
        lines.append("")

    # --- reflexões ---------------------------------------------------------
    refl = tables["reflections"]
    if refl:
        lines += ["## Reflexões: comprimento e diversidade lexical", "",
                  "| profundidade | n | palavras (média) | TTR médio | tokens gerados |",
                  "|---|---|---|---|---|"]
        by_depth: dict[str, list[dict[str, Any]]] = {}
        for row in refl:
            by_depth.setdefault(str(row["depth"]), []).append(row)
        for depth, rows in sorted(by_depth.items()):
            n = sum(r["n"] for r in rows)
            words = sum(r["mean_words"] * r["n"] for r in rows) / max(n, 1)
            ttr = sum(r["mean_ttr"] * r["n"] for r in rows) / max(n, 1)
            toks = sum(r["mean_completion_tokens"] * r["n"] for r in rows) / max(n, 1)
            lines.append(f"| {depth} | {n:,} | {words:.0f} | {ttr:.3f} | {toks:.0f} |")
        lines.append("")

    # --- extração ----------------------------------------------------------
    test_rows = [r for r in acc if r.get("split") == "test" and r.get("n")]
    if test_rows:
        lines += [
            "## Como a resposta foi detectada",
            "",
            "`strict` é o formato exigido (`FINAL ANSWER: X`). O resto são regras "
            "de resgate, em ordem de aplicação. Se `strict` for baixo para algum "
            "modelo, isso é resultado a reportar, não bug a esconder no extrator.",
            "",
            "| modelo | n | strict | loose | prosa | valor | letra solta | abstenção |",
            "|---|---|---|---|---|---|---|---|",
        ]
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in test_rows:
            by_model.setdefault(str(row["student"]), []).append(row)
        for model, rows in sorted(by_model.items()):
            n = sum(r["n"] for r in rows)
            g = lambda key: sum(r.get(key, 0) for r in rows)  # noqa: E731
            lines.append(
                f"| {model} | {n:,} | {g('extr_strict') / max(n, 1):.1%} | "
                f"{g('extr_loose_final') / max(n, 1):.1%} | "
                f"{g('extr_answer_is') / max(n, 1):.1%} | "
                f"{g('extr_value_match') / max(n, 1):.1%} | "
                f"{g('extr_bare_letter') / max(n, 1):.1%} | "
                f"{g('extr_none') / max(n, 1):.1%} |"
            )
        lines.append("")

        # value_match existe por causa do GSM8K, cujas alternativas são números.
        vm = [r for r in test_rows if r.get("extr_value_match")]
        if vm:
            by_ds: dict[str, tuple[int, int]] = {}
            for row in vm:
                cur = by_ds.get(str(row["dataset"]), (0, 0))
                by_ds[str(row["dataset"])] = (cur[0] + row["extr_value_match"], cur[1] + row["n"])
            frag = ", ".join(f"{d} {c}/{t} ({c / max(t, 1):.1%})" for d, (c, t) in sorted(by_ds.items()))
            lines += [
                f"Respostas resolvidas pelo VALOR da alternativa em vez da letra: {frag}. "
                "Concentração no GSM8K é esperada — lá as alternativas são números e o "
                "modelo tende a terminar com o resultado da conta.",
                "",
            ]

    # --- custo -------------------------------------------------------------
    cost = [r for r in tables["cost"] if r.get("split") == "test" and r.get("n")]
    if cost:
        lines += ["## Custo por condição, por questão de teste", "",
                  "| condição | tokens entrada | tokens saída | total |",
                  "|---|---|---|---|"]
        by_c: dict[str, list[dict[str, Any]]] = {}
        for row in cost:
            by_c.setdefault(str(row["condition"]), []).append(row)
        for cond, rows in sorted(by_c.items()):
            n = sum(r["n"] for r in rows)
            pin = sum((r["mean_prompt_tokens"] or 0) * r["n"] for r in rows) / max(n, 1)
            pout = sum((r["mean_completion_tokens"] or 0) * r["n"] for r in rows) / max(n, 1)
            lines.append(f"| {cond} | {pin:.0f} | {pout:.0f} | {pin + pout:.0f} |")
        lines.append("")

    lines += [
        "## Como ler",
        "",
        "- `accuracy` conta abstenção como erro; é o número do paper.",
        "- `utility` já subtrai as viradas certo→errado. Utility zero com muitas "
        "viradas nas duas direções significa que a reflexão mexe no resultado sem melhorá-lo.",
        "- Compare sempre com `retry_feedback` antes de atribuir ganho à reflexão: "
        "dizer que a resposta estava errada já elimina uma alternativa.",
        "- `self_consistency` gasta tokens de saída, a reflexão gasta de entrada. "
        "A tabela de custo tem os dois lados; nenhum dos dois é orçamento igual por construção.",
    ]
    return "\n".join(lines)
