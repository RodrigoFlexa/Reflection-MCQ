#!/usr/bin/env python3
"""
Grade de debug (versao do prompt, k, limiar de similaridade, com/sem Key
Takeaway e questao de origem) como processo standalone, para rodar em
tmux/nohup em vez de dentro do notebook.

--prompt-version escolhe o layout do prompt de avaliacao (rmcq/common.py):
v1 e o antigo (as licoes abrem o prompt, antes de o modelo saber qual e a
tarefa; sem corte por orcamento; letras de alternativa da questao de origem
passam inteiras) e v2 e o revisto para modelos pequenos (enquadramento
primeiro, <notes> delimitadas com o enunciado de origem, questao e formato de
saida por ultimo, reflexao cortada em 120 palavras pela cabeca e letras
neutralizadas). As linhas ja gravadas com v1 continuam validas e nao sao
refeitas: a chave de retomada so ganha sufixo a partir do v2.

A variante 'with' pede a um modelo local um resumo curto de cada reflexao ja
escrita (uma linha comecando com Takeaway:), e concatena esse resumo a ela
antes de injetar no prompt do aluno. Por padrao, se o professor da reflexao
for gpt-5-petrobras (Azure, indisponivel no momento), quem escreve o Takeaway
e o llama3-8b, nao o professor original -- use --takeaway-model para trocar.
Os takeaways sao gerados uma vez por (aluno, modelo do takeaway, profundidade,
dataset) e cacheados em notebooks/debugger_results/takeaways/, reaproveitados
entre valores de k e de limiar.

Por que existe: dentro do notebook, uma queda do servidor Jupyter derruba a
grade inteira e o que não foi salvo em disco se perde. Este script grava cada
lote de respostas assim que é gerado, com fsync, então rodar de novo com os
mesmos argumentos retoma exatamente de onde parou — nada é refeito.

Uso típico (dentro de uma sessão tmux):

    source venv/bin/activate
    python scripts/run_debugger_grid.py --backend hf
    python scripts/run_debugger_grid.py --backend vllm --cuda-devices 5

Os resultados vão para notebooks/debugger_results/responses/{student}/{dataset}.jsonl.
O notebook notebooks/debugger.ipynb lê esses arquivos; ele não roda mais
inferência.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["arc", "gsm8k"])
    parser.add_argument("--students", nargs="+", default=["phi4-mini", "llama3-8b"])
    parser.add_argument("--teacher", default="gpt-5-petrobras")
    parser.add_argument("--depths", nargs="+", default=["simple", "complex"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1,2])
    parser.add_argument(
        "--thresholds", nargs="+", default=["none", "0.80"],
        help="Use 'none' para nao filtrar por similaridade.",
    )
    parser.add_argument(
        "--takeaway-variants", nargs="+", default=["without", "with"], choices=["without", "with"],
        help="'with' concatena um Key Takeaway a cada reflexao recuperada.",
    )
    parser.add_argument(
        "--takeaway-model", default=None,
        help="Modelo que escreve o Takeaway. Default: llama3-8b se --teacher for "
             "gpt-5-petrobras (Azure indisponivel), senao o proprio --teacher.",
    )
    parser.add_argument(
        "--include-source-question", action=argparse.BooleanOptionalAction, default=True,
        help="Inclui o enunciado de origem junto a cada reflexao recuperada.",
    )
    parser.add_argument(
        "--include-source-outcome", action=argparse.BooleanOptionalAction, default=False,
        help="Indica se a resposta que gerou cada reflexao estava correta.",
    )
    parser.add_argument(
        "--prompt-version", default="v2", choices=["v1", "v2"],
        help="v1: layout antigo (licoes antes do enquadramento, sem corte nem "
             "neutralizacao de letra). v2: layout revisto para modelos pequenos.",
    )
    parser.add_argument("--n-items", type=int, default=50, help="Itens de teste por dataset.")
    parser.add_argument("--backend", default=None, help="Sobrepoe RMCQ_BACKEND (hf, vllm, stub).")
    parser.add_argument("--cuda-devices", default=None, help="Sobrepoe CUDA_VISIBLE_DEVICES.")
    parser.add_argument(
        "--out-dir", default=str(ROOT / "notebooks" / "debugger_results"),
        help="Raiz onde as respostas e o log ficam.",
    )
    return parser.parse_args()


def _parse_thresholds(raw: list[str]) -> list[float | None]:
    out: list[float | None] = []
    for value in raw:
        out.append(None if value.lower() == "none" else float(value))
    return out


def main() -> None:
    args = parse_args()

    # Precisa acontecer ANTES de qualquer import que toque CUDA/torch.
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    if args.backend:
        os.environ["RMCQ_BACKEND"] = args.backend

    sys.path.insert(0, str(ROOT))

    import numpy as np

    from rmcq.backends import GenParams, get_backend
    from rmcq.common import (
        build_eval_prompt,
        extract_final_answer,
        format_question,
        read_jsonl,
    )
    from rmcq.config import BACKEND, SEED, STUDENT_GEN
    from rmcq.store import JsonlStore, get_logger, log_to_file

    log = get_logger("debugger_grid")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_to_file("debugger_grid")
    log.info("log espelhado em %s", log_path)
    log.info("backend efetivo: %s", BACKEND)

    thresholds = _parse_thresholds(args.thresholds)
    takeaway_variants = list(dict.fromkeys(args.takeaway_variants))
    takeaway_model = args.takeaway_model or (
        "llama3-8b" if args.teacher == "gpt-5-petrobras" else args.teacher
    )
    if "with" in takeaway_variants:
        log.info("Key Takeaway sera escrito por: %s", takeaway_model)

    # Prompt curto pedido ao professor DEPOIS da reflexao ja escrita: destila
    # a reflexao num unico principio geral, para testar se o resumo ajuda o
    # aluno mais do que a reflexao inteira ou atrapalha por diluicao.
    TAKEAWAY_PROMPT = (
        "Extract a single, reusable rule of thumb (a 'Takeaway') from the reflection below.\n\n"
        "RULES:\n"
        "1. Focus on the GENERAL LESSON or reasoning principle.\n"
        "2. Abstract away specific details. Replace specific names, numbers, or scenarios with general categories (e.g., use 'units' instead of 'eggs').\n"
        "3. Keep general subject-matter concepts (e.g., 'cellular respiration', 'natural selection') if essential to the lesson.\n\n"
        "EXAMPLES:\n"
        "- Bad: Track the number of eggs and convert them to dozens.\n"
        "- Good: Track units throughout a calculation and verify the final quantity.\n"
        "- Bad: Remember that mitochondria produce ATP in this scenario.\n"
        "- Good: When answering biology questions, distinguish the function of each organelle.\n\n"
        "REFLECTION TO ANALYZE:\n"
        "<reflection>\n"
        "{reflection}\n"
        "</reflection>\n\n"
        "OUTPUT INSTRUCTIONS:\n"
        "Output ONLY a single line starting with 'Takeaway:'. Do not add any explanation.\n\n"
        "Takeaway:"
    )


    def extract_takeaway_line(text: str) -> str:
        for line in text.strip().splitlines():
            line = line.strip()
            if line.lower().startswith("takeaway:"):
                return line.split(":", 1)[1].strip()
        stripped = text.strip()
        return stripped.splitlines()[-1].strip() if stripped else ""

    def append_takeaway(reflection_text: str, takeaway_text: str) -> str:
        return f"{reflection_text.rstrip()}\nKey Takeaway: {takeaway_text}" if takeaway_text else reflection_text

    def load_map(path: Path) -> dict:
        return {row["uid"]: row for row in read_jsonl(path)} if path.exists() else {}

    def find_index(dataset: str) -> Path:
        candidates = sorted((ROOT / "results" / "index").glob(f"*/{dataset}/neighbors.npz"))
        if not candidates:
            raise FileNotFoundError(f"Indice ausente para {dataset}")
        return candidates[0]

    def load_neighbors(dataset: str, required_k: int):
        neighbors_path = find_index(dataset)
        cached = np.load(neighbors_path, allow_pickle=False)
        train_uids = cached["train_uids"].tolist()
        test_uids = cached["test_uids"].tolist()
        if cached["top_idx"].shape[1] >= required_k:
            return train_uids, test_uids, cached["top_idx"], cached["top_sim"]

        train_embeddings = np.load(neighbors_path.with_name("train.npy"))
        test_embeddings = np.load(neighbors_path.with_name("test.npy"))
        similarities = test_embeddings @ train_embeddings.T
        ranking = np.argsort(-similarities, axis=1, kind="stable")[:, :required_k]
        row_indices = np.arange(similarities.shape[0])[:, None]
        return (
            train_uids, test_uids, ranking.astype(np.int32),
            similarities[row_indices, ranking].astype(np.float32),
        )

    def combo_key(row: dict) -> str:
        # O sufixo de versao so entra a partir do v2, para nao invalidar as
        # chaves das linhas ja gravadas com o prompt antigo: o arquivo continua
        # retomavel, e v1 e v2 convivem sem colidir.
        suffix = "" if args.prompt_version == "v1" else f"::prompt={args.prompt_version}"
        return (
            f"{row['uid']}::{row['depth']}::{row['k']}::{row['similarity_threshold']}"
            f"::{row['takeaway']}::source_question={args.include_source_question}"
            f"::source_outcome={args.include_source_outcome}{suffix}"
        )

    def takeaway_path(student: str, depth: str, dataset: str) -> Path:
        return out_dir / "takeaways" / f"{student}__{takeaway_model}__{depth}" / f"{dataset}.jsonl"

    def load_takeaways(student: str, depth: str, dataset: str) -> dict[str, str]:
        path = takeaway_path(student, depth, dataset)
        return {row["uid"]: row["takeaway"] for row in read_jsonl(path)} if path.exists() else {}

    max_k = max(args.ks)
    params = GenParams.from_config(STUDENT_GEN, seed=SEED)
    takeaway_params = GenParams.from_config({**STUDENT_GEN, "max_new_tokens": 96}, seed=SEED)
    log.info("questao de origem nas reflexoes: %s", args.include_source_question)
    log.info("acerto da resposta de origem nas reflexoes: %s", args.include_source_outcome)
    log.info("versao do prompt de avaliacao: %s", args.prompt_version)

    def build_takeaways(teacher_backend, student: str, depth: str, dataset: str, refl: dict) -> int:
        take_store = JsonlStore(takeaway_path(student, depth, dataset))
        have = take_store.done_keys()
        missing = [
            uid for uid, row in refl.items()
            if row.get("reflection_text") and uid not in have
        ]
        if missing:
            prompts = [TAKEAWAY_PROMPT.format(reflection=refl[uid]["reflection_text"]) for uid in missing]
            generations = teacher_backend.generate(
                prompts, takeaway_params, desc=f"takeaway {student}/{dataset}/{depth}",
            )
            new_rows = [
                {"uid": uid, "takeaway": extract_takeaway_line(generation.text)}
                for uid, generation in zip(missing, generations)
            ]
            take_store.append(new_rows)
        return len(missing)

    # Fase separada, ANTES da grade de avaliacao: escreve todos os Key
    # Takeaways de uma vez com takeaway_model, e fecha o backend antes de
    # carregar qualquer aluno. Evita ter dois modelos na mesma GPU ao mesmo
    # tempo, o que um vLLM/HF local nao aceita como o Azure aceitava.
    if "with" in takeaway_variants:
        log.info("preparando Key Takeaways com %s", takeaway_model)
        with get_backend(takeaway_model) as teacher_backend:
            for dataset in args.datasets:
                for student in args.students:
                    for depth in args.depths:
                        refl = load_map(
                            ROOT / "results" / "reflections" / f"{student}__{args.teacher}__{depth}" / f"{dataset}.jsonl"
                        )
                        if not refl:
                            continue
                        n_new = build_takeaways(teacher_backend, student, depth, dataset, refl)
                        if n_new:
                            log.info("  %s/%s/%s: %d takeaways novos", student, dataset, depth, n_new)

    for dataset in args.datasets:
        items = load_map(ROOT / "data" / "splits" / dataset / "test.jsonl")
        train_items = load_map(ROOT / "data" / "splits" / dataset / "train.jsonl")
        selected = list(items.values())[: args.n_items]
        train_uids, test_uids, top_idx, top_sim = load_neighbors(dataset, max_k)
        pos = {uid: i for i, uid in enumerate(test_uids)}

        for student in args.students:
            store = JsonlStore(out_dir / "responses" / student / f"{dataset}.jsonl", key_fn=combo_key)
            done = store.done_keys()

            # Monta o trabalho pendente antes de carregar o modelo, para nao
            # pagar o custo de carga se nao houver nada a fazer.
            pending_combos = []
            for depth, k, threshold, takeaway in itertools.product(args.depths, args.ks, thresholds, takeaway_variants):
                pending_uids = [
                    item["uid"] for item in selected
                    if combo_key({
                        "uid": item["uid"], "depth": depth, "k": k,
                        "similarity_threshold": threshold, "takeaway": takeaway,
                    }) not in done
                ]
                if pending_uids:
                    pending_combos.append((depth, k, threshold, takeaway, pending_uids))

            if not pending_combos:
                log.info("[%s/%s] nada pendente, pulando carga do backend", student, dataset)
                continue

            log.info(
                "[%s/%s] %d configuracoes pendentes, %d respostas no total",
                student, dataset, len(pending_combos), sum(len(c[4]) for c in pending_combos),
            )

            refl_cache: dict[str, dict] = {}
            takeaway_cache: dict[str, dict[str, str]] = {}
            with get_backend(student) as backend:
                for depth, k, threshold, takeaway, pending_uids in pending_combos:
                    if depth not in refl_cache:
                        refl_cache[depth] = load_map(
                            ROOT / "results" / "reflections" / f"{student}__{args.teacher}__{depth}" / f"{dataset}.jsonl"
                        )
                    refl = refl_cache[depth]
                    if takeaway == "with" and depth not in takeaway_cache:
                        takeaway_cache[depth] = load_takeaways(student, depth, dataset)
                    take_map = takeaway_cache.get(depth, {})

                    pending_items = [items[uid] for uid in pending_uids]
                    prompts, meta = [], []
                    for item in pending_items:
                        row = pos[item["uid"]]
                        pairs = [
                            (train_uids[j], float(s))
                            for j, s in zip(top_idx[row, :k], top_sim[row, :k])
                            if threshold is None or float(s) >= threshold
                        ]
                        pairs = [(u, s) for u, s in pairs if (refl.get(u) or {}).get("reflection_text")]
                        pairs.sort(key=lambda pair: pair[1])
                        texts = [refl[u]["reflection_text"] for u, _ in pairs]
                        source_questions = [format_question(train_items[u]) for u, _ in pairs]
                        # O acerto da resposta de origem virou parametro do
                        # montador (antes era prefixado no texto na mao, o que
                        # o corte por orcamento do v2 poderia comer).
                        source_outcomes = (
                            [(refl[u].get("extra") or {}).get("source_was_correct") for u, _ in pairs]
                            if args.include_source_outcome else None
                        )
                        if takeaway == "with":
                            texts = [append_takeaway(t, take_map.get(u, "")) for t, (u, _) in zip(texts, pairs)]
                        prompts.append(build_eval_prompt(
                            item,
                            texts,
                            source_questions,
                            source_outcomes,
                            include_source=args.include_source_question,
                            version=args.prompt_version,
                        ))
                        meta.append((item, pairs))

                    desc = f"{student}/{dataset}/{depth}/k{k}/thr{threshold}/takeaway={takeaway}"
                    generations = backend.generate(prompts, params, desc=desc)

                    records = []
                    for (item, pairs), generation, prompt in zip(meta, generations, prompts):
                        extraction = extract_final_answer(generation.text, item["choices"])
                        records.append({
                            "uid": item["uid"], "dataset": dataset, "student": student,
                            "teacher": args.teacher, "depth": depth,
                            "k": k, "similarity_threshold": threshold, "takeaway": takeaway,
                            "takeaway_model": takeaway_model if takeaway == "with" else None,
                            "include_source_question": args.include_source_question,
                            "include_source_outcome": args.include_source_outcome,
                            "prompt_version": args.prompt_version,
                            "predicted": extraction.letter, "gold": item["answerKey"],
                            "is_correct": extraction.letter == item["answerKey"],
                            "n_reflections": len(pairs),
                            "top1_similarity": max((s for _, s in pairs), default=None),
                            "prompt": prompt, "raw_output": generation.text,
                        })
                    store.append(records)
                    n_ok = sum(1 for r in records if r["is_correct"])
                    log.info("  %s: %d respostas, acerto %.1f%%", desc, len(records), 100 * n_ok / max(len(records), 1))

    log.info("grade concluida. Resultados em %s", out_dir / "responses")


if __name__ == "__main__":
    main()
