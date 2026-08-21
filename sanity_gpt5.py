#!/usr/bin/env python
"""
Teste de sanidade: o GPT-5 escreve reflexões sobre respostas reais de um aluno.

Para rodar ANTES da grade de 14 mil chamadas. Pega algumas questões de treino,
a resposta que o aluno de verdade deu no baseline, monta o mesmo prompt que a
etapa `reflect` montaria, chama o GPT-5 e imprime tudo lado a lado — pergunta,
resposta do aluno, acerto ou erro, e a reflexão gerada.

O que você está conferindo com os próprios olhos:

  1. A reflexão fala do raciocínio DAQUELE aluno, não é texto genérico.
  2. Ela NÃO entrega a resposta certa (o prompt proíbe; modelos de reasoning
     às vezes desobedecem, e é melhor descobrir agora).
  3. O tamanho bate com a profundidade pedida: simple = 3-6 frases.
  4. `completion_tokens` > 0 e o texto não está vazio.

Por padrão NÃO grava nada em results/ — é diagnóstico, não dado experimental.
Use --save se quiser que as reflexões contem para a grade.

Uso:
    python sanity_gpt5.py                              # 3 questões de arc, phi4-mini, simple
    python sanity_gpt5.py -n 5 --dataset gsm8k
    python sanity_gpt5.py --depth complex --student llama3-8b
    python sanity_gpt5.py --only-wrong                 # só respostas que o aluno errou
    python sanity_gpt5.py --backend stub               # sem tocar na API, só para ver o formato
"""

from __future__ import annotations

import argparse
import sys

import rmcq  # noqa: F401 — carrega o .env antes de qualquer coisa
from rmcq.backends import GenParams, get_backend
from rmcq.common import build_reflection_prompt, format_options, format_question, strip_think
from rmcq.config import SEED, TEACHER_GEN, condition_for, perspective_for
from rmcq.data import load_index
from rmcq.stages.baseline import load_baseline

LINHA = "=" * 78
TRACO = "-" * 78


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sanidade: GPT-5 refletindo sobre respostas reais de um aluno.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-n", "--num", type=int, default=3, help="quantas questões (padrão: 3)")
    p.add_argument("--student", default="phi4-mini", help="aluno cujas respostas serão refletidas")
    p.add_argument("--teacher", default="gpt5", help="professor (padrão: gpt5)")
    p.add_argument("--dataset", default="arc", help="dataset (padrão: arc)")
    p.add_argument("--depth", default="simple", choices=("simple", "complex"))
    p.add_argument("--only-wrong", action="store_true",
                   help="só questões que o aluno errou (onde a reflexão mais importa)")
    p.add_argument("--only-right", action="store_true",
                   help="só questões que o aluno acertou")
    p.add_argument("--backend", default=None, choices=("vllm", "hf", "stub", "azure"),
                   help="stub não chama a API; serve para ver o formato de graça")
    p.add_argument("--full-answer", action="store_true",
                   help="imprimir a resposta do aluno inteira, sem truncar")
    p.add_argument("--show-prompt", action="store_true",
                   help="imprimir também o prompt exato enviado ao professor")
    p.add_argument("--save", action="store_true",
                   help="gravar em results/reflections/ (padrão: não grava nada)")
    return p.parse_args(argv)


def truncar(texto: str, limite: int, inteiro: bool = False) -> str:
    texto = (texto or "").strip()
    if inteiro or len(texto) <= limite:
        return texto
    return texto[:limite].rstrip() + f"\n[... +{len(texto) - limite} caracteres]"


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.only_wrong and args.only_right:
        print("--only-wrong e --only-right se excluem.", file=sys.stderr)
        return 2

    # --- carregar questões e respostas do aluno --------------------------
    try:
        items = load_index(args.dataset, "train")
        base = load_baseline(args.student, args.dataset, "train")
    except FileNotFoundError as exc:
        print(f"\nERRO: {exc}\n", file=sys.stderr)
        print("Nesta máquina os dados chegam pelo pacote de troca. Rode antes:", file=sys.stderr)
        print("  python -m rmcq import-bundle --direction to-azure", file=sys.stderr)
        return 1

    uids = [u for u in items if u in base]
    if args.only_wrong:
        uids = [u for u in uids if not base[u].get("is_correct")]
    elif args.only_right:
        uids = [u for u in uids if base[u].get("is_correct")]

    if not uids:
        print("nenhuma questão bate com o filtro pedido.", file=sys.stderr)
        return 1
    uids = uids[: args.num]

    perspective = perspective_for(args.student, args.teacher)
    condition = condition_for(args.student, args.teacher)

    print(f"\n{LINHA}")
    print(f"  SANIDADE — {args.teacher} refletindo sobre {args.student}")
    print(f"{LINHA}")
    print(f"  dataset      {args.dataset} (treino)")
    print(f"  profundidade {args.depth}")
    print(f"  perspectiva  {perspective}   |   condição  {condition}")
    print(f"  questões     {len(uids)}")
    if args.backend == "stub":
        print("  backend      stub — NÃO chama a API, texto é falso")
    print(f"  gravando     {'sim, em results/reflections/' if args.save else 'não (diagnóstico)'}")
    print(LINHA)

    # --- montar os prompts, exatamente como reflect.py faria -------------
    prompts, contextos = [], []
    for uid in uids:
        item, answer = items[uid], base[uid]
        prompts.append(
            build_reflection_prompt(
                item,
                previous_answer=answer["raw_output"],
                was_correct=bool(answer.get("is_correct")),
                depth=args.depth,
                perspective=perspective,
            )
        )
        contextos.append((item, answer))

    # --- chamar o professor ---------------------------------------------
    params = GenParams.from_config(TEACHER_GEN, seed=SEED)
    try:
        with get_backend(args.teacher, args.backend) as backend:
            gens = backend.generate(prompts, params, desc=f"{args.teacher} sanidade")
    except Exception as exc:
        print(f"\nFALHOU: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 1

    # --- imprimir -------------------------------------------------------
    vazias = 0
    for i, ((item, answer), prompt, gen) in enumerate(zip(contextos, prompts, gens), 1):
        reflexao = strip_think(gen.text)
        acertou = bool(answer.get("is_correct"))
        veredito = "ACERTOU" if acertou else ("ABSTEVE-SE" if answer.get("predicted") is None else "ERROU")

        print(f"\n{LINHA}")
        print(f"  [{i}/{len(uids)}]  {item['uid']}   —   aluno {veredito}")
        print(LINHA)

        print(f"\nPERGUNTA\n{format_question(item)}\n")
        print(f"OPÇÕES\n{format_options(item['choices'])}")
        print(f"\ngabarito: {item['answerKey']}   |   aluno respondeu: {answer.get('predicted')}")

        if args.show_prompt:
            print(f"\n{TRACO}\nPROMPT ENVIADO AO PROFESSOR\n{TRACO}\n{prompt}")

        print(f"\n{TRACO}\nRESPOSTA DO ALUNO ({args.student})\n{TRACO}")
        print(truncar(answer["raw_output"], 700, args.full_answer))

        print(f"\n{TRACO}\nREFLEXÃO DO PROFESSOR ({args.teacher}, {args.depth})\n{TRACO}")
        print(reflexao if reflexao else ">>> VAZIA <<<")

        n_palavras = len(reflexao.split())
        n_frases = reflexao.count(".") + reflexao.count("!") + reflexao.count("?")
        print(
            f"\n[{n_palavras} palavras, ~{n_frases} frases, "
            f"{gen.completion_tokens} tokens de saída, {gen.latency_s:.1f}s, "
            f"finish_reason={gen.finish_reason!r}]"
        )
        if not reflexao:
            vazias += 1

    # --- veredito -------------------------------------------------------
    print(f"\n{LINHA}")
    media = sum(len(strip_think(g.text).split()) for g in gens) / len(gens)
    print(f"  {len(gens)} reflexões, {media:.0f} palavras em média, {vazias} vazia(s)")
    print(LINHA)
    print("\n  Confira com os próprios olhos, antes de liberar a grade:")
    print("    [ ] a reflexão comenta o raciocínio DESTE aluno (não é texto genérico)")
    print("    [ ] ela NÃO revela qual é a alternativa correta")
    if args.depth == "simple":
        print("    [ ] o tamanho é de 3 a 6 frases, como o prompt pede")
    print("    [ ] nenhuma reflexão veio vazia")
    print()

    if args.save:
        _gravar(args, uids, contextos, prompts, gens, perspective, condition)

    return 1 if vazias else 0


def _gravar(args, uids, contextos, prompts, gens, perspective, condition) -> None:
    """Só com --save. Grava no mesmo formato da etapa reflect."""
    from rmcq.common import Record
    from rmcq.data import reflections_path
    from rmcq.store import JsonlStore

    registros = [
        Record(
            uid=item["uid"], dataset=args.dataset, split="train",
            problem_type=item.get("problem_type", ""),
            stage="reflect", condition=condition,
            student_model=args.student, teacher_model=args.teacher,
            prompt=prompt, raw_output=gen.text,
            predicted=None, gold=item["answerKey"], is_correct=None,
            reflection_depth=args.depth, reflection_perspective=perspective,
            reflection_text=strip_think(gen.text),
            prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
            latency_s=gen.latency_s, seed=SEED, temperature=TEACHER_GEN["temperature"],
            extra={
                "source_was_correct": answer.get("is_correct"),
                "source_predicted": answer.get("predicted"),
            },
        )
        for (item, answer), prompt, gen in zip(contextos, prompts, gens)
    ]
    caminho = reflections_path(args.student, args.teacher, args.depth, args.dataset)
    JsonlStore(caminho).append(registros)
    print(f"  gravadas {len(registros)} reflexões em {caminho}\n")


if __name__ == "__main__":
    sys.exit(main())
