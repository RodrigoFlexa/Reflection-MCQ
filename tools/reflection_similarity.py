#!/usr/bin/env python
"""Similaridade semântica entre a questão de treino e a reflexão escrita sobre ela.

    python tools/reflection_similarity.py --experiment-id 21186c44cfbb
    python tools/reflection_similarity.py --experiment-id 1f84f440d6e7 --out validation.csv

Lê o exchange de uma run, embeda cada questão de treino e cada reflexão com o
mesmo modelo da recuperação (BAAI/bge-large-en-v1.5) e grava um CSV com uma
linha por (aluno, dataset, fonte de treino, braço):

    model,dataset,source_uid,arm,similarity,tokens,truncated,chars

`arm` é `self_simple`, `self_complex` ou `teacher_<profundidade>@<professor>`,
a mesma convenção da análise. O CSV é pequeno, alguns MB, e é ele que volta
para a máquina de análise; nenhuma reflexão viaja junto.

Duas decisões que valem estar escritas:

- **Sem prefixo de consulta dos dois lados.** Na recuperação, a questão de
  avaliação é consulta (leva o prefixo do BGE) e a de treino é passagem (não
  leva). Aqui as duas pontas são documentos, e comparar documento com documento
  é simétrico. `--query-prefix` põe o prefixo na reflexão para quem quiser a
  leitura assimétrica; o padrão não põe.
- **Nada é cortado em silêncio.** O BGE só enxerga 512 tokens, e as reflexões
  complexas passam disso com frequência. `--pooling chunks`, que é o padrão,
  parte o texto longo em pedaços de 512, embeda cada um e usa a média
  normalizada, então a reflexão inteira entra na conta. `--pooling truncate`
  volta ao corte simples, para comparação. As colunas `tokens`, `truncated` e
  `chunks` registram o que aconteceu em cada linha, e `pooling` registra o modo,
  para que a figura nunca dependa de lembrar qual foi.

Reflexões ausentes (filtro de conteúdo, limite de comprimento, geração vazia)
não entram: não há texto para embedar, e inventar zero ali seria afirmar
distância semântica onde houve ausência de geração.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEPTHS = ("simple", "complex")
DEFAULT_DATASETS = "aqua,arc,logiqa2,openbookqa"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--exchange-root", default="experiment_exchange")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS,
                        help="CSV de datasets; o padrão já deixa o RACE de fora.")
    parser.add_argument("--models", help="CSV de alunos; o padrão é todos os que têm treino.")
    parser.add_argument("--teachers", help="CSV de professores; o padrão é todos os do exchange.")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pooling", choices=("chunks", "truncate"), default="chunks",
                        help="chunks: média normalizada dos pedaços de 512 tokens, texto inteiro. "
                             "truncate: só os 512 primeiros tokens, como o encoder faria sozinho.")
    parser.add_argument("--query-prefix", action="store_true",
                        help="Trata a reflexão como consulta do BGE. O padrão é simétrico.")
    parser.add_argument("--limit", type=int, help="Só as N primeiras fontes de cada arquivo; teste de fumaça.")
    parser.add_argument("--out", type=Path, help="CSV de saída (padrão: reflection_similarity_<id>.csv).")
    return parser.parse_args()


def embedding_text(item: dict) -> str:
    """A mesma composição usada na recuperação: contexto, questão e opções."""
    options = " | ".join(choice["text"].strip() for choice in item["choices"])
    context = (item.get("context") or "").strip()
    return "\n".join(p for p in (context, item["question"].strip(), options) if p)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def student_files(exchange: Path, wanted: list[str] | None) -> list[tuple[str, Path]]:
    found = sorted((p.parent.name, p) for p in exchange.glob("students/*/train.jsonl"))
    if wanted:
        fora = [m for m in wanted if m not in {name for name, _ in found}]
        if fora:
            raise SystemExit(f"sem treino no exchange: {', '.join(fora)}")
        found = [(name, path) for name, path in found if name in wanted]
    return found


def teacher_files(exchange: Path, wanted: list[str] | None,
                  students: list[str]) -> list[tuple[str, str, Path]]:
    """(professor, aluno, arquivo), cobrindo o layout atual e o de um professor só.

    O recorte de alunos vale dos dois lados: pedir `--models phi2` e ainda assim
    embedar as reflexões escritas para os outros alunos seria trabalho que
    ninguém pediu, e o CSV sairia com linhas fora do recorte.
    """
    found = [(path.parents[1].name, path.stem, path)
             for path in sorted(exchange.glob("teacher/*/student_reflections/*.jsonl"))]
    found += [(None, path.stem, path)
              for path in sorted(exchange.glob("teacher/student_reflections/*.jsonl"))]
    found = [(t, s, p) for t, s, p in found if s in students]
    if wanted:
        found = [(t, s, p) for t, s, p in found if (t or "") in wanted]
    return found


def main() -> None:
    args = parse_args()
    datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}
    exchange = Path(args.exchange_root) / args.experiment_id
    if not exchange.exists():
        raise SystemExit(f"{exchange} não existe; restaure o exchange dessa run primeiro")
    manifest = exchange / "manifest.json"
    legacy_teacher = "gpt-5-4-petrobras"
    if manifest.exists():
        legacy_teacher = json.loads(manifest.read_text(encoding="utf-8")).get("teacher_model", legacy_teacher)

    alunos = student_files(exchange, args.models.split(",") if args.models else None)
    if not alunos:
        raise SystemExit(f"nenhum students/<aluno>/train.jsonl em {exchange}")
    professores = teacher_files(exchange, args.teachers.split(",") if args.teachers else None,
                                [nome for nome, _ in alunos])
    print(f"{len(alunos)} aluno(s), {len(professores)} par(es) professor-aluno, "
          f"datasets: {', '.join(sorted(datasets))}", flush=True)

    # As questões de treino são as mesmas para todos os alunos; embeda uma vez.
    perguntas: dict[str, str] = {}
    dataset_de: dict[str, str] = {}
    for _, path in alunos:
        for row in load_jsonl(path):
            if row["dataset"] in datasets and row["source_uid"] not in perguntas:
                perguntas[row["source_uid"]] = embedding_text(row["item"])
                dataset_de[row["source_uid"]] = row["dataset"]
    print(f"{len(perguntas)} questão(ões) de treino únicas", flush=True)

    import numpy as np
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    limite_tokens = encoder.max_seq_length

    def embed(texts: list[str]) -> "np.ndarray":
        return encoder.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)

    def partir(texto: str) -> list[str]:
        """Pedaços de no máximo `limite_tokens`, sem sobreposição."""
        ids = encoder.tokenizer(texto, add_special_tokens=False, truncation=False)["input_ids"]
        passo = max(1, limite_tokens - 2)
        return [encoder.tokenizer.decode(ids[i:i + passo]) for i in range(0, len(ids), passo)]

    def vetores_de(textos: list[str], n_tokens: list[int]):
        """Um vetor por texto, com os longos entrando inteiros quando pedido.

        A média dos pedaços é renormalizada, então o cosseno continua sendo
        cosseno. Sem isso, o braço mais longo seria medido sobre menos texto que
        o mais curto, e a comparação entre profundidades mediria o encoder.
        """
        longos = [i for i, n in enumerate(n_tokens) if n > limite_tokens]
        if args.pooling == "truncate" or not longos:
            return embed(textos), [1] * len(textos)
        pedacos, dono = [], []
        for i, (texto, n) in enumerate(zip(textos, n_tokens)):
            partes = partir(texto) if n > limite_tokens else [texto]
            pedacos.extend(partes)
            dono.extend([i] * len(partes))
        vetores = embed(pedacos)
        soma = np.zeros((len(textos), vetores.shape[1]), dtype="float32")
        contagem = np.zeros(len(textos), dtype="int32")
        for vetor, i in zip(vetores, dono):
            soma[i] += vetor
            contagem[i] += 1
        normas = np.linalg.norm(soma, axis=1, keepdims=True)
        return soma / np.where(normas == 0, 1.0, normas), contagem.tolist()

    def contar_tokens(texts: list[str]) -> list[int]:
        saida = []
        for start in range(0, len(texts), 128):
            encoded = encoder.tokenizer(texts[start:start + 128], truncation=False, padding=False)
            saida.extend(len(ids) for ids in encoded["input_ids"])
        return saida

    uids = list(perguntas)
    vetor_pergunta = dict(zip(uids, embed([perguntas[u] for u in uids])))

    destino = args.out or Path(f"reflection_similarity_{args.experiment_id}.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    escritos = 0
    with destino.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "source_uid", "arm", "similarity",
                         "tokens", "truncated", "chunks", "chars", "pooling"])

        def processar(model: str, arm_de, linhas) -> None:
            """Embeda as reflexões de um arquivo e grava uma linha por reflexão."""
            nonlocal escritos
            chaves, textos = [], []
            for row in linhas:
                if row["dataset"] not in datasets or row["source_uid"] not in vetor_pergunta:
                    continue
                for depth in DEPTHS:
                    texto = (row.get("reflections") or {}).get(depth)
                    if not texto or not texto.strip():
                        continue
                    chaves.append((row["source_uid"], arm_de(depth)))
                    textos.append(QUERY_PREFIX + texto if args.query_prefix else texto)
            if not textos:
                return
            tokens = contar_tokens(textos)
            for start in range(0, len(textos), args.batch_size * 8):
                fatia = slice(start, start + args.batch_size * 8)
                vetores, pedacos = vetores_de(textos[fatia], tokens[fatia])
                for (uid, arm), vetor, texto, n_tokens, n_pedacos in zip(
                        chaves[fatia], vetores, textos[fatia], tokens[fatia], pedacos):
                    writer.writerow([model, dataset_de[uid], uid, arm,
                                     f"{float(vetor_pergunta[uid] @ vetor):.6f}",
                                     n_tokens, n_tokens > limite_tokens, n_pedacos,
                                     len(texto), args.pooling])
                    escritos += 1
            fh.flush()

        def ler(path: Path):
            linhas = load_jsonl(path)
            if args.limit:
                linhas = (row for i, row in enumerate(linhas) if i < args.limit)
            return linhas

        for model, path in alunos:
            print(f"  {model}: autorreflexão", flush=True)
            processar(model, lambda depth: f"self_{depth}", ler(path))
        for teacher, model, path in professores:
            nome = teacher or legacy_teacher
            print(f"  {model}: reflexão de {nome}", flush=True)
            processar(model, lambda depth, t=nome: f"teacher_{depth}@{t}", ler(path))

    print(f"{escritos} linha(s) em {destino}", flush=True)
    if not escritos:
        sys.exit("nenhuma reflexão embedada; confira --datasets e o exchange")


if __name__ == "__main__":
    main()
