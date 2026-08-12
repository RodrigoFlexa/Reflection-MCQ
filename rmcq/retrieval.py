"""
Etapa 3: índice de similaridade semântica entre questões.

**Por que isto custa quase nada:** o embedding de uma questão não depende de
quem respondeu, de quem refletiu, nem da profundidade da reflexão. Ele depende
só do texto da questão. Logo o índice é calculado UMA VEZ por dataset e
reaproveitado nas 96 configurações da etapa de avaliação. O que varia entre
configurações é apenas qual texto de reflexão está pendurado em cada uid de
treino, e isso é uma consulta a dicionário.

O que fica cacheado em disco:

- `train.npy`, `test.npy`: embeddings normalizados
- `neighbors.npz`: os max(K_VALUES) vizinhos mais próximos de cada questão de
  teste, com as similaridades
- `meta.json`: embedder, dimensão, contagens e ordem dos uids

Sem faiss de propósito: as matrizes aqui são no máximo 4.801 × 383, e um produto
de matrizes denso resolve em milissegundos. Uma dependência a menos.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from rmcq.config import EMBED_BATCH_SIZE, EMBEDDER, K_VALUES, ensure_dirs
from rmcq.data import index_paths, load_split, resolve_datasets  # noqa: F401
from rmcq.store import Timer, get_logger

log = get_logger(__name__)

# Prefixo de consulta recomendado pelos modelos BGE. Aplicado apenas ao lado da
# consulta (as questões de teste), como manda o card do modelo; aplicar aos dois
# lados piora o retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _needs_query_prefix(embedder: str) -> bool:
    return "bge" in embedder.lower() and "en" in embedder.lower()


def _embed_text(item: dict[str, Any]) -> str:
    """
    O texto que representa uma questão no índice.

    Inclui o contexto porque, no LogiQA2, a premissa é o que define o problema —
    duas perguntas idênticas ("qual das seguintes deve ser verdadeira?") sobre
    premissas diferentes são questões completamente diferentes. Não inclui as
    alternativas: elas variam em formato entre datasets (números no GSM8K, frases
    no ARC) e adicionariam ruído lexical ao eixo de similaridade.
    """
    context = (item.get("context") or "").strip()
    question = item["question"].strip()
    return f"{context}\n\n{question}".strip() if context else question


# ---------------------------------------------------------------------------


def _index_ready(dataset: str, embedder: str) -> bool:
    """Índice utilizável: os quatro arquivos existem e nenhum está vazio."""
    paths = index_paths(dataset, embedder)
    return all(
        paths[name].exists() and paths[name].stat().st_size > 0
        for name in ("train_emb", "test_emb", "neighbors", "meta")
    )


def build(
    datasets: Sequence[str] | None = None,
    embedder: str = EMBEDDER,
    max_k: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Calcula embeddings e vizinhos para cada dataset. Idempotente."""
    import numpy as np

    ensure_dirs()
    datasets = resolve_datasets(datasets)
    max_k = max_k or max(K_VALUES)

    # Testa tamanho, não só existência. Um processo morto no meio do
    # savez_compressed deixa um .npz de 0 bytes: ele "existe", o build pularia
    # para sempre, e a falha só apareceria como EOFError na etapa de avaliação.
    pending = [d for d in datasets if force or not _index_ready(d, embedder)]
    if not pending:
        log.info("índice já existe para %s (use --force para recalcular)", ", ".join(datasets))
        return {"built": [], "skipped": list(datasets)}

    model = _load_embedder(embedder)
    stats: dict[str, Any] = {"built": [], "skipped": [d for d in datasets if d not in pending]}

    for dataset in pending:
        paths = index_paths(dataset, embedder)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        train = load_split(dataset, "train")
        test = load_split(dataset, "test")

        with Timer() as timer:
            train_emb = _encode(model, [_embed_text(i) for i in train], embedder, is_query=False)
            test_emb = _encode(model, [_embed_text(i) for i in test], embedder, is_query=True)

        # Embeddings normalizados, então o produto interno já é o cosseno.
        sims = test_emb @ train_emb.T
        k = min(max_k, len(train))

        # argpartition acha os k maiores sem ordenar tudo; depois ordena só os k.
        top_unsorted = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row_idx = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[row_idx, top_unsorted], axis=1)
        top_idx = top_unsorted[row_idx, order]
        top_sim = sims[row_idx, top_idx]

        np.save(paths["train_emb"], train_emb)
        np.save(paths["test_emb"], test_emb)
        np.savez_compressed(
            paths["neighbors"],
            test_uids=np.array([i["uid"] for i in test]),
            train_uids=np.array([i["uid"] for i in train]),
            # Índices em train_uids, do mais similar para o menos.
            top_idx=top_idx.astype(np.int32),
            top_sim=top_sim.astype(np.float32),
        )
        paths["meta"].write_text(
            json.dumps({
                "dataset": dataset,
                "embedder": embedder,
                "dim": int(train_emb.shape[1]),
                "n_train": len(train),
                "n_test": len(test),
                "max_k": int(k),
                "query_prefix": BGE_QUERY_PREFIX if _needs_query_prefix(embedder) else None,
                "embed_text": "context + question, sem alternativas",
                "elapsed_s": round(timer.elapsed, 1),
                "sim_mean": float(top_sim.mean()),
                "sim_top1_mean": float(top_sim[:, 0].mean()),
                "sim_top1_max": float(top_sim[:, 0].max()),
                "n_test_with_top1_above_0_95": int((top_sim[:, 0] > 0.95).sum()),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        log.info(
            "[%s] %d treino x %d teste, dim %d, %.1fs | cosseno top-1 médio %.3f, máx %.3f",
            dataset, len(train), len(test), train_emb.shape[1], timer.elapsed,
            top_sim[:, 0].mean(), top_sim[:, 0].max(),
        )
        near_dupes = int((top_sim[:, 0] > 0.95).sum())
        if near_dupes:
            # Depois da deduplicação do notebook 01 isto deve ser pequeno. Se for
            # grande, sobrou vazamento e a etapa de avaliação vai medir
            # memorização em vez de generalização.
            log.warning(
                "  %d questões de teste (%0.1f%%) têm vizinho de treino com cosseno > 0.95",
                near_dupes, 100 * near_dupes / len(test),
            )
        stats["built"].append(dataset)

    return stats


class HashingEmbedder:
    """
    Embedder lexical determinístico, em numpy puro. NÃO USE NO PAPER.

    Existe só para validar o pipeline sem GPU e sem baixar 1,3 GB de pesos:
    hashing de bigramas de caracteres com pesagem TF-IDF, normalizado. Captura
    sobreposição de superfície, não semântica — duas questões que dizem a mesma
    coisa com palavras diferentes ficam distantes. Serve para conferir que o
    índice, os vizinhos e a injeção no prompt funcionam.

    Selecionado por `--embedder hashing`.
    """

    NAME = "hashing"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        log.warning(
            "usando HashingEmbedder (dim=%d): lexical, apenas para teste do pipeline. "
            "Para resultados, use %s", dim, EMBEDDER,
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def encode(self, texts: list[str], **_: object):
        import numpy as np

        n_docs = len(texts)
        rows = []
        df = np.zeros(self.dim, dtype=np.float64)

        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float64)
            for tok in self._tokens(text):
                idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dim
                vec[idx] += 1.0
            rows.append(vec)
            df += vec > 0

        mat = np.vstack(rows) if rows else np.zeros((0, self.dim))
        idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        mat *= idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return (mat / np.maximum(norms, 1e-12)).astype("float32")


def _load_embedder(name: str):
    if name == HashingEmbedder.NAME:
        return HashingEmbedder()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers não instalado. Rode: pip install sentence-transformers\n"
            "Para só testar o pipeline sem GPU: --embedder hashing"
        ) from exc

    from rmcq.config import HF_HOME, hf_token

    log.info("carregando embedder %s", name)
    return SentenceTransformer(name, cache_folder=str(HF_HOME), token=hf_token())


def _encode(model, texts: list[str], embedder: str, is_query: bool):
    if is_query and _needs_query_prefix(embedder):
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    if isinstance(model, HashingEmbedder):
        return model.encode(texts)
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,   # cosseno = produto interno
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# ---------------------------------------------------------------------------
# Consulta
# ---------------------------------------------------------------------------


class Neighbors:
    """
    Vizinhos de treino de cada questão de teste, carregados do cache.

    `for_uid` devolve em ordem CRESCENTE de similaridade, porque é assim que o
    prompt injeta: a reflexão mais relevante fica por último, adjacente à
    questão nova, onde o modelo atende mais. Ver common.build_retrieval_prefix.
    """

    def __init__(self, dataset: str, embedder: str = EMBEDDER) -> None:
        import numpy as np

        paths = index_paths(dataset, embedder)
        if not _index_ready(dataset, embedder):
            empty = [
                n for n in ("train_emb", "test_emb", "neighbors", "meta")
                if paths[n].exists() and paths[n].stat().st_size == 0
            ]
            detail = (
                f"arquivo(s) vazio(s): {', '.join(empty)} — provavelmente uma execução "
                f"interrompida.\nRode: python -m rmcq index --force --datasets {dataset}"
                if empty else
                f"Rode: python -m rmcq index --datasets {dataset}"
            )
            raise FileNotFoundError(f"índice inutilizável em {paths['dir']}\n{detail}")

        blob = np.load(paths["neighbors"], allow_pickle=False)
        self.dataset = dataset
        self.embedder = embedder
        self.test_uids: list[str] = [str(u) for u in blob["test_uids"]]
        self.train_uids: list[str] = [str(u) for u in blob["train_uids"]]
        self.top_idx = blob["top_idx"]
        self.top_sim = blob["top_sim"]
        self.max_k = int(self.top_idx.shape[1])
        self._row = {uid: i for i, uid in enumerate(self.test_uids)}
        self.meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

    def for_uid(self, test_uid: str, k: int) -> list[tuple[str, float]]:
        """[(train_uid, similaridade)], em similaridade crescente."""
        if k > self.max_k:
            raise ValueError(
                f"índice de {self.dataset} tem apenas {self.max_k} vizinhos por questão; "
                f"pediram k={k}. Recalcule com --max-k {k}"
            )
        row = self._row.get(test_uid)
        if row is None:
            raise KeyError(f"{test_uid} não está no índice de {self.dataset}")

        pairs = [
            (self.train_uids[int(self.top_idx[row, j])], float(self.top_sim[row, j]))
            for j in range(k)
        ]
        return list(reversed(pairs))  # crescente: o mais similar fica no fim

    def top_k_for_all(
        self,
        k: int,
        allowed_uids: set[str] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Vizinhos de todas as questões de teste, opcionalmente restritos.

        `allowed_uids` limita os candidatos aos itens de treino que realmente
        têm reflexão gerada. Isso importa em dois casos:

        - **pilotos com --limit**: só uma fatia do treino foi refletida, e sem o
          filtro os k vizinhos cairiam quase sempre fora dela, produzindo um
          arquivo rotulado k=3 com zero reflexões injetadas;
        - **runs parciais**: se a etapa de reflexão foi interrompida, o filtro
          garante que o k pedido seja o k entregue, em vez de degradar silencioso.

        Quando o conjunto permitido cobre todo o treino, usa o cache de vizinhos
        direto. Caso contrário recalcula do embedding — 1.319 × 366 é um produto
        de matrizes que resolve em milissegundos.
        """
        import numpy as np

        if allowed_uids is None or allowed_uids >= set(self.train_uids):
            return {uid: self.for_uid(uid, min(k, self.max_k)) for uid in self.test_uids}

        keep = np.array([i for i, u in enumerate(self.train_uids) if u in allowed_uids])
        if keep.size == 0:
            return {uid: [] for uid in self.test_uids}

        paths = index_paths(self.dataset, self.embedder)
        train_emb = np.load(paths["train_emb"])[keep]
        test_emb = np.load(paths["test_emb"])
        sims = test_emb @ train_emb.T

        eff_k = min(k, keep.size)
        top_unsorted = np.argpartition(-sims, eff_k - 1, axis=1)[:, :eff_k]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, top_unsorted], axis=1)
        top = top_unsorted[rows, order]
        top_s = sims[rows, top]

        out: dict[str, list[tuple[str, float]]] = {}
        for r, uid in enumerate(self.test_uids):
            pairs = [
                (self.train_uids[int(keep[int(top[r, j])])], float(top_s[r, j]))
                for j in range(eff_k)
            ]
            out[uid] = list(reversed(pairs))  # crescente
        return out

    def __len__(self) -> int:
        return len(self.test_uids)

    def __repr__(self) -> str:
        return (
            f"Neighbors({self.dataset}, {len(self)} questões de teste, "
            f"max_k={self.max_k}, embedder={self.embedder})"
        )
