#!/usr/bin/env python
"""
Grade do notebook 06 como script: reuso de resposta em vez de regeracao.

Por que existe
--------------
O 06 nasceu como notebook e a grade completa tem 128 condicoes x ~300 itens x 2
datasets. Notebook nao e o lugar disso: nao da para rodar por `nohup`, nao da
para retomar de um `Ctrl-C` limpo e o estado vive numa sessao de kernel. Aqui a
grade e um script e o notebook fica so com leitura e graficos.

A ideia central: UMA geracao por PROMPT DISTINTO
------------------------------------------------
O `05` mediu ~5,6% de nao-determinismo entre execucoes: 10.908 itens caiam em
`fallback_to_baseline` -- prompt byte a byte identico ao do baseline, temperatura
0 -- e mesmo assim so 94,44% reproduziram a letra do baseline. Com n=300 isso da
um desvio-padrao de utility de +-0,0136, do tamanho dos efeitos que a grade quer
medir.

Flags de determinismo no vLLM (RMCQ_VLLM_DETERMINISTIC) atacam o sintoma. A cura
e nao gerar duas vezes: se dois pedidos tem o MESMO prompt, a resposta e a mesma
por construcao, entao gera-se uma vez e reusa-se. Isso resolve tres casos de uma
vez:

1. **Condicao sem nota recuperada.** O prompt e identico ao de `build_answer_prompt`
   -- e a resposta do controle `k=0`, copiada. Contribui exatamente 0 para a
   utility, que e o que conceitualmente deveria acontecer quando nao ha
   tratamento. No `05` esses itens contribuiam ruido nos dois sentidos.
2. **Condicoes que colidem.** `pool="errors"` e um subconjunto de `pool="all"`:
   quando o vizinho mais proximo ja e uma reflexao de erro, as duas condicoes
   montam o MESMO prompt. Idem para `lesson` com k=1 em profundidades cuja licao
   coincide. Hoje isso e gerado duas vezes e as duas respostas podem divergir --
   ruido puro entre condicoes que deveriam ser identicas.
3. **Retomada.** O cache e por hash de prompt, entao reexecutar depois de uma
   queda nao regenera nada do que ja existe, mesmo que a grade tenha mudado.

Efeito colateral util: da para saber o custo ANTES de ligar a GPU. A fase `plan`
monta todos os prompts, conta quantos sao distintos e quantos reusam o controle,
sem carregar modelo nenhum.

Fases
-----
  plan         monta e hasheia todos os prompts; grava o plano; relatorio de custo
  control      gera as respostas sem nota (k=0), uma por (aluno, dataset, item)
  grid         gera uma vez cada prompt distinto que ainda nao esta no cache
  materialize  junta plano + cache + controle nos JSONL por condicao
  summary      utility/McNemar por condicao -> diagnostics/summary_v3.csv
  all          as cinco, na ordem

Uso
---
  python scripts/run_grid_v3.py plan --grid-stage triagem
  python scripts/run_grid_v3.py all  --grid-stage triagem
  nohup python scripts/run_grid_v3.py all --grid-stage completa > run_v3.log 2>&1 &

  # smoke test sem GPU, em arvore separada:
  python scripts/run_grid_v3.py all --backend stub --limit 8 --out results/reflection_v3_smoke

Le de `results/reflection_v2/` (dados, baselines, reflexoes, matriz de
similaridade do notebook 05) e NAO escreve nada la.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rmcq  # noqa: F401  -- carrega o .env antes de torch/vllm
from rmcq.backends import GenParams, get_backend
from rmcq.common import (
    EVAL_TAIL_V2, NOTE_OUTCOME_V2, NOTE_SOURCE_V2, NOTE_V2, NOTES_HEADER_V2,
    build_answer_prompt, compact_reflection, extract_lesson, format_options,
    format_question, format_source_question, make_record, neutralize_option_letters,
)
from rmcq.config import EMBED_BATCH_SIZE, EMBEDDER, HF_HOME, RESULTS_DIR, SEED, STUDENT_GEN, hf_token
from rmcq.stages.analyze import accuracy_block, utility
from rmcq.store import JsonlStore, Timer, get_logger

# O nome PRECISA comecar com "rmcq.": `rmcq.store.get_logger` pendura o handler
# no logger "rmcq" e marca propagate=False nele. Um logger chamado
# "run_grid_v3" nao esta nessa hierarquia, entao propaga para o root -- que nao
# tem handler e esta em WARNING -- e todo log.info() some sem erro nenhum.
log = get_logger("rmcq.grid_v3")

STUDENTS = ["phi4-mini", "llama3-8b"]
DATASETS = ["arc", "logiqa2"]
DEPTHS = ["simple", "complex"]
CONTENTS = ["full", "lesson", "lesson_source", "playbook"]
POOLS = ["all", "errors"]
K_GRID = [1, 3]

RESULTS_V2 = RESULTS_DIR / "reflection_v2"

# Cabecalho alternativo do playbook: sao regras destiladas, nao notas sobre
# questoes especificas. Dizer o contrario seria mentir para o modelo sobre o que
# ele esta lendo -- o tipo de incoerencia que confunde um 4B.
PLAYBOOK_HEADER = """You are answering a multiple-choice question.

First, a short list of reasoning rules distilled from earlier attempts at other questions. They are general advice about how to reason. None of them is about the question below, and none of them contains its answer.

<notes>
{notes}
</notes>

"""


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def load_rows(path: Path) -> dict[str, dict]:
    store = JsonlStore(path)
    return {r["uid"]: r for r in store.read_all()} if store.exists() else {}


# ---------------------------------------------------------------------------
# Corpus: tudo que o 05 produziu, so leitura
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    datasets: list[str]
    students: list[str]
    train: dict = field(default_factory=dict)          # ds -> [item]
    val: dict = field(default_factory=dict)            # ds -> [item]
    train_by_uid: dict = field(default_factory=dict)   # ds -> {uid: item}
    baseline: dict = field(default_factory=dict)       # (student, ds) -> {uid: row}
    reflections: dict = field(default_factory=dict)    # (ds, student, depth) -> {uid: row}
    lessons: dict = field(default_factory=dict)        # (ds, student, depth) -> {uid: licao}
    sim: dict = field(default_factory=dict)            # ds -> matriz
    train_uids: dict = field(default_factory=dict)
    val_row: dict = field(default_factory=dict)        # ds -> {uid: linha da matriz}


def load_corpus(datasets: list[str], students: list[str]) -> Corpus:
    c = Corpus(datasets=datasets, students=students)
    for ds in datasets:
        c.train[ds] = read_items(RESULTS_V2 / "data" / ds / "train.jsonl")
        c.val[ds] = read_items(RESULTS_V2 / "data" / ds / "validation.jsonl")
        c.train_by_uid[ds] = {i["uid"]: i for i in c.train[ds]}

        blob = np.load(RESULTS_V2 / "index" / EMBEDDER.replace("/", "_") / ds / "train_validation_sim.npz",
                       allow_pickle=False)
        c.sim[ds] = blob["sim"]
        c.train_uids[ds] = [str(u) for u in blob["train_uids"]]
        val_uids = [str(u) for u in blob["val_uids"]]
        c.val_row[ds] = {u: i for i, u in enumerate(val_uids)}

        # Um desalinhamento entre a matriz e os uids produziria uma grade inteira
        # de vizinhos errados sem levantar erro nenhum -- por isso e assercao.
        assert c.sim[ds].shape == (len(c.val[ds]), len(c.train[ds])), f"{ds}: matriz fora de forma"
        assert val_uids == [i["uid"] for i in c.val[ds]], f"{ds}: ordem da validacao mudou"
        assert c.train_uids[ds] == [i["uid"] for i in c.train[ds]], f"{ds}: ordem do treino mudou"

        for s in students:
            c.baseline[(s, ds)] = load_rows(RESULTS_V2 / "baseline" / s / f"{ds}_validation.jsonl")
            assert len(c.baseline[(s, ds)]) == len(c.val[ds]), f"baseline incompleto: {s}/{ds}"
            for d in DEPTHS:
                rows = load_rows(RESULTS_V2 / "reflections" / f"{s}__{s}__{d}" / f"{ds}.jsonl")
                assert len(rows) == len(c.train[ds]), f"reflexoes incompletas: {ds}/{s}/{d}"
                c.reflections[(ds, s, d)] = rows
                c.lessons[(ds, s, d)] = {u: extract_lesson(r["reflection_text"] or "")
                                         for u, r in rows.items()}
    log.info("corpus carregado: %s | %s", datasets, students)
    return c


# ---------------------------------------------------------------------------
# Fator B (pool), fator A (conteudo da nota) e recuperacao
# ---------------------------------------------------------------------------

def in_pool(refl_row: dict, pool: str) -> bool:
    if pool == "all":
        return True
    return not (refl_row.get("extra") or {}).get("source_was_correct")


def retrieve(c: Corpus, ds, student, depth, pool, val_uid, k, content, threshold=0.0):
    """Ate k vizinhos de treino do pool com similaridade >= threshold, em ordem
    CRESCENTE de similaridade.

    Mais similar por ultimo, perto de onde a geracao comeca -- mesma politica de
    `rmcq.retrieval` e do 05. Variantes que so usam a licao pulam reflexoes sem
    linha `Lesson:` (2-7% delas, dependendo do modelo).

    Com `threshold > 0` a busca pode voltar VAZIA. Nesse caso o prompt montado e
    identico ao do controle e a resposta e copiada em vez de gerada (ver
    `phase_materialize`) -- entao um item sem vizinho contribui exatamente 0 para
    a utility, em vez de contribuir ruido de decodificacao nos dois sentidos como
    acontecia no 05.
    """
    refl = c.reflections[(ds, student, depth)]
    licoes = c.lessons[(ds, student, depth)]
    precisa_licao = content in ("lesson", "lesson_source")
    sims = c.sim[ds][c.val_row[ds][val_uid]]
    escolhidos = []
    for idx in np.argsort(-sims):
        if float(sims[idx]) < threshold:
            break
        uid = c.train_uids[ds][idx]
        row = refl.get(uid)
        if row is None or not in_pool(row, pool):
            continue
        if precisa_licao and not licoes.get(uid):
            continue
        escolhidos.append((uid, float(sims[idx])))
        if len(escolhidos) >= k:
            break
    escolhidos.reverse()
    return escolhidos


def build_note_body(content: str, refl_row: dict, source_item: dict) -> str:
    was_correct = (refl_row.get("extra") or {}).get("source_was_correct")
    texto = refl_row["reflection_text"] or ""

    if content == "full":   # identica ao 05
        linhas = [NOTE_SOURCE_V2.format(
            source_question=format_source_question(format_question(source_item)))]
        marca = NOTE_OUTCOME_V2.get(was_correct)
        if marca:
            linhas.append(marca)
        linhas.append(neutralize_option_letters(compact_reflection(texto)))
        return "\n".join(linhas)

    licao = extract_lesson(texto)
    if content == "lesson":
        return neutralize_option_letters(licao)
    if content == "lesson_source":
        linhas = [NOTE_SOURCE_V2.format(
            source_question=format_source_question(format_question(source_item)))]
        marca = NOTE_OUTCOME_V2.get(was_correct)
        if marca:
            linhas.append(marca)
        linhas.append(neutralize_option_letters(licao))
        return "\n".join(linhas)
    raise ValueError(f"conteudo sem montagem por item: {content}")


def build_eval_prompt_v3(item: dict, note_bodies: list[str], content: str) -> str:
    """Sem nota nenhuma, devolve build_answer_prompt() byte a byte -- e o que
    permite reusar a resposta do controle em vez de gerar de novo."""
    if not note_bodies:
        return build_answer_prompt(item)
    notas = "\n\n".join(NOTE_V2.format(i=i, body=b) for i, b in enumerate(note_bodies, start=1))
    cabecalho = PLAYBOOK_HEADER if content == "playbook" else NOTES_HEADER_V2
    return cabecalho.format(notes=notas) + EVAL_TAIL_V2.format(
        question=format_question(item), options=format_options(item["choices"]))


# ---------------------------------------------------------------------------
# Playbook: destilacao das licoes em k regras distintas
# ---------------------------------------------------------------------------

def _kmeans(x: np.ndarray, k: int, seed: int = SEED, iters: int = 50):
    """k-means++ enxuto em numpy. Deterministico dada a seed, sem dependencia nova."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(((x[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        total = d.sum()
        centers.append(x[rng.choice(n, p=d / total) if total > 0 else rng.integers(n)])
    cen = np.array(centers)
    for _ in range(iters):
        lab = np.argmax(x @ cen.T, axis=1)   # vetores normalizados: cosseno == produto interno
        novo = np.stack([x[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        novo /= np.linalg.norm(novo, axis=1, keepdims=True) + 1e-12
        if np.allclose(novo, cen):
            break
        cen = novo
    return cen, np.argmax(x @ cen.T, axis=1)


_embedder = None


def _embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDER, cache_folder=str(HF_HOME), token=hf_token())
    return _embedder.encode(list(texts), batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)


def build_playbook(c: Corpus, paths, ds, student, depth, pool, k, method="kmeans"):
    """As k regras do playbook, cacheadas em disco. A regra e uma licao REAL do
    corpus (a mais proxima do centroide), nunca uma parafrase inventada."""
    path = paths["playbook"] / f"{ds}__{student}__{depth}__{pool}__k{k}__{method}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["rules"]

    licoes = c.lessons[(ds, student, depth)]
    refl = c.reflections[(ds, student, depth)]
    textos = sorted({licoes[u] for u in licoes
                     if licoes[u] and in_pool(refl[u], pool)})   # ordenado: reprodutivel

    if method == "first" or len(textos) <= k:
        regras = textos[:k]
    else:
        emb = _embed(textos)
        cen, lab = _kmeans(emb, k)
        regras = []
        for j in range(k):
            idx = np.where(lab == j)[0]
            if len(idx):
                regras.append(textos[int(idx[np.argmax(emb[idx] @ cen[j])])])

    path.write_text(json.dumps({"dataset": ds, "student": student, "depth": depth, "pool": pool,
                                "k": k, "method": method, "n_licoes_no_pool": len(textos),
                                "rules": regras}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("playbook %s/%s/%s/%s k=%d: %d regras de %d licoes",
             ds, student, depth, pool, k, len(regras), len(textos))
    return regras


def playbook_method_used(paths, default="kmeans"):
    """Qual metodo de destilacao a rodada usou, lido do proprio cache.

    O nome do arquivo carrega o metodo, entao a analise nao precisa adivinhar --
    e, mais importante, nao recalcula um playbook com metodo diferente do que
    foi usado na grade (o que exigiria `sentence_transformers` e produziria
    regras que nunca entraram em prompt nenhum).
    """
    metodos = {p.stem.rsplit("__", 1)[-1] for p in paths["playbook"].glob("*.json")}
    if len(metodos) == 1:
        return metodos.pop()
    if len(metodos) > 1:
        log.warning("playbooks com metodos diferentes no cache (%s); usando %s",
                    sorted(metodos), default)
    return default


# ---------------------------------------------------------------------------
# Grade e caminhos
# ---------------------------------------------------------------------------

def make_grid(stage: str, datasets, students, thresholds=(0.0,)):
    if stage == "triagem":
        contents, pools, ks = CONTENTS, ["all"], [1]
    elif stage == "completa":
        contents, pools, ks = CONTENTS, POOLS, K_GRID
    else:
        raise ValueError(f"grid-stage desconhecido: {stage}")
    # `playbook` nao recupera nada -- injeta k regras fixas -- entao limiar nao o
    # afeta e cruzar os dois so criaria copias identicas da mesma condicao. Ele
    # entra uma vez so, com threshold 0.
    return [{"dataset": ds, "student": s, "content": ct, "pool": p, "depth": d, "k": k,
             "threshold": float(thr)}
            for ds in datasets for s in students
            for ct in contents for p in pools for d in DEPTHS for k in ks for thr in thresholds
            if not (ct == "playbook" and float(thr) != 0.0)]


def cond_tag(cfg) -> str:
    tag = f'{cfg["student"]}__{cfg["depth"]}__{cfg["content"]}__{cfg["pool"]}__k{cfg["k"]}'
    thr = float(cfg.get("threshold", 0.0))
    # Sem sufixo quando o limiar e 0: mantem os nomes de diretorio de antes.
    return tag if thr == 0.0 else tag + "__t" + f"{thr:.2f}".replace(".", "p")


def make_paths(out_root: Path) -> dict:
    p = {"root": out_root, "plan": out_root / "plan", "control": out_root / "control",
         "cache": out_root / "cache", "diag": out_root / "diagnostics",
         "playbook": out_root / "playbook"}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def cond_path(paths, cfg) -> Path:
    d = paths["diag"] / cond_tag(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{cfg["dataset"]}_validation.jsonl'


def control_path(paths, student, ds) -> Path:
    d = paths["control"] / student
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ds}_validation.jsonl"


# ---------------------------------------------------------------------------
# Fase 1 -- plan: monta e hasheia todos os prompts, sem tocar em GPU
# ---------------------------------------------------------------------------

def prompt_for(c: Corpus, paths, cfg, item, playbook_method="kmeans"):
    """(prompt, uids_recuperados, similaridades) de UM item numa condicao."""
    ds, s, d = cfg["dataset"], cfg["student"], cfg["depth"]
    ct, pool, k = cfg["content"], cfg["pool"], cfg["k"]
    thr = float(cfg.get("threshold", 0.0))
    if ct == "control" or k == 0:
        return build_answer_prompt(item), [], []
    if ct == "playbook":
        regras = build_playbook(c, paths, ds, s, d, pool, k, playbook_method)
        return build_eval_prompt_v3(item, regras, ct), [], []
    picked = retrieve(c, ds, s, d, pool, item["uid"], k, ct, thr)
    bodies = [build_note_body(ct, c.reflections[(ds, s, d)][u], c.train_by_uid[ds][u])
              for u, _ in picked]
    return build_eval_prompt_v3(item, bodies, ct), [u for u, _ in picked], [x for _, x in picked]


def phase_plan(c: Corpus, paths, grid, limit=None, playbook_method="kmeans"):
    por_par = defaultdict(list)
    for cfg in grid:
        por_par[(cfg["student"], cfg["dataset"])].append(cfg)

    relatorio = []
    for (student, ds), cfgs in sorted(por_par.items()):
        itens = c.val[ds][:limit] if limit else c.val[ds]
        base_sha = {i["uid"]: sha(build_answer_prompt(i)) for i in itens}
        linhas = []
        for cfg in cfgs:
            tag = cond_tag(cfg)
            for item in itens:
                p, uids, sims = prompt_for(c, paths, cfg, item, playbook_method)
                h = sha(p)
                linhas.append({
                    "cond_tag": tag, "uid": item["uid"], "sha": h,
                    "content": cfg["content"], "pool": cfg["pool"],
                    "depth": cfg["depth"], "k": cfg["k"],
                    "threshold": float(cfg.get("threshold", 0.0)),
                    "n_notes": len(uids) if cfg["content"] != "playbook" else cfg["k"],
                    "retrieved_uids": uids,
                    "retrieved_similarities": [round(x, 6) for x in sims],
                    "mean_similarity": (sum(sims) / len(sims)) if sims else None,
                    # O coracao da economia: prompt igual ao do controle => nao gera.
                    "reuse_control": h == base_sha[item["uid"]],
                })
        path = paths["plan"] / f"{student}__{ds}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in linhas:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        pares = len(linhas)
        reuso = sum(1 for r in linhas if r["reuse_control"])
        distintos = len({r["sha"] for r in linhas if not r["reuse_control"]})
        relatorio.append({
            "student": student, "dataset": ds, "condicoes": len(cfgs), "itens": len(itens),
            "pares (cond x item)": pares, "reusam o controle": reuso,
            "prompts distintos a gerar": distintos,
            "geracoes economizadas": pares - distintos,
            "% economizado": round(100 * (pares - distintos) / max(pares, 1), 1),
        })
        log.info("[plan] %s/%s: %d pares -> %d prompts distintos (%d reusam o controle, %.1f%% economizado)",
                 student, ds, pares, distintos, reuso, 100 * (pares - distintos) / max(pares, 1))

    (paths["plan"] / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=1), encoding="utf-8")

    # Em stdout, nao so no log: `plan` existe para ser lido por quem esta olhando
    # o terminal antes de decidir se liga a GPU.
    campos = ["student", "dataset", "condicoes", "itens", "pares (cond x item)",
              "reusam o controle", "prompts distintos a gerar", "% economizado"]
    larg = [max(len(c), max((len(str(r[c])) for r in relatorio), default=0)) for c in campos]
    print()
    print("  ".join(c.rjust(w) for c, w in zip(campos, larg)))
    for r in relatorio:
        print("  ".join(str(r[c]).rjust(w) for c, w in zip(campos, larg)))
    total_pares = sum(r["pares (cond x item)"] for r in relatorio)
    total_ger = sum(r["prompts distintos a gerar"] for r in relatorio)
    total_ctrl = sum(r["itens"] for r in relatorio)   # uma por (aluno, dataset, item)
    log.info("[plan] TOTAL: %d pares condicao x item -> %d geracoes de grade "
             "(%.1f%% economizado por deduplicacao e reuso do controle)",
             total_pares, total_ger, 100 * (total_pares - total_ger) / max(total_pares, 1))
    log.info("[plan] mais %d geracoes de controle -> %d geracoes no total, contra %d sem esta otimizacao",
             total_ctrl, total_ger + total_ctrl, total_pares + total_ctrl)
    print(f"\n  pares condicao x item .......... {total_pares:,}")
    print(f"  geracoes de grade ............. {total_ger:,}")
    print(f"  geracoes de controle .......... {total_ctrl:,}")
    print(f"  TOTAL a gerar ................. {total_ger + total_ctrl:,}")
    print(f"  seria, sem reuso .............. {total_pares + total_ctrl:,}")
    print(f"  economia ...................... "
          f"{100 * (1 - (total_ger + total_ctrl) / max(total_pares + total_ctrl, 1)):.1f}%")
    print(f"\n  plano gravado em {paths['plan']}")
    return relatorio


def load_plan(paths, student, ds):
    path = paths["plan"] / f"{student}__{ds}.jsonl"
    if not path.exists():
        raise SystemExit(f"plano ausente: {path}. Rode a fase `plan` primeiro.")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# ---------------------------------------------------------------------------
# Fase 2 -- control: a resposta sem nota, uma por (aluno, dataset, item)
# ---------------------------------------------------------------------------

def phase_control(c: Corpus, paths, students, datasets, limit=None, backend_kind=None):
    params = GenParams.from_config(STUDENT_GEN, seed=SEED)
    for student in students:
        falta = {}
        for ds in datasets:
            itens = c.val[ds][:limit] if limit else c.val[ds]
            feito = JsonlStore(control_path(paths, student, ds)).done_keys()
            pend = [i for i in itens if i["uid"] not in feito]
            if pend:
                falta[ds] = pend
        if not falta:
            log.info("[control] %s: ja completo", student)
            continue
        with Timer() as t, get_backend(student, backend_kind) as backend:
            for ds, pend in falta.items():
                prompts = [build_answer_prompt(i) for i in pend]
                gens = backend.generate(prompts, params, desc=f"{student} control {ds}")
                recs = [
                    make_record(item, stage="control_v3", condition="control_no_notes",
                                student_model=student, teacher_model=student,
                                prompt=p, output=g.text, k=0,
                                prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens,
                                latency_s=g.latency_s, seed=SEED, temperature=params.temperature,
                                extra={"content": "control", "pool": "-", "n_notes_injected": 0,
                                       "prompt_sha1": sha(p)})
                    for item, p, g in zip(pend, prompts, gens)
                ]
                JsonlStore(control_path(paths, student, ds)).append(recs)
                ok = sum(1 for r in recs if r.is_correct)
                log.info("[control] %s/%s: %d respostas, acerto %.1f%%",
                         student, ds, len(recs), 100 * ok / max(len(recs), 1))
        log.info("[control] %s: %.1f min", student, t.elapsed / 60)


# ---------------------------------------------------------------------------
# Fase 3 -- grid: uma geracao por prompt distinto que ainda nao esta no cache
# ---------------------------------------------------------------------------

def cache_path(paths, student, ds) -> Path:
    d = paths["cache"] / student
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ds}.jsonl"


def load_cache(paths, student, ds) -> dict[str, dict]:
    path = cache_path(paths, student, ds)
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # linha truncada por kill no meio do write
        out[r["sha"]] = r
    return out


def phase_grid(c: Corpus, paths, grid, limit=None, backend_kind=None, chunk=256,
               playbook_method="kmeans"):
    por_aluno = defaultdict(list)
    for cfg in grid:
        por_aluno[cfg["student"]].append(cfg)

    params = GenParams.from_config(STUDENT_GEN, seed=SEED)
    for student, cfgs in sorted(por_aluno.items()):
        datasets = sorted({cfg["dataset"] for cfg in cfgs})

        # Um prompt distinto -> um pedido. Se duas condicoes montam o mesmo
        # prompt (pool="errors" e um subconjunto de "all"), a segunda nao gera.
        trabalho = {}
        for ds in datasets:
            cache = load_cache(paths, student, ds)
            itens = {i["uid"]: i for i in (c.val[ds][:limit] if limit else c.val[ds])}
            pend = {}
            for linha in load_plan(paths, student, ds):
                if linha["reuse_control"] or linha["sha"] in cache or linha["sha"] in pend:
                    continue
                if linha["uid"] not in itens:
                    continue
                cfg = {"dataset": ds, "student": student, "content": linha["content"],
                       "pool": linha["pool"], "depth": linha["depth"], "k": linha["k"],
                       "threshold": linha.get("threshold", 0.0)}
                p, _, _ = prompt_for(c, paths, cfg, itens[linha["uid"]], playbook_method)
                if sha(p) != linha["sha"]:
                    raise SystemExit(
                        f"plano desatualizado ({student}/{ds}, {linha['cond_tag']}, {linha['uid']}): "
                        "o prompt reconstruido nao bate com o hash gravado. Rode `plan` de novo.")
                pend[linha["sha"]] = p
            if pend:
                trabalho[ds] = pend
                log.info("[grid] %s/%s: %d prompts distintos a gerar (%d ja no cache)",
                         student, ds, len(pend), len(cache))
            else:
                log.info("[grid] %s/%s: nada a gerar (%d no cache)", student, ds, len(cache))

        if not trabalho:
            continue

        with Timer() as t, get_backend(student, backend_kind) as backend:
            for ds, pend in trabalho.items():
                store = JsonlStore(cache_path(paths, student, ds), key_fn=lambda r: r["sha"])
                shas = sorted(pend)
                for ini in range(0, len(shas), chunk):
                    lote = shas[ini:ini + chunk]
                    prompts = [pend[h] for h in lote]
                    gens = backend.generate(
                        prompts, params,
                        desc=f"{student} grid {ds} {ini // chunk + 1}/{-(-len(shas) // chunk)}")
                    store.append([
                        {"sha": h, "prompt": p, "text": g.text,
                         "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                         "latency_s": g.latency_s}
                        for h, p, g in zip(lote, prompts, gens)
                    ])
        log.info("[grid] %s: %.1f min", student, t.elapsed / 60)


# ---------------------------------------------------------------------------
# Fase 4 -- materialize: plano + cache + controle -> JSONL por condicao
# ---------------------------------------------------------------------------

def phase_materialize(c: Corpus, paths, grid, limit=None):
    por_par = defaultdict(list)
    for cfg in grid:
        por_par[(cfg["student"], cfg["dataset"])].append(cfg)

    for (student, ds), cfgs in sorted(por_par.items()):
        cache = load_cache(paths, student, ds)
        ctrl = load_rows(control_path(paths, student, ds))
        itens = {i["uid"]: i for i in (c.val[ds][:limit] if limit else c.val[ds])}
        por_tag = {cond_tag(cfg): cfg for cfg in cfgs}

        agrupado = defaultdict(list)
        for linha in load_plan(paths, student, ds):
            if linha["cond_tag"] in por_tag and linha["uid"] in itens:
                agrupado[linha["cond_tag"]].append(linha)

        for tag, linhas in agrupado.items():
            cfg = por_tag[tag]
            path = cond_path(paths, cfg)
            store = JsonlStore(path)
            feito = store.done_keys()
            recs, faltando = [], 0
            for linha in linhas:
                if linha["uid"] in feito:
                    continue
                extra = {
                    "content": cfg["content"], "pool": cfg["pool"],
                    "threshold": float(cfg.get("threshold", 0.0)),
                    "n_notes_injected": linha["n_notes"],
                    "mean_similarity": linha["mean_similarity"],
                    "prompt_sha1": linha["sha"],
                    "reused_control": bool(linha["reuse_control"]),
                    "grid_stage": cfg.get("grid_stage", ""),
                }
                if linha["reuse_control"]:
                    # Prompt identico ao do controle: a resposta E a do controle.
                    # Copia-se em vez de gerar -- e o que zera o ruido nesses itens.
                    base = ctrl.get(linha["uid"])
                    if base is None:
                        faltando += 1
                        continue
                    rec = dict(base)
                    rec.update(stage="eval_v3", condition="self_reflection",
                               reflection_depth=cfg["depth"], reflection_perspective="student",
                               k=cfg["k"], retrieved_uids=linha["retrieved_uids"],
                               retrieved_similarities=linha["retrieved_similarities"])
                    rec["extra"] = {**(base.get("extra") or {}), **extra}
                    recs.append(rec)
                    continue

                gen = cache.get(linha["sha"])
                if gen is None:
                    faltando += 1
                    continue
                recs.append(make_record(
                    itens[linha["uid"]], stage="eval_v3", condition="self_reflection",
                    student_model=student, teacher_model=student,
                    prompt=gen["prompt"], output=gen["text"],
                    reflection_depth=cfg["depth"], reflection_perspective="student",
                    retrieved_uids=linha["retrieved_uids"],
                    retrieved_similarities=linha["retrieved_similarities"],
                    k=cfg["k"], prompt_tokens=gen["prompt_tokens"],
                    completion_tokens=gen["completion_tokens"], latency_s=gen["latency_s"],
                    seed=SEED, temperature=GenParams.from_config(STUDENT_GEN, seed=SEED).temperature,
                    extra=extra))
            if recs:
                store.append(recs)
            if faltando:
                log.warning("[materialize] %s/%s: %d itens sem geracao no cache (rode `grid`)",
                            tag, ds, faltando)
        log.info("[materialize] %s/%s: %d condicoes gravadas", student, ds, len(agrupado))


# ---------------------------------------------------------------------------
# Fase 5 -- summary: utility contra o controle k=0, McNemar, custo
# ---------------------------------------------------------------------------

def phase_summary(c: Corpus, paths, grid, limit=None):
    import pandas as pd

    registros = []
    for cfg in grid:
        linhas = load_rows(cond_path(paths, cfg))
        if not linhas:
            continue
        ctrl = load_rows(control_path(paths, cfg["student"], cfg["dataset"]))
        if not ctrl:
            log.warning("sem controle para %s/%s", cfg["student"], cfg["dataset"])
            continue
        u_ctrl = utility(ctrl, linhas)
        u_b05 = utility(c.baseline[(cfg["student"], cfg["dataset"])], linhas)
        acc = accuracy_block(list(linhas.values()))
        sims = [(r.get("extra") or {}).get("mean_similarity") for r in linhas.values()]
        sims = [x for x in sims if x is not None]
        n_notes = [(r.get("extra") or {}).get("n_notes_injected", 0) for r in linhas.values()]
        reusados = [bool((r.get("extra") or {}).get("reused_control")) for r in linhas.values()]
        ctrl_len = np.median([len(r["prompt"].split()) for r in ctrl.values()])
        prompt_len = np.median([len(r["prompt"].split()) for r in linhas.values()])

        registros.append({
            **cfg, **u_ctrl,
            "utility_vs_baseline05": u_b05["utility"],
            "accuracy": acc["accuracy"], "accuracy_answered": acc["accuracy_answered"],
            "notas_media": float(np.mean(n_notes)) if n_notes else 0.0,
            "pct_reusou_controle": float(np.mean(reusados)) if reusados else 0.0,
            "similaridade_media": float(np.mean(sims)) if sims else None,
            "palavras_prompt": float(prompt_len),
            "palavras_de_nota": float(prompt_len - ctrl_len),
        })

    df = pd.DataFrame(registros)
    if df.empty:
        log.warning("nada a resumir")
        return df

    try:
        from scipy.stats import binomtest
        df["mcnemar_p"] = df.apply(
            lambda r: 1.0 if r["wrong_to_right"] + r["right_to_wrong"] == 0
            else binomtest(int(min(r["wrong_to_right"], r["right_to_wrong"])),
                           int(r["wrong_to_right"] + r["right_to_wrong"]), 0.5).pvalue, axis=1)
    except ImportError:
        log.warning("scipy indisponivel: mcnemar_p nao calculado")
        df["mcnemar_p"] = None

    out = paths["diag"] / "summary_v3.csv"
    df.to_csv(out, index=False)
    log.info("[summary] %d condicoes -> %s", len(df), out)

    # Ruido residual: controle deste run x baseline do 05. Prompts identicos,
    # temperatura 0, execucoes diferentes -> o que faltar para 100% e o
    # nao-determinismo que sobrou ENTRE execucoes. Dentro deste run ele nao
    # existe mais, porque prompt igual nunca e gerado duas vezes.
    ruido = []
    for student in sorted({cfg["student"] for cfg in grid}):
        for ds in sorted({cfg["dataset"] for cfg in grid}):
            ctrl = load_rows(control_path(paths, student, ds))
            base = c.baseline[(student, ds)]
            comuns = [u for u in ctrl if u in base]
            if not comuns:
                continue
            disc = 1 - float(np.mean([bool(ctrl[u]["is_correct"]) == bool(base[u]["is_correct"])
                                      for u in comuns]))
            ruido.append({
                "student": student, "dataset": ds, "n": len(comuns),
                "mesma_letra_pct": round(100 * float(np.mean(
                    [ctrl[u]["predicted"] == base[u]["predicted"] for u in comuns])), 2),
                "mesmo_acerto_pct": round(100 * (1 - disc), 2),
                "acc_controle_v3": round(float(np.mean([bool(ctrl[u]["is_correct"]) for u in comuns])), 4),
                "acc_baseline_v2": round(float(np.mean([bool(base[u]["is_correct"]) for u in comuns])), 4),
                "piso_sd_utility": round(float(np.sqrt(disc) / np.sqrt(len(comuns))), 4),
            })
    pd.DataFrame(ruido).to_csv(paths["diag"] / "ruido_controle.csv", index=False)
    log.info("[summary] ruido controle x baseline05 -> %s", paths["diag"] / "ruido_controle.csv")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fase", choices=["plan", "control", "grid", "materialize", "summary", "all"])
    ap.add_argument("--grid-stage", default="triagem", choices=["triagem", "completa"])
    ap.add_argument("--students", nargs="+", default=STUDENTS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--limit", type=int, default=None, help="so os N primeiros itens de validacao")
    ap.add_argument("--backend", default=None, help="vllm | hf | stub (padrao: RMCQ_BACKEND do .env)")
    ap.add_argument("--chunk", type=int, default=256, help="prompts por chamada de generate()")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0],
                    help="limiares de similaridade a cruzar com a grade (padrao: so 0.0). "
                         "Com limiar > 0 alguns itens ficam sem vizinho: esses reusam a "
                         "resposta do controle em vez de serem gerados de novo, entao "
                         "contribuem 0 para a utility em vez de ruido.")
    ap.add_argument("--out", default=None, help="raiz de saida (padrao: results/reflection_v3)")
    ap.add_argument("--playbook-method", default=None, choices=["kmeans", "first"],
                    help="padrao: kmeans, ou first quando --backend stub")
    a = ap.parse_args(argv)

    from rmcq.store import log_to_file
    log_to_file(f"grid_v3_{a.fase}")

    paths = make_paths(Path(a.out) if a.out else RESULTS_DIR / "reflection_v3")
    method = a.playbook_method or ("first" if a.backend == "stub" else "kmeans")

    c = load_corpus(a.datasets, a.students)
    grid = make_grid(a.grid_stage, a.datasets, a.students, a.thresholds)
    for cfg in grid:
        cfg["grid_stage"] = a.grid_stage
    log.info("grid-stage=%s thresholds=%s: %d condicoes | saida: %s",
             a.grid_stage, a.thresholds, len(grid), paths["root"])

    fases = ["plan", "control", "grid", "materialize", "summary"] if a.fase == "all" else [a.fase]
    print(f"grid-stage={a.grid_stage}  thresholds={a.thresholds}  playbook={method}")
    print(f"{len(grid)} condicoes  |  saida: {paths['root']}")
    for f in fases:
        log.info("========== fase %s ==========", f)
        print(f"\n========== fase {f} ==========", flush=True)
        if f == "plan":
            phase_plan(c, paths, grid, a.limit, method)
        elif f == "control":
            phase_control(c, paths, a.students, a.datasets, a.limit, a.backend)
        elif f == "grid":
            phase_grid(c, paths, grid, a.limit, a.backend, a.chunk, method)
        elif f == "materialize":
            phase_materialize(c, paths, grid, a.limit)
        elif f == "summary":
            df = phase_summary(c, paths, grid, a.limit)
            if df is not None and not df.empty:
                cols = ["dataset", "student", "content", "pool", "depth", "k", "threshold",
                        "utility", "mcnemar_p", "pct_reusou_controle", "palavras_de_nota"]
                print(df.sort_values("utility", ascending=False)[cols].head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
