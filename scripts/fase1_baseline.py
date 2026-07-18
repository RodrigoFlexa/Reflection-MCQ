#!/usr/bin/env python3
"""
Fase 1 — Respostas baseline (versão script do notebook 02; backend HuggingFace).

Modelos respondem TODAS as questões dos splits congelados (data/splits/,
gerados pelo notebook 01) com CoT explícito e decodificação greedy
(determinística). As respostas do TREINO alimentam a geração de reflexões
(Fase 2); as do TESTE são o baseline "sem reflexão" (decisão D3a).

SAÍDA: um JSONL por (modelo, dataset, split) em results/fase1/
  <modelo>__<dataset>__<split>.jsonl — 1 linha por questão:
    id, dataset, split, model, hf_model, mock, prompt_sha, seed,
    gold, pred, valid, extraction_method, strict_format, correct, ts,
    raw_response (resposta bruta completa, insumo da Fase 2),
    wall_time_s, prompt_tokens, gen_tokens
  O notebook 02 (seção de métricas/plots) lê esses arquivos diretamente.

Matriz de modelos (MODEL_CONFIGS) — três eixos com lógica experimental:

  EIXO A — Comparabilidade com o paper original
    - deepseek-r1-8b: DeepSeek-R1-Distill-Llama-8B, o "DeepSeek-8B" do paper.
    - phi4-mini (3.8B): sucessor do Phi-2 (2.7B) do paper — Phi-2 não segue
      instruções de formato de forma confiável, o que contaminaria a métrica
      de inválidas; phi4-mini preserva o papel de "pequeno da família Phi".

  EIXO B — Par controlado: mesmo backbone ± distilação de raciocínio
    - llama3.1-8b vs. deepseek-r1-8b: o R1-Distill-8B é destilado SOBRE o
      Llama-3.1-8B. Mesma arquitetura e escala; muda só o treinamento de
      raciocínio. Isola "reflexão externa ajuda menos quem já raciocina?".

  EIXO C — Escada de tamanho dentro de uma família
    - qwen3-1.7b / 4b / 8b: mesma família, tokenizador e dados; só muda a
      escala. Testa se o ganho da reflexão decresce com a capacidade do
      respondedor — previsão natural da hipótese central.

  Modelos de raciocínio (deepseek-r1, qwen3) pensam em blocos <think>:
  recebem max_new_tokens maior; o extrator ignora o conteúdo de <think>.

Preparação no servidor:
  pip install torch transformers accelerate huggingface_hub pandas pyarrow
  export HF_TOKEN=hf_...      # necessário para Llama-3.1 (repo gated:
                              # aceite a licença em huggingface.co/meta-llama)
  python fase1_baseline.py --download          # 1º passo: baixa os 6 modelos
  python fase1_baseline.py --dry-run           # plano + custo estimado
  python fase1_baseline.py                     # roda tudo (resumável)
  python fase1_baseline.py --models phi4-mini qwen3-1.7b --datasets gsm8k
  python fase1_baseline.py --batch-size 16     # mais throughput se a VRAM der
  python fase1_baseline.py --status            # progresso + acurácia parcial
  python fase1_baseline.py --mock --limit 5    # valida o pipeline sem GPU
"""
import argparse
import hashlib
import json
import re
import signal
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- configuração
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
RES = ROOT / "results" / "fase1"

# max_new_tokens por modelo: modelos de raciocínio precisam de espaço para o
# bloco <think>; truncar o pensamento derruba a taxa de respostas válidas.
MODEL_CONFIGS = {
    "phi4-mini":      {"hf_id": "microsoft/Phi-4-mini-instruct",
                       "max_new_tokens": 1024,
                       "eixo": "A: sucessor do Phi-2 do paper"},
    "llama3.1-8b":    {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",
                       "max_new_tokens": 1024, "gated": True,
                       "eixo": "B: backbone do par controlado"},
    "deepseek-r1-8b": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                       "max_new_tokens": 4096,
                       "eixo": "A+B: âncora do paper; par com llama3.1-8b"},
    "qwen3-1.7b":     {"hf_id": "Qwen/Qwen3-1.7B",
                       "max_new_tokens": 4096,
                       "eixo": "C: escada qwen3 (menor)"},
    "qwen3-4b":       {"hf_id": "Qwen/Qwen3-4B",
                       "max_new_tokens": 4096,
                       "eixo": "C: escada qwen3 (médio)"},
    "qwen3-8b":       {"hf_id": "Qwen/Qwen3-8B",
                       "max_new_tokens": 4096,
                       "eixo": "C: escada qwen3 (maior)"},
}
DEFAULT_MODELS = list(MODEL_CONFIGS)

# ------------------------------------------------------------------ prompt (D9)
PROMPT_TEMPLATE = """You are solving a multiple-choice question. Think through the problem step by step, showing all your reasoning. Then give your final answer.

Question:
{question}

Options:
{options}

Rules:
- Reason step by step BEFORE answering.
- You must choose exactly one option.
- The LAST line of your response must be exactly: FINAL ANSWER: <letter>"""

PROMPT_SHA = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:12]


def format_prompt(row):
    ch = row["choices"]
    opts = "\n".join(f"{l}) {t}" for l, t in zip(ch["label"], ch["text"]))
    return PROMPT_TEMPLATE.format(question=row["question"], options=opts)


# ------------------------------------------------------- extração da resposta (D10)
def _norm_val(s):
    s = re.sub(r"[\s,$]", "", str(s).strip().lower()).rstrip(".")
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def strip_think(text):
    """Remove blocos <think>...</think> (deepseek-r1, qwen3) antes da extração,
    para não casar letras mencionadas durante o pensamento."""
    return re.sub(r"<think>.*?(?:</think>|$)", "", str(text), flags=re.DOTALL)


def extract_answer(text, labels, texts=None):
    """Retorna (letra, método) ou (None, None). NUNCA retorna o texto completo.
    Cascata em ordem de confiança: final_answer > answer_is > option
    > value_match > tail_letter. Igual ao notebook 02 + strip de <think>."""
    labels = [str(l).upper() for l in labels]
    lab = "".join(labels)
    t = strip_think(text)

    for method, pat in [
        ("final_answer", rf"FINAL\s*ANSWER\s*[:\-]?\s*\(?([{lab}])\b\)?"),
        ("answer_is",    rf"(?:the\s+)?answer\s+is\s*[:\-]?\s*\(?([{lab}])\b\)?"),
        ("option",       rf"(?:correct\s+option|option)\s*[:\-]?\s*\(?([{lab}])\b\)?"),
    ]:
        m = list(re.finditer(pat, t, flags=re.IGNORECASE))
        if m:
            return m[-1].group(1).upper(), method

    if texts is not None:
        cand = None
        for pat in [r"FINAL\s*ANSWER\s*[:\-]?\s*(.+)",
                    r"(?:the\s+)?answer\s+is\s*[:\-]?\s*([^\n.]+)"]:
            m = list(re.finditer(pat, t, flags=re.IGNORECASE))
            if m:
                cand = m[-1].group(1)
                break
        if cand:
            cn = _norm_val(cand)
            hits = [l for l, tx in zip(labels, texts) if cn != "" and _norm_val(tx) == cn]
            if len(hits) == 1:
                return hits[0], "value_match"

    tail = t[-200:]
    m = list(re.finditer(rf"\(([{lab}])\)|(?:^|\s)([{lab}])[).:]?\s*$", tail,
                         flags=re.MULTILINE))
    if m:
        g = m[-1]
        return (g.group(1) or g.group(2)).upper(), "tail_letter"

    return None, None


def selftest():
    L = list("ABCDE")
    T = ["12", "15", "18", "21", "25"]
    assert extract_answer("bla\nFINAL ANSWER: C", L) == ("C", "final_answer")
    assert extract_answer("final answer: (b)", L)[0] == "B"
    assert extract_answer("So the answer is A.", L)[0] == "A"
    assert extract_answer("... I conclude it must be D", L)[0] == "D"
    assert extract_answer("A) is wrong... FINAL ANSWER: E", L)[0] == "E"
    assert extract_answer("FINAL ANSWER: 18", L, T) == ("C", "value_match")
    assert extract_answer("the answer is $1,500", L, ["1500", "2", "3", "4", "5"])[0] == "A"
    assert extract_answer("answer is 18 or 21, not sure", L, T)[0] is None
    assert extract_answer("A common mistake would be to guess here.", L)[0] is None
    assert extract_answer("Between the options, hard to say.", L)[0] is None
    assert extract_answer("no letters here 123", L)[0] is None
    assert extract_answer("<think>maybe B? no...</think>\nFINAL ANSWER: D", L) == ("D", "final_answer")
    assert extract_answer("<think>it is B it is B</think>", L)[0] is None
    print("extrator OK (13 asserts)")


# ------------------------------------------------------------ backend HF
def download_models(models):
    """Passo de instalação: baixa os pesos para o cache local do HF.
    Rode uma vez (ou num nó com internet) antes dos experimentos."""
    from huggingface_hub import snapshot_download
    for m in models:
        cfg = MODEL_CONFIGS[m]
        print(f"→ {m}  ({cfg['hf_id']})"
              + ("  [GATED: exige HF_TOKEN + licença aceita]" if cfg.get("gated") else ""))
        try:
            path = snapshot_download(cfg["hf_id"])
            print(f"  ok: {path}")
        except Exception as e:
            print(f"  FALHOU: {type(e).__name__}: {e}")
            if cfg.get("gated"):
                print("  aceite a licença em https://huggingface.co/"
                      f"{cfg['hf_id']} e exporte HF_TOKEN.")
    print("\nDownload concluído. Cache padrão: ~/.cache/huggingface/ "
          "(defina HF_HOME para mudar).")


class HFModel:
    """Carrega um modelo uma vez e gera em lotes (greedy, determinístico)."""

    def __init__(self, alias, device=None, dtype="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        cfg = MODEL_CONFIGS[alias]
        self.alias, self.hf_id = alias, cfg["hf_id"]
        self.max_new_tokens = cfg["max_new_tokens"]
        set_seed(SEED)
        print(f"Carregando {alias} ({self.hf_id})…", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=(torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)),
            device_map=device or "auto",
        )
        self.model.eval()
        self._torch = torch

    def generate_batch(self, prompts):
        torch = self._torch
        texts = [self.tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                      # greedy = temperatura 0
                pad_token_id=self.tok.pad_token_id,
            )
        wall = time.perf_counter() - t0
        gen = out[:, enc["input_ids"].shape[1]:]
        results = []
        for i in range(len(prompts)):
            toks = gen[i]
            n_gen = int((toks != self.tok.pad_token_id).sum())
            results.append({
                "raw_response": self.tok.decode(toks, skip_special_tokens=True),
                "wall_time_s": round(wall / len(prompts), 3),   # média do lote
                "prompt_tokens": int((enc["input_ids"][i] != self.tok.pad_token_id).sum()),
                "gen_tokens": n_gen,
            })
        return results

    def unload(self):
        torch = self._torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mock_batch(model, prompts, qids, labels_list):
    results = []
    for prompt, qid, labels in zip(prompts, qids, labels_list):
        h = int(hashlib.sha256(f"{model}|{qid}".encode()).hexdigest(), 16)
        letter = labels[h % len(labels)]
        body = f"Step 1: mock reasoning. Step 2: more mock steps ({h % 97} tokens)."
        text = body + (f"\nFINAL ANSWER: {letter}" if h % 10 else
                       "\n(no final line - simulated failure)")
        results.append({"raw_response": text,
                        "wall_time_s": 0.01 + (h % 100) / 1000,
                        "prompt_tokens": len(prompt.split()),
                        "gen_tokens": len(text.split())})
    return results


# --------------------------------------------------------------------- runner
def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def out_path(res_out, model, ds, sp):
    return res_out / f"{slug(model)}__{ds}__{sp}.jsonl"


def load_done(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


INTERRUPTED = False


def _sigint(_sig, _frm):
    global INTERRUPTED
    INTERRUPTED = True
    print("\n[Ctrl-C] terminando o lote atual e salvando…", flush=True)


def load_splits(datasets, splits_wanted):
    man = json.loads((SPLITS / "manifest.json").read_text())
    available = sorted({k.rsplit("_", 1)[0] for k in man if k != "_meta"})
    datasets = datasets or available
    unknown = set(datasets) - set(available)
    if unknown:
        sys.exit(f"datasets desconhecidos: {unknown}. Disponíveis: {available}")
    out = {}
    for ds in datasets:
        for sp in splits_wanted:
            df = pd.read_parquet(SPLITS / f"{ds}_{sp}.parquet")
            assert len(df) == man[f"{ds}_{sp}"]["n"], f"{ds}_{sp}: n difere do manifest"
            out[(ds, sp)] = df
    return out


def run(model_alias, hf_model, ds, sp, df, res_out, mock, batch_size, limit=None):
    if limit:
        df = df.head(limit)
    path = out_path(res_out, model_alias, ds, sp)
    done = load_done(path)
    todo = df[~df["id"].astype(str).isin(done)]
    print(f"{model_alias} × {ds}/{sp}: {len(done)} feitas, {len(todo)} restantes "
          f"-> {path.name}", flush=True)
    if todo.empty:
        return True
    t_start, n_done = time.perf_counter(), 0
    rows = list(todo.iterrows())
    with open(path, "a", encoding="utf-8") as f:
        for b0 in range(0, len(rows), batch_size):
            if INTERRUPTED:
                return False
            batch = [r for _, r in rows[b0:b0 + batch_size]]
            prompts = [format_prompt(r) for r in batch]
            labels_list = [[str(l).upper() for l in r["choices"]["label"]] for r in batch]
            try:
                outs = (mock_batch(model_alias, prompts,
                                   [r["id"] for r in batch], labels_list) if mock
                        else hf_model.generate_batch(prompts))
            except Exception as e:
                outs = [{"raw_response": f"__ERROR__ {e}", "wall_time_s": None,
                         "prompt_tokens": None, "gen_tokens": None}] * len(batch)
            for row, labels, out in zip(batch, labels_list, outs):
                texts = [str(x) for x in row["choices"]["text"]]
                pred, method = extract_answer(out["raw_response"], labels, texts)
                rec = {"id": str(row["id"]), "dataset": ds, "split": sp,
                       "model": model_alias,
                       "hf_model": MODEL_CONFIGS.get(model_alias, {}).get("hf_id"),
                       "mock": mock, "prompt_sha": PROMPT_SHA, "seed": SEED,
                       "gold": str(row["answerKey"]).upper(), "pred": pred,
                       "valid": pred is not None, "extraction_method": method,
                       "strict_format": method == "final_answer",
                       "correct": pred == str(row["answerKey"]).upper(),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **out}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            if (b0 // batch_size) % 5 == 0 or n_done == len(rows):
                el = time.perf_counter() - t_start
                eta = el / n_done * (len(rows) - n_done)
                print(f"  {n_done}/{len(rows)}  ({el/60:.1f} min | "
                      f"{el/n_done:.1f} s/q | ETA {eta/60:.0f} min)", flush=True)
    return True


# --------------------------------------------------------------------- status
def status(res_out):
    rows = []
    for p in sorted(res_out.glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    if not rows:
        print(f"Sem resultados em {res_out}/")
        return
    df = pd.DataFrame(rows)
    if df["mock"].any():
        print("⚠️  Contém resultados MOCK — números apenas de validação de pipeline!")
    g = (df.groupby(["model", "dataset", "split"])
         .agg(n=("id", "count"), acc=("correct", "mean"),
              invalidas=("valid", lambda s: 1 - s.mean()),
              formato=("strict_format", "mean"),
              s_por_q=("wall_time_s", "mean"))
         .round(3))
    print(g.to_string())


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODEL_CONFIGS),
                    help="default: todos os 6 da matriz")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: todos os do manifest de data/splits/")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "test"])
    ap.add_argument("--batch-size", type=int, default=8,
                    help="questoes por lote de geracao (suba se a VRAM permitir)")
    ap.add_argument("--limit", type=int, default=None,
                    help="limita questoes por (modelo, dataset, split)")
    ap.add_argument("--device", default=None, help='ex.: "cuda:0"; default: auto')
    ap.add_argument("--download", action="store_true",
                    help="so baixa os pesos dos modelos (passo de instalacao) e sai")
    ap.add_argument("--mock", action="store_true",
                    help="respostas sinteticas (valida pipeline sem GPU; saida separada)")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra o plano e o custo estimado, nao executa")
    ap.add_argument("--status", action="store_true",
                    help="progresso e metricas parciais dos JSONL")
    ap.add_argument("--selftest", action="store_true",
                    help="roda os testes do extrator e sai")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    models = args.models or list(DEFAULT_MODELS)

    if args.download:
        download_models(models)
        return

    res_out = RES.parent / "fase1_mock" if args.mock else RES
    res_out.mkdir(parents=True, exist_ok=True)

    if args.status:
        status(res_out)
        return

    splits = load_splits(args.datasets, args.splits)
    n_q = sum(min(args.limit or 10**9, len(df)) for df in splits.values()) * len(models)
    print(f"Plano: {len(models)} modelos x {len(splits)} (dataset,split) "
          f"= {n_q:,} chamadas | prompt_sha={PROMPT_SHA} | batch={args.batch_size} | "
          f"{'MOCK' if args.mock else 'HF/transformers'}")
    for m in models:
        cfg = MODEL_CONFIGS[m]
        print(f"  - {m:15s} {cfg['hf_id']:44s} max_new={cfg['max_new_tokens']:5d}"
              f"{'  [GATED]' if cfg.get('gated') else ''}  [{cfg['eixo']}]")
    if args.dry_run:
        print(f"Custo estimado (2-15 s/questao com batch em GPU): "
              f"{n_q*2/3600:.1f}-{n_q*15/3600:.1f} h")
        return

    signal.signal(signal.SIGINT, _sigint)
    # loop externo por MODELO: carrega uma vez, roda todos os (dataset, split)
    for m in models:
        hf_model = None
        if not args.mock:
            hf_model = HFModel(m, device=args.device)
        try:
            for (ds, sp), df in splits.items():
                if not run(m, hf_model, ds, sp, df, res_out, args.mock,
                           args.batch_size, args.limit):
                    print("Interrompido - progresso salvo; reexecute para retomar.")
                    return
        finally:
            if hf_model is not None:
                hf_model.unload()
    print("\nResumo:")
    status(res_out)


if __name__ == "__main__":
    main()
