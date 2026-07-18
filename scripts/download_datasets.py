#!/usr/bin/env python3
"""
Aquisição dos datasets MCQ — coerente com o notebook 01
(01_datasets_criacao_analise_splits.ipynb) e com a matriz experimental:

  processo:      gsm8k, aqua, logiqa2
  conhecimento:  arc, openbookqa, oil_and_gas (v2)

Duas origens, refletindo a natureza de cada dataset:

  1. HuggingFace (público, conversão determinística — mesmo código do notebook 01):
     - logiqa2    <- jeggers/logiqa2_formatted (MRC do csitfun/LogiQA2.0)
     - openbookqa <- allenai/openbookqa (config "main")

  2. Drive do Rodrigo (fallback: cópia local do repo) — datasets já convertidos
     (arc, gsm8k-MCQ, aqua) e o proprietário (oil_and_gas_v2):
     - o GSM8K MCQ é uma CONVERSÃO própria (distratores gerados); baixar do
       Drive garante que todos usem a mesma conversão auditada (decisão D6).

agievalar/ e mmlu/ foram SUBSTITUÍDOS (por logiqa2 e openbookqa — ver
racional no notebook 01); se existirem no disco, são ignorados.

Uso:
    pip install gdown pandas pyarrow huggingface_hub datasets
    python scripts/download_datasets.py            # Drive + HF -> data/mcq
    python scripts/download_datasets.py --local    # fonte local + HF
    python scripts/download_datasets.py --verify   # só verifica integridade

Depois de baixar: rode o notebook 01 para gerar os splits congelados
(data/splits/), que são o que os scripts de experimento consomem.
"""

import argparse
import shutil
import sys
from pathlib import Path

DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1ykTzmLKtYdhmarDvMsdHc1kp7VSdVVWy"
)

ROOT = Path(__file__).resolve().parents[1]          # journal_extension/
DATA_DIR = ROOT / "data" / "mcq"
LOCAL_SOURCE = ROOT.parent / "sandbox" / "experiments" / "datasets" / "mcq"

# dataset -> (origem, arquivos esperados)
EXPECTED = {
    "aqua":           ("drive", ["train.parquet", "validation.parquet", "test.parquet"]),
    "arc":            ("drive", ["arc_train.parquet", "arc_validation.parquet",
                                 "arc_test.parquet"]),
    "gsm8k":          ("drive", ["train.parquet", "test.parquet"]),
    "oil_and_gas_v2": ("drive", ["test.parquet"]),
    "logiqa2":        ("hf",    ["train.parquet", "validation.parquet", "test.parquet"]),
    "openbookqa":     ("hf",    ["train.parquet", "validation.parquet", "test.parquet"]),
}
DEPRECATED = ["agievalar", "mmlu"]  # substituídos por logiqa2 / openbookqa


# ------------------------------------------------- origem 1: HuggingFace
def _hf_load(repo, config=None):
    """Igual ao notebook 01: tenta load_dataset; cai para os parquets da
    conversão automática (refs/convert/parquet) se o repo for script-based."""
    import pandas as pd
    try:
        from datasets import load_dataset
        d = load_dataset(repo, config) if config else load_dataset(repo)
        return {s: d[s].to_pandas() for s in d}
    except Exception as e:
        print(f"  load_dataset falhou ({type(e).__name__}); tentando parquet do Hub…")
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        base = f"datasets/{repo}@refs%2Fconvert%2Fparquet"
        out = {}
        for split_dir in fs.ls(f"{base}/{config or 'default'}", detail=False):
            split = Path(split_dir).name
            files = [f for f in fs.ls(split_dir, detail=False) if f.endswith(".parquet")]
            out[split] = pd.concat([pd.read_parquet(fs.open(f)) for f in files],
                                   ignore_index=True)
        return out


def fetch_openbookqa() -> bool:
    import pandas as pd
    if (DATA_DIR / "openbookqa" / "train.parquet").exists():
        print("openbookqa: já em disco")
        return True
    print("openbookqa: baixando de allenai/openbookqa…")
    d = _hf_load("allenai/openbookqa", "main")
    (DATA_DIR / "openbookqa").mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        raw = d[split]
        df = pd.DataFrame({
            "id":        [f"obqa_{split}_{i:05d}" for i in range(len(raw))],
            "question":  raw["question_stem"],
            "choices":   [{"label": list(c["label"]), "text": list(c["text"])}
                          for c in raw["choices"]],
            "answerKey": raw["answerKey"],
        })
        df.to_parquet(DATA_DIR / "openbookqa" / f"{split}.parquet", index=False)
    return True


TYPE_COLS = ["Categorical Reasoning", "Disjunctive Reasoning", "Conjunctive Reasoning",
             "Necessry Condtional Reasoning", "Sufficient Conditional Reasoning"]


def fetch_logiqa2() -> bool:
    import pandas as pd
    if (DATA_DIR / "logiqa2" / "train.parquet").exists():
        print("logiqa2: já em disco")
        return True
    print("logiqa2: baixando de jeggers/logiqa2_formatted…")
    d = _hf_load("jeggers/logiqa2_formatted")
    (DATA_DIR / "logiqa2").mkdir(parents=True, exist_ok=True)
    LBL = list("ABCD")
    for split in ("train", "validation", "test"):
        raw = d[split]
        ctx = raw["text"].astype(str).str.strip()
        types = raw[TYPE_COLS].fillna(False).apply(
            lambda r: "+".join(t.split()[0].lower() for t, v in r.items() if v)
                      or "unspecified", axis=1)
        df = pd.DataFrame({
            "id":        [f"logiqa2_{split}_{i:05d}" for i in range(len(raw))],
            "question":  ctx + "\n\n" + raw["question"].astype(str).str.strip(),
            "context":   ctx,
            "choices":   [{"label": LBL, "text": [str(o) for o in ops]}
                          for ops in raw["options"]],
            "answerKey": [LBL[int(i)] for i in raw["answer"]],
            "subject":   types.values,
        })
        df.to_parquet(DATA_DIR / "logiqa2" / f"{split}.parquet", index=False)
    return True


# ------------------------------------------------- origem 2: Drive / local
def download_from_drive() -> bool:
    try:
        import gdown
    except ImportError:
        print("gdown não instalado. Rode: pip install gdown")
        return False
    print(f"Baixando pasta do Drive para {DATA_DIR} …")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(url=DRIVE_FOLDER_URL, output=str(DATA_DIR), quiet=False)
        return True
    except Exception as e:
        print(f"Falha no download do Drive: {e}")
        return False


def copy_from_local() -> bool:
    if not LOCAL_SOURCE.exists():
        print(f"Fonte local não encontrada: {LOCAL_SOURCE}")
        return False
    print(f"Copiando {LOCAL_SOURCE} -> {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOCAL_SOURCE, DATA_DIR, dirs_exist_ok=True)
    return True


# --------------------------------------------------------------- verificação
def verify() -> bool:
    import pandas as pd

    ok = True
    for ds, (origem, files) in EXPECTED.items():
        for f in files:
            p = DATA_DIR / ds / f
            if not p.exists():
                print(f"[FALTA]  {ds}/{f}  (origem: {origem})")
                ok = False
                continue
            try:
                df = pd.read_parquet(p)
                assert {"id", "question", "choices", "answerKey"} <= set(df.columns), \
                    "schema canônico incompleto"
                c = df.iloc[0]["choices"]
                assert "label" in c and "text" in c, "choices sem label/text"
                print(f"[OK]     {ds}/{f}  ({len(df):,} linhas)")
            except Exception as e:
                print(f"[ERRO]   {ds}/{f}: {e}")
                ok = False
    for ds in DEPRECATED:
        if (DATA_DIR / ds).exists():
            print(f"[AVISO]  {ds}/ presente mas SUBSTITUÍDO — ignorado pelo "
                  f"notebook 01; pode remover")
    splits = ROOT / "data" / "splits" / "manifest.json"
    print(f"\nSplits congelados: {'presentes' if splits.exists() else 'AUSENTES'} "
          f"({splits.parent})" + ("" if splits.exists() else
          " -> rode o notebook 01 para gerá-los"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("Uso:")[0])
    ap.add_argument("--local", action="store_true",
                    help="copiar da fonte local em vez do Drive")
    ap.add_argument("--verify", action="store_true",
                    help="apenas verificar integridade")
    args = ap.parse_args()

    if not args.verify:
        got = copy_from_local() if args.local else download_from_drive()
        if not got and not args.local:
            print("Tentando fallback local…")
            got = copy_from_local()
        if not got:
            sys.exit(1)
        if not (fetch_openbookqa() and fetch_logiqa2()):
            sys.exit(1)

    sys.exit(0 if verify() else 1)
