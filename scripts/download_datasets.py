"""
Download automatizado dos datasets MCQ para a extensão do journal.

Fonte primária : Google Drive do Rodrigo
Fonte fallback : cópia local em sandbox/experiments/datasets/mcq (repo eagle)

Uso:
    pip install gdown pandas pyarrow
    python scripts/download_datasets.py            # baixa do Drive -> data/mcq
    python scripts/download_datasets.py --local    # copia do sandbox local
    python scripts/download_datasets.py --verify   # só verifica integridade

Obs.: o link do Drive precisa estar como "Qualquer pessoa com o link".
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

# dataset -> arquivos parquet esperados (split: nome do arquivo)
EXPECTED = {
    "agievalar":   ["dev.parquet", "test.parquet"],
    "aqua":        ["train.parquet", "validation.parquet", "test.parquet"],
    "arc":         ["arc_train.parquet", "arc_validation.parquet", "arc_test.parquet"],
    "gsm8k":       ["train.parquet", "test.parquet"],
    "mmlu":        ["validation.parquet", "test.parquet"],
    "oil_and_gas": ["test.parquet"],
}


def download_from_drive() -> bool:
    try:
        import gdown
    except ImportError:
        print("gdown não instalado. Rode: pip install gdown")
        return False
    print(f"Baixando pasta do Drive para {DATA_DIR} ...")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
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
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOCAL_SOURCE, DATA_DIR, dirs_exist_ok=True)
    return True


def verify() -> bool:
    import pandas as pd

    ok = True
    for ds, files in EXPECTED.items():
        for f in files:
            p = DATA_DIR / ds / f
            if not p.exists():
                print(f"[FALTA]  {ds}/{f}")
                ok = False
                continue
            try:
                df = pd.read_parquet(p)
                assert {"id", "question", "choices", "answerKey"} <= set(df.columns)
                print(f"[OK]     {ds}/{f}  ({len(df)} linhas)")
            except Exception as e:
                print(f"[ERRO]   {ds}/{f}: {e}")
                ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="copiar da fonte local em vez do Drive")
    ap.add_argument("--verify", action="store_true", help="apenas verificar integridade")
    args = ap.parse_args()

    if not args.verify:
        got = copy_from_local() if args.local else download_from_drive()
        if not got and not args.local:
            print("Tentando fallback local...")
            got = copy_from_local()
        if not got:
            sys.exit(1)

    sys.exit(0 if verify() else 1)
