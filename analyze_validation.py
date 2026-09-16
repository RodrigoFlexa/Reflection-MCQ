#!/usr/bin/env python
"""Tabelas e figuras de threshold a partir do arquivo canônico de validação.

    python analyze_validation.py

Lê `results_definitivos/validation/all_outcomes.jsonl` e escreve em
`results_definitivos/validation/analysis/<digest>/`, onde o digest é o hash da
configuração: duas execuções com a mesma configuração caem na mesma pasta, e
uma mudança de política nunca sobrescreve o resultado da anterior.

Não chama modelo nenhum e não altera o arquivo canônico.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rmcq import results_store as store

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fallback", action="store_true",
                        help="Usa o baseline em falhas e abaixo do threshold (padrão: excluir)")
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--min-coverage", type=float, default=.1)
    parser.add_argument("--outcomes", type=Path,
                        help="Sobrepõe o arquivo canônico; para inspecionar um recorte.")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from rmcq.test_analysis import read_outcomes
    from rmcq.validation_analysis import study, export_study, plot_study, CLEAN_FLAGS, THRESHOLDS

    path = args.outcomes or store.outcomes_path(ROOT, "validation")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode: python consolidate_results.py build --split validation")
    frame = read_outcomes(path, "validation")
    if frame.empty:
        raise ValueError(f"{path} não tem linhas de validação")

    config = dict(source=str(path.relative_to(ROOT)), fallback=args.fallback, min_n=args.min_n,
                  min_coverage=args.min_coverage, clean_flags=CLEAN_FLAGS, thresholds=THRESHOLDS)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    directory = store.split_dir(ROOT, "validation") / "analysis" / digest
    arms = sorted(frame.arm.dropna().unique())
    print(f"{len(frame):,} linhas | {frame.model.nunique()} modelos | {len(arms)} braços -> {directory}", flush=True)
    views = study(frame, fallback=args.fallback, min_n=args.min_n, min_coverage=args.min_coverage)
    export_study(views, directory)
    (directory / "config.json").write_text(
        json.dumps({**config, "arms": arms}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_study(views, directory, show=False)
    print(f"Pronto: {directory}", flush=True)


if __name__ == "__main__":
    main()
