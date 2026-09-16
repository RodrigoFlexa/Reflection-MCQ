#!/usr/bin/env python
"""Monta e audita results_definitivos a partir dos runs brutos.

    python consolidate_results.py build  --split validation
    python consolidate_results.py audit  --split validation
    python consolidate_results.py status

`build` reconstrói o arquivo canônico do split a partir das fontes declaradas
abaixo. É reprodutível: as mesmas fontes na mesma ordem dão o mesmo arquivo, e a
ordem da lista é a ordem de autoridade — a primeira fonte que traz uma medida
vence, as repetições depois dela são contadas e descartadas.

`audit` não reconstrói nada; relê o arquivo canônico e refaz cobertura e
lacunas. Use depois de uma rodada nova para ver o que ainda falta.

As fontes ficam aqui, em código, e não num argumento de linha de comando, para
que a procedência do arquivo final seja uma coisa que se lê e se revisa junto
com o resto do repositório.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rmcq import grid
from rmcq import results_store as store
from rmcq.results_store import Source

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "data/results/reflection_top1"
EXCHANGE = ROOT / "experiment_exchange"

# Revisão corrente do pipeline. Um run gerado sob ela tem autoridade sobre
# qualquer run mais antigo, por mais recente que o arquivo do outro seja.
CURRENT_PIPELINE = "top1-two-server-v5"

# Run v4, anterior à revisão dos prompts de professor e de transferência.
LEGACY_RUN = "91ccab5e5028"


def run_manifest(run_id: str) -> dict:
    path = EXCHANGE / run_id / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def declared_split(path: Path) -> str | None:
    """O split que o próprio arquivo de resultados afirma, lendo a primeira linha.

    Um run que chegou por handoff não traz o manifest junto, e sem isto ficaria
    invisível para a consolidação. A linha sabe a que split pertence — é o campo
    que o pipeline grava em cada uma — então é ela quem responde quando não há
    manifest para perguntar.
    """
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    return json.loads(line).get("eval_split")
    except (OSError, json.JSONDecodeError):
        return None
    return None


def discovered_runs(split: str) -> list[Source]:
    """Runs do disco que produziram resultados deste split, em ordem de autoridade.

    A descoberta é automática de propósito: uma rodada nova entra no arquivo
    definitivo sem que ninguém precise lembrar de editar uma lista aqui. A ordem
    é revisão do pipeline primeiro, recência depois — um run velho não
    sobrescreve um novo só por ter sido tocado por último, e um run da revisão
    corrente tem precedência sobre qualquer um de revisão anterior.
    """
    found = []
    for directory in RUNS.iterdir() if RUNS.exists() else []:
        if not directory.is_dir() or directory.name == LEGACY_RUN:
            continue
        manifest = run_manifest(directory.name)
        # O arquivo do run inteiro vale mais que o recorte de autorreflexão, que
        # é o mesmo run antes do professor entrar.
        for relative, phase in (("analysis/all_outcomes.jsonl", "completo"),
                                ("self_eval/analysis/all_outcomes.jsonl", "só autorreflexão")):
            path = directory / relative
            if not path.exists():
                continue
            # Igualdade estrita, nunca "não sei": aceitar um run por omissão foi
            # exatamente como linhas de teste entraram no arquivo de validação.
            # Sem manifest, quem responde é a primeira linha do próprio arquivo.
            belongs = manifest.get("eval_split") or declared_split(path)
            if belongs != split:
                if belongs is None:
                    print(f"AVISO: {directory.name} não declara split nem no manifest nem nas "
                          f"linhas; fora da consolidação", flush=True)
                break
            version = manifest.get("pipeline_version") or "desconhecida"
            found.append((
                version == CURRENT_PIPELINE, path.stat().st_mtime,
                Source(path=path,
                       note=f"Run {directory.name} ({phase}), pipeline {version}.",
                       drop_models=grid.RETIRED_STUDENTS,
                       stamp={"source_run": directory.name, "pipeline_version": version}),
            ))
            break
    found.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [source for _, _, source in found]


def validation_sources() -> list[Source]:
    return discovered_runs("validation")


VALIDATION_EXCLUDED = [
    {
        "path": "data/results/reflection_top1/*/self_eval/analysis/all_outcomes.jsonl",
        "what": "todas as linhas de deepseek-r1-0528-qwen3-8b",
        "why": ("o DeepSeek da grade é a destilação sobre Llama, por padronização; o "
                "0528 é sobre Qwen3, outra base. Não é o mesmo modelo, então as linhas "
                "dele não contam como deepseek-r1-8b nem entram como um décimo aluno"),
        "rows": 22461,
    },
    {
        "path": f"data/results/reflection_top1/{LEGACY_RUN}/analysis/all_outcomes.jsonl",
        "what": "o run v4 inteiro (phi2, deepseek-r1-distill-llama-8b, llama3.1-8b)",
        "why": ("gerado com transfer_prompt e teacher_reflection_prompts do pipeline v4, "
                "substituídos no v5. É a única destilação-Llama de 8B que existe em "
                "disco hoje, mas cobre quatro datasets (sem race) e sob prompts antigos: "
                "entraria como revisão misturada, não como dado faltante. O modelo "
                "precisa ser gerado sob o v5"),
        "rows": 39000,
    },
]


def test_sources() -> list[Source]:
    return discovered_runs("test")


SOURCES = {"validation": validation_sources, "test": test_sources}
EXCLUDED = {"validation": VALIDATION_EXCLUDED, "test": []}


def build(split: str) -> None:
    sources = [s for s in SOURCES[split]() if Path(s.path).exists()]
    absent = [str(s.path) for s in SOURCES[split]() if not Path(s.path).exists()]
    if not sources:
        raise FileNotFoundError(f"Nenhuma fonte de {split} existe em disco: {absent}")
    for missing in absent:
        print(f"AVISO: fonte ausente, seguindo sem ela: {missing}", flush=True)

    print(f"Fundindo {len(sources)} fonte(s) -> {store.outcomes_path(ROOT, split)}", flush=True)
    report = store.merge(ROOT, split, sources)
    print(f"  {report.total_written:,} linhas gravadas, "
          f"{report.duplicates_skipped:,} duplicatas, "
          f"{report.rejected_by_filter:,} fora do recorte", flush=True)

    table, universe, provenance = store.coverage(store.stream_outcomes(ROOT, split))
    holes = store.gaps(table, universe)
    store.write_coverage(store.coverage_path(ROOT, split), table)
    store.write_json(store.gaps_path(ROOT, split), holes)
    store.write_json(store.manifest_path(ROOT, split), {
        "split": split,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": report.total_written,
        "sources": report.per_source,
        "excluded_on_purpose": EXCLUDED[split],
        "normalizations": {
            "eval_split_filled_from_file": report.eval_split_filled,
            "teacher_model_filled_with_legacy_default": report.teacher_model_filled,
        },
        "row_key": list(store.ROW_KEY),
        "duplicates_skipped": report.duplicates_skipped,
        "provenance": provenance,
        "datasets": holes["datasets"],
        "universe_per_dataset": holes["universe_per_dataset"],
    })
    report_status(split, table, holes, provenance)


def audit(split: str) -> None:
    table, universe, provenance = store.coverage(store.stream_outcomes(ROOT, split))
    holes = store.gaps(table, universe)
    store.write_coverage(store.coverage_path(ROOT, split), table)
    store.write_json(store.gaps_path(ROOT, split), holes)
    report_status(split, table, holes, provenance)


def report_status(split: str, table, holes, provenance) -> None:
    rows = sum(r["rows"] for r in table)
    print(f"\n=== {split}: {rows:,} linhas ===", flush=True)
    print(f"universo por dataset: {holes['universe_per_dataset']}", flush=True)
    mixed = {p["model"] for p in provenance if p["pipeline_version"]}
    versions = {}
    for entry in provenance:
        versions.setdefault(entry["model"], set()).add(entry["pipeline_version"])
    divergent = sorted(m for m in mixed if len(versions.get(m, set())) > 1 or
                       versions.get(m) == {"top1-two-server-v4"})
    if divergent:
        print(f"ATENCAO, revisao de pipeline diferente da grade: {', '.join(divergent)}", flush=True)
    print(f"celulas da grade sem nenhuma linha: {len(holes['cells_missing'])}", flush=True)
    print(f"celulas parciais: {len(holes['cells_partial'])}", flush=True)
    print(f"linhas que faltam para fechar a grade: {holes['rows_missing_total']:,}", flush=True)
    if holes["cells_outside_grid"]:
        print(f"celulas fora da grade final: {len(holes['cells_outside_grid'])}", flush=True)
    for cell in holes["cells_missing"][:12]:
        teacher = f" <- {cell['teacher_model']}" if cell["teacher_model"] else ""
        print(f"   falta: {cell['model']} {cell['condition']}{teacher} ({cell['rows_missing']:,} linhas)", flush=True)
    if len(holes["cells_missing"]) > 12:
        print(f"   ... e mais {len(holes['cells_missing']) - 12}; veja {store.gaps_path(ROOT, split)}", flush=True)
    for cell in holes["cells_partial"][:6]:
        teacher = f" <- {cell['teacher_model']}" if cell["teacher_model"] else ""
        print(f"   parcial: {cell['model']} {cell['condition']}{teacher} ({cell['rows_missing']:,} linhas)", flush=True)


def status() -> None:
    print(grid.describe(), flush=True)
    for split in store.SPLITS:
        path = store.gaps_path(ROOT, split)
        if not path.exists():
            print(f"\n=== {split}: ainda não consolidado "
                  f"(python consolidate_results.py build --split {split}) ===", flush=True)
            continue
        holes = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(store.manifest_path(ROOT, split).read_text(encoding="utf-8"))
        print(f"\n=== {split}: {manifest['rows']:,} linhas, montado em {manifest['built_at_utc']} ===", flush=True)
        print(f"celulas sem linha: {len(holes['cells_missing'])} | "
              f"parciais: {len(holes['cells_partial'])} | "
              f"linhas faltando: {holes['rows_missing_total']:,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("build", "audit", "status"))
    parser.add_argument("--split", choices=store.SPLITS)
    args = parser.parse_args()
    grid.validate()
    if args.action == "status":
        status()
        return
    if not args.split:
        parser.error("--split é obrigatório para build e audit")
    (build if args.action == "build" else audit)(args.split)


if __name__ == "__main__":
    main()
