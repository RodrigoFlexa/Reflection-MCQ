"""results_definitivos: um lugar só para os resultados que valem.

Antes havia um arquivo de resultado por run, cada um com um id opaco no
caminho, e a pergunta "qual é o número final?" dependia de lembrar qual run era
o bom. Aqui a resposta é um caminho fixo por split:

    results_definitivos/validation/all_outcomes.jsonl
    results_definitivos/test/all_outcomes.jsonl

e, ao lado de cada um, o que torna esse arquivo auditável sem abri-lo:

    manifest.json   de onde veio cada linha, quantas vieram, o que foi normalizado
    coverage.csv    contagem por aluno x condição x professor x dataset
    gaps.json       o que a grade final pede e ainda não está no arquivo

Os arquivos são grandes demais para caber na memória (centenas de MB), então
tudo aqui é streaming: uma linha por vez, entra e sai.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from rmcq import grid

SPLITS = ("validation", "test")
STORE_DIRNAME = "results_definitivos"

# Identidade de uma linha de resultado. Duas linhas com a mesma chave são a
# mesma medida; a segunda é duplicata, não dado novo.
ROW_KEY = ("model", "dataset", "val_uid", "condition", "teacher_model")


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------


def store_root(root: Path) -> Path:
    return Path(root) / STORE_DIRNAME


def split_dir(root: Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split deve ser um de {SPLITS}, não {split!r}")
    return store_root(root) / split


def outcomes_path(root: Path, split: str) -> Path:
    return split_dir(root, split) / "all_outcomes.jsonl"


def manifest_path(root: Path, split: str) -> Path:
    return split_dir(root, split) / "manifest.json"


def coverage_path(root: Path, split: str) -> Path:
    return split_dir(root, split) / "coverage.csv"


def gaps_path(root: Path, split: str) -> Path:
    return split_dir(root, split) / "gaps.json"


# ---------------------------------------------------------------------------
# Leitura e normalização
# ---------------------------------------------------------------------------


def normalize(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Põe a linha na forma canônica, sem inventar nada que ela não afirme.

    Duas correções, ambas de forma e não de conteúdo:

    `teacher_model` passa a ser explícito. Nas linhas antigas ele não existia
    porque só havia um professor possível; deixar em branco hoje faria a linha
    colidir com a de outro professor na mesma questão.

    `eval_split` passa a ser preenchido. Um run de validação escreveu nulo aí
    quando o campo veio do item em vez do par; o split é o do arquivo, e é o
    arquivo que sabe disso, não a linha.

    E `model` passa a ser o nome canônico da grade. O DeepSeek-R1 de 8B apareceu
    em rodadas diferentes sob duas destilações, e é um modelo só para efeito de
    comparação; o checkpoint de onde a linha saiu fica em `model_checkpoint`,
    para que a fusão não apague de onde o número veio.
    """
    out = dict(row)
    for campo, guarda in (("model", "model_checkpoint"), ("teacher_model", "teacher_checkpoint")):
        original = out.get(campo)
        nome = grid.canonical(original)
        if nome != original:
            out.setdefault(guarda, original)
        out[campo] = nome
    if grid.is_teacher_condition(out.get("condition", "")):
        out["teacher_model"] = out.get("teacher_model") or grid.LEGACY_TEACHER
    else:
        out["teacher_model"] = None
        out.pop("teacher_checkpoint", None)
    if not out.get("eval_split"):
        out["eval_split"] = split
    return out


def row_key(row: dict[str, Any]) -> tuple:
    return tuple(row.get(k) for k in ROW_KEY)


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def stream_outcomes(root: Path, split: str) -> Iterator[dict[str, Any]]:
    """Linhas canônicas do split, já normalizadas."""
    path = outcomes_path(root, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não existe. Rode: python consolidate_results.py build --split {split}"
        )
    for row in stream_jsonl(path):
        yield normalize(row, split)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """Uma origem de linhas, com a regra de recorte que a torna legítima."""

    path: Path
    note: str = ""
    # Recortes opcionais: quando presentes, só passam as linhas que casam.
    models: tuple[str, ...] | None = None
    conditions: tuple[str, ...] | None = None
    drop_models: tuple[str, ...] = ()
    # Carimbo de proveniência gravado em cada linha que vem daqui. Sem ele,
    # linhas geradas sob revisões diferentes do pipeline ficam indistinguíveis
    # dentro do arquivo fundido, e a comparação entre modelos passa a esconder
    # de qual prompt cada número saiu.
    stamp: dict[str, Any] = field(default_factory=dict)

    def accepts(self, row: dict[str, Any]) -> bool:
        # Compara pelos dois nomes: um recorte escrito com o nome canônico
        # precisa casar com linhas gravadas sob o nome histórico, e vice-versa.
        nomes = {row.get("model"), grid.canonical(row.get("model"))}
        if self.models is not None and not nomes & set(self.models):
            return False
        if self.conditions is not None and row.get("condition") not in self.conditions:
            return False
        return not nomes & set(self.drop_models)

    def describe(self, root: Path) -> dict[str, Any]:
        try:
            shown = str(Path(self.path).resolve().relative_to(Path(root).resolve()))
        except ValueError:
            shown = str(self.path)
        info: dict[str, Any] = {"path": shown, "note": self.note}
        if self.stamp:
            info["stamp"] = dict(self.stamp)
        if self.models is not None:
            info["only_models"] = list(self.models)
        if self.conditions is not None:
            info["only_conditions"] = list(self.conditions)
        if self.drop_models:
            info["dropped_models"] = list(self.drop_models)
        return info


@dataclass
class MergeReport:
    total_written: int = 0
    per_source: list[dict[str, Any]] = field(default_factory=list)
    duplicates_skipped: int = 0
    rejected_by_filter: int = 0
    eval_split_filled: int = 0
    teacher_model_filled: int = 0


def merge(root: Path, split: str, sources: list[Source], destination: Path | None = None) -> MergeReport:
    """Funde as fontes num arquivo canônico. A primeira fonte que traz uma
    chave vence: a ordem da lista é a ordem de autoridade, declarada e não
    inferida de data de modificação.
    """
    destination = destination or outcomes_path(root, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")

    seen: set[tuple] = set()
    report = MergeReport()
    with temp.open("w", encoding="utf-8") as out:
        for source in sources:
            kept = skipped = rejected = 0
            for raw in stream_jsonl(source.path):
                if not source.accepts(raw):
                    rejected += 1
                    continue
                declared = raw.get("eval_split")
                if declared and declared != split:
                    raise ValueError(
                        f"{source.path} tem linha de split {declared!r} sendo fundida em "
                        f"{split!r} (modelo {raw.get('model')!r}). Uma medida não muda de "
                        f"split por estar noutro arquivo; corrija a fonte.")
                row = normalize(raw, split)
                row.update(source.stamp)
                if not raw.get("eval_split"):
                    report.eval_split_filled += 1
                if grid.is_teacher_condition(raw.get("condition", "")) and not raw.get("teacher_model"):
                    report.teacher_model_filled += 1
                key = row_key(row)
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
            report.per_source.append({**source.describe(root), "rows_kept": kept,
                                      "rows_duplicate": skipped, "rows_filtered_out": rejected})
            report.duplicates_skipped += skipped
            report.rejected_by_filter += rejected
            report.total_written += kept
    temp.replace(destination)
    return report


# ---------------------------------------------------------------------------
# Cobertura e lacunas
# ---------------------------------------------------------------------------


def coverage(rows: Iterator[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, set], list[dict[str, Any]]]:
    """Conta as linhas por célula da grade e devolve o universo de itens visto.

    O universo por dataset é o conjunto de `val_uid` que qualquer condição
    alcançou. É contra ele que uma célula é chamada de completa ou parcial: o
    denominador sai dos dados, não de uma constante escrita à mão.

    A terceira saída é a proveniência: quantas linhas de cada modelo vieram de
    qual run e sob qual revisão do pipeline. Um modelo que aparece com duas
    revisões tem números que não se comparam entre si, e isso precisa estar
    visível sem reabrir o arquivo.
    """
    counts: Counter = Counter()
    resolved: Counter = Counter()
    origins: Counter = Counter()
    universe: dict[str, set] = defaultdict(set)
    for row in rows:
        cell = (row.get("model"), row.get("condition"), row.get("teacher_model"), row.get("dataset"))
        counts[cell] += 1
        if row.get("correct") is not None:
            resolved[cell] += 1
        origins[(row.get("model"), row.get("model_checkpoint") or row.get("model"),
                 row.get("source_run"), row.get("pipeline_version"))] += 1
        universe[row.get("dataset")].add(row.get("val_uid"))
    provenance = [
        {"model": model, "checkpoint": checkpoint, "source_run": run or "",
         "pipeline_version": version or "", "rows": n}
        for (model, checkpoint, run, version), n in sorted(
            origins.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
    ]
    table = [
        {"model": model, "condition": condition, "teacher_model": teacher or "",
         "dataset": dataset, "rows": n, "resolved": resolved[(model, condition, teacher, dataset)],
         "universe": len(universe[dataset]),
         "complete": n >= len(universe[dataset])}
        for (model, condition, teacher, dataset), n in sorted(
            counts.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
    ]
    return table, dict(universe), provenance


def gaps(table: list[dict[str, Any]], universe: dict[str, set]) -> dict[str, Any]:
    """O que a grade final pede e o arquivo ainda não tem.

    Distingue duas coisas que soam iguais e não são: a célula que não existe
    (nada foi gerado) e a célula que existe pela metade (faltam itens de algum
    dataset). A primeira é trabalho inteiro; a segunda é retomada.
    """
    datasets = sorted(universe)
    have = {(r["model"], r["condition"], r["teacher_model"] or None, r["dataset"]): r for r in table}
    missing: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for student, condition, teacher in grid.expected_cells():
        for dataset in datasets:
            found = have.get((student, condition, teacher, dataset))
            expected = len(universe[dataset])
            if found is None:
                missing.append({"model": student, "condition": condition,
                                "teacher_model": teacher, "dataset": dataset,
                                "rows": 0, "expected": expected})
            elif found["rows"] < expected:
                partial.append({"model": student, "condition": condition,
                                "teacher_model": teacher, "dataset": dataset,
                                "rows": found["rows"], "expected": expected,
                                "missing": expected - found["rows"]})
    extra = sorted({
        (r["model"], r["condition"], r["teacher_model"] or None) for r in table
        if (r["model"], r["condition"], r["teacher_model"] or None) not in
        {(s, c, t) for s, c, t in grid.expected_cells()}
    }, key=lambda cell: tuple(str(x) for x in cell))

    def by_cell(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple, dict[str, Any]] = {}
        for entry in entries:
            cell = (entry["model"], entry["condition"], entry["teacher_model"])
            slot = grouped.setdefault(cell, {"model": cell[0], "condition": cell[1],
                                             "teacher_model": cell[2], "datasets": {},
                                             "rows_missing": 0})
            slot["datasets"][entry["dataset"]] = entry.get("missing", entry["expected"])
            slot["rows_missing"] += entry.get("missing", entry["expected"])
        return sorted(grouped.values(), key=lambda s: (s["model"], s["condition"], str(s["teacher_model"])))

    return {
        "grid": {
            "students": list(grid.STUDENTS),
            "small_students": list(grid.SMALL_STUDENTS),
            "external_teacher": grid.EXTERNAL_TEACHER,
            "open_teachers": list(grid.OPEN_TEACHERS),
            "expected_cells": len(grid.expected_cells()),
        },
        "datasets": datasets,
        "universe_per_dataset": {d: len(universe[d]) for d in datasets},
        "cells_missing": by_cell(missing),
        "cells_partial": by_cell(partial),
        "cells_outside_grid": [
            {"model": m, "condition": c, "teacher_model": t} for m, c, t in extra
        ],
        "rows_missing_total": sum(e["expected"] for e in missing) + sum(e["missing"] for e in partial),
    }


def write_coverage(path: Path, table: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "condition", "teacher_model", "dataset", "rows", "resolved", "universe", "complete"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
