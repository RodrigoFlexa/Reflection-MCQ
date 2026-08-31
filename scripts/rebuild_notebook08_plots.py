"""Build the plot-only companion notebook for experiment 08."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "08_similarity_reflection_threshold_plots.ipynb"


def md(text: str) -> dict:
    value = text.strip() + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code(text: str) -> dict:
    value = text.strip() + "\n"
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": value.splitlines(keepends=True),
    }


cells = [
    md(
        r"""
# 08 — Plots do threshold de similaridade

Este notebook **não executa modelos, embeddings, recuperação ou bootstrap**.
Ele apenas carrega os CSVs produzidos por `run_similarity_threshold.py` e
recria figuras e resumos.

Por padrão, ele abre a execução concluída mais recente. Para escolher outra,
preencha `EXPERIMENT_ID` na célula seguinte ou defina a variável de ambiente
`RMCQ_THRESHOLD_EXPERIMENT_ID` antes de iniciar o Jupyter.
"""
    ),
    code(
        r"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "rmcq").is_dir():
        ROOT = candidate
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Não encontrei a raiz do repositório contendo rmcq/")

RESULTS_ROOT = ROOT / "data" / "results" / "similarity_threshold_v2"

# Informe um ID para fixar a execução, ou deixe None para usar a concluída mais recente.
EXPERIMENT_ID = os.environ.get("RMCQ_THRESHOLD_EXPERIMENT_ID") or None

REQUIRED = [
    "analysis/all_outcomes.csv",
    "analysis/thresholds_calibration.csv",
    "analysis/effect_curves_calibration.csv",
    "analysis/heldout_policies.csv",
    "analysis/decision_transitions.csv",
]


def is_complete(path):
    return all((path / relative).exists() for relative in REQUIRED)


if EXPERIMENT_ID:
    OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID
    if not is_complete(OUT_DIR):
        missing = [x for x in REQUIRED if not (OUT_DIR / x).exists()]
        raise FileNotFoundError(f"Execução {EXPERIMENT_ID} incompleta; faltam: {missing}")
else:
    candidates = [path for path in RESULTS_ROOT.iterdir() if path.is_dir() and is_complete(path)]
    if not candidates:
        raise FileNotFoundError(
            "Nenhuma execução concluída. Rode primeiro: python -u run_similarity_threshold.py"
        )
    OUT_DIR = max(candidates, key=lambda path: (path / "analysis/heldout_policies.csv").stat().st_mtime)
    EXPERIMENT_ID = OUT_DIR.name

PLOT_DIR = OUT_DIR / "plots_from_notebook"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))

print("experiment_id:", EXPERIMENT_ID)
print("diretório:", OUT_DIR.relative_to(ROOT))
print("modelos:", manifest["models"])
print("datasets:", manifest["datasets"])
print("pipeline:", manifest.get("pipeline_version"))
"""
    ),
    code(
        r"""
results_df = pd.read_csv(OUT_DIR / "analysis" / "all_outcomes.csv")
threshold_df = pd.read_csv(OUT_DIR / "analysis" / "thresholds_calibration.csv")
curves_df = pd.read_csv(OUT_DIR / "analysis" / "effect_curves_calibration.csv")
policy_df = pd.read_csv(OUT_DIR / "analysis" / "heldout_policies.csv")
transition_df = pd.read_csv(OUT_DIR / "analysis" / "decision_transitions.csv")

dedup_path = OUT_DIR / "retrieval" / "dedup_audit.csv"
truncation_path = OUT_DIR / "retrieval" / "embedding_truncation.csv"
pairs_path = OUT_DIR / "retrieval" / "pairs.csv"
dedup_df = pd.read_csv(dedup_path) if dedup_path.exists() else pd.DataFrame()
truncation_df = pd.read_csv(truncation_path) if truncation_path.exists() else pd.DataFrame()
pairs_df = pd.read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame()

print(f"{len(results_df):,} resultados individuais")
print(f"{len(threshold_df):,} estimativas de threshold")
print(f"{len(policy_df):,} avaliações holdout")
display(threshold_df.head())
"""
    ),
    md(
        r"""
## Auditoria dos dados e da recuperação

Antes de interpretar os thresholds, confira quantos exemplos duplicados foram
removidos, quanto texto ultrapassou o limite do encoder e quais similaridades
foram realmente alcançadas em cada braço.
"""
    ),
    code(
        r"""
if not dedup_df.empty:
    display(dedup_df)
if not truncation_df.empty:
    display(truncation_df)
if not pairs_df.empty:
    retrieval_summary = (
        pairs_df.groupby(["dataset", "arm", "level"])["similarity"]
        .agg(["count", "min", "median", "max"]).reset_index()
    )
    display(retrieval_summary)
"""
    ),
    md(
        r"""
## Curvas por modelo, dataset e profundidade

Linha contínua: ganho pareado estimado na calibração. Faixa azul: IC 95% do
pool `all`. Linhas verticais: thresholds identificados. O painel direito usa
somente o teste holdout.
"""
    ),
    code(
        r"""
POOL_COLORS = {"all": "tab:blue", "errors": "tab:red", "correct": "tab:green"}
POLICY_LABELS = {
    "always_top": "sempre top-1",
    "threshold_top": "política threshold",
    "placebo": "placebo",
}


def plot_result(model, dataset, depth, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.axhline(0, color="black", linewidth=1)
    for pool, color in POOL_COLORS.items():
        curve = curves_df[
            (curves_df["model"] == model) & (curves_df["dataset"] == dataset)
            & (curves_df["depth"] == depth) & (curves_df["pool"] == pool)
        ].sort_values("similarity")
        if curve.empty:
            continue
        ax.plot(curve["similarity"], curve["effect"], color=color, label=pool)
        if pool == "all":
            ax.fill_between(
                curve["similarity"].to_numpy(), curve["ci_low"].to_numpy(),
                curve["ci_high"].to_numpy(), color=color, alpha=0.18,
            )
        estimate = threshold_df[
            (threshold_df["model"] == model) & (threshold_df["dataset"] == dataset)
            & (threshold_df["depth"] == depth) & (threshold_df["pool"] == pool)
        ]
        if not estimate.empty and pd.notna(estimate.iloc[0]["threshold"]):
            ax.axvline(estimate.iloc[0]["threshold"], color=color, linestyle="--", alpha=0.7)
    ax.set_xlabel("similaridade da memória")
    ax.set_ylabel("ganho pareado de acurácia")
    ax.set_title("Calibração: efeito vs. similaridade")
    ax.legend(title="origem da memória")

    ax2 = axes[1]
    heldout = policy_df[
        (policy_df["model"] == model) & (policy_df["dataset"] == dataset)
        & (policy_df["depth"] == depth) & (policy_df["pool"] == "all")
    ].copy()
    order = ["always_top", "threshold_top", "placebo"]
    heldout["order"] = heldout["policy"].map({name: i for i, name in enumerate(order)})
    heldout = heldout.sort_values("order")
    if not heldout.empty:
        x = np.arange(len(heldout))
        y = heldout["difference"].to_numpy()
        low = y - heldout["ci_low"].to_numpy()
        high = heldout["ci_high"].to_numpy() - y
        colors = ["gray" if p == "placebo" else "tab:orange" for p in heldout["policy"]]
        ax2.bar(x, y, color=colors, alpha=0.85)
        ax2.errorbar(x, y, yerr=np.vstack([low, high]), fmt="none", color="black", capsize=4)
        ax2.set_xticks(x, [POLICY_LABELS[p] for p in heldout["policy"]], rotation=20)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("ganho de acurácia no teste")
    ax2.set_title("Teste holdout")

    fig.suptitle(f"{model} | {dataset} | {depth} | uma memória")
    fig.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / f"{model}__{dataset}__{depth}.png", dpi=170, bbox_inches="tight")
    plt.show()


for model in manifest["models"]:
    for dataset in manifest["datasets"]:
        for depth in manifest["depths"]:
            plot_result(model, dataset, depth)
"""
    ),
    md(
        r"""
## Visão geral dos thresholds

O forest plot mostra o corte pontual e seu intervalo bootstrap. Ausência de um
ponto significa que a calibração não identificou cruzamento sustentado ou não
tinha suporte suficiente.
"""
    ),
    code(
        r"""
overview = threshold_df[
    (threshold_df["pool"] == "all") & threshold_df["threshold"].notna()
].copy()
overview["label"] = (
    overview["model"] + " | " + overview["dataset"] + " | " + overview["depth"]
)
overview = overview.sort_values(["model", "depth", "threshold"])

if overview.empty:
    print("Nenhum threshold foi identificado no pool all.")
else:
    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(overview))))
    y = np.arange(len(overview))
    center = overview["threshold"].to_numpy()
    left = center - overview["threshold_ci_low"].to_numpy()
    right = overview["threshold_ci_high"].to_numpy() - center
    valid = np.isfinite(left) & np.isfinite(right) & (left >= 0) & (right >= 0)
    ax.scatter(center, y, color="tab:blue", zorder=3)
    if valid.any():
        ax.errorbar(center[valid], y[valid], xerr=np.vstack([left[valid], right[valid]]),
                    fmt="none", color="tab:blue", capsize=3)
    ax.set_yticks(y, overview["label"])
    ax.set_xlabel("threshold de similaridade")
    ax.set_title("Thresholds aprendidos na calibração — pool all")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "threshold_forest.png", dpi=180, bbox_inches="tight")
    plt.show()

display(overview[[
    "model", "dataset", "depth", "threshold", "threshold_ci_low",
    "threshold_ci_high", "threshold_identification_rate",
    "confident_help_threshold", "confident_harm_region_start",
]])
"""
    ),
    md(
        r"""
## Ganho holdout em todos os datasets

Este painel compara as políticas usando dados que não participaram da escolha
do threshold. Células sem `threshold_top` indicam que nenhum corte utilizável
foi identificado na calibração.
"""
    ),
    code(
        r"""
heldout_threshold = policy_df[
    (policy_df["policy"] == "threshold_top") & (policy_df["pool"] == "all")
].copy()
heldout_threshold["column"] = heldout_threshold["model"] + " | " + heldout_threshold["depth"]
pivot = heldout_threshold.pivot(index="dataset", columns="column", values="difference")

if not pivot.empty:
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(pivot.columns)), 4.8))
    values = pivot.to_numpy()
    image = ax.imshow(values, cmap="RdBu", vmin=-max(0.01, np.nanmax(np.abs(values))),
                      vmax=max(0.01, np.nanmax(np.abs(values))), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{100*values[i, j]:+.1f} pp", ha="center", va="center", fontsize=9)
    ax.set_title("Ganho da política de threshold no teste holdout")
    fig.colorbar(image, ax=ax, label="diferença de acurácia")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "heldout_threshold_heatmap.png", dpi=180, bbox_inches="tight")
    plt.show()

display(heldout_threshold.sort_values("difference", ascending=False)[[
    "model", "dataset", "depth", "threshold", "memory_use_rate", "difference",
    "ci_low", "ci_high", "helped", "harmed", "mcnemar_p",
]])
"""
    ),
    md(
        r"""
## Como a memória alterou decisões

As barras usam todos os resultados válidos e mostram a fração que passou de
errada para correta, de correta para errada ou permaneceu igual em cada nível.
"""
    ),
    code(
        r"""
transition_rates = transition_df.copy()
transition_rates["total"] = transition_rates.groupby(
    ["model", "dataset", "depth", "arm", "level"]
)["n"].transform("sum")
transition_rates["rate"] = transition_rates["n"] / transition_rates["total"]

for model in manifest["models"]:
    for depth in manifest["depths"]:
        sub = transition_rates[
            (transition_rates["model"] == model)
            & (transition_rates["depth"] == depth)
            & (transition_rates["arm"] == "retrieved")
        ]
        if sub.empty:
            continue
        table = sub.pivot_table(
            index=["dataset", "level"], columns="transition", values="rate", fill_value=0
        )
        table = table.reindex(columns=["helped", "harmed", "unchanged"], fill_value=0)
        fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(table))))
        y = np.arange(len(table))
        left = np.zeros(len(table))
        colors = {"helped": "tab:green", "harmed": "tab:red", "unchanged": "lightgray"}
        for name in table.columns:
            ax.barh(y, table[name], left=left, label=name, color=colors[name])
            left += table[name].to_numpy()
        ax.set_yticks(y, [f"{a} | {b}" for a, b in table.index])
        ax.set_xlabel("fração das decisões")
        ax.set_xlim(0, 1)
        ax.set_title(f"Mudanças de decisão — {model} | {depth}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"transitions__{model}__{depth}.png", dpi=170, bbox_inches="tight")
        plt.show()
"""
    ),
    md(
        r"""
## Resumo automático

Uma política é destacada como benefício/dano apenas quando todo o IC 95% fica
acima/abaixo de zero no teste holdout. O resultado final continua devendo ser
interpretado em conjunto com placebo, estabilidade do threshold e replicação
entre modelos.
"""
    ),
    code(
        r"""
def pp(value):
    return f"{100*value:+.1f} pp"


evaluated = policy_df[policy_df["policy"] == "threshold_top"].copy()
positive = evaluated[evaluated["ci_low"] > 0].sort_values("difference", ascending=False)
negative = evaluated[evaluated["ci_high"] < 0].sort_values("difference")

print("BENEFÍCIOS COM IC 95% ACIMA DE ZERO")
if positive.empty:
    print("- Nenhum.")
for row in positive.itertuples():
    print(
        f"- {row.model}/{row.dataset}/{row.depth}/{row.pool}: {pp(row.difference)} "
        f"(IC {pp(row.ci_low)} a {pp(row.ci_high)}), threshold={row.threshold:.3f}"
    )

print("\nDANOS COM IC 95% ABAIXO DE ZERO")
if negative.empty:
    print("- Nenhum.")
for row in negative.itertuples():
    print(
        f"- {row.model}/{row.dataset}/{row.depth}/{row.pool}: {pp(row.difference)} "
        f"(IC {pp(row.ci_low)} a {pp(row.ci_high)})"
    )

print("\nPlots salvos em:", PLOT_DIR.relative_to(ROOT))
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "venv (3.10.12)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"rebuilt {TARGET}")
