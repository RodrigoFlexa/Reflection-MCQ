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
# 08 — Análise comparativa do threshold de similaridade

Este notebook **só lê os resultados**: não carrega modelos, não gera reflexões e
não refaz bootstrap. A análise compara `compact`, `simple`, `diagnostic`, `complex`
e `external_reflection` (quando disponível), sempre com **uma única memória**.

Além das curvas individuais, os painéis abaixo respondem quatro perguntas:

1. Em que similaridade surge um benefício estável?
2. O corte aprendido generaliza para o holdout e supera o placebo?
3. O ganho vem de corrigir mais respostas do que estragar respostas corretas?
4. Algum estilo parece melhor apenas porque suas gerações foram truncadas?
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
EXPERIMENT_ID = os.environ.get("RMCQ_THRESHOLD_EXPERIMENT_ID") or None
COMPARISON_POOLS = ["all", "errors"]
SAVE_FIGURES = True

REQUIRED = [
    "manifest.json",
    "analysis/all_outcomes.csv",
    "analysis/thresholds_calibration.csv",
    "analysis/effect_curves_calibration.csv",
    "analysis/heldout_policies.csv",
]


def is_complete(path):
    return all((path / relative).exists() for relative in REQUIRED)


if EXPERIMENT_ID:
    OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID
    if not is_complete(OUT_DIR):
        missing = [x for x in REQUIRED if not (OUT_DIR / x).exists()]
        raise FileNotFoundError(f"Execução {EXPERIMENT_ID} incompleta; faltam: {missing}")
else:
    candidates = [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and is_complete(p)]
    if not candidates:
        raise FileNotFoundError("Nenhuma execução concluída foi encontrada.")
    OUT_DIR = max(candidates, key=lambda p: (p / "analysis/heldout_policies.csv").stat().st_mtime)
    EXPERIMENT_ID = OUT_DIR.name

# Diretório novo: não sobrescreve os plots da análise anterior.
PLOT_DIR = OUT_DIR / "plots_comparison_v2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))

preferred_depths = ["compact", "simple", "diagnostic", "complex", "external_reflection"]
available_depths = list(manifest.get("analysis_depths", manifest.get("depths", [])))
DEPTH_ORDER = [d for d in preferred_depths if d in available_depths]
DEPTH_ORDER += [d for d in available_depths if d not in DEPTH_ORDER]
MODELS = list(manifest["models"])
DATASETS = list(manifest["datasets"])
DEPTH_COLORS = dict(zip(DEPTH_ORDER, plt.cm.viridis(np.linspace(0.12, 0.88, len(DEPTH_ORDER)))))
if "external_reflection" in DEPTH_COLORS:
    DEPTH_COLORS["external_reflection"] = "tab:purple"
DEPTH_LABELS = {"external_reflection": "external (GPT)"}

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.2})


def finish(fig, filename):
    fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(PLOT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def short_path(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


print("experiment_id:", EXPERIMENT_ID)
print("diretório:", short_path(OUT_DIR))
print("modelos:", MODELS)
print("datasets:", DATASETS)
print("reflexões:", DEPTH_ORDER)
"""
    ),
    code(
        r"""
results_df = pd.read_csv(OUT_DIR / "analysis" / "all_outcomes.csv")
threshold_df = pd.read_csv(OUT_DIR / "analysis" / "thresholds_calibration.csv")
curves_df = pd.read_csv(OUT_DIR / "analysis" / "effect_curves_calibration.csv")
policy_df = pd.read_csv(OUT_DIR / "analysis" / "heldout_policies.csv")

optional = {}
for name, relative in {
    "transitions": "analysis/decision_transitions.csv",
    "dedup": "retrieval/dedup_audit.csv",
    "truncation": "retrieval/embedding_truncation.csv",
    "pairs": "retrieval/pairs.csv",
}.items():
    path = OUT_DIR / relative
    optional[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()

print(f"{len(results_df):,} resultados individuais")
print(f"{len(threshold_df):,} estimativas de threshold")
print(f"{len(policy_df):,} avaliações holdout")
display(threshold_df.groupby(["depth", "pool"], dropna=False)["threshold"].agg(["count", "median"]))
"""
    ),
    md(
        r"""
## 1. Qualidade operacional das reflexões

O gráfico não tenta julgar semanticamente a reflexão. Ele verifica um confundidor
importante em modelos pequenos: tamanho da saída e término por limite de tokens.
Uma variante muito truncada não está sendo comparada em condições equivalentes.
"""
    ),
    code(
        r"""
quality_rows = []
for model in MODELS:
    for depth in DEPTH_ORDER:
        path = OUT_DIR / "generations" / model / f"reflections_{depth}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        words = [len(str(r.get("text", "")).split()) for r in records]
        tokens = [r.get("completion_tokens", np.nan) for r in records]
        reasons = [str(r.get("finish_reason", "")).lower() for r in records]
        quality_rows.append({
            "model": model, "depth": depth, "n": len(records),
            "mean_words": np.mean(words), "mean_tokens": np.nanmean(tokens),
            "truncation_rate": np.mean([x in {"length", "max_tokens"} for x in reasons]),
        })

quality_df = pd.DataFrame(quality_rows)
if quality_df.empty:
    print("Caches de reflexão não foram encontrados; auditoria ignorada.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / max(1, len(MODELS))
    x = np.arange(len(DEPTH_ORDER))
    for i, model in enumerate(MODELS):
        sub = quality_df[quality_df.model == model].set_index("depth").reindex(DEPTH_ORDER)
        offset = (i - (len(MODELS) - 1) / 2) * width
        axes[0].bar(x + offset, sub["mean_tokens"], width, label=model)
        axes[1].bar(x + offset, 100 * sub["truncation_rate"], width, label=model)
    axes[0].set(title="Comprimento médio", ylabel="tokens de saída")
    axes[1].set(title="Término por limite", ylabel="reflexões truncadas (%)")
    for ax in axes:
        ax.set_xticks(x, [DEPTH_LABELS.get(d, d) for d in DEPTH_ORDER], rotation=20)
        ax.legend()
    finish(fig, "01_reflection_generation_audit.png")
    display(quality_df.sort_values(["model", "depth"]))
"""
    ),
    md(
        r"""
## 2. Cobertura da recuperação

Essas tabelas mostram duplicatas, truncamento do encoder e a faixa de
similaridades disponível. Um threshold próximo do extremo observado deve ser
interpretado com cautela porque possui pouco suporte empírico.
"""
    ),
    code(
        r"""
if not optional["dedup"].empty:
    display(optional["dedup"])
if not optional["truncation"].empty:
    display(optional["truncation"])
if not optional["pairs"].empty:
    retrieval_summary = (
        optional["pairs"].groupby(["dataset", "arm", "level"])["similarity"]
        .agg(["count", "min", "median", "max"]).reset_index()
    )
    display(retrieval_summary)
"""
    ),
    md(
        r"""
## 3. Curvas de calibração — reflexões na mesma escala

mesmos limites, tornando diferenças de formato e estabilidade visíveis. A linha
Cada figura fixa modelo, dataset e origem da memória. Todos os painéis usam os
mesmos limites, tornando diferenças de formato, gerador e estabilidade visíveis. A linha
mesmos limites, tornando diferenças de formato e estabilidade visíveis. A linha
vertical é o threshold aprendido; a faixa é o IC 95% do efeito pareado.
"""
    ),
    code(
        r"""
def plot_calibration_dashboard(model, dataset, pool):
    ncols = 2
    nrows = int(np.ceil(len(DEPTH_ORDER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.4 * nrows), sharex=True,
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    selected = curves_df[
        (curves_df.model == model) & (curves_df.dataset == dataset)
        & (curves_df.pool == pool) & (curves_df.depth.isin(DEPTH_ORDER))
    ]
    if selected.empty:
        plt.close(fig)
        return
    y_values = selected[["ci_low", "ci_high"]].to_numpy(float)
    finite = y_values[np.isfinite(y_values)]
    y_limit = max(0.05, np.max(np.abs(finite)) * 1.08) if len(finite) else 0.1
    for ax, depth in zip(axes, DEPTH_ORDER):
        curve = selected[selected.depth == depth].sort_values("similarity")
        ax.axhline(0, color="black", lw=1)
        if curve.empty:
            ax.set_title(f"{depth} — sem dados")
            continue
        x = curve.similarity.to_numpy(float)
        effect = curve.effect.to_numpy(float)
        ax.plot(x, effect, color=DEPTH_COLORS[depth], lw=2)
        ax.fill_between(x, curve.ci_low.to_numpy(float), curve.ci_high.to_numpy(float),
                        color=DEPTH_COLORS[depth], alpha=0.20)
        estimate = threshold_df[
            (threshold_df.model == model) & (threshold_df.dataset == dataset)
            & (threshold_df.depth == depth) & (threshold_df.pool == pool)
        ]
        if not estimate.empty and pd.notna(estimate.iloc[0].threshold):
            t = float(estimate.iloc[0].threshold)
            rate = estimate.iloc[0].get("threshold_identification_rate", np.nan)
            ax.axvline(t, color="tab:red", ls="--", lw=1.5)
            ax.text(0.03, 0.95, f"t={t:.3f} | ident.={rate:.0%}", transform=ax.transAxes,
                    va="top", fontsize=9)
        ax.set_title(DEPTH_LABELS.get(depth, depth))
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel("similaridade")
        ax.set_ylabel("efeito na acurácia")
    for ax in axes[len(DEPTH_ORDER):]:
        ax.axis("off")
    fig.suptitle(f"Calibração | {model} | {dataset} | pool={pool}", fontsize=14)
    finish(fig, f"02_calibration__{model}__{dataset}__{pool}.png")


for model in MODELS:
    for dataset in DATASETS:
        for pool in COMPARISON_POOLS:
            plot_calibration_dashboard(model, dataset, pool)
"""
    ),
    md(
        r"""
## 4. Onde os thresholds aparecem?

Cada célula traz `threshold / taxa de identificação no bootstrap`. O primeiro
número diz **onde** usar a reflexão; o segundo diz quão estável foi encontrar
esse ponto. Células vazias significam que não houve corte sustentado.
"""
    ),
    code(
        r"""
def threshold_heatmap(model, pool):
    sub = threshold_df[(threshold_df.model == model) & (threshold_df.pool == pool)]
    values = sub.pivot(index="dataset", columns="depth", values="threshold").reindex(
        index=DATASETS, columns=DEPTH_ORDER
    )
    rates = sub.pivot(index="dataset", columns="depth", values="threshold_identification_rate").reindex(
        index=DATASETS, columns=DEPTH_ORDER
    )
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(DEPTH_ORDER)), max(4, 0.8 * len(DATASETS))))
    masked = np.ma.masked_invalid(values.to_numpy(float))
    image = ax.imshow(masked, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DEPTH_ORDER)), [DEPTH_LABELS.get(d, d) for d in DEPTH_ORDER], rotation=20)
    ax.set_yticks(range(len(DATASETS)), DATASETS)
    for i in range(len(DATASETS)):
        for j in range(len(DEPTH_ORDER)):
            t, rate = values.iloc[i, j], rates.iloc[i, j]
            label = "—" if pd.isna(t) else f"{t:.3f}\n{rate:.0%}"
            color = "white" if pd.notna(t) and (t < 0.28 or t > 0.72) else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    ax.set_title(f"Threshold / estabilidade bootstrap | {model} | pool={pool}")
    fig.colorbar(image, ax=ax, label="threshold de similaridade")
    finish(fig, f"03_threshold_map__{model}__{pool}.png")


for model in MODELS:
    for pool in COMPARISON_POOLS:
        threshold_heatmap(model, pool)
"""
    ),
    md(
        r"""
## 5. O threshold generaliza? — efeito no holdout

O forest plot é a evidência principal: usa apenas perguntas que não escolheram
o corte. Pontos à direita ajudam; à esquerda atrapalham. O IC precisa ficar
inteiramente de um lado do zero para uma conclusão confirmatória simples.
"""
    ),
    code(
        r"""
def holdout_forest(pool):
    selected = policy_df[(policy_df.policy == "threshold_top") & (policy_df.pool == pool)].copy()
    if selected.empty:
        print(f"Sem política threshold_top para pool={pool}")
        return
    fig, axes = plt.subplots(1, len(MODELS), figsize=(8 * len(MODELS),
                             max(6, 0.42 * len(DATASETS) * len(DEPTH_ORDER))), squeeze=False)
    for ax, model in zip(axes.ravel(), MODELS):
        sub = selected[selected.model == model].copy()
        sub["dataset"] = pd.Categorical(sub.dataset, DATASETS, ordered=True)
        sub["depth"] = pd.Categorical(sub.depth, DEPTH_ORDER, ordered=True)
        sub = sub.sort_values(["dataset", "depth"])
        y = np.arange(len(sub))
        center = 100 * sub.difference.to_numpy(float)
        low = center - 100 * sub.ci_low.to_numpy(float)
        high = 100 * sub.ci_high.to_numpy(float) - center
        colors = [DEPTH_COLORS[str(d)] for d in sub.depth]
        ax.axvline(0, color="black", lw=1)
        ax.errorbar(center, y, xerr=np.vstack([low, high]), fmt="none", color="gray", capsize=3)
        ax.scatter(center, y, c=colors, s=42, zorder=3)
        ax.set_yticks(y, [f"{r.dataset} | {r.depth}" for r in sub.itertuples()])
        ax.invert_yaxis()
        ax.set_xlabel("ganho de acurácia (pontos percentuais)")
        ax.set_title(model)
    fig.suptitle(f"Política de threshold no holdout | pool={pool}", fontsize=14)
    finish(fig, f"04_holdout_forest__{pool}.png")


for pool in COMPARISON_POOLS:
    holdout_forest(pool)
"""
    ),
    code(
        r"""
def holdout_heatmap(model, pool):
    sub = policy_df[(policy_df.model == model) & (policy_df.pool == pool)
                    & (policy_df.policy == "threshold_top")]
    values = (100 * sub.pivot(index="dataset", columns="depth", values="difference")).reindex(
        index=DATASETS, columns=DEPTH_ORDER
    )
    if values.empty:
        return
    arr = values.to_numpy(float)
    finite = arr[np.isfinite(arr)]
    limit = max(1, np.max(np.abs(finite))) if len(finite) else 1
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(DEPTH_ORDER)), max(4, 0.8 * len(DATASETS))))
    image = ax.imshow(np.ma.masked_invalid(arr), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(DEPTH_ORDER)), [DEPTH_LABELS.get(d, d) for d in DEPTH_ORDER], rotation=20)
    ax.set_yticks(range(len(DATASETS)), DATASETS)
    for i in range(len(DATASETS)):
        for j in range(len(DEPTH_ORDER)):
            value = arr[i, j]
            ax.text(j, i, "—" if not np.isfinite(value) else f"{value:+.1f} pp",
                    ha="center", va="center", fontsize=9)
    ax.set_title(f"Ganho holdout da política | {model} | pool={pool}")
    fig.colorbar(image, ax=ax, label="pontos percentuais")
    finish(fig, f"05_holdout_map__{model}__{pool}.png")


for model in MODELS:
    for pool in COMPARISON_POOLS:
        holdout_heatmap(model, pool)
"""
    ),
    md(
        r"""
## 6. Threshold versus placebo

Cada ponto compara, na mesma combinação, o efeito da memória selecionada pelo
threshold (eixo vertical) ao efeito do placebo (horizontal). Acima da diagonal,
a recuperação por similaridade foi melhor do que uma memória sem relação.
"""
    ),
    code(
        r"""
threshold_policy = policy_df[policy_df.policy == "threshold_top"].copy()
placebo_policy = policy_df[policy_df.policy == "placebo"].copy()
join_keys = ["model", "dataset", "depth", "pool"]
comparison = threshold_policy.merge(
    placebo_policy[join_keys + ["difference"]], on=join_keys, how="inner",
    suffixes=("_threshold", "_placebo")
)

for pool in COMPARISON_POOLS:
    sub_pool = comparison[comparison.pool == pool]
    if sub_pool.empty:
        continue
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6), squeeze=False)
    all_values = 100 * sub_pool[["difference_placebo", "difference_threshold"]].to_numpy(float)
    finite = all_values[np.isfinite(all_values)]
    limit = max(3, np.max(np.abs(finite)) * 1.10) if len(finite) else 3
    for ax, model in zip(axes.ravel(), MODELS):
        sub = sub_pool[sub_pool.model == model]
        ax.plot([-limit, limit], [-limit, limit], color="black", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        for row in sub.itertuples():
            x, y = 100 * row.difference_placebo, 100 * row.difference_threshold
            ax.scatter(x, y, color=DEPTH_COLORS[row.depth], s=48)
            ax.annotate(row.dataset, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), title=model,
               xlabel="efeito do placebo (pp)", ylabel="efeito do threshold (pp)")
    fig.suptitle(f"Recuperação por similaridade versus placebo | pool={pool}", fontsize=14)
    finish(fig, f"06_threshold_vs_placebo__{pool}.png")
"""
    ),
    md(
        r"""
## 7. GPT externo versus a reflexão diagnóstica do próprio estudante

O prompt de `external_reflection` é propositalmente o mesmo de `diagnostic`.
Assim, este painel compara principalmente **quem escreveu a memória**: no eixo
horizontal, o próprio modelo pequeno; no vertical, o GPT externo. Pontos acima
da diagonal favorecem a reflexão externa naquela combinação.
"""
    ),
    code(
        r"""
direct = policy_df[
    (policy_df.policy == "threshold_top")
    & (policy_df.depth.isin(["diagnostic", "external_reflection"]))
].pivot_table(
    index=["model", "dataset", "pool"], columns="depth", values="difference"
).dropna(subset=["diagnostic", "external_reflection"], how="any")

for pool in COMPARISON_POOLS:
    sub_pool = direct.reset_index()
    sub_pool = sub_pool[sub_pool.pool == pool]
    if sub_pool.empty:
        continue
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6), squeeze=False)
    values = 100 * sub_pool[["diagnostic", "external_reflection"]].to_numpy(float)
    limit = max(3, np.max(np.abs(values)) * 1.12)
    for ax, model in zip(axes.ravel(), MODELS):
        sub = sub_pool[sub_pool.model == model]
        ax.plot([-limit, limit], [-limit, limit], color="black", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        for row in sub.itertuples():
            x, y = 100 * row.diagnostic, 100 * row.external_reflection
            ax.scatter(x, y, color="tab:purple", s=55)
            ax.annotate(row.dataset, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), title=model,
               xlabel="diagnostic do estudante (pp)", ylabel="external GPT (pp)")
    fig.suptitle(f"Mesmo prompt: estudante versus GPT externo | pool={pool}", fontsize=14)
    finish(fig, f"07_external_vs_diagnostic__{pool}.png")
"""
    ),
    md(
        r"""
## 8. Como a reflexão ajuda ou atrapalha

O eixo vertical é a fração de respostas corrigidas; o horizontal é a fração de
respostas corretas que foram estragadas. Acima da diagonal há saldo positivo.
O tamanho do ponto representa quantas perguntas receberam memória.
"""
    ),
    code(
        r"""
selected = policy_df[policy_df.policy == "threshold_top"].copy()
selected["helped_rate"] = selected.helped / selected.n
selected["harmed_rate"] = selected.harmed / selected.n

for pool in COMPARISON_POOLS:
    sub_pool = selected[selected.pool == pool]
    if sub_pool.empty:
        continue
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6), squeeze=False)
    limit = max(0.01, sub_pool[["helped_rate", "harmed_rate"]].max().max() * 1.12)
    for ax, model in zip(axes.ravel(), MODELS):
        sub = sub_pool[sub_pool.model == model]
        ax.plot([0, limit], [0, limit], color="black", ls="--", lw=1)
        for row in sub.itertuples():
            size = 35 + 100 * float(getattr(row, "memory_use_rate", 0) or 0)
            ax.scatter(row.harmed_rate, row.helped_rate, color=DEPTH_COLORS[row.depth], s=size, alpha=0.85)
            ax.annotate(row.dataset, (row.harmed_rate, row.helped_rate),
                        xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set(xlim=(0, limit), ylim=(0, limit), title=model,
               xlabel="fração prejudicada", ylabel="fração corrigida")
    fig.suptitle(f"Respostas corrigidas versus prejudicadas | pool={pool}", fontsize=14)
    finish(fig, f"08_helped_vs_harmed__{pool}.png")
"""
    ),
    md(
        r"""
## 9. Tabela final com controle de comparações múltiplas

Como há vários modelos, datasets e reflexões, também reportamos `q_bh`, o
p-valor de McNemar corrigido por Benjamini–Hochberg dentro de cada pool. A
ordenação prioriza efeito holdout, mas a interpretação deve combinar IC 95%,
`q_bh`, estabilidade do threshold e comparação com placebo.
"""
    ),
    code(
        r"""
def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


ranking = policy_df[policy_df.policy == "threshold_top"].copy()
ranking["q_bh"] = ranking.groupby("pool")["mcnemar_p"].transform(bh_adjust)
ranking["effect_pp"] = 100 * ranking.difference
ranking["ci_low_pp"] = 100 * ranking.ci_low
ranking["ci_high_pp"] = 100 * ranking.ci_high
ranking["conclusion"] = np.select(
    [ranking.ci_low > 0, ranking.ci_high < 0],
    ["benefício", "dano"], default="inconclusivo"
)
columns = [
    "model", "dataset", "depth", "pool", "threshold", "memory_use_rate",
    "effect_pp", "ci_low_pp", "ci_high_pp", "mcnemar_p", "q_bh", "conclusion",
]
display(ranking.sort_values(["pool", "effect_pp"], ascending=[True, False])[columns])

for pool in COMPARISON_POOLS:
    sub = ranking[ranking.pool == pool]
    benefit = sub[sub.ci_low > 0].sort_values("difference", ascending=False)
    harm = sub[sub.ci_high < 0].sort_values("difference")
    print(f"\nPOOL={pool}")
    print("Benefícios com IC 95% acima de zero:", len(benefit))
    for row in benefit.itertuples():
        print(f"- {row.model}/{row.dataset}/{row.depth}: {100*row.difference:+.1f} pp; "
              f"IC [{100*row.ci_low:+.1f}, {100*row.ci_high:+.1f}]; q={row.q_bh:.3g}")
    print("Danos com IC 95% abaixo de zero:", len(harm))
    for row in harm.itertuples():
        print(f"- {row.model}/{row.dataset}/{row.depth}: {100*row.difference:+.1f} pp; "
              f"IC [{100*row.ci_low:+.1f}, {100*row.ci_high:+.1f}]; q={row.q_bh:.3g}")

if not quality_df.empty:
    high_truncation = quality_df[quality_df.truncation_rate > 0.10]
    print("\nReflexões com mais de 10% de truncamento:")
    if high_truncation.empty:
        print("- Nenhuma.")
    else:
        for row in high_truncation.itertuples():
            print(f"- {row.model}/{row.depth}: {row.truncation_rate:.1%}")

print("\nPlots salvos em:", short_path(PLOT_DIR))
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
