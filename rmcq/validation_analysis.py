"""Validation-only threshold study, including same-question baseline comparisons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rmcq.test_analysis import (GROUP, KEY, build_panel, apply_policy, summarize_policy,
                               average_datasets, select_validation_thresholds)

CLEAN_FLAGS = ["length_exhausted", "empty_exhausted", "context_exceeded", "content_filter",
               "partial_think", "reflection_clipped_for_context", "embedding_truncated"]
THRESHOLDS = [-1.0] + [i / 20 for i in range(21)]


def matched_summary(rows, groups=GROUP):
    summary = summarize_policy(rows, groups)
    rows = rows.copy()
    rows["baseline_selected"] = rows.baseline_score.fillna(0).where(rows.included)
    rows["baseline_unanswered"] = rows.included & rows.baseline_score.isna()
    rows["corrected"] = rows.included & rows.score.eq(1) & rows.baseline_selected.eq(0)
    rows["harmed"] = rows.included & rows.score.eq(0) & rows.baseline_selected.eq(1)
    matched = rows.groupby(groups, dropna=False, observed=True).agg(
        matched_baseline_accuracy=("baseline_selected", "mean"),
        n_baseline_unanswered=("baseline_unanswered", "sum"),
        n_corrected=("corrected", "sum"), n_harmed=("harmed", "sum"),
    ).reset_index()
    summary = summary.merge(matched, on=groups, validate="one_to_one")
    summary["delta_vs_baseline"] = summary.accuracy - summary.matched_baseline_accuracy
    return summary


def study(frame, *, fallback=False, thresholds=THRESHOLDS, clean_flags=CLEAN_FLAGS,
          min_n=30, min_coverage=.1):
    if set(frame.eval_split) != {"validation"}:
        raise ValueError("Threshold selection here must use validation only")
    views = {}
    for name, flags in (("completo", []), ("filtrado", clean_flags)):
        panel = build_panel(frame, flags)
        no_cut = matched_summary(apply_policy(panel, None, fallback))
        sweep = []
        for threshold in thresholds:
            table = matched_summary(apply_policy(panel, threshold, fallback))
            table["threshold"] = threshold
            sweep.append(table)
        sweep = pd.concat(sweep, ignore_index=True)
        # Disjoint intervals complement the cumulative threshold curves.
        band_rows = apply_policy(panel, None, fallback)
        band_rows["similarity_band"] = pd.cut(band_rows.similarity,
            [-np.inf, .6, .7, .8, .85, .9, .95, np.inf], right=False,
            labels=["<0.60", "[0.60,0.70)", "[0.70,0.80)", "[0.80,0.85)",
                    "[0.85,0.90)", "[0.90,0.95)", ">=0.95"])
        bands = matched_summary(band_rows, [*GROUP, "similarity_band"])
        chosen = select_validation_thresholds(sweep, min_n, min_coverage)
        views[name] = dict(accuracy=no_cut, thresholds=sweep, averages=average_datasets(sweep),
                           bands=bands, selected=chosen)
    return views


def export_study(views, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for view, tables in views.items():
        for name, table in tables.items():
            table.to_csv(directory / f"{view}_{name}.csv", index=False)


def plot_study(views, directory: Path, *, auto_ylim=True, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, FuncFormatter
    from rmcq.plotting import accuracy_ylim
    from rmcq.test_analysis import CONDITIONS
    colors = dict(zip(CONDITIONS, ["#222222", "#1f77b4", "#17becf", "#d62728", "#ff7f0e"]))
    labels = dict(zip(CONDITIONS, ["Baseline", "Self simples", "Self complexa", "Teacher simples", "Teacher complexa"]))
    plt.style.use("seaborn-v0_8-whitegrid")
    pool = pd.concat([v["thresholds"] for v in views.values()])
    for view, tables in views.items():
        for model, group in tables["thresholds"].groupby("model"):
            datasets = sorted(group.dataset.unique())
            fig, axes = plt.subplots(len(datasets), 3, figsize=(16, 3.2 * len(datasets)), squeeze=False)
            for i, dataset in enumerate(datasets):
                part = group.loc[group.dataset.eq(dataset) & group.threshold.ge(0)]
                shared = pool.loc[pool.model.eq(model) & pool.dataset.eq(dataset) & pool.threshold.ge(0)]
                for j, metric in enumerate(["accuracy", "delta_vs_baseline", "coverage"]):
                    ax = axes[i, j]
                    for condition in CONDITIONS:
                        if metric == "delta_vs_baseline" and condition == "baseline":
                            continue
                        curve = part.loc[part.condition.eq(condition)]
                        if curve.empty:
                            continue
                        ax.plot(curve.threshold, curve[metric], label=labels[condition], color=colors[condition], marker=".")
                    if metric == "delta_vs_baseline":
                        ax.axhline(0, color="black", lw=1)
                        maximum = shared[metric].abs().max()
                        maximum = max(.04, float(maximum) + .02) if pd.notna(maximum) else .04
                        ax.set_ylim(-maximum, maximum)
                    elif metric == "coverage":
                        ax.set_ylim(0, 1.02)
                    else:
                        ax.set_ylim(*(accuracy_ylim(shared[metric]) if auto_ylim else (0, 1)))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.1f}")
                                                if metric == "delta_vs_baseline" else PercentFormatter(1))
                    if metric == "delta_vs_baseline":
                        ax.set_ylabel("Diferença (p.p.)")
                    ax.set_title(f"{dataset} | " + {"accuracy": "Acurácia", "delta_vs_baseline": "Ganho vs. baseline nas mesmas questões", "coverage": "Cobertura"}[metric], fontsize=10)
                    ax.set_xlabel("Threshold mínimo")
                    if i == 0 and j == 0:
                        ax.legend(fontsize=8)
            fig.suptitle(f"Validação | {model} | {view}")
            fig.tight_layout(rect=(0, 0, 1, .98))
            fig.savefig(directory / f"{view}_{model}.png", dpi=130, bbox_inches="tight")
            fig.savefig(directory / f"{view}_{model}.pdf", bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
