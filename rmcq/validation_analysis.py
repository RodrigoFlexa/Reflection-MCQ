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


def plot_study(views, directory: Path, *, auto_ylim=True, show=True, models_per_row=2, datasets_per_row=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from rmcq.plotting import accuracy_ylim
    from rmcq.test_analysis import CONDITIONS
    colors = dict(zip(CONDITIONS, ["#222222", "#1f77b4", "#17becf", "#d62728", "#ff7f0e"]))
    labels = dict(zip(CONDITIONS, ["Baseline", "Self simples", "Self complexa", "Teacher simples", "Teacher complexa"]))
    plt.style.use("seaborn-v0_8-whitegrid")
    pool = pd.concat([v["thresholds"] for v in views.values()])
    for view, tables in views.items():
        grouped = list(tables["thresholds"].groupby("model"))
        if not grouped:
            continue
        n_cols = max(1, int(datasets_per_row if datasets_per_row is not None else models_per_row))
        for model, group in grouped:
            datasets = sorted(group.dataset.unique())
            n_rows = (len(datasets) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.6 * n_rows), squeeze=False)
            handles, legend_labels = [], []
            for index, dataset in enumerate(datasets):
                row = index // n_cols
                col = index % n_cols
                ax = axes[row, col]
                part = group.loc[group.dataset.eq(dataset) & group.threshold.ge(0)]
                shared = pool.loc[pool.model.eq(model) & pool.dataset.eq(dataset) & pool.threshold.ge(0)]
                for condition in CONDITIONS:
                    curve = part.loc[part.condition.eq(condition)]
                    if curve.empty:
                        continue
                    ax.plot(
                        curve.threshold,
                        curve["accuracy"],
                        label=labels[condition],
                        color=colors[condition],
                        marker=".",
                    )
                if not shared.empty:
                    ax.set_ylim(*(accuracy_ylim(shared["accuracy"]) if auto_ylim else (0, 1)))
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_title(f"{dataset} | Acurácia", fontsize=10)
                ax.set_xlabel("Threshold mínimo")
                if index == 0:
                    handles, legend_labels = ax.get_legend_handles_labels()
            for index in range(len(datasets), n_rows * n_cols):
                row = index // n_cols
                col = index % n_cols
                axes[row, col].axis("off")
            fig.suptitle(f"Validação | {view} | {model}")
            if handles:
                fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=5, fontsize=7)
            fig.tight_layout(rect=(0, 0, 1, .98))
            fig.savefig(directory / f"{view}_{model}_grid_accuracy.png", dpi=130, bbox_inches="tight")
            fig.savefig(directory / f"{view}_{model}_grid_accuracy.pdf", bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
