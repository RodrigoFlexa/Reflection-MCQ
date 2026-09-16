"""Offline, auditable analysis of final MCQ outcomes (no model calls)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rmcq.analysis import outcome_flags
from rmcq.grid import LEGACY_TEACHER, canonical

KEY = ["model", "dataset", "val_uid"]
# O eixo da análise é o braço, não a condição. Com um professor só os dois
# coincidiam; com cinco, "teacher_simple" deixaria de identificar uma medida —
# cinco linhas diferentes disputariam o mesmo nome na mesma questão.
GROUP = ["model", "dataset", "arm"]
CONDITIONS = ["baseline", "self_simple", "self_complex", "teacher_simple", "teacher_complex"]
FAILURE_WORDS = ("exceeded", "exhausted", "content_filter", "unavailable", "not_generated", "unresolved", "error")

CONDITION_LABELS = {"baseline": "Baseline", "self_simple": "Self simples",
                    "self_complex": "Self complexa", "teacher_simple": "Teacher simples",
                    "teacher_complex": "Teacher complexa"}
CONDITION_COLORS = {"baseline": "#222222", "self_simple": "#1f77b4", "self_complex": "#17becf",
                    "teacher_simple": "#d62728", "teacher_complex": "#ff7f0e"}


def arm_of(condition: str, teacher_model=None) -> str:
    """Nome único de um braço experimental: a condição e, se houver, o professor."""
    if condition in ("teacher_simple", "teacher_complex") and teacher_model:
        return f"{condition}@{teacher_model}"
    return condition


def split_arm(arm: str) -> tuple[str, str | None]:
    condition, _, teacher = str(arm).partition("@")
    return condition, teacher or None


def arm_label(arm: str) -> str:
    condition, teacher = split_arm(arm)
    base = CONDITION_LABELS.get(condition, condition)
    return f"{base} ({teacher})" if teacher else base


def read_outcomes(path: Path, split: str) -> pd.DataFrame:
    """Stream large JSONL; retain metrics and audits, never response texts."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            actual = raw.get("eval_split") or split
            if actual != split:
                raise ValueError(f"Unexpected split {actual!r} in {path}; expected {split}")
            row = {k: raw.get(k) for k in [*KEY, "condition", "similarity", "correct", "source_uid", "race_subset"]}
            row["eval_split"] = actual
            # A família é a unidade de comparação; o checkpoint fica ao lado.
            row["model_checkpoint"] = raw.get("model_checkpoint") or raw.get("model")
            row["model"] = canonical(row["model"])
            teacher = canonical(raw.get("teacher_model"))
            if row["condition"] in ("teacher_simple", "teacher_complex") and not teacher:
                teacher = LEGACY_TEACHER
            row["teacher_model"] = teacher if row["condition"] in ("teacher_simple", "teacher_complex") else None
            row["arm"] = arm_of(row["condition"], row["teacher_model"])
            row["source_run"] = raw.get("source_run")
            row["pipeline_version"] = raw.get("pipeline_version")
            row["audit_flags"] = outcome_flags(raw)
            statuses = [str(raw.get(k) or "") for k in ("eval_method", "finish_reason", "reflection_finish_reason", "source_answer_finish_reason")]
            row["failure"] = any(word in status for word in FAILURE_WORDS for status in statuses)
            row["audit_detail_available"] = "reflection_audit" in raw or "evaluation_generation" in raw
            row["eval_method"] = raw.get("eval_method")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if frame[[*KEY, "condition", "arm"]].isna().any().any():
        raise ValueError(f"Missing identifiers in {path}")
    if frame.duplicated([*KEY, "arm"]).any():
        raise ValueError(f"Duplicate outcomes in {path}")
    frame["correct"] = pd.to_numeric(frame["correct"], errors="raise")
    if not frame["correct"].dropna().isin([0, 1]).all():
        raise ValueError("Correctness must be 0, 1 or null")
    frame["similarity"] = pd.to_numeric(frame["similarity"], errors="coerce")
    return frame


def load_run(root: Path, run_id: str, split: str, restore: bool = False, phase: str = "auto"):
    """Prefer consolidated outcomes, otherwise existing per-model / teacher files."""
    directory = root / "data/results/reflection_top1" / run_id
    if phase not in ("auto", "self", "full"):
        raise ValueError("phase must be auto, self or full")
    if phase == "self":
        directory = directory / "self_eval"
    consolidated = directory / "analysis/all_outcomes.jsonl"
    if not consolidated.exists() and restore:
        bundle = root / "experiment_handoff" / run_id / "finish"
        if (bundle / "bundle.json").exists():
            from rmcq.handoff import unpack
            unpack(root, bundle, run_id, "finish")
    if consolidated.exists():
        paths = [consolidated]
    elif phase in ("self", "full"):
        return pd.DataFrame(), []
    elif phase == "auto" and (directory / "self_eval/analysis/all_outcomes.jsonl").exists():
        paths = [directory / "self_eval/analysis/all_outcomes.jsonl"]
    else:
        paths = sorted((directory / "models").glob(f"*/{split}.jsonl"))
        teacher = root / "experiment_exchange" / run_id / "teacher" / f"{split}.jsonl"
        if teacher.exists():
            paths.append(teacher)
    if not paths:
        return pd.DataFrame(), []
    frame = pd.concat([read_outcomes(p, split) for p in paths], ignore_index=True)
    if frame.duplicated([*KEY, "arm"]).any():
        raise ValueError(f"Overlapping outcome files in run {run_id}")
    frame["run_id"] = run_id
    return frame, [str(p.relative_to(root)) for p in paths]


def ensure_arm(frame: pd.DataFrame) -> pd.DataFrame:
    """Garante a coluna do braço para quadros montados fora de `read_outcomes`."""
    if "arm" in frame:
        return frame
    frame = frame.copy()
    teacher = frame["teacher_model"] if "teacher_model" in frame else pd.Series(None, index=frame.index, dtype=object)
    frame["teacher_model"] = teacher
    frame["arm"] = [arm_of(c, t) for c, t in zip(frame.condition, teacher)]
    return frame


def build_panel(frame: pd.DataFrame, exclude_flags=()) -> pd.DataFrame:
    """Baseline defines the question universe; restore missing condition rows explicitly."""
    frame = ensure_arm(frame)
    baseline = frame.loc[frame.arm.eq("baseline")].copy()
    if baseline.empty:
        raise ValueError("No baseline available")
    if frame.merge(baseline[KEY], on=KEY, how="left", indicator=True)["_merge"].eq("left_only").any():
        raise ValueError("An experimental question has no baseline row")
    if frame.groupby(KEY).similarity.nunique().gt(1).any():
        raise ValueError("Similarity differs across conditions for the same question")
    available = frame[["model", "dataset", "condition", "teacher_model", "arm"]].drop_duplicates()
    grid = baseline[KEY + ["similarity", "race_subset", "correct", "failure"]].rename(
        columns={"correct": "baseline_correct", "failure": "baseline_failure"})
    grid = grid.merge(available, on=["model", "dataset"], how="inner")
    original = frame[[*KEY, "arm", "correct", "failure", "audit_flags", "audit_detail_available"]]
    panel = grid.merge(original, on=[*KEY, "arm"], how="left", validate="one_to_one", indicator=True)
    panel["missing_row"] = panel.pop("_merge").eq("left_only")
    panel["flagged"] = panel.audit_flags.map(lambda v: bool(set(v if isinstance(v, list) else []).intersection(exclude_flags)))
    panel["usable"] = panel.correct.notna() & panel.failure.eq(False) & ~panel.flagged & ~panel.missing_row
    panel["baseline_score"] = panel.baseline_correct.where(~panel.baseline_failure, np.nan)
    return panel


def apply_policy(panel: pd.DataFrame, threshold=None, fallback: bool = True) -> pd.DataFrame:
    """Rejected / unavailable reflections reuse this model's same-question baseline.

    With fallback enabled unresolved baselines remain in the denominator as failures.
    Without fallback, only answered eligible rows remain; there is no paired intersection.
    Thresholds never gate baseline. Equality is accepted (similarity >= threshold).
    """
    out = panel.copy()
    base = out.arm.eq("baseline")
    above = pd.Series(True, index=out.index) if threshold is None else out.similarity.ge(threshold)
    out["below_threshold"] = ~base & ~above
    use = out.usable & (base | above)
    # Baseline is the unchanged reference, even in the optional audit-filtered view.
    use = use.where(~base, out.baseline_score.notna())
    out["fallback_used"] = ~base & ~use & fallback
    out["included"] = use | fallback
    out["reflection_used"] = ~base & use
    score = out.correct.where(use)
    score = score.where(~out.fallback_used, out.baseline_score)
    score = score.where(~base, out.baseline_score)
    out["unanswered"] = out.included & score.isna()
    out["score"] = score.fillna(0).where(out.included)
    return out


def summarize_policy(rows: pd.DataFrame, groups=GROUP) -> pd.DataFrame:
    result = rows.groupby(groups, dropna=False, observed=True).agg(
        n_total=("included", "size"), n=("included", "sum"), correct=("score", "sum"),
        n_reflection=("reflection_used", "sum"), n_fallback=("fallback_used", "sum"),
        n_unanswered=("unanswered", "sum"), n_below_threshold=("below_threshold", "sum"),
    ).reset_index()
    result["accuracy"] = result.correct / result.n.replace(0, np.nan)
    result["coverage"] = result.n / result.n_total
    result["fallback_rate"] = result.n_fallback / result.n.replace(0, np.nan)
    return result


def threshold_sweep(panel, thresholds, fallback=True):
    summaries = []
    for threshold in thresholds:
        summary = summarize_policy(apply_policy(panel, threshold, fallback))
        summary["threshold"] = threshold
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def average_datasets(summary: pd.DataFrame) -> pd.DataFrame:
    """Macro treats datasets equally; micro treats selected questions equally."""
    keys = [k for k in ["view", "model", "arm", "threshold"] if k in summary]
    result = summary.groupby(keys, dropna=False).agg(
        macro_accuracy=("accuracy", "mean"), correct=("correct", "sum"), n=("n", "sum"),
        n_total=("n_total", "sum"), n_datasets=("dataset", "nunique"),
        n_datasets_scored=("accuracy", "count"), n_fallback=("n_fallback", "sum"),
    ).reset_index()
    result["micro_accuracy"] = result.correct / result.n.replace(0, np.nan)
    # Do not silently increase macro accuracy by dropping a dataset with zero coverage.
    result.loc[result.n_datasets_scored.ne(result.n_datasets), "macro_accuracy"] = np.nan
    result["coverage"] = result.n / result.n_total
    return result


def select_validation_thresholds(sweep, min_n=30, min_coverage=0.0):
    eligible = sweep.loc[~sweep.arm.eq("baseline") & sweep.n.ge(min_n)
                         & sweep.coverage.ge(min_coverage) & sweep.accuracy.notna()].copy()
    # Deterministic tie break: lower threshold, preserving wider applicability.
    return eligible.sort_values([*GROUP, "accuracy", "threshold"],
                                ascending=[True, True, True, False, True]).drop_duplicates(GROUP)


def transfer_thresholds(test_panel, selected, fallback=True):
    """Only transfer configurations actually selected using validation."""
    records = []
    for choice in selected.to_dict("records"):
        match = test_panel.model.eq(choice["model"]) & test_panel.dataset.eq(choice["dataset"])
        sub = test_panel.loc[match & test_panel.arm.isin(["baseline", choice["arm"]])]
        if sub.empty or not sub.arm.eq(choice["arm"]).any():
            continue
        scored = apply_policy(sub, choice["threshold"], fallback)
        experiment = scored.loc[scored.arm.eq(choice["arm"])]
        metric = summarize_policy(experiment).iloc[0].to_dict()
        # Matched baseline on the exact selected test questions, even without fallback.
        baseline = scored.loc[scored.arm.eq("baseline")].set_index(KEY)
        chosen_keys = pd.MultiIndex.from_frame(experiment.loc[experiment.included, KEY])
        matched = baseline.reindex(chosen_keys).baseline_score.fillna(0)
        metric.update(threshold=choice["threshold"], validation_accuracy=choice["accuracy"],
                      validation_n=choice["n"], matched_baseline_accuracy=matched.mean())
        metric["delta_vs_matched_baseline"] = metric["accuracy"] - metric["matched_baseline_accuracy"]
        records.append(metric)
    return pd.DataFrame(records)
