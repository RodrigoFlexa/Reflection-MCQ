"""Auditable, non-destructive outcome filtering shared by scripts and notebooks."""
from __future__ import annotations

from collections import defaultdict


def outcome_flags(row: dict) -> list[str]:
    flags = set(row.get("audit_flags") or [])
    method = str(row.get("eval_method") or "")
    reason = str(row.get("finish_reason") or "")
    if row.get("correct") is None:
        flags.add("unresolved")
    if "context" in method or "context" in reason or "token_limit_exceeded" in method:
        flags.add("context_exceeded")
    if "content_filter" in method or "content_filter" in reason:
        flags.add("content_filter")
    if "length" in method or "length" in reason:
        flags.add("length_exhausted")
    if "empty" in method or "empty" in reason:
        flags.add("empty_exhausted")
    if method.startswith("judge") or method == "unresolved":
        flags.add("judge_fallback")
    if row.get("reflection_clipped_for_context"):
        flags.add("reflection_clipped_for_context")
    for name in ("evaluation_generation", "source_answer_audit", "reflection_audit"):
        gen = row.get(name) or {}
        for flag in ("thinking_removed", "partial_think", "budget_reduced_for_context"):
            if gen.get(flag):
                flags.add(flag)
        if "length" in str(gen.get("finish_reason", "")):
            flags.add("length_exhausted")
    if row.get("embedding_truncated"):
        flags.add("embedding_truncated")
    return sorted(flags)


def compact_audit(gen: dict) -> dict:
    return {k: v for k, v in gen.items() if k not in {"raw_text", "text", "visible_text", "prompt_hash", "key"}}


def annotate_outcomes(rows: list[dict], pairs: list[dict], students: dict, teachers: dict) -> list[dict]:
    by_uid = {p["val_uid"]: p for p in pairs}
    annotated = []
    for original in rows:
        row = dict(original)
        pair = by_uid[row["val_uid"]]
        item = pair["validation_item"]
        row.update(eval_uid=row["val_uid"], eval_split=pair.get("eval_split", item["split"]),
                   race_subset=item.get("race_subset"), article_uid=item.get("article_uid"),
                   source_article_uid=pair["source_item"].get("article_uid"),
                   embedding_truncated=pair.get("embedding_truncated", False))
        if row["condition"] != "baseline":
            author, depth = row["condition"].split("_", 1)
            attempt = students.get(row["model"], {}).get(row["source_uid"], {})
            reflection = attempt if author == "self" else teachers.get(row["model"], {}).get(row["source_uid"], {})
            row["source_answer_audit"] = compact_audit(attempt.get("answer_generation", {}))
            row["source_answer_finish_reason"] = attempt.get("answer_finish_reason")
            row["reflection_audit"] = compact_audit(reflection.get("reflection_generations", {}).get(depth, {}))
            row["reflection_finish_reason"] = reflection.get("reflection_status", {}).get(depth)
        row["audit_flags"] = outcome_flags(row)
        annotated.append(row)
    return annotated


def filter_outcomes(rows: list[dict], *, exclude_flags=(), exclude_methods=(),
                    conditions=(), race_subsets=(), paired=False, resolved_only=False) -> tuple[list[dict], list[dict]]:
    """Keep the original denominator and optionally intersect conditions per model/item.

    Pairing is within a model/dataset, across the requested (or available) conditions.
    A resolved-only intersection is explicit; missing outcomes never become correct.
    """
    ids = [(r["model"], r["dataset"], r.get("eval_split", "validation"), r["val_uid"], r["condition"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model/dataset/split/item/condition outcomes")
    scoped = [r for r in rows if (not conditions or r["condition"] in conditions)
              and (not race_subsets or r["dataset"] != "race" or r.get("race_subset") in race_subsets)]
    def eligible(r):
        return (not set(exclude_flags).intersection(outcome_flags(r))
                and r.get("eval_method") not in exclude_methods
                and (not resolved_only or r.get("correct") is not None))
    kept = [dict(r) for r in scoped if eligible(r)]
    if paired:
        expected, present = defaultdict(set), defaultdict(set)
        for r in scoped:
            expected[(r["model"], r["dataset"], r.get("eval_split", "validation"))].add(r["condition"])
        for r in kept:
            group = (r["model"], r["dataset"], r.get("eval_split", "validation"))
            present[(*group, r["val_uid"])].add(r["condition"])
        kept = [r for r in kept if present[(r["model"], r["dataset"], r.get("eval_split", "validation"), r["val_uid"])]
                == expected[(r["model"], r["dataset"], r.get("eval_split", "validation"))]]
    originals, selected = defaultdict(list), defaultdict(list)
    for r in scoped:
        originals[(r["model"], r["dataset"], r.get("eval_split", "validation"), r["condition"])].append(r)
    for r in kept:
        selected[(r["model"], r["dataset"], r.get("eval_split", "validation"), r["condition"])].append(r)
    audit = []
    for (model, dataset, split, condition), group in sorted(originals.items()):
        subset = selected[(model, dataset, split, condition)]
        resolved = [r for r in subset if r.get("correct") is not None]
        correct = sum(bool(r["correct"]) for r in resolved)
        audit.append({"model": model, "dataset": dataset, "eval_split": split, "condition": condition,
                      "n_original": len(group), "n_selected": len(subset), "n_excluded": len(group)-len(subset),
                      "resolved": len(resolved), "coverage_original": len(resolved)/len(group),
                      "coverage_selected": len(resolved)/len(subset) if subset else None,
                      "accuracy": correct/len(resolved) if resolved else None,
                      "accuracy_all": correct/len(subset) if subset else None})
    return kept, audit
