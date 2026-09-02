"""Utilities for the similarity-threshold experiment in notebook 08.

The functions here are deliberately model/backend independent so the
retrieval design and statistical analysis can be unit-tested without a GPU.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*\(?([A-H])\)?", re.IGNORECASE)


def normalize_stem(item: dict[str, Any]) -> str:
    """Normalized context + question key used to prevent split leakage."""
    text = f"{item.get('context') or ''}\n{item.get('question') or ''}"
    return " ".join(text.casefold().split())


def extract_final_answer(text: str) -> str | None:
    """Return the last explicitly formatted answer letter."""
    matches = _FINAL_ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def stable_fraction(value: str, seed: int = 42) -> float:
    """Stable pseudo-random number in [0, 1), independent of Python hashing."""
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def calibration_or_test(value: str, fraction: float = 0.60, seed: int = 42) -> str:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    return "calibration" if stable_fraction(value, seed) < fraction else "test"


def select_similarity_ladder(
    similarities: Sequence[float],
    source_ids: Sequence[str],
    targets: Sequence[float],
    *,
    seed_key: str,
    placebo_quantile: float = 0.20,
) -> list[dict[str, Any]]:
    """Select one unique source near each requested similarity, top-1, and placebo.

    The placebo is sampled deterministically from the bottom similarity tail.
    Every returned record represents exactly one source memory.
    """
    sims = np.asarray(similarities, dtype=float)
    ids = np.asarray(source_ids, dtype=object)
    if sims.ndim != 1 or len(sims) != len(ids) or len(sims) == 0:
        raise ValueError("similarities and source_ids must be non-empty aligned vectors")
    if not 0 < placebo_quantile <= 1:
        raise ValueError("placebo_quantile must be in (0, 1]")

    # Reserve the true top-1 before filling the ladder. Otherwise a target
    # close to the maximum could consume it and the arm named ``top`` would
    # silently become top-2.
    top_idx = int(np.argmax(sims))
    used: set[int] = {top_idx}
    selected: list[dict[str, Any]] = []

    for target in targets:
        order = np.argsort(np.abs(sims - float(target)), kind="stable")
        idx = next((int(i) for i in order if int(i) not in used), None)
        if idx is None:
            break
        used.add(idx)
        selected.append({
            "arm": "retrieved",
            "level": f"target_{float(target):.2f}",
            "requested_similarity": float(target),
            "source_uid": str(ids[idx]),
            "similarity": float(sims[idx]),
        })

    selected.append({
        "arm": "retrieved",
        "level": "top",
        "requested_similarity": None,
        "source_uid": str(ids[top_idx]),
        "similarity": float(sims[top_idx]),
    })

    cutoff = float(np.quantile(sims, placebo_quantile))
    candidates = [int(i) for i in np.flatnonzero(sims <= cutoff) if int(i) not in used]
    if not candidates:
        candidates = [int(i) for i in np.argsort(sims, kind="stable") if int(i) not in used]
    if candidates:
        pick = int(stable_fraction(seed_key) * len(candidates))
        placebo_idx = candidates[min(pick, len(candidates) - 1)]
        selected.append({
            "arm": "placebo",
            "level": "placebo",
            "requested_similarity": None,
            "source_uid": str(ids[placebo_idx]),
            "similarity": float(sims[placebo_idx]),
        })
    return selected


def gaussian_kernel_curve(
    similarity: Sequence[float],
    delta: Sequence[float],
    grid: Sequence[float],
    bandwidth: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Nadaraya-Watson estimate and effective sample size on a fixed grid."""
    x = np.asarray(similarity, dtype=float)
    y = np.asarray(delta, dtype=float)
    g = np.asarray(grid, dtype=float)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("similarity and delta must be non-empty aligned vectors")

    weights = np.exp(-0.5 * ((x[:, None] - g[None, :]) / bandwidth) ** 2)
    denom = weights.sum(axis=0)
    estimate = np.divide(weights.T @ y, denom, out=np.full(len(g), np.nan), where=denom > 0)
    effective_n = np.divide(
        denom**2,
        (weights**2).sum(axis=0),
        out=np.zeros(len(g)),
        where=(weights**2).sum(axis=0) > 0,
    )
    return estimate, effective_n


def clustered_bootstrap_curve(
    rows: Sequence[dict[str, Any]],
    grid: Sequence[float],
    *,
    value_key: str = "delta",
    bandwidth: float = 0.08,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Kernel curve for ``value_key``, resampling validation items as clusters."""
    if not rows:
        raise ValueError("rows cannot be empty")
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(str(row["val_uid"]), []).append(row)
    cluster_ids = list(by_cluster)
    flat = [r for rs in by_cluster.values() for r in rs]
    estimate, effective_n = gaussian_kernel_curve(
        [r["similarity"] for r in flat], [r[value_key] for r in flat], grid, bandwidth
    )

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(grid)), np.nan)
    for b in range(n_boot):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled = [r for cid in sampled_ids for r in by_cluster[str(cid)]]
        boot[b], _ = gaussian_kernel_curve(
            [r["similarity"] for r in sampled],
            [r[value_key] for r in sampled],
            grid,
            bandwidth,
        )
    return {
        "grid": np.asarray(grid, dtype=float),
        "estimate": estimate,
        "low": np.nanpercentile(boot, 2.5, axis=0),
        "high": np.nanpercentile(boot, 97.5, axis=0),
        "effective_n": effective_n,
        "bootstrap": boot,
    }


def sustained_threshold(
    grid: Sequence[float],
    effect: Sequence[float],
    effective_n: Sequence[float],
    *,
    min_effective_n: float = 20,
    consecutive: int = 5,
) -> float | None:
    """First similarity with a sustained non-negative estimated effect."""
    g = np.asarray(grid, dtype=float)
    e = np.asarray(effect, dtype=float)
    n = np.asarray(effective_n, dtype=float)
    valid = np.isfinite(e) & (e >= 0) & (n >= min_effective_n)
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    for i in range(0, len(g) - consecutive + 1):
        if valid[i : i + consecutive].all():
            return float(g[i])
    return None


def threshold_bootstrap_ci(
    grid: Sequence[float],
    boot_effects: np.ndarray,
    effective_n: Sequence[float],
    *,
    min_effective_n: float = 20,
    consecutive: int = 5,
) -> tuple[float | None, float | None, float]:
    """Percentile CI and identification rate for bootstrap thresholds."""
    values = [
        sustained_threshold(
            grid, row, effective_n,
            min_effective_n=min_effective_n,
            consecutive=consecutive,
        )
        for row in np.asarray(boot_effects)
    ]
    identified = np.asarray([v for v in values if v is not None], dtype=float)
    rate = len(identified) / max(len(values), 1)
    if not len(identified):
        return None, None, rate
    return float(np.percentile(identified, 2.5)), float(np.percentile(identified, 97.5)), rate


def paired_bootstrap_difference(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired accuracy difference and percentile confidence interval."""
    b = np.asarray(baseline, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if len(b) != len(t) or len(b) == 0:
        raise ValueError("baseline and treatment must be non-empty aligned vectors")
    delta = t - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    sampled = delta[indices].mean(axis=1)
    return {
        "n": int(len(delta)),
        "baseline_accuracy": float(b.mean()),
        "treatment_accuracy": float(t.mean()),
        "difference": float(delta.mean()),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "helped": int(np.sum((b == 0) & (t == 1))),
        "harmed": int(np.sum((b == 1) & (t == 0))),
    }


def rows_to_records(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Small convenience for pandas namedtuples/dicts in notebook code."""
    out = []
    for row in rows:
        out.append(dict(row) if isinstance(row, dict) else row._asdict())
    return out
