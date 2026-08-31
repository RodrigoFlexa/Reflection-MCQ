import numpy as np

from rmcq.thresholds import (
    calibration_or_test,
    extract_final_answer,
    gaussian_kernel_curve,
    normalize_stem,
    paired_bootstrap_difference,
    select_similarity_ladder,
    sustained_threshold,
)


def test_extract_final_answer_uses_last_formatted_occurrence():
    text = "FINAL ANSWER: A\nI reconsidered.\nFINAL ANSWER: C"
    assert extract_final_answer(text) == "C"
    assert extract_final_answer("Answer: C") is None


def test_normalize_stem_ignores_case_and_whitespace():
    a = {"context": " Some  Context ", "question": "Why?"}
    b = {"context": "some context", "question": "  WHY?  "}
    assert normalize_stem(a) == normalize_stem(b)


def test_calibration_split_is_stable():
    first = calibration_or_test("item-1", seed=7)
    assert first == calibration_or_test("item-1", seed=7)
    assert first in {"calibration", "test"}


def test_similarity_ladder_uses_one_unique_source_per_arm():
    sims = [0.10, 0.30, 0.51, 0.72, 0.91]
    ids = [f"s{i}" for i in range(len(sims))]
    selected = select_similarity_ladder(sims, ids, [0.30, 0.50, 0.70], seed_key="v1")
    retrieved = [r for r in selected if r["arm"] == "retrieved"]
    assert [r["source_uid"] for r in retrieved[:3]] == ["s1", "s2", "s3"]
    assert retrieved[-1]["level"] == "top"
    assert len({r["source_uid"] for r in selected}) == len(selected)
    assert sum(r["arm"] == "placebo" for r in selected) == 1


def test_top_arm_is_true_top_even_when_target_is_near_maximum():
    sims = [0.10, 0.30, 0.51, 0.70, 0.71, 0.91]
    ids = [f"s{i}" for i in range(len(sims))]
    selected = select_similarity_ladder(sims, ids, [0.30, 0.50, 0.90], seed_key="v2")
    top = next(r for r in selected if r["level"] == "top")
    assert top["source_uid"] == "s5"
    assert top["similarity"] == 0.91


def test_kernel_curve_and_sustained_threshold():
    x = np.linspace(0.2, 0.9, 200)
    y = np.where(x >= 0.6, 1.0, -1.0)
    grid = np.linspace(0.25, 0.85, 61)
    effect, n_eff = gaussian_kernel_curve(x, y, grid, bandwidth=0.025)
    threshold = sustained_threshold(grid, effect, n_eff, min_effective_n=5, consecutive=3)
    assert threshold is not None
    assert 0.55 <= threshold <= 0.65


def test_paired_bootstrap_counts_help_and_harm():
    result = paired_bootstrap_difference([0, 1, 1, 0], [1, 0, 1, 0], n_boot=100, seed=1)
    assert result["difference"] == 0
    assert result["helped"] == 1
    assert result["harmed"] == 1
