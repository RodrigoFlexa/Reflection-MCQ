import pandas as pd
import pytest

from rmcq.test_analysis import (apply_policy, build_panel, summarize_policy,
    threshold_sweep, select_validation_thresholds, transfer_thresholds, average_datasets)


def panel():
    rows = []
    for uid, b, r, sim in [("a", 1, None, .8), ("b", 0, 1, .4), ("c", None, 0, .9)]:
        for condition, correct in [("baseline", b), ("self_simple", r)]:
            rows.append(dict(model="m", dataset="d", val_uid=uid, condition=condition,
                             correct=correct, similarity=sim, race_subset=None,
                             failure=correct is None, audit_flags=[], audit_detail_available=True))
    return build_panel(pd.DataFrame(rows))


def test_fallback_uses_same_question_and_retains_unanswered_baseline():
    scored = apply_policy(panel(), .5, True)
    reflection = scored.loc[scored.condition.eq("self_simple")].set_index("val_uid")
    assert reflection.score.to_dict() == {"a": 1, "b": 0, "c": 0}
    assert reflection.fallback_used.sum() == 2
    summary = summarize_policy(scored).set_index("arm")
    assert summary.loc["self_simple", "n"] == 3
    assert summary.loc["baseline", "n_unanswered"] == 1


def test_no_fallback_keeps_independent_samples_and_wrong_answers():
    scored = apply_policy(panel(), .5, False)
    summary = summarize_policy(scored).set_index("arm")
    assert summary.loc["baseline", "n"] == 2
    assert summary.loc["self_simple", "n"] == 1
    assert summary.loc["self_simple", "accuracy"] == 0  # wrong is answered, never a fallback trigger


def test_missing_condition_and_audit_filters():
    p = panel()
    frame = p[["model", "dataset", "val_uid", "condition", "correct", "similarity", "race_subset",
               "failure", "audit_flags", "audit_detail_available"]].copy()
    frame = frame.loc[~(frame.val_uid.eq("a") & frame.condition.eq("self_simple"))]
    frame.loc[frame.val_uid.eq("b") & frame.condition.eq("self_simple"), "audit_flags"] = pd.Series(
        [["embedding_truncated"]], index=frame.index[frame.val_uid.eq("b") & frame.condition.eq("self_simple")])
    rebuilt = build_panel(frame, ["embedding_truncated"])
    assert rebuilt.missing_row.sum() == 1
    scored = apply_policy(rebuilt, None, True)
    assert scored.fallback_used.sum() == 2


def test_validation_selection_ties_and_missing_configurations():
    sweep = threshold_sweep(panel(), [0, .5, 1], True)
    selected = select_validation_thresholds(sweep, min_n=1)
    assert selected.iloc[0].threshold == 0
    assert select_validation_thresholds(sweep, min_n=99).empty
    selected.loc[:, "arm"] = "teacher_simple@gpt-5-4-petrobras"
    assert transfer_thresholds(panel(), selected).empty


def test_transfer_matched_baseline_without_fallback():
    selected = select_validation_thresholds(threshold_sweep(panel(), [.5], False), min_n=1)
    result = transfer_thresholds(panel(), selected, False)
    assert result.iloc[0]["n"] == 1
    assert result.iloc[0].matched_baseline_accuracy == 0


def test_zero_coverage_is_nan_and_not_dropped_from_macro():
    summary = summarize_policy(apply_policy(panel(), 1, False))
    assert pd.isna(summary.set_index("arm").loc["self_simple", "accuracy"])
    copy = summary.copy()
    copy["dataset"] = "second"
    copy.loc[copy.arm.eq("self_simple"), ["accuracy", "n", "correct"]] = [1, 2, 2]
    averaged = average_datasets(pd.concat([summary, copy])).set_index("arm")
    assert pd.isna(averaged.loc["self_simple", "macro_accuracy"])


def test_exact_threshold_is_accepted_and_baseline_not_gated():
    summary = summarize_policy(apply_policy(panel(), .4, False)).set_index("arm")
    assert summary.loc["self_simple", "n"] == 2
    assert summary.loc["baseline", "n"] == 2


def test_missing_baseline_rejected():
    p = panel()
    frame = p[["model", "dataset", "val_uid", "condition", "correct", "similarity", "race_subset",
               "failure", "audit_flags", "audit_detail_available"]]
    frame = frame.loc[~(frame.val_uid.eq("a") & frame.condition.eq("baseline"))]
    with pytest.raises(ValueError, match="no baseline"):
        build_panel(frame)
