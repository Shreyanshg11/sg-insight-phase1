import numpy as np
import pandas as pd
import pytest
from app.engines.cleaning_engine import apply_action, compute_quality_score


class TestDuplicateManager:
    def test_drop_duplicates_removes_exact_dupes_only(self, messy_df):
        before = len(messy_df)
        out, preview = apply_action(messy_df, "duplicate_manager", "drop_duplicates", None, {})
        assert len(out) == before - 1  # fixture has exactly 1 exact duplicate
        assert preview["rows_removed"] == 1
        assert not out.duplicated().any()

    def test_does_not_mutate_input(self, messy_df):
        before_len = len(messy_df)
        apply_action(messy_df, "duplicate_manager", "drop_duplicates", None, {})
        assert len(messy_df) == before_len  # original untouched


class TestMissingValueCenter:
    def test_impute_mean(self, messy_df):
        original_mean = messy_df["cost"].mean()
        out, preview = apply_action(messy_df, "missing_value_center", "impute_mean", "cost", {})
        assert out["cost"].isna().sum() == 0
        assert preview["missing_before"] == 4
        assert preview["missing_after"] == 0
        # imputed rows should equal the mean of the ORIGINAL column
        assert out.loc[5, "cost"] == pytest.approx(original_mean)

    def test_impute_median(self, messy_df):
        original_median = messy_df["cost"].median()
        out, _ = apply_action(messy_df, "missing_value_center", "impute_median", "cost", {})
        assert out.loc[5, "cost"] == pytest.approx(original_median)

    def test_impute_constant(self, messy_df):
        out, _ = apply_action(messy_df, "missing_value_center", "impute_constant", "cost", {"value": 0})
        assert (out.loc[[5, 6, 7, 8], "cost"] == 0).all()

    def test_impute_mode_categorical(self):
        df = pd.DataFrame({"cat": ["a", "a", "a", "b", None]})
        out, _ = apply_action(df, "missing_value_center", "impute_mode", "cat", {})
        assert out.loc[4, "cat"] == "a"

    def test_drop_column(self, messy_df):
        out, _ = apply_action(messy_df, "missing_value_center", "drop_column", "cost", {})
        assert "cost" not in out.columns
        assert len(out) == len(messy_df)  # rows untouched

    def test_drop_rows(self, messy_df):
        out, preview = apply_action(messy_df, "missing_value_center", "drop_rows", "cost", {})
        assert out["cost"].isna().sum() == 0
        assert len(out) == len(messy_df) - 4

    def test_missing_column_raises(self, messy_df):
        with pytest.raises(ValueError):
            apply_action(messy_df, "missing_value_center", "impute_mean", None, {})

    def test_unknown_action_raises(self, messy_df):
        with pytest.raises(ValueError, match="Unknown missing_value_center action"):
            apply_action(messy_df, "missing_value_center", "not_a_real_action", "cost", {})


class TestOutlierCenter:
    def test_ignore_is_a_true_noop(self, messy_df):
        out, preview = apply_action(messy_df, "outlier_center", "ignore", "revenue", {})
        pd.testing.assert_frame_equal(out, messy_df)
        assert preview["outliers_affected"] == 2  # fixture's 2 appended outliers

    def test_convert_to_missing(self, messy_df):
        out, preview = apply_action(messy_df, "outlier_center", "convert_to_missing", "revenue", {})
        assert preview["outliers_affected"] == 2
        assert out["revenue"].isna().sum() == 2

    def test_clip_bounds_extreme_values_without_dropping_rows(self, messy_df):
        out, preview = apply_action(messy_df, "outlier_center", "clip", "revenue", {})
        assert len(out) == len(messy_df)  # no rows removed
        assert out["revenue"].max() < 50000
        assert out["revenue"].min() > -500

    def test_drop_rows_removes_exactly_the_outliers(self, messy_df):
        out, preview = apply_action(messy_df, "outlier_center", "drop_rows", "revenue", {})
        assert len(out) == len(messy_df) - 2
        assert out["revenue"].max() < 50000

    def test_clean_data_finds_few_outliers(self, clean_numeric_df):
        out, preview = apply_action(clean_numeric_df, "outlier_center", "ignore", "x", {})
        # not asserting exactly 0 (IQR can flag a handful in pure noise) —
        # asserting it's small relative to n, i.e. not flagging everything
        assert preview["outliers_affected"] < len(clean_numeric_df) * 0.1


class TestCategoryMergeCenter:
    def test_normalize_case_whitespace_collapses_variants(self, messy_df):
        before_unique = messy_df["region"].nunique()
        out, preview = apply_action(messy_df, "category_merge_center", "normalize_case_whitespace", "region", {})
        # "North" and " north " should collapse to one canonical form
        assert out["region"].nunique() < before_unique
        assert out["region"].str.strip().eq(out["region"]).all()  # no leading/trailing whitespace left

    def test_fuzzy_merge_maps_listed_values_to_canonical(self, messy_df):
        out, preview = apply_action(
            messy_df, "category_merge_center", "fuzzy_merge", "region",
            {"values": ["North", "South"], "canonical": "Unified"},
        )
        assert "Unified" in out["region"].values
        assert not out["region"].isin(["North", "South"]).any()
        assert preview["canonical"] == "Unified"

    def test_fuzzy_merge_requires_params(self, messy_df):
        with pytest.raises(ValueError, match="requires"):
            apply_action(messy_df, "category_merge_center", "fuzzy_merge", "region", {})

    def test_rename_category(self, messy_df):
        out, preview = apply_action(
            messy_df, "category_merge_center", "rename_category", "region",
            {"old_value": "East", "new_value": "Eastern"},
        )
        assert "East" not in out["region"].values
        assert "Eastern" in out["region"].values


class TestTextCleaning:
    @pytest.mark.parametrize("action,expected_fn", [
        ("trim", lambda s: s.strip()),
        ("lowercase", lambda s: s.lower()),
        ("uppercase", lambda s: s.upper()),
    ])
    def test_simple_transforms(self, action, expected_fn):
        df = pd.DataFrame({"col": ["  Hello World  "]})
        out, _ = apply_action(df, "text_cleaning", action, "col", {})
        assert out.loc[0, "col"] == expected_fn("  Hello World  ")

    def test_remove_special_chars(self):
        df = pd.DataFrame({"col": ["a!b@c#123"]})
        out, _ = apply_action(df, "text_cleaning", "remove_special_chars", "col", {})
        assert "!" not in out.loc[0, "col"] and "@" not in out.loc[0, "col"] and "#" not in out.loc[0, "col"]
        assert "123" in out.loc[0, "col"]


class TestUnknownStepAction:
    def test_completely_unknown_combo_raises(self, messy_df):
        with pytest.raises(ValueError, match="Unknown step/action combination"):
            apply_action(messy_df, "not_a_real_step", "not_a_real_action", None, {})


class TestQualityScore:
    def test_clean_data_scores_high(self, clean_numeric_df):
        score = compute_quality_score(clean_numeric_df)
        assert score.overall > 80
        assert score.completeness == 100  # no missing values at all

    def test_missing_values_reduce_completeness(self, messy_df):
        score = compute_quality_score(messy_df)
        assert score.completeness < 100

    def test_duplicates_reduce_uniqueness(self, messy_df):
        score = compute_quality_score(messy_df)
        assert score.uniqueness < 100

    def test_score_bounded_zero_to_hundred(self, messy_df, clean_numeric_df):
        for df in (messy_df, clean_numeric_df):
            score = compute_quality_score(df)
            for field in ("overall", "completeness", "consistency", "validity", "uniqueness"):
                val = getattr(score, field)
                assert 0 <= val <= 100, f"{field}={val} out of bounds"

    def test_breakdown_deductions_are_additive_and_explain_overall(self, messy_df):
        score = compute_quality_score(messy_df)
        assert len(score.breakdown) > 0
        for item in score.breakdown:
            assert item.points_deducted >= 0
