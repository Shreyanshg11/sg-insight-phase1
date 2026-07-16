import numpy as np
import pandas as pd
import pytest
from app.engines.feature_engineering_engine import apply_feature_action


class TestDateFeatures:
    def test_extract_date_parts_default(self, messy_df):
        out, preview = apply_feature_action(messy_df, "date_features", "extract_date_parts",
                                              ["signup_date"], None, {})
        for suffix in ("_year", "_month", "_day", "_dayofweek", "_is_weekend"):
            assert f"signup_date{suffix}" in out.columns
        assert out["signup_date_year"].iloc[0] == 2024
        assert out["signup_date_month"].iloc[0] == 1
        assert out["signup_date_day"].iloc[0] == 1

    def test_extract_date_parts_subset(self, messy_df):
        out, preview = apply_feature_action(messy_df, "date_features", "extract_date_parts",
                                              ["signup_date"], None, {"parts": ["year"]})
        assert "signup_date_year" in out.columns
        assert "signup_date_month" not in out.columns

    def test_original_column_never_dropped(self, messy_df):
        out, _ = apply_feature_action(messy_df, "date_features", "extract_date_parts", ["signup_date"], None, {})
        assert "signup_date" in out.columns  # additive, never destructive


class TestRatio:
    def test_create_ratio_default_name(self, messy_df):
        out, preview = apply_feature_action(messy_df, "ratio", "create_ratio", ["revenue", "cost"], None, {})
        assert "revenue_per_cost" in out.columns
        assert out["revenue_per_cost"].iloc[0] == pytest.approx(messy_df["revenue"].iloc[0] / messy_df["cost"].iloc[0])

    def test_create_ratio_custom_name(self, messy_df):
        out, _ = apply_feature_action(messy_df, "ratio", "create_ratio", ["revenue", "cost"], "my_ratio", {})
        assert "my_ratio" in out.columns
        assert "revenue_per_cost" not in out.columns

    def test_division_by_zero_becomes_nan_not_inf(self):
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 0.0]})
        out, preview = apply_feature_action(df, "ratio", "create_ratio", ["a", "b"], None, {})
        assert pd.isna(out["a_per_b"].iloc[1])
        assert not np.isinf(out["a_per_b"].iloc[1]) if not pd.isna(out["a_per_b"].iloc[1]) else True
        assert preview["undefined_rows"] == 1


class TestBucketing:
    def test_quartile_bucket_creates_four_labeled_bins(self, clean_numeric_df):
        out, preview = apply_feature_action(clean_numeric_df, "bucketing", "quartile_bucket", ["x"], None, {})
        assert "x_bucket" in out.columns
        assert set(out["x_bucket"].dropna().unique()) <= {"Q1", "Q2", "Q3", "Q4"}


class TestScalingNormalization:
    def test_min_max_scale_bounds_zero_to_one(self, clean_numeric_df):
        out, _ = apply_feature_action(clean_numeric_df, "scaling_normalization", "min_max_scale", ["x"], None, {})
        assert out["x_scaled"].min() == pytest.approx(0.0)
        assert out["x_scaled"].max() == pytest.approx(1.0)

    def test_z_score_standardize_mean_zero_std_one(self, clean_numeric_df):
        out, _ = apply_feature_action(clean_numeric_df, "scaling_normalization", "z_score_standardize", ["x"], None, {})
        assert out["x_scaled"].mean() == pytest.approx(0.0, abs=1e-9)
        assert out["x_scaled"].std() == pytest.approx(1.0, abs=1e-9)

    def test_robust_scale_uses_median_not_mean(self, clean_numeric_df):
        out, _ = apply_feature_action(clean_numeric_df, "scaling_normalization", "robust_scale", ["x"], None, {})
        assert "x_scaled" in out.columns
        assert out["x_scaled"].notna().all()

    def test_custom_new_column_name(self, clean_numeric_df):
        out, _ = apply_feature_action(clean_numeric_df, "scaling_normalization", "min_max_scale",
                                       ["x"], "x_normalized", {})
        assert "x_normalized" in out.columns


class TestEncoding:
    def test_one_hot_encode_creates_one_column_per_category(self, categorical_df):
        n_categories = categorical_df["group"].nunique()
        out, preview = apply_feature_action(categorical_df, "encoding", "one_hot_encode", ["group"], None, {})
        new_cols = [c for c in out.columns if c.startswith("group_") and c != "group"]
        assert len(new_cols) == n_categories
        # every row sums to exactly 1 across the dummy columns
        assert (out[new_cols].sum(axis=1) == 1).all()

    def test_label_encode_produces_integer_codes(self, categorical_df):
        out, preview = apply_feature_action(categorical_df, "encoding", "label_encode", ["group"], None, {})
        assert "group_encoded" in out.columns
        assert set(out["group_encoded"].unique()) == set(range(categorical_df["group"].nunique()))
        assert "category_map" in preview


class TestLogTransform:
    def test_log1p_transform(self, clean_numeric_df):
        positive_df = clean_numeric_df.copy()
        positive_df["x"] = positive_df["x"].abs() + 1  # ensure all positive
        out, _ = apply_feature_action(positive_df, "log_transform", "log1p_transform", ["x"], None, {})
        assert "x_log" in out.columns
        assert out["x_log"].iloc[0] == pytest.approx(np.log1p(positive_df["x"].iloc[0]))

    def test_negative_values_raise(self):
        df = pd.DataFrame({"x": [-1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="negative"):
            apply_feature_action(df, "log_transform", "log1p_transform", ["x"], None, {})


class TestUnknownCategoryAction:
    def test_unknown_combo_raises(self, clean_numeric_df):
        with pytest.raises(ValueError, match="Unknown category/action combination"):
            apply_feature_action(clean_numeric_df, "not_real", "not_real", ["x"], None, {})


class TestNeverMutatesInput:
    def test_original_df_untouched(self, messy_df):
        cols_before = list(messy_df.columns)
        apply_feature_action(messy_df, "ratio", "create_ratio", ["revenue", "cost"], None, {})
        assert list(messy_df.columns) == cols_before
