"""
Tests for value-tier classification in feature_engineering_engine —
generate_feature_recommendations() now tags every suggestion with a
value_tier ("high_value" | "situational" | "risky" | "redundant") separate
from `confidence`. Fixtures are built so the correct tier is provable from
construction, same convention as the other new-engine test files.
"""
import numpy as np
import pandas as pd
import pytest
from app.engines.feature_engineering_engine import generate_feature_recommendations, VALUE_TIERS


def _rec(recs, category, new_column_name):
    return next(r for r in recs if r.category == category and r.new_column_name == new_column_name)


class TestValueTierIsAlwaysValid:
    def test_every_recommendation_has_a_known_tier(self, messy_df):
        recs = generate_feature_recommendations(messy_df)
        for r in recs:
            assert r.value_tier in VALUE_TIERS
            assert r.value_tier_reason  # never blank


class TestRatioValueTier:
    @pytest.fixture
    def derived_revenue_df(self):
        rng = np.random.RandomState(0)
        n = 200
        price = rng.normal(50, 10, n).round(2)
        quantity = rng.randint(1, 20, n)
        return pd.DataFrame({
            "price": price, "quantity": quantity, "revenue": price * quantity,  # exactly derivable
        })

    def test_ratio_duplicating_existing_column_is_redundant(self, derived_revenue_df):
        recs = generate_feature_recommendations(derived_revenue_df)
        rec = _rec(recs, "ratio", "revenue_per_price")
        assert rec.value_tier == "redundant"
        assert "correlates" in rec.value_tier_reason.lower()

    def test_unrelated_clean_ratio_is_high_value(self, derived_revenue_df):
        recs = generate_feature_recommendations(derived_revenue_df)
        rec = _rec(recs, "ratio", "price_per_quantity")
        assert rec.value_tier == "high_value"

    def test_high_zero_denominator_rate_is_risky(self):
        rng = np.random.RandomState(0)
        n = 200
        a = rng.uniform(1, 100, n)
        b = rng.uniform(1, 100, n)
        b[: int(n * 0.2)] = 0  # 20% zero-denominator, above the 15% risky cutoff
        df = pd.DataFrame({"a": a, "b": b})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "ratio", "a_per_b")
        assert rec.value_tier == "risky"


class TestBucketingValueTier:
    def test_high_outlier_column_is_high_value(self):
        rng = np.random.RandomState(0)
        n = 200
        values = rng.normal(50, 5, n)
        values[:15] = rng.uniform(500, 1000, 15)  # inject a real outlier tail (>5%)
        df = pd.DataFrame({"x": values})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "bucketing", "x_bucket")
        assert rec.value_tier == "high_value"

    def test_clean_distribution_is_situational(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"x": rng.normal(50, 5, 200)})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "bucketing", "x_bucket")
        assert rec.value_tier == "situational"


class TestScalingValueTier:
    def test_high_outlier_rate_is_risky(self):
        rng = np.random.RandomState(0)
        n = 200
        values = rng.normal(50, 5, n)
        values[:40] = rng.uniform(500, 1000, 40)  # 20% far outliers, above 15% risky cutoff
        df = pd.DataFrame({"x": values})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "scaling_normalization", "x_scaled")
        assert rec.value_tier == "risky"

    def test_clean_column_is_high_value(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"x": rng.normal(50, 5, 200)})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "scaling_normalization", "x_scaled")
        assert rec.value_tier == "high_value"


class TestEncodingValueTier:
    def test_low_cardinality_one_hot_is_high_value(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], 100)})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "encoding", "cat_onehot")
        assert rec.value_tier == "high_value"

    def test_label_encode_is_always_risky(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"cat": rng.choice([f"g{i}" for i in range(20)], 200)})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "encoding", "cat_encoded")
        assert rec.value_tier == "risky"
        assert "ordering" in rec.value_tier_reason.lower() or "order" in rec.value_tier_reason.lower()

    def test_moderate_cardinality_one_hot_is_situational(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"cat": rng.choice([f"g{i}" for i in range(8)], 200)})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "encoding", "cat_onehot")
        assert rec.value_tier == "situational"


class TestLogTransformValueTier:
    def test_strong_skew_correction_is_high_value(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"x": rng.lognormal(0, 1.2, 300)})  # strongly right-skewed (skew > 2)
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "log_transform", "x_log")
        assert rec.value_tier == "high_value"


class TestDateFeaturesValueTier:
    def test_full_parse_high_variety_is_high_value(self):
        df = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=200, freq="D")})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "date_features", "d_year")
        assert rec.value_tier == "high_value"

    def test_single_constant_date_is_redundant(self):
        # Every non-null value is literally the same date -> extracted parts
        # (year/month/day/weekday) can't vary at all.
        df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01"] * 150 + [pd.NaT] * 10)})
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "date_features", "d_year")
        assert rec.value_tier == "redundant"

    def test_borderline_parse_rate_is_risky(self):
        df = pd.DataFrame({
            "d": pd.to_datetime(["2020-01-0" + str((i % 9) + 1) for i in range(55)] + [None] * 45)
        })
        recs = generate_feature_recommendations(df)
        rec = _rec(recs, "date_features", "d_year")
        assert rec.value_tier == "risky"
