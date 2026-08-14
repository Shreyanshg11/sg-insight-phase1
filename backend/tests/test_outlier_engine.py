"""
Tests for outlier_engine — fixtures are built so the "right" tier is provable
from how the data was constructed (same approach as test_missingness_engine.py).
"""
import numpy as np
import pandas as pd
import pytest
from app.engines.outlier_engine import classify_outliers, classify_outliers_for_column


@pytest.fixture
def tiered_df():
    """One column engineered to contain one of each tier:
    - row 0: a decimal-point-slip error (~100x the median), far outside the fence
    - rows 1-3: a repeated 'legitimate' rare value (a real premium tier, appears 3x)
    - row 4: a lone statistically-unusual value with no repeat and no error signature
    """
    rng = np.random.RandomState(0)
    n = 100
    prices = rng.normal(50, 5, n).round(2)
    prices[0] = 5000.0   # ~100x median -> error signature, far out
    prices[1] = 200.0
    prices[2] = 200.0
    prices[3] = 200.0    # repeats 3x, far out but no error signature -> legitimate rare
    prices[4] = 85.0     # singleton, no repeat, no error signature -> statistically unusual
    return pd.DataFrame({"price": prices})


class TestOutlierTiers:
    def test_decimal_slip_classified_as_error(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        row0 = next(p for p in summary.points if p.row_index == 0)
        assert row0.tier == "likely_data_entry_error"
        assert row0.confidence >= 0.8

    def test_repeated_far_value_classified_as_legitimate(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        repeated_points = [p for p in summary.points if p.row_index in (1, 2, 3)]
        assert len(repeated_points) == 3
        assert all(p.tier == "legitimate_rare_value" for p in repeated_points)

    def test_singleton_classified_as_statistically_unusual(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        row4 = next(p for p in summary.points if p.row_index == 4)
        assert row4.tier == "statistically_unusual"

    def test_tier_counts_match_points(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        assert summary.tier_counts["likely_data_entry_error"] == 1
        assert summary.tier_counts["legitimate_rare_value"] == 3
        assert summary.tier_counts["statistically_unusual"] == 1
        assert summary.total_outliers == 5

    def test_points_sorted_worst_tier_first(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        tiers_in_order = [p.tier for p in summary.points]
        assert tiers_in_order[0] == "likely_data_entry_error"

    def test_evidence_is_nonempty_for_every_point(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        assert all(len(p.evidence) >= 1 for p in summary.points)

    def test_far_out_fence_wider_than_standard_fence(self, tiered_df):
        summary = classify_outliers_for_column(tiered_df, "price")
        assert summary.far_out_lower < summary.lower_fence
        assert summary.far_out_upper > summary.upper_fence


class TestNegativeInNonnegativeColumn:
    def test_negative_value_flagged_as_error_signature(self):
        rng = np.random.RandomState(3)
        values = rng.uniform(10, 20, 50)
        values[0] = -5.0  # negative in an otherwise all-nonnegative column
        df = pd.DataFrame({"quantity": values})
        summary = classify_outliers_for_column(df, "quantity")
        row0 = next((p for p in summary.points if p.row_index == 0), None)
        assert row0 is not None
        assert row0.tier == "likely_data_entry_error"
        assert any("negative" in e.lower() for e in row0.evidence)


class TestNoOutliers:
    def test_clean_column_returns_none(self):
        df = pd.DataFrame({"x": list(range(20))})  # no IQR outliers by construction
        assert classify_outliers_for_column(df, "x") is None

    def test_too_few_rows_returns_none(self):
        df = pd.DataFrame({"x": [1, 2, 3, 1000]})
        assert classify_outliers_for_column(df, "x") is None

    def test_zero_iqr_returns_none(self):
        df = pd.DataFrame({"x": [5] * 20 + [500]})  # IQR is 0, can't fence
        assert classify_outliers_for_column(df, "x") is None


class TestClassifyOutliers:
    def test_only_includes_numeric_columns_with_outliers(self, tiered_df):
        df = tiered_df.copy()
        df["clean_col"] = list(range(len(df)))  # no outliers
        df["cat"] = "x"  # non-numeric, excluded regardless
        result = classify_outliers(df)
        assert "price" in result
        assert "clean_col" not in result
        assert "cat" not in result

    def test_empty_when_nothing_flagged(self):
        df = pd.DataFrame({"x": list(range(20))})
        assert classify_outliers(df) == {}
