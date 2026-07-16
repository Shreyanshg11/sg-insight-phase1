import numpy as np
import pandas as pd
import pytest
from app.engines.eda_engine import (
    compute_overview, compute_distributions, compute_correlation,
    compute_missing_patterns, compute_relationships,
)


class TestComputeOverview:
    def test_classifies_column_kinds(self, messy_df):
        overview = compute_overview(messy_df)
        assert "revenue" in overview["numeric_columns"]
        assert "cost" in overview["numeric_columns"]
        assert "region" in overview["categorical_columns"]
        assert "signup_date" in overview["date_columns"]

    def test_row_and_column_counts(self, messy_df):
        overview = compute_overview(messy_df)
        assert overview["rows"] == len(messy_df)
        assert overview["columns"] == len(messy_df.columns)

    def test_missing_pct_reported_per_column(self, messy_df):
        overview = compute_overview(messy_df)
        cost_row = next(r for r in overview["summary"] if r["column"] == "cost")
        assert cost_row["missing_pct"] == pytest.approx(4 / len(messy_df) * 100, abs=0.1)

    def test_empty_dataframe_does_not_crash(self):
        overview = compute_overview(pd.DataFrame())
        assert overview["rows"] == 0
        assert overview["columns"] == 0


class TestComputeDistributions:
    def test_numeric_column_has_stats_and_bins(self, clean_numeric_df):
        dist = compute_distributions(clean_numeric_df, "x")
        assert "stats" in dist
        assert "bins" in dist
        assert sum(b["count"] for b in dist["bins"]) == len(clean_numeric_df)

    def test_categorical_column_has_bins(self, categorical_df):
        dist = compute_distributions(categorical_df, "group")
        assert sum(b["count"] for b in dist["bins"]) == len(categorical_df)


class TestComputeCorrelation:
    def test_perfect_self_correlation_is_one(self, clean_numeric_df):
        result = compute_correlation(clean_numeric_df)
        idx = result["columns"].index("x")
        assert result["matrix"][idx][idx] == 1.0

    def test_fewer_than_two_numeric_columns_returns_empty(self):
        df = pd.DataFrame({"only_col": [1, 2, 3]})
        result = compute_correlation(df)
        assert result["matrix"] == []
        assert "Fewer than 2 numeric" in result["note"]

    def test_top_pairs_sorted_by_absolute_strength(self, clean_numeric_df):
        result = compute_correlation(clean_numeric_df)
        strengths = [abs(p["correlation"]) for p in result["top_pairs"]]
        assert strengths == sorted(strengths, reverse=True)

    def test_known_strong_correlation_detected(self):
        n = 100
        x = np.random.normal(size=n)
        df = pd.DataFrame({"x": x, "y": x * 2 + 1})  # perfectly correlated
        result = compute_correlation(df)
        pair = result["top_pairs"][0]
        assert abs(pair["correlation"]) > 0.99
        assert pair["strength"] == "strong"


class TestComputeMissingPatterns:
    def test_identifies_columns_with_missing_values(self, messy_df):
        result = compute_missing_patterns(messy_df)
        assert "cost" in result["columns_with_missing"]
        assert "revenue" not in result["columns_with_missing"]

    def test_no_missing_data_returns_empty_columns_list(self, clean_numeric_df):
        result = compute_missing_patterns(clean_numeric_df)
        assert result["columns_with_missing"] == []

    def test_row_bucket_counts_sum_to_total_rows(self, messy_df):
        result = compute_missing_patterns(messy_df)
        total = sum(b["count"] for b in result["missing_by_row_bucket"])
        assert total == len(messy_df)


class TestComputeRelationships:
    def test_numeric_numeric_returns_points_and_correlation(self, clean_numeric_df):
        result = compute_relationships(clean_numeric_df, "x", "y")
        assert result["kind"] == "numeric_numeric"
        assert "correlation" in result["stats"]
        assert len(result["points"]) > 0

    def test_numeric_categorical_groups_by_category(self, categorical_df):
        result = compute_relationships(categorical_df, "value", "group")
        assert result["kind"] == "numeric_categorical"
        categories_seen = {p["category"] for p in result["points"]}
        assert categories_seen <= set(categorical_df["group"].unique())

    def test_categorical_categorical_returns_crosstab(self, messy_df):
        df = messy_df.copy()
        df["region2"] = df["region"]
        result = compute_relationships(df, "region", "region2")
        assert result["kind"] == "categorical_categorical"

    def test_nonexistent_column_raises(self, clean_numeric_df):
        with pytest.raises(ValueError):
            compute_relationships(clean_numeric_df, "x", "does_not_exist")
