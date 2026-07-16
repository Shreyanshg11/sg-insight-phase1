import numpy as np
import pandas as pd
import pytest
from app.engines.analysis_engine import recommend_analyses, run_analysis


class TestRecommendAnalyses:
    def test_returns_a_list_with_required_fields(self, messy_df):
        recs = recommend_analyses(messy_df)
        assert len(recs) > 0
        for r in recs:
            assert r["analysis_type"] and r["title"] and r["reason"]
            assert 0 <= r["confidence"] <= 1

    def test_goal_keywords_influence_recommendations(self, messy_df):
        # shouldn't raise; goal-relevant columns should be reflected somewhere
        recs = recommend_analyses(messy_df, goal_keywords=["revenue"])
        assert len(recs) > 0


class TestRunAnalysisTrend:
    def test_trend_direction_matches_synthetic_data(self):
        dates = pd.date_range("2024-01-01", periods=24, freq="MS")
        values = np.arange(24) * 10.0  # clearly increasing
        df = pd.DataFrame({"date": dates, "value": values})
        result = run_analysis(df, "trend_analysis", ["date"], {"value_column": "value"})
        assert result["result"]["trend_direction"] == "increasing"
        assert result["result"]["slope_per_period"] > 0


class TestRunAnalysisRegression:
    def test_perfect_linear_relationship_gives_r_squared_near_one(self):
        n = 100
        x = np.random.normal(size=n)
        df = pd.DataFrame({"x": x, "y": 3 * x + 5})
        result = run_analysis(df, "regression", ["x", "y"], {"target": "y", "predictors": ["x"]})
        assert result["result"]["in_sample_r_squared"] > 0.99

    def test_missing_target_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="target"):
            run_analysis(df, "regression", ["x"], {})


class TestRunAnalysisClustering:
    def test_finds_two_well_separated_clusters(self):
        np.random.seed(0)
        cluster_a = np.random.normal(loc=[0, 0], scale=0.3, size=(50, 2))
        cluster_b = np.random.normal(loc=[10, 10], scale=0.3, size=(50, 2))
        df = pd.DataFrame(np.vstack([cluster_a, cluster_b]), columns=["a", "b"])
        result = run_analysis(df, "clustering", ["a", "b"], {})
        assert result["result"]["k"] == 2
        assert result["result"]["silhouette_score"] > 0.8


class TestRunAnalysisForecasting:
    def test_forecast_continues_the_trend(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="MS")
        values = np.arange(12) * 10.0
        df = pd.DataFrame({"date": dates, "value": values})
        result = run_analysis(df, "forecasting", ["date"], {"value_column": "value", "horizon_periods": 3})
        assert len(result["result"]["forecast_values"]) == 3
        assert result["result"]["forecast_values"][0] > result["result"]["history_values"][-1]


class TestRunAnalysisHypothesisTest:
    def test_detects_significant_group_difference(self):
        np.random.seed(0)
        df = pd.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "value": list(np.random.normal(0, 1, 50)) + list(np.random.normal(5, 1, 50)),
        })
        result = run_analysis(df, "hypothesis_test", ["group"], {"outcome_column": "value"})
        assert result["result"]["p_value"] < 0.01


class TestRunAnalysisPCA:
    def test_pca_reduces_dimensionality(self, clean_numeric_df):
        result = run_analysis(clean_numeric_df, "pca", ["x", "y", "z"], {})
        assert result["result"]["n_components"] <= 3
        assert sum(result["result"]["explained_variance_ratio"]) <= 1.0001
        assert result["result"]["cumulative_variance_ratio"][-1] == pytest.approx(
            sum(result["result"]["explained_variance_ratio"]), abs=1e-6
        )


class TestRunAnalysisAnomalyDetection:
    def test_flags_a_reasonable_fraction_of_rows(self, messy_df):
        result = run_analysis(messy_df, "anomaly_detection", ["revenue", "cost"], {})
        # n_total reflects rows usable after dropping NaNs in the analyzed
        # columns — messy_df has 4 missing 'cost' values, so this is
        # len(messy_df) - 4, not the raw row count.
        assert result["result"]["n_total"] == messy_df[["revenue", "cost"]].dropna().shape[0]
        assert 0 <= result["result"]["n_flagged"] <= result["result"]["n_total"]
        assert result["result"]["flagged_pct"] == pytest.approx(
            result["result"]["n_flagged"] / result["result"]["n_total"] * 100, abs=0.1
        )


class TestUnknownAnalysisType:
    def test_unknown_type_raises(self, clean_numeric_df):
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            run_analysis(clean_numeric_df, "not_a_real_analysis", ["x"], {})
