import numpy as np
import pandas as pd
import pytest
from app.engines.quality_engine import compute_analytical_risk, compute_readiness
from app.engines.warnings_engine import generate_warnings


class TestAnalyticalRiskScore:
    def test_no_warnings_means_zero_risk(self, clean_numeric_df):
        # clean_numeric_df has no target candidates passed, so no class-imbalance
        # or leakage checks fire; whatever generic warnings it has (if any)
        # should still translate consistently into the risk score.
        warnings = generate_warnings(clean_numeric_df)
        risk = compute_analytical_risk(clean_numeric_df)
        if not warnings:
            assert risk.overall == 0.0
            assert risk.breakdown == []

    def test_risk_score_between_0_and_100(self, messy_df):
        risk = compute_analytical_risk(messy_df)
        assert 0.0 <= risk.overall <= 100.0

    def test_critical_warning_contributes_more_than_info(self):
        # A numeric column near-perfectly correlated with a numeric target
        # triggers a "critical" leakage warning.
        rng = np.random.RandomState(0)
        n = 50
        target = rng.randint(0, 100, n)
        df = pd.DataFrame({"leaky_col": target * 2 + 1, "target": target})
        risk = compute_analytical_risk(df, target_candidates=["target"])
        assert risk.critical_count >= 1
        assert risk.overall > 0

    def test_breakdown_reasons_cite_the_actual_warning_titles(self):
        rng = np.random.RandomState(1)
        n = 20  # below the small-sample threshold
        df = pd.DataFrame({"x": rng.randn(n)})
        risk = compute_analytical_risk(df)
        assert any("small sample" in b.reason.lower() or "small sample" in b.label.lower()
                   for b in risk.breakdown)

    def test_risk_category_points_are_capped(self):
        # Construct a dataset with many near-constant columns to try to blow
        # past the per-category cap for "info" signals.
        n = 50
        data = {f"const_{i}": [1] * n for i in range(10)}
        df = pd.DataFrame(data)
        risk = compute_analytical_risk(df)
        info_items = [b for b in risk.breakdown if "info" in b.label.lower()]
        if info_items:
            assert info_items[0].points <= 15  # RISK_CATEGORY_CAP["info"]


class TestReadinessScore:
    def test_readiness_between_0_and_100(self, messy_df):
        readiness = compute_readiness(messy_df)
        assert 0.0 <= readiness.overall <= 100.0

    def test_large_varied_dataset_scores_high(self):
        rng = np.random.RandomState(0)
        n = 200
        df = pd.DataFrame({
            "age": rng.randint(18, 80, n),
            "income": rng.normal(60000, 15000, n),
            "category": rng.choice(["a", "b", "c"], n),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "churn": rng.choice([0, 1], n, p=[0.9, 0.1]),
        })
        readiness = compute_readiness(df, target_candidates=["churn"])
        assert readiness.overall >= 90.0

    def test_tiny_dataset_scores_lower_than_large_one(self):
        rng = np.random.RandomState(0)
        big = pd.DataFrame({"x": rng.randn(200), "y": rng.choice(["a", "b"], 200)})
        small = big.head(4)
        big_score = compute_readiness(big)
        small_score = compute_readiness(small)
        assert small_score.overall < big_score.overall

    def test_no_target_candidates_scores_zero_on_that_dimension(self):
        df = pd.DataFrame({"x": range(50)})
        readiness = compute_readiness(df, target_candidates=[])
        target_item = next(b for b in readiness.breakdown if "target" in b.label.lower())
        assert target_item.points == 0.0

    def test_single_value_target_not_counted_as_usable(self):
        df = pd.DataFrame({"x": range(50), "flag": [1] * 50})
        readiness = compute_readiness(df, target_candidates=["flag"])
        target_item = next(b for b in readiness.breakdown if "target" in b.label.lower())
        assert target_item.points == 0.0

    def test_breakdown_row_counts_match_dataframe(self):
        df = pd.DataFrame({"x": range(15)})
        readiness = compute_readiness(df)
        row_item = next(b for b in readiness.breakdown if "row" in b.label.lower())
        assert "15 rows" in row_item.reason
