"""
Synthetic multi-dataset regression suite.

The original spec's complaint: every analytical engine here was validated
against exactly one real dataset (an IPL cricket CSV) during development.
The engines are written to be dataset-agnostic (see e.g. missingness_engine's
and outlier_engine's docstrings — no hardcoded column/dataset logic anywhere
in this codebase, verified by grep), but "written to be dataset-agnostic"
and "actually behaves sanely across genuinely different data" are different
claims. This file tests the second one directly.

For every dataset in synthetic_datasets.ALL_SYNTHETIC_DATASET_BUILDERS, this
runs the full non-AI pipeline (profiling, quality/risk/readiness scoring,
cleaning + missingness + outlier recommendations, feature engineering,
warnings, analysis recommendations, and a couple of EDA computations) and
checks two tiers of things:

1. Structural sanity, for every dataset, with no domain-specific assumptions:
   nothing crashes, every score stays in its stated bounds, every confidence
   is 0-1, every value_tier/classification comes from the declared enum, and
   every recommendation carries a non-empty reason (the "no arbitrary claims"
   principle used throughout the app, checked mechanically here).

2. A handful of domain-specific spot checks on the datasets that were built
   with a provable answer baked in — e.g. the two datasets with a genuine
   skip-logic structural-missingness column (real estate's pool_size_sqft,
   survey's car_brand) should actually get classified as "structural", not
   just "not crash". A suite that only checks (1) would pass even if
   structural detection quietly stopped firing on non-cricket data — the
   exact regression this suite exists to catch.

(AI/Copilot engines are excluded — those call an external API and are
covered by their own mocked tests elsewhere; this suite covers every engine
that runs on the dataframe alone.)
"""
import numpy as np
import pandas as pd
import pytest

from synthetic_datasets import ALL_SYNTHETIC_DATASET_BUILDERS

from app.engines.profiling_engine import profile_dataset
from app.engines.cleaning_engine import compute_quality_score, generate_recommendations
from app.engines.missingness_engine import classify_missingness
from app.engines.outlier_engine import classify_outliers
from app.engines.feature_engineering_engine import generate_feature_recommendations, VALUE_TIERS
from app.engines.warnings_engine import generate_warnings
from app.engines.quality_engine import compute_analytical_risk, compute_readiness
from app.engines.analysis_engine import recommend_analyses
from app.engines.eda_engine import compute_overview, compute_correlation, compute_missing_patterns

DATASET_NAMES = list(ALL_SYNTHETIC_DATASET_BUILDERS.keys())


@pytest.fixture(params=DATASET_NAMES)
def synthetic_df(request):
    return request.param, ALL_SYNTHETIC_DATASET_BUILDERS[request.param]()


class TestFullPipelineDoesNotCrash:
    """Runs every non-AI engine against every synthetic dataset. A failure
    here means some engine makes an assumption that holds on cricket-shaped
    data but breaks on a genuinely different column mix."""

    def test_profiling(self, synthetic_df):
        name, df = synthetic_df
        profile = profile_dataset(name, df)
        assert profile.rows == len(df)
        assert profile.columns == len(df.columns)
        assert len(profile.columns_detail) == len(df.columns)

    def test_quality_score(self, synthetic_df):
        name, df = synthetic_df
        score = compute_quality_score(df)
        assert 0.0 <= score.overall <= 100.0
        for dim in (score.completeness, score.consistency, score.validity, score.uniqueness):
            assert 0.0 <= dim <= 100.0

    def test_cleaning_recommendations(self, synthetic_df):
        name, df = synthetic_df
        recs = generate_recommendations(df)
        for r in recs:
            assert 0.0 <= r.confidence <= 1.0
            assert r.reason  # never blank, per the "no arbitrary claims" convention

    def test_missingness_classification(self, synthetic_df):
        name, df = synthetic_df
        result = classify_missingness(df)
        for col, c in result.items():
            assert c.classification in ("structural", "conditional", "random", "insufficient_data")
            assert 0.0 <= c.confidence <= 1.0
            assert c.evidence

    def test_outlier_classification(self, synthetic_df):
        name, df = synthetic_df
        result = classify_outliers(df)
        for col, summary in result.items():
            assert summary.total_outliers > 0
            for p in summary.points:
                assert p.tier in ("likely_data_entry_error", "legitimate_rare_value", "statistically_unusual")
                assert 0.0 <= p.confidence <= 1.0

    def test_feature_engineering(self, synthetic_df):
        name, df = synthetic_df
        recs = generate_feature_recommendations(df)
        for r in recs:
            assert 0.0 <= r.confidence <= 1.0
            assert r.value_tier in VALUE_TIERS
            assert r.value_tier_reason

    def test_warnings(self, synthetic_df):
        name, df = synthetic_df
        profile = profile_dataset(name, df)
        warnings = generate_warnings(df, profile.target_candidates)
        for w in warnings:
            assert w.severity in ("info", "warning", "critical")
            assert w.reason

    def test_analytical_risk_and_readiness(self, synthetic_df):
        name, df = synthetic_df
        profile = profile_dataset(name, df)
        risk = compute_analytical_risk(df, profile.target_candidates)
        readiness = compute_readiness(df, profile.target_candidates)
        assert 0.0 <= risk.overall <= 100.0
        assert 0.0 <= readiness.overall <= 100.0

    def test_analysis_recommendations(self, synthetic_df):
        name, df = synthetic_df
        recs = recommend_analyses(df)
        for r in recs:
            assert 0.0 <= r["confidence"] <= 1.0
            assert r["reason"]

    def test_eda_overview_and_correlation(self, synthetic_df):
        name, df = synthetic_df
        overview = compute_overview(df)
        assert overview["rows"] == len(df)
        compute_correlation(df)          # just check it doesn't raise
        compute_missing_patterns(df)     # ditto


class TestDomainSpecificSpotChecks:
    """A handful of assertions with a provable right answer, on datasets
    deliberately built to have one — catches a suite that only checks
    'nothing crashed' from silently missing a real regression."""

    def test_real_estate_pool_size_detected_as_structural(self):
        df = ALL_SYNTHETIC_DATASET_BUILDERS["real_estate_listings"]()
        result = classify_missingness(df)
        assert result["pool_size_sqft"].classification == "structural"
        assert result["pool_size_sqft"].predictor_column == "has_pool"

    def test_survey_car_brand_detected_as_structural(self):
        df = ALL_SYNTHETIC_DATASET_BUILDERS["survey_responses"]()
        result = classify_missingness(df)
        assert result["car_brand"].classification == "structural"
        assert result["car_brand"].predictor_column == "owns_car"

    def test_healthcare_icu_days_detected_as_structural_or_conditional(self):
        # icu_days depends on a NUMERIC threshold (length_of_stay > 5), not a
        # clean categorical split, so this exercises the numeric-correlation
        # path rather than the group-split path — a real but softer signal.
        df = ALL_SYNTHETIC_DATASET_BUILDERS["healthcare_patient"]()
        result = classify_missingness(df)
        assert result["icu_days"].classification in ("structural", "conditional")

    def test_iot_sensor_spikes_detected_as_error_tier(self):
        df = ALL_SYNTHETIC_DATASET_BUILDERS["iot_sensor_readings"]()
        result = classify_outliers(df)
        temp_summary = result["temperature_c"]
        assert temp_summary.tier_counts["likely_data_entry_error"] >= 1

    def test_customer_churn_imbalance_flagged(self):
        df = ALL_SYNTHETIC_DATASET_BUILDERS["customer_churn"]()
        profile = profile_dataset("customer_churn", df)
        warnings = generate_warnings(df, profile.target_candidates)
        assert any("imbalance" in w.title.lower() for w in warnings)
        risk = compute_analytical_risk(df, profile.target_candidates)
        assert risk.overall > 0

    def test_sales_pricing_errors_detected_as_outliers(self):
        df = ALL_SYNTHETIC_DATASET_BUILDERS["sales_transactions"]()
        result = classify_outliers(df)
        assert "price" in result
        assert result["price"].total_outliers >= 5  # the 5 injected pricing errors

    def test_web_traffic_and_finance_timeseries_get_trend_analysis_recommended(self):
        for name in ("web_traffic_timeseries", "finance_stock_prices"):
            df = ALL_SYNTHETIC_DATASET_BUILDERS[name]()
            recs = recommend_analyses(df)
            trend_recs = [r for r in recs if r["analysis_type"] == "trend_analysis"]
            assert trend_recs and trend_recs[0]["recommended"], \
                f"{name} has a clear date span but trend_analysis wasn't recommended"

    def test_every_dataset_produces_at_least_one_recommendation_of_some_kind(self):
        # A dataset producing literally zero recommendations across all
        # engines would suggest something silently isn't firing.
        for name, builder in ALL_SYNTHETIC_DATASET_BUILDERS.items():
            df = builder()
            cleaning = generate_recommendations(df)
            features = generate_feature_recommendations(df)
            assert cleaning or features, f"{name} produced no recommendations at all"
