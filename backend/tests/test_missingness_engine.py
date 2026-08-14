"""
Tests for missingness_engine — each fixture is constructed so the "right"
classification is provable from how it was built, mirroring the rest of the
suite's approach (see conftest.py docstring).
"""
import numpy as np
import pandas as pd
import pytest
from app.engines.missingness_engine import classify_missingness, classify_column_missingness


@pytest.fixture
def structural_df():
    """'dismissal_type' is populated ONLY when 'event' == 'wicket' — a clean
    on/off split, i.e. textbook structural (missing-by-design) missingness."""
    rng = np.random.RandomState(0)
    n = 200
    event = rng.choice(["wicket", "no_wicket"], size=n, p=[0.25, 0.75])
    dismissal = [rng.choice(["bowled", "caught", "lbw"]) if e == "wicket" else np.nan for e in event]
    return pd.DataFrame({"event": event, "dismissal_type": dismissal, "runs": rng.randint(0, 7, n)})


@pytest.fixture
def conditional_df():
    """'income' is missing much more often for older respondents (55%) than
    younger ones (~8%) — a real but imperfect correlation with 'age'."""
    rng = np.random.RandomState(1)
    n = 300
    age = rng.randint(18, 80, n)
    income = np.where(
        age > 55,
        np.where(rng.rand(n) < 0.55, np.nan, rng.randint(30, 120, n)),
        np.where(rng.rand(n) < 0.08, np.nan, rng.randint(30, 120, n)),
    )
    return pd.DataFrame({"age": age, "income": income})


@pytest.fixture
def random_df():
    """'val' has missingness applied uniformly at random, unrelated to 'cat'
    or to val's own distribution — no predictor should explain it."""
    rng = np.random.RandomState(2)
    n = 300
    df = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], n), "val": rng.randn(n)})
    mask = rng.rand(n) < 0.15
    df.loc[mask, "val"] = np.nan
    return df


class TestStructuralMissingness:
    def test_classified_structural(self, structural_df):
        result = classify_column_missingness(structural_df, "dismissal_type")
        assert result.classification == "structural"
        assert result.predictor_column == "event"
        assert result.confidence >= 0.85

    def test_evidence_cites_the_actual_group_rates(self, structural_df):
        result = classify_column_missingness(structural_df, "dismissal_type")
        joined = " ".join(result.evidence)
        assert "event" in joined
        assert "100.0%" in joined  # no_wicket group is 100% missing by construction

    def test_recommendation_hint_warns_against_dropping(self, structural_df):
        result = classify_column_missingness(structural_df, "dismissal_type")
        assert "missing by design" in result.recommendation_hint.lower() or \
               "not applicable" in result.recommendation_hint.lower()


class TestConditionalMissingness:
    def test_classified_conditional(self, conditional_df):
        result = classify_column_missingness(conditional_df, "income")
        assert result.classification == "conditional"
        assert result.predictor_column == "age"

    def test_not_misclassified_as_structural(self, conditional_df):
        # The split is real but not clean (55% vs 8%, not ~100% vs ~0%),
        # so this must NOT clear the stricter structural bar.
        result = classify_column_missingness(conditional_df, "income")
        assert result.classification != "structural"


class TestRandomMissingness:
    def test_classified_random(self, random_df):
        result = classify_column_missingness(random_df, "val")
        assert result.classification == "random"

    def test_evidence_is_honest_about_absence_of_proof(self, random_df):
        result = classify_column_missingness(random_df, "val")
        joined = " ".join(result.evidence).lower()
        assert "not a statistical proof" in joined or "absence-of-evidence" in joined


class TestInsufficientData:
    def test_too_few_missing_values(self):
        df = pd.DataFrame({"a": list(range(20)) + [None], "b": list(range(21))})
        result = classify_column_missingness(df, "a")
        assert result.classification == "insufficient_data"

    def test_no_missing_values_at_all(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = classify_column_missingness(df, "a")
        assert result.classification == "insufficient_data"


class TestClassifyMissingness:
    def test_only_includes_columns_with_missing_values(self, structural_df):
        result = classify_missingness(structural_df)
        assert "dismissal_type" in result
        assert "event" not in result  # no missing values in 'event'
        assert "runs" not in result   # no missing values in 'runs'

    def test_empty_when_nothing_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert classify_missingness(df) == {}
