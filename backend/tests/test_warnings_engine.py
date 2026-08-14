import numpy as np
import pandas as pd
from app.engines.warnings_engine import generate_warnings


class TestGenerateWarnings:
    def test_returns_a_list_of_trust_warnings(self, messy_df):
        warnings = generate_warnings(messy_df)
        assert isinstance(warnings, list)
        for w in warnings:
            assert w.severity in ("info", "warning", "critical")
            assert w.id and w.title and w.description and w.reason

    def test_clean_data_produces_fewer_or_no_critical_warnings(self, clean_numeric_df, messy_df):
        clean_warnings = generate_warnings(clean_numeric_df)
        messy_warnings = generate_warnings(messy_df)
        clean_critical = sum(1 for w in clean_warnings if w.severity == "critical")
        messy_critical = sum(1 for w in messy_warnings if w.severity == "critical")
        assert clean_critical <= messy_critical

    def test_empty_dataframe_does_not_crash(self):
        warnings = generate_warnings(pd.DataFrame())
        assert isinstance(warnings, list)

    def test_target_candidates_param_accepted(self, messy_df):
        # shouldn't raise, whether or not it changes output
        generate_warnings(messy_df, target_candidates=["region"])

    def test_zero_variance_pair_does_not_raise_or_warn(self):
        # A numeric column that's only non-null within one value of a
        # boolean-like target candidate is constant after the dropna-aligned
        # subset — correlation is undefined there, and this should be
        # skipped cleanly rather than let numpy divide by zero.
        import warnings as pywarnings
        df = pd.DataFrame({
            "has_pool": [True] * 20 + [False] * 20,
            "pool_size_sqft": [300.0] * 20 + [np.nan] * 20,  # constant among non-null rows
        })
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("error")
            result = generate_warnings(df, target_candidates=["has_pool"])
        assert isinstance(result, list)
