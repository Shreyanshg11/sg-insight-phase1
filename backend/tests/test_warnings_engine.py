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
