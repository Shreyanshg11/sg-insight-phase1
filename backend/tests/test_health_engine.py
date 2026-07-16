import pandas as pd
from app.engines.health_engine import compute_health_dashboard


class TestComputeHealthDashboard:
    def test_returns_categories_with_valid_status(self, messy_df):
        categories = compute_health_dashboard(messy_df)
        assert len(categories) > 0
        for c in categories:
            assert c.status in ("good", "warning", "critical")
            assert c.key and c.label and c.summary and c.detail

    def test_missing_values_category_reports_affected_columns(self, messy_df):
        categories = compute_health_dashboard(messy_df)
        missing_cat = next(c for c in categories if "missing" in c.key.lower())
        assert "cost" in missing_cat.affected_columns
        # messy_df's overall missing rate is low (~2%) — correctly still "good"
        # by this category's dataset-wide-rate threshold, not per-column presence.
        assert missing_cat.status == "good"

    def test_high_missing_rate_is_flagged_not_good(self):
        import numpy as np
        df = pd.DataFrame({"a": [1.0] * 100})
        df.loc[:29, "a"] = np.nan  # 30% missing — should cross the "good" threshold
        categories = compute_health_dashboard(df)
        missing_cat = next(c for c in categories if "missing" in c.key.lower())
        assert missing_cat.status != "good"

    def test_duplicates_category_flags_the_known_duplicate(self, messy_df):
        categories = compute_health_dashboard(messy_df)
        dup_cat = next(c for c in categories if "duplicate" in c.key.lower())
        assert dup_cat.status != "good"

    def test_clean_data_scores_well_on_missing_and_duplicates(self, clean_numeric_df):
        categories = compute_health_dashboard(clean_numeric_df)
        missing_cat = next(c for c in categories if "missing" in c.key.lower())
        dup_cat = next(c for c in categories if "duplicate" in c.key.lower())
        assert missing_cat.status == "good"
        assert dup_cat.status == "good"

    def test_empty_dataframe_does_not_crash(self):
        categories = compute_health_dashboard(pd.DataFrame())
        assert isinstance(categories, list)
