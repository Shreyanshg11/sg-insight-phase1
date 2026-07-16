from app.engines.profiling_engine import profile_dataset


class TestProfileDataset:
    def test_reports_row_and_column_counts(self, messy_df):
        profile = profile_dataset("ds1", messy_df)
        assert profile.rows == len(messy_df)
        assert profile.columns == len(messy_df.columns)

    def test_detects_duplicate_rows(self, messy_df):
        profile = profile_dataset("ds1", messy_df)
        assert profile.duplicate_rows == 1
        assert profile.duplicate_pct > 0

    def test_no_duplicates_in_clean_data(self, clean_numeric_df):
        profile = profile_dataset("ds1", clean_numeric_df)
        assert profile.duplicate_rows == 0

    def test_infers_date_column_role(self, messy_df):
        profile = profile_dataset("ds1", messy_df)
        date_col = next(c for c in profile.columns_detail if c.name == "signup_date")
        assert date_col.inferred_role == "date"
        assert "signup_date" in profile.date_columns

    def test_infers_numeric_column_role(self, messy_df):
        profile = profile_dataset("ds1", messy_df)
        revenue_col = next(c for c in profile.columns_detail if c.name == "revenue")
        assert revenue_col.inferred_role == "numeric"

    def test_goal_keywords_mark_relevant_columns(self, messy_df):
        profile = profile_dataset("ds1", messy_df, investigation_goal="find why revenue dropped",
                                   goal_keywords=["revenue"])
        revenue_col = next(c for c in profile.columns_detail if c.name == "revenue")
        assert revenue_col.goal_relevant is True
        assert "revenue" in profile.goal_relevant_columns

    def test_missing_values_reported_per_column(self, messy_df):
        profile = profile_dataset("ds1", messy_df)
        cost_col = next(c for c in profile.columns_detail if c.name == "cost")
        assert cost_col.missing_count == 4
