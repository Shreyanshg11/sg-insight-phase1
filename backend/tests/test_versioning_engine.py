import pandas as pd
from app.engines.versioning_engine import compare_versions


class TestCompareVersions:
    def test_detects_added_and_removed_columns(self):
        df_a = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_b = pd.DataFrame({"a": [1, 2], "c": [5, 6]})
        result = compare_versions(df_a, df_b)
        assert result["columns_added"] == ["c"]
        assert result["columns_removed"] == ["b"]

    def test_identical_dataframes_show_no_changes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = compare_versions(df, df.copy())
        assert result["columns_added"] == []
        assert result["columns_removed"] == []

    def test_row_counts_reported(self):
        df_a = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df_b = pd.DataFrame({"a": [1, 2, 3]})
        result = compare_versions(df_a, df_b)
        assert result["rows_a"] == 5
        assert result["rows_b"] == 3

    def test_value_changes_detected_in_common_column(self):
        df_a = pd.DataFrame({"a": [1, 2, 3]})
        df_b = pd.DataFrame({"a": [1, 99, 3]})
        result = compare_versions(df_a, df_b)
        changed = next(c for c in result["column_changes"] if c["column"] == "a")
        assert changed["cells_changed"] == 1
