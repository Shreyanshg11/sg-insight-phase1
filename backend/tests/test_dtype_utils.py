import numpy as np
import pandas as pd
from app.core.dtype_utils import _is_stringy, stringy_columns


class TestIsStringy:
    def test_plain_object_dtype_is_stringy(self):
        s = pd.Series(["a", "b", "c"], dtype=object)
        assert _is_stringy(s) is True

    def test_pandas_string_dtype_is_stringy(self):
        s = pd.Series(["a", "b", "c"], dtype="string")
        assert _is_stringy(s) is True

    def test_numeric_dtype_is_not_stringy(self):
        s = pd.Series([1, 2, 3], dtype="int64")
        assert _is_stringy(s) is False

    def test_float_dtype_is_not_stringy(self):
        s = pd.Series([1.0, 2.0], dtype="float64")
        assert _is_stringy(s) is False

    def test_bool_dtype_is_not_stringy(self):
        s = pd.Series([True, False], dtype="bool")
        assert _is_stringy(s) is False

    def test_datetime_dtype_is_not_stringy(self):
        s = pd.Series(pd.date_range("2024-01-01", periods=3))
        assert _is_stringy(s) is False

    def test_category_of_strings_is_currently_not_stringy(self):
        # Documents ACTUAL behavior, not necessarily ideal behavior: a
        # category-dtype Series (e.g. what pd.qcut() produces — see
        # feature_engineering_engine.py's quartile_bucket) is neither
        # is_object_dtype nor is_string_dtype, so it's NOT currently treated
        # as text-like here even when its categories are strings. Flagged
        # separately as a possible gap — see README — rather than silently
        # changed as a side effect of adding tests.
        s = pd.Series(["a", "b", "a"], dtype="category")
        assert _is_stringy(s) is False

    def test_bare_dtype_object_input(self):
        assert _is_stringy(np.dtype("object")) is True
        assert _is_stringy(np.dtype("int64")) is False


class TestStringyColumns:
    def test_returns_only_text_columns(self):
        df = pd.DataFrame({
            "name": ["a", "b"],
            "age": [1, 2],
            "score": [1.5, 2.5],
            "active": [True, False],
        })
        assert stringy_columns(df) == ["name"]

    def test_empty_df_returns_empty_list(self):
        assert stringy_columns(pd.DataFrame()) == []

    def test_all_numeric_df_returns_empty_list(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert stringy_columns(df) == []

    def test_preserves_column_order(self):
        df = pd.DataFrame({"z_text": ["a"], "a_num": [1], "m_text": ["b"]})
        assert stringy_columns(df) == ["z_text", "m_text"]
