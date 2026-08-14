import numpy as np
import pandas as pd
import pytest
from app.core.filtering import apply_filter, rows_to_json_safe_records, _json_safe
from app.models.schemas import FilterCondition


@pytest.fixture
def df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West", "North"],
        "revenue": [100.0, 200.0, None, 400.0, 500.0],
        "signed_up": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
    })


class TestApplyFilterOps:
    def test_no_filters_returns_same_frame(self, df):
        out = apply_filter(df, [])
        assert len(out) == len(df)

    def test_eq(self, df):
        out = apply_filter(df, [FilterCondition(column="region", op="eq", value="North")])
        assert len(out) == 2
        assert (out["region"] == "North").all()

    def test_in(self, df):
        out = apply_filter(df, [FilterCondition(column="region", op="in", value=["North", "South"])])
        assert len(out) == 3

    def test_gte(self, df):
        out = apply_filter(df, [FilterCondition(column="revenue", op="gte", value=300)])
        assert len(out) == 2
        assert (out["revenue"] >= 300).all()

    def test_lte(self, df):
        out = apply_filter(df, [FilterCondition(column="revenue", op="lte", value=200)])
        assert len(out) == 2

    def test_between(self, df):
        out = apply_filter(df, [FilterCondition(column="revenue", op="between", value=[150, 450])])
        assert len(out) == 2

    def test_multiple_conditions_and_together(self, df):
        out = apply_filter(df, [
            FilterCondition(column="region", op="eq", value="North"),
            FilterCondition(column="revenue", op="gte", value=300),
        ])
        assert len(out) == 1

    def test_unknown_column_raises(self, df):
        with pytest.raises(ValueError, match="isn't in this dataset"):
            apply_filter(df, [FilterCondition(column="does_not_exist", op="eq", value=1)])

    def test_unknown_op_raises(self, df):
        with pytest.raises(ValueError, match="Unknown filter op"):
            apply_filter(df, [FilterCondition(column="region", op="not_a_real_op", value=1)])

    def test_nan_rows_excluded_from_numeric_filters(self, df):
        out = apply_filter(df, [FilterCondition(column="revenue", op="gte", value=0)])
        assert out["revenue"].isna().sum() == 0


class TestJsonSafeConversion:
    def test_nan_becomes_none(self):
        assert _json_safe(float("nan")) is None

    def test_numpy_int_becomes_native_int(self):
        val = _json_safe(np.int64(42))
        assert val == 42
        assert isinstance(val, int)

    def test_numpy_float_becomes_native_float(self):
        val = _json_safe(np.float64(3.14))
        assert isinstance(val, float)

    def test_timestamp_becomes_isoformat_string(self):
        val = _json_safe(pd.Timestamp("2024-01-01"))
        assert val == "2024-01-01T00:00:00"

    def test_plain_string_passes_through(self):
        assert _json_safe("hello") == "hello"

    def test_rows_to_json_safe_records_full_row(self, df):
        records = rows_to_json_safe_records(df.iloc[[2]])
        assert records[0]["revenue"] is None
        assert isinstance(records[0]["signed_up"], str)
        assert records[0]["region"] == "East"
