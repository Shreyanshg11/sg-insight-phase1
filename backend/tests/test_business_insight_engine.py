import pandas as pd
import pytest

from app.engines.business_insight_engine import (
    _numeric_insights,
    build_business_insights,
)


def test_segment_distribution_uses_actual_category_counts():
    df = pd.DataFrame({
        "Segment": ["A"] * 6 + ["B"] * 4,
        "Revenue": list(range(1, 11)),
    })

    result = build_business_insights(df)
    segment = next(
        item for item in result["segment_intelligence"]
        if item["column"] == "Segment"
    )

    assert segment["top_segments"][0]["value"] == "A"
    assert segment["top_segments"][0]["count"] == 6
    assert segment["top_segments"][0]["share_pct"] == 60.0


def test_recognised_spending_column_gets_spending_analysis():
    df = pd.DataFrame({
        "CustomerType": ["Retail"] * 10,
        "Revenue": [10, 12, 14, 16, 18, 20, 22, 24, 26, 1000],
    })

    result = build_business_insights(df)

    assert "Revenue" in result["spending_intelligence"]
    assert result["spending_intelligence"]["Revenue"]["outlier_count"] >= 1


def test_arbitrary_numeric_column_is_not_labelled_spending():
    df = pd.DataFrame({"temperature": [10, 11, 12, 13, 14, 15]})

    result = build_business_insights(df)

    assert result["spending_intelligence"] == {}
    assert "temperature" in result["numeric_intelligence"]


def test_category_numeric_comparison_requires_minimum_group_size():
    df = pd.DataFrame({
        "Region": ["North"] * 10 + ["South"] * 10,
        "Sales": list(range(10)) + list(range(10, 20)),
    })

    result = build_business_insights(df)

    comparison = next(
        item for item in result["category_numeric_comparisons"]
        if item["category_column"] == "Region"
        and item["metric_column"] == "Sales"
    )

    assert len(comparison["groups"]) == 2
    assert all(group["count"] >= 5 for group in comparison["groups"])


def test_outlier_opportunity_is_qualified_not_called_fraud():
    df = pd.DataFrame({
        "TransactionAmount": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]
    })

    result = build_business_insights(df)

    assert any(
        item["type"] == "spending_outlier_review"
        for item in result["business_opportunities"]
    )
    assert all(
        "fraud" not in item["recommendation"].lower()
        for item in result["business_opportunities"]
    )


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        build_business_insights(pd.DataFrame())


def test_engine_does_not_modify_input_dataframe():
    df = pd.DataFrame({
        "Segment": ["A", "B", "A", "B", "A", "B"],
        "Revenue": [10, 20, 30, 40, 50, 60],
    })
    original = df.copy(deep=True)

    build_business_insights(df)

    pd.testing.assert_frame_equal(df, original)


def test_identifier_columns_are_excluded_from_category_numeric_comparisons():
    df = pd.DataFrame({
        "Segment": ["A"] * 10 + ["B"] * 10,
        "CustomerID": list(range(1, 21)),
        "Revenue": list(range(10, 20)) + list(range(20, 30)),
    })

    result = build_business_insights(df)

    comparison_metrics = {
        item["metric_column"]
        for item in result["category_numeric_comparisons"]
        if item["category_column"] == "Segment"
    }

    assert "CustomerID" not in comparison_metrics
    assert "Revenue" in comparison_metrics

    # ID remains available in the general numeric profile.
    assert "CustomerID" in result["numeric_intelligence"]

def test_numeric_insights_ignore_nan_and_infinite_values():
    import warnings

    df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, float("inf"), float("-inf"), float("nan")]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        info = _numeric_insights(df)["value"]

    assert info["count"] == 3
    assert info["mean"] == 2.0
    assert info["minimum"] == 1.0
    assert info["maximum"] == 3.0
