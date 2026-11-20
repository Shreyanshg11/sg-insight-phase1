"""Deterministic, read-only business insights computed from the actual DataFrame."""

import math
import re

import numpy as np
import pandas as pd
from app.core.iqr import iqr_bounds, outlier_mask


SPENDING_NAME_PATTERN = re.compile(
    r"spend|sales|revenue|amount|price|cost|monetary|"
    r"purchase|order.?value|transaction.?value|total.?value",
    re.IGNORECASE,
)


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _numeric_insights(df):
    result = {}
    for column in df.select_dtypes(include="number").columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        series = series[np.isfinite(series)]
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        median = float(series.median())
        q3 = float(series.quantile(0.75))
        bounds = iqr_bounds(series)  # shared IQR policy (finite values, >= 8, non-zero IQR)
        skewness = _safe_float(series.skew()) if len(series) >= 3 else None

        result[str(column)] = {
            "count": int(series.count()),
            "mean": _safe_float(series.mean()),
            "median": _safe_float(median),
            "minimum": _safe_float(series.min()),
            "maximum": _safe_float(series.max()),
            "q1": _safe_float(q1),
            "q3": _safe_float(q3),
            "skewness": skewness,
            "outlier_count": int(outlier_mask(series, bounds).sum()),
            "outlier_method": "1.5x IQR",
            "outlier_status": "evaluated" if bounds is not None else "not_evaluated",
        }
    return result


def _segment_insights(df, max_categories=5):
    result = []
    row_count = len(df)

    for column in df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns:
        series = df[column].dropna()
        if series.empty:
            continue

        counts = series.value_counts(dropna=True)
        top = counts.head(max_categories)

        result.append({
            "column": str(column),
            "unique_values": int(series.nunique()),
            "observed_count": int(series.count()),
            "top_segments": [
                {
                    "value": str(value),
                    "count": int(count),
                    "share_pct": round(float(count) / row_count * 100, 2),
                }
                for value, count in top.items()
            ],
        })

    return result


def _is_identifier_column(column):
    """Identify common ID/key columns that should not be compared as metrics."""
    name = str(column).strip()

    if re.search(r"(?:^|[_\W])(id|uuid|guid|key)(?:$|[_\W])", name, re.IGNORECASE):
        return True

    # Handle common camelCase/PascalCase forms such as CustomerID.
    if re.search(r"ID$", name):
        return True

    if re.fullmatch(r"uuid|guid", name, re.IGNORECASE):
        return True

    return False


def _category_numeric_comparisons(df, max_pairs=20):
    result = []
    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if not _is_identifier_column(column)
    ]
    categorical_columns = list(
        df.select_dtypes(include=["object", "category", "string", "bool"]).columns
    )

    for category_column in categorical_columns:
        distinct = df[category_column].nunique(dropna=True)
        if distinct < 2 or distinct > 12:
            continue

        for numeric_column in numeric_columns:
            pair = df[[category_column, numeric_column]].dropna()
            if pair.empty:
                continue

            minimum_group_size = max(5, math.ceil(len(pair) * 0.01))
            grouped = pair.groupby(category_column)[numeric_column].agg(
                ["count", "mean", "median"]
            )
            grouped = grouped[grouped["count"] >= minimum_group_size]

            if len(grouped) < 2:
                continue

            grouped = grouped.sort_values("mean", ascending=False).head(8)

            result.append({
                "category_column": str(category_column),
                "metric_column": str(numeric_column),
                "minimum_group_size": int(minimum_group_size),
                "groups": [
                    {
                        "category": str(index),
                        "count": int(row["count"]),
                        "mean": _safe_float(row["mean"]),
                        "median": _safe_float(row["median"]),
                    }
                    for index, row in grouped.iterrows()
                ],
                "interpretation_limit": (
                    "Descriptive group differences only; not causal evidence."
                ),
            })

            if len(result) >= max_pairs:
                return result

    return result


def build_business_insights(df):
    """Build generic, evidence-backed insights without changing the DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot analyse an empty dataset.")

    numeric = _numeric_insights(df)
    segments = _segment_insights(df)
    comparisons = _category_numeric_comparisons(df)

    spending_metrics = {
        column: stats
        for column, stats in numeric.items()
        if SPENDING_NAME_PATTERN.search(column)
    }

    opportunities = []

    for segment in segments:
        if not segment["top_segments"]:
            continue

        dominant = segment["top_segments"][0]
        if dominant["share_pct"] >= 50:
            opportunities.append({
                "type": "segment_concentration",
                "column": segment["column"],
                "evidence": dominant,
                "recommendation": (
                    "Review dependence on the dominant group and assess "
                    "whether other groups offer diversification opportunities."
                ),
                "caveat": "Concentration alone does not establish business risk.",
            })

    for column, stats in spending_metrics.items():
        if stats["outlier_count"] > 0:
            opportunities.append({
                "type": "spending_outlier_review",
                "column": column,
                "evidence": {
                    "outlier_count": stats["outlier_count"],
                    "median": stats["median"],
                    "maximum": stats["maximum"],
                },
                "recommendation": (
                    "Review unusually high or low values in this metric "
                    "to distinguish legitimate high-value activity from "
                    "possible data-quality issues."
                ),
                "caveat": "IQR outliers are not automatically errors or fraud.",
            })

        if stats["skewness"] is not None and abs(stats["skewness"]) >= 1:
            opportunities.append({
                "type": "spending_distribution_review",
                "column": column,
                "evidence": {
                    "skewness": stats["skewness"],
                    "mean": stats["mean"],
                    "median": stats["median"],
                },
                "recommendation": (
                    "Inspect the distribution and high-value observations; "
                    "the mean may not represent a typical record well."
                ),
                "caveat": "Skewness does not explain why the distribution occurs.",
            })

    return {
        "method": "Deterministic Pandas analysis",
        "read_only": True,
        "row_count": int(len(df)),
        "segment_intelligence": segments,
        "numeric_intelligence": numeric,
        "spending_intelligence": spending_metrics,
        "category_numeric_comparisons": comparisons,
        "business_opportunities": opportunities,
        "limitations": [
            "Only columns identifiable by their data types are analysed.",
            "Spending-specific analysis is limited to recognisable metric names.",
            "Group differences are descriptive and do not establish causation.",
            "Statistical outliers require review and are not automatically errors.",
            "No clustering, PCA, forecasting, or predictive model is run here.",
        ],
    }
