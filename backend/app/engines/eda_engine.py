"""
EDA Engine (#13 — "Improve EDA Navigation").

Each function below is independently fetchable (its own endpoint in
routers/eda.py) rather than one monolithic "run EDA" call, matching the
brief's Overview / Distributions / Correlation / Missing Patterns /
Relationships / Category Analysis / Time Analysis sections and the frontend
EDA sub-navigation. Read-only — this module never mutates the dataset.
"""
import numpy as np
import pandas as pd

from app.core.dtype_utils import stringy_columns
from app.engines.profiling_engine import _is_date_column

MAX_CORRELATION_COLUMNS = 15   # keeps the heatmap/matrix readable and O(n^2) bounded
MAX_CATEGORY_BARS = 10         # top-N categories shown; rest collapse into "Other"
SCATTER_SAMPLE_SIZE = 200      # cap points returned for relationship scatter plots


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def _date_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if _is_date_column(df[col], col)]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    date_cols = set(_date_columns(df))
    return [col for col in stringy_columns(df) if col not in date_cols]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def compute_overview(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    numeric_cols = _numeric_columns(df)
    date_cols = _date_columns(df)
    categorical_cols = _categorical_columns(df)

    summary = []
    for col in df.columns:
        series = df[col]
        missing_pct = round(float(series.isna().mean() * 100), 1) if n_rows else 0.0
        row = {"column": col, "missing_pct": missing_pct}
        if col in numeric_cols:
            s = series.dropna()
            row.update({
                "kind": "numeric",
                "mean": round(float(s.mean()), 3) if len(s) else None,
                "median": round(float(s.median()), 3) if len(s) else None,
                "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 3) if len(s) else None,
                "max": round(float(s.max()), 3) if len(s) else None,
            })
        elif col in date_cols:
            parsed = pd.to_datetime(series, errors="coerce").dropna()
            row.update({
                "kind": "date",
                "min": parsed.min().date().isoformat() if len(parsed) else None,
                "max": parsed.max().date().isoformat() if len(parsed) else None,
            })
        else:
            s = series.dropna()
            top = s.value_counts().head(1)
            row.update({
                "kind": "categorical",
                "unique_count": int(s.nunique()),
                "top_value": str(top.index[0]) if len(top) else None,
                "top_value_pct": round(float(top.iloc[0] / len(s) * 100), 1) if len(s) else None,
            })
        summary.append(row)

    return {
        "rows": n_rows,
        "columns": len(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def compute_distributions(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    series = df[column]

    if column in _numeric_columns(df):
        s = series.dropna()
        if s.empty:
            return {"column": column, "kind": "numeric", "stats": {}, "bins": [],
                    "note": "No non-missing values to compute a distribution from."}
        counts, edges = np.histogram(s, bins=10)
        bins = [{"label": f"{round(edges[i], 2)} - {round(edges[i+1], 2)}", "count": int(counts[i])}
                for i in range(len(counts))]
        stats = {
            "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
            "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
            "skew": round(float(s.skew()), 3) if len(s) > 2 else 0.0,
        }
        return {"column": column, "kind": "numeric", "stats": stats, "bins": bins, "note": ""}

    s = series.dropna().astype(str)
    if s.empty:
        return {"column": column, "kind": "categorical", "stats": {}, "bins": [],
                "note": "No non-missing values to compute a distribution from."}
    counts = s.value_counts()
    top = counts.head(MAX_CATEGORY_BARS)
    bins = [{"label": str(k), "count": int(v)} for k, v in top.items()]
    other_count = int(counts.iloc[MAX_CATEGORY_BARS:].sum())
    note = ""
    if other_count > 0:
        bins.append({"label": "Other", "count": other_count})
        note = f"Showing top {MAX_CATEGORY_BARS} of {counts.shape[0]} distinct values; the rest are grouped into 'Other'."
    stats = {"unique_count": int(counts.shape[0]), "missing_pct": round(float(series.isna().mean() * 100), 1)}
    return {"column": column, "kind": "categorical", "stats": stats, "bins": bins, "note": note}


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def _strength_label(corr: float) -> str:
    abs_corr = abs(corr)
    if abs_corr >= 0.6:
        return "strong"
    if abs_corr >= 0.3:
        return "moderate"
    return "weak"


def compute_correlation(df: pd.DataFrame) -> dict:
    numeric_cols = _numeric_columns(df)
    truncated_note = ""
    if len(numeric_cols) > MAX_CORRELATION_COLUMNS:
        # Keep the highest-variance columns — they're the ones most likely to
        # carry a real linear relationship worth looking at.
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(MAX_CORRELATION_COLUMNS).index.tolist()
        truncated_note = (f"Limited to the {MAX_CORRELATION_COLUMNS} highest-variance numeric columns "
                           "to keep the matrix readable. ")

    if len(numeric_cols) < 2:
        return {"columns": numeric_cols, "matrix": [], "top_pairs": [],
                "note": truncated_note + "Fewer than 2 numeric columns — nothing to correlate."}

    corr = df[numeric_cols].corr(method="pearson")
    matrix = [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in corr.values]

    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            pairs.append({"column_a": numeric_cols[i], "column_b": numeric_cols[j],
                          "correlation": round(float(val), 3), "strength": _strength_label(val)})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    return {
        "columns": numeric_cols,
        "matrix": matrix,
        "top_pairs": pairs[:8],
        "note": truncated_note + "Pearson correlation measures linear relationships only — "
                "two columns can be strongly (non-linearly) related and still show a low value here.",
    }


# ---------------------------------------------------------------------------
# Missing Patterns
# ---------------------------------------------------------------------------

def compute_missing_patterns(df: pd.DataFrame) -> dict:
    missing_mask = df.isna()
    cols_with_missing = [c for c in df.columns if missing_mask[c].any()]

    co_missing_pairs = []
    for i in range(len(cols_with_missing)):
        for j in range(i + 1, len(cols_with_missing)):
            a, b = cols_with_missing[i], cols_with_missing[j]
            both = int((missing_mask[a] & missing_mask[b]).sum())
            either = int((missing_mask[a] | missing_mask[b]).sum())
            if either == 0 or both == 0:
                continue
            jaccard = round(both / either, 3)
            co_missing_pairs.append({"column_a": a, "column_b": b, "rows_missing_both": both,
                                      "co_missing_ratio": jaccard})
    co_missing_pairs.sort(key=lambda p: p["co_missing_ratio"], reverse=True)

    missing_per_row = missing_mask.sum(axis=1)
    buckets = [
        {"label": "0 missing", "count": int((missing_per_row == 0).sum())},
        {"label": "1-2 missing", "count": int(((missing_per_row >= 1) & (missing_per_row <= 2)).sum())},
        {"label": "3+ missing", "count": int((missing_per_row >= 3).sum())},
    ]

    return {
        "columns_with_missing": cols_with_missing,
        "co_missing_pairs": co_missing_pairs[:10],
        "missing_by_row_bucket": buckets,
        "note": "co_missing_ratio is the Jaccard overlap of missingness between two columns (rows missing "
                "both / rows missing either) — a high ratio suggests the same rows/records are affected, "
                "not that the columns cause each other to be missing.",
    }


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def compute_relationships(df: pd.DataFrame, column_x: str, column_y: str) -> dict:
    if column_x not in df.columns or column_y not in df.columns:
        raise ValueError("Both columns must exist in the dataset")

    numeric_cols = set(_numeric_columns(df))
    x_numeric, y_numeric = column_x in numeric_cols, column_y in numeric_cols

    if x_numeric and y_numeric:
        pair = df[[column_x, column_y]].dropna()
        sample = pair.sample(min(len(pair), SCATTER_SAMPLE_SIZE), random_state=42) if len(pair) else pair
        points = [{"x": round(float(r[column_x]), 3), "y": round(float(r[column_y]), 3)}
                  for _, r in sample.iterrows()]
        corr = round(float(pair[column_x].corr(pair[column_y])), 3) if len(pair) > 1 else None
        return {"column_x": column_x, "column_y": column_y, "kind": "numeric_numeric",
                "points": points, "stats": {"correlation": corr, "n": len(pair)},
                "note": f"Showing {len(points)} sampled point(s)" +
                        (f" of {len(pair)} total." if len(pair) > SCATTER_SAMPLE_SIZE else ".")}

    if x_numeric != y_numeric:
        num_col, cat_col = (column_x, column_y) if x_numeric else (column_y, column_x)
        pair = df[[num_col, cat_col]].dropna()
        grouped = pair.groupby(cat_col)[num_col].agg(["mean", "median", "count"]).sort_values("count", ascending=False)
        top = grouped.head(MAX_CATEGORY_BARS)
        points = [{"category": str(idx), "mean": round(float(row["mean"]), 3),
                   "median": round(float(row["median"]), 3), "count": int(row["count"])}
                  for idx, row in top.iterrows()]
        note = ""
        if len(grouped) > MAX_CATEGORY_BARS:
            note = f"Showing the {MAX_CATEGORY_BARS} most frequent categories of {len(grouped)} total."
        return {"column_x": column_x, "column_y": column_y, "kind": "numeric_categorical",
                "points": points, "stats": {"categories_shown": len(top), "categories_total": len(grouped)},
                "note": note}

    pair = df[[column_x, column_y]].dropna().astype(str)
    ctab = pd.crosstab(pair[column_x], pair[column_y])
    top_rows = ctab.sum(axis=1).sort_values(ascending=False).head(8).index
    top_cols = ctab.sum(axis=0).sort_values(ascending=False).head(8).index
    ctab = ctab.loc[top_rows, top_cols]
    points = [{"row": str(r), "col": str(c), "count": int(ctab.loc[r, c])} for r in ctab.index for c in ctab.columns]
    note = "Showing up to the 8 most frequent categories on each axis." if (
        len(pair[column_x].unique()) > 8 or len(pair[column_y].unique()) > 8) else ""
    return {"column_x": column_x, "column_y": column_y, "kind": "categorical_categorical",
            "points": points, "stats": {"rows_shown": len(top_rows), "cols_shown": len(top_cols)}, "note": note}


# ---------------------------------------------------------------------------
# Category Analysis
# ---------------------------------------------------------------------------

def compute_category_analysis(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    series = df[column].dropna().astype(str)
    n = len(series)
    counts = series.value_counts()
    top = counts.head(MAX_CATEGORY_BARS)
    categories = [{"value": str(k), "count": int(v), "pct": round(float(v / n * 100), 1) if n else 0.0}
                  for k, v in top.items()]
    coverage = round(float(top.sum() / n * 100), 1) if n else 0.0
    note = ""
    if len(counts) > MAX_CATEGORY_BARS:
        note = f"Top {MAX_CATEGORY_BARS} of {len(counts)} distinct values cover {coverage}% of non-missing rows."
    return {"column": column, "categories": categories, "unique_count": int(len(counts)),
            "top_n_coverage_pct": coverage, "note": note}


# ---------------------------------------------------------------------------
# Time Analysis
# ---------------------------------------------------------------------------

def compute_time_analysis(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    parsed = pd.to_datetime(df[column], errors="coerce").dropna()
    if parsed.empty:
        return {"column": column, "granularity": "none", "buckets": [],
                "note": "No values in this column parsed as dates."}

    span_days = (parsed.max() - parsed.min()).days
    if span_days <= 60:
        granularity, freq = "daily", "D"
    elif span_days <= 730:
        granularity, freq = "monthly", "MS"
    else:
        granularity, freq = "yearly", "YS"

    counts = parsed.dt.to_period(freq[0] if freq != "D" else "D").value_counts().sort_index()
    buckets = [{"period": str(period), "count": int(count)} for period, count in counts.items()]

    return {
        "column": column, "granularity": granularity, "buckets": buckets,
        "note": f"Auto-selected {granularity} buckets based on a {span_days}-day span "
                f"({parsed.min().date().isoformat()} to {parsed.max().date().isoformat()}).",
    }
