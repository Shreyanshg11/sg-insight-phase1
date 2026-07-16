"""
Visualization Engine (#16) + Dashboard Builder (#17).

Chart TYPE selection follows the same explainability pattern as every other
engine in this app: why this chart (reason), how confident (confidence, tied
to a concrete data-shape check), what triggered it (how_detected), and what
else could have been used instead (alternative_chart_types) - never a bare
dropdown of chart options.

Chart DATA reuses eda_engine's already-built computations wherever the shape
matches (distributions, correlation, category analysis, time buckets)
instead of recomputing the same statistics a second way - the two engines
should never be able to disagree about the same numbers.

The Dashboard Builder (#17) stores chart SPECS (chart_type + columns), not
cached data - every dashboard view recomputes live off the dataset's current
version, so a pinned chart never silently shows stale numbers after a
cleaning/feature-engineering action changes the data. It's flagged `stale`
(recomputed anyway, just annotated) if the dataset version has moved on
since the chart was pinned.
"""
import uuid
import numpy as np
import pandas as pd
from scipy import stats

from app.engines.feature_engineering_engine import (
    _usable_numeric_columns, _usable_categorical_columns, _usable_date_columns,
)
from app.engines.eda_engine import (
    compute_distributions, compute_correlation, compute_category_analysis,
    compute_time_analysis, compute_relationships,
)

MAX_BOXPLOT_GROUPS = 8
MIN_ROWS_FOR_SCATTER = 10


def _is_goal_relevant(columns: list[str], goal_keywords: list[str]) -> bool:
    if not goal_keywords:
        return False
    return any(any(kw in c.lower() or c.lower() in kw for kw in goal_keywords) for c in columns)


def _iqr_outlier_indices(df: pd.DataFrame, col: str) -> list[int]:
    series = df[col].dropna()
    if len(series) < 8:
        return []
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)].index.tolist()


# ---------------------------------------------------------------------------
# Chart builders — one per chart_type, shared by recommend/explain/dashboard
# ---------------------------------------------------------------------------

def _build_histogram(df: pd.DataFrame, columns: list[str]) -> dict:
    col = columns[0]
    dist = compute_distributions(df, col)
    outliers = _iqr_outlier_indices(df, col)
    return {"data": {"bins": dist["bins"]}, "summary_stats": dist["stats"],
            "outlier_indices": outliers, "trend": None}


def _build_bar_chart(df: pd.DataFrame, columns: list[str]) -> dict:
    col = columns[0]
    cat = compute_category_analysis(df, col)
    return {"data": {"categories": cat["categories"], "note": cat["note"]},
            "summary_stats": {"unique_count": cat["unique_count"], "top_n_coverage_pct": cat["top_n_coverage_pct"]},
            "outlier_indices": [], "trend": None}


def _build_line_chart(df: pd.DataFrame, columns: list[str]) -> dict:
    col = columns[0]
    time_data = compute_time_analysis(df, col)
    counts = [b["count"] for b in time_data["buckets"]]
    x = np.arange(len(counts))
    if len(counts) >= 2:
        slope, intercept, r, p, se = stats.linregress(x, counts)
    else:
        slope, r = 0.0, 0.0
    direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "flat"
    return {"data": {"buckets": time_data["buckets"], "granularity": time_data["granularity"]},
            "summary_stats": {"n_periods": len(counts)},
            "outlier_indices": [],
            "trend": {"slope_per_period": round(float(slope), 4), "direction": direction,
                      "r_squared": round(float(r ** 2), 3)}}


def _build_scatter_plot(df: pd.DataFrame, columns: list[str]) -> dict:
    col_a, col_b = columns[0], columns[1]
    pair = df[[col_a, col_b]].dropna()
    outliers_a, outliers_b = set(_iqr_outlier_indices(df, col_a)), set(_iqr_outlier_indices(df, col_b))
    combined_outliers = outliers_a | outliers_b
    sample = pair.sample(min(len(pair), 300), random_state=42) if len(pair) else pair
    points = [{"x": round(float(r[col_a]), 3), "y": round(float(r[col_b]), 3),
               "row_index": int(idx), "is_outlier": idx in combined_outliers}
              for idx, r in sample.iterrows()]
    correlation = round(float(pair[col_a].corr(pair[col_b])), 3) if len(pair) > 1 else None
    return {"data": {"points": points}, "summary_stats": {"correlation": correlation, "n": len(pair)},
            "outlier_indices": sorted(combined_outliers), "trend": None}


def _build_box_plot(df: pd.DataFrame, columns: list[str]) -> dict:
    cat_col, num_col = columns[0], columns[1]
    frame = df[[cat_col, num_col]].dropna()
    sizes = frame.groupby(cat_col).size().sort_values(ascending=False)
    top_groups = sizes.head(MAX_BOXPLOT_GROUPS).index
    groups_data = []
    outlier_indices = []
    for name in top_groups:
        sub = frame[frame[cat_col] == name][num_col]
        q1, q2, q3 = sub.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        group_outliers = sub[(sub < lower) | (sub > upper)]
        outlier_indices.extend(group_outliers.index.tolist())
        groups_data.append({
            "category": str(name), "n": int(len(sub)),
            "min": round(float(sub.min()), 3), "q1": round(float(q1), 3), "median": round(float(q2), 3),
            "q3": round(float(q3), 3), "max": round(float(sub.max()), 3), "outlier_count": int(len(group_outliers)),
        })
    note = "" if len(sizes) <= MAX_BOXPLOT_GROUPS else f"Showing the {MAX_BOXPLOT_GROUPS} most frequent groups of {len(sizes)} total."
    return {"data": {"groups": groups_data, "note": note},
            "summary_stats": {"groups_shown": len(groups_data), "groups_total": int(len(sizes))},
            "outlier_indices": sorted(set(outlier_indices)), "trend": None}


def _build_heatmap(df: pd.DataFrame, columns: list[str]) -> dict:
    # Restrict to exactly the requested columns (via a sub-dataframe) rather
    # than letting compute_correlation recompute its own numeric-column list
    # from the full df - otherwise the chart's `data` could silently cover a
    # different column set than what `columns` on the recommendation says.
    corr = compute_correlation(df[columns])
    return {"data": {"columns": corr["columns"], "matrix": corr["matrix"], "note": corr["note"]},
            "summary_stats": {"top_pairs": corr["top_pairs"]}, "outlier_indices": [], "trend": None}


_BUILDERS = {
    "histogram": _build_histogram, "bar_chart": _build_bar_chart, "line_chart": _build_line_chart,
    "scatter_plot": _build_scatter_plot, "box_plot": _build_box_plot, "heatmap": _build_heatmap,
}


def build_chart(df: pd.DataFrame, chart_type: str, columns: list[str]) -> dict:
    if chart_type not in _BUILDERS:
        raise ValueError(f"Unknown chart_type: {chart_type}")
    return _BUILDERS[chart_type](df, columns)


# ---------------------------------------------------------------------------
# Recommendations (#16)
# ---------------------------------------------------------------------------

def recommend_charts(df: pd.DataFrame, goal_keywords: list[str] | None = None) -> list[dict]:
    goal_keywords = goal_keywords or []
    n_rows = len(df)
    numeric_cols = _usable_numeric_columns(df)
    categorical_cols = _usable_categorical_columns(df)
    date_cols = _usable_date_columns(df)
    recs = []

    def add(chart_type, columns, title, reason, confidence, how_detected, alternatives):
        built = build_chart(df, chart_type, columns)
        recs.append({
            "id": str(uuid.uuid4()), "chart_type": chart_type, "columns": columns, "title": title,
            "reason": reason, "confidence": confidence, "how_detected": how_detected,
            "alternative_chart_types": alternatives, "goal_relevant": _is_goal_relevant(columns, goal_keywords),
            **built,
        })

    # -- Histograms: one per usable numeric column ---------------------------
    for col in numeric_cols:
        skew = float(df[col].dropna().skew()) if df[col].dropna().shape[0] > 2 else 0.0
        add("histogram", [col], f"Distribution of {col}",
            f"'{col}' is a continuous numeric column - a histogram is the standard way to see its "
            f"shape (skew={round(skew,2)}) at a glance.",
            0.8, f"Numeric dtype, {df[col].dropna().shape[0]} non-null values.",
            ["box_plot (if you mainly care about outliers/quartiles rather than the full shape)"])

    # -- Bar charts: one per usable categorical column ------------------------
    for col in categorical_cols:
        cardinality = df[col].dropna().nunique()
        if cardinality < 2:
            continue
        add("bar_chart", [col], f"Category breakdown of {col}",
            f"'{col}' is categorical with {cardinality} distinct value(s) - a bar chart ranks them by "
            "frequency, which a table of numbers doesn't make as immediately scannable.",
            0.75, f"Categorical dtype, {cardinality} distinct values.",
            ["pie_chart (avoided here - bar charts are more accurately comparable than pie slices)"])

    # -- Line charts: one per usable date column (row-count trend) -----------
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(parsed) < 8:
            continue
        add("line_chart", [col], f"Trend over {col}",
            f"'{col}' is a date column with {len(parsed)} parseable value(s) - a line chart is the "
            "standard way to show how row volume moves over time.",
            0.75, f"Date column with {len(parsed)} parseable values (>= 8 minimum).",
            ["bar_chart (if you want period-by-period comparison rather than a continuous trend line)"])

    # -- Scatter plots: the single strongest numeric-numeric pair -------------
    if len(numeric_cols) >= 2 and n_rows >= MIN_ROWS_FOR_SCATTER:
        corr = compute_correlation(df)
        if corr["top_pairs"]:
            top = corr["top_pairs"][0]
            add("scatter_plot", [top["column_a"], top["column_b"]],
                f"{top['column_a']} vs {top['column_b']}",
                f"Of all numeric column pairs, '{top['column_a']}' and '{top['column_b']}' have the "
                f"{top['strength']} correlation ({top['correlation']}) - a scatter plot shows that "
                "relationship directly instead of just a single number.",
                round(min(0.85, 0.5 + abs(top["correlation"]) * 0.4), 2),
                f"Strongest of {len(corr['top_pairs'])} numeric pair(s) by |Pearson correlation|.",
                ["heatmap (to see every numeric pair's correlation at once, not just the strongest one)"])

    # -- Box plots: numeric outcome across a low-cardinality categorical col --
    for col in categorical_cols:
        counts = df[col].dropna().value_counts()
        groups_with_enough_rows = counts[counts >= 5]
        if 2 <= len(groups_with_enough_rows) <= 8 and numeric_cols:
            outcome = numeric_cols[0]
            add("box_plot", [col, outcome], f"{outcome} by {col}",
                f"'{col}' splits into {len(groups_with_enough_rows)} group(s) with 5+ rows each - a box "
                f"plot compares '{outcome}'s spread and outliers across those groups at once "
                "(swap the outcome column if a different one matters more).",
                0.7, f"{len(groups_with_enough_rows)} groups with 5+ rows each in '{col}'.",
                ["bar_chart of group means (simpler, but hides spread/outliers within each group)"])

    # -- Heatmap: dataset-level correlation overview --------------------------
    if len(numeric_cols) >= 3:
        add("heatmap", numeric_cols, "Correlation heatmap",
            f"{len(numeric_cols)} usable numeric columns is enough for a correlation heatmap to be "
            "more useful than checking pairs one at a time.",
            0.7, f"{len(numeric_cols)} usable numeric columns (>= 3).",
            ["scatter_plot (for a closer look at one specific pair from the heatmap)"])

    recs.sort(key=lambda r: (not r["goal_relevant"], -r["confidence"]))
    return recs


# ---------------------------------------------------------------------------
# "Explain this chart" (#16) — deterministic, not AI-generated
# ---------------------------------------------------------------------------

def explain_chart(df: pd.DataFrame, chart_type: str, columns: list[str]) -> str:
    built = build_chart(df, chart_type, columns)
    data, stats_ = built["data"], built["summary_stats"]

    if chart_type == "histogram":
        col = columns[0]
        s = stats_
        n_outliers = len(built["outlier_indices"])
        return (f"This histogram shows the distribution of '{col}' across {df[col].notna().sum()} non-missing "
                f"rows. The mean is {s.get('mean')} and the median is {s.get('median')}"
                + (f", with a skew of {s.get('skew')} (a longer tail on one side)." if abs(s.get("skew", 0)) > 0.5 else ".")
                + (f" {n_outliers} row(s) fall outside the typical (1.5×IQR) range and are highlighted as outliers." if n_outliers else ""))

    if chart_type == "bar_chart":
        col = columns[0]
        top = data["categories"][0] if data["categories"] else None
        return (f"This bar chart ranks '{col}' by frequency across {stats_['unique_count']} distinct value(s). "
                + (f"'{top['value']}' is the most common at {top['pct']}% of rows. " if top else "")
                + f"The values shown cover {stats_['top_n_coverage_pct']}% of non-missing rows."
                + (f" {data['note']}" if data.get("note") else ""))

    if chart_type == "line_chart":
        col = columns[0]
        trend = built["trend"]
        return (f"This line chart tracks row volume over '{col}' in {stats_['n_periods']} {data['granularity']} "
                f"period(s). The trend is {trend['direction']} at roughly {trend['slope_per_period']} rows per "
                f"period (R²={trend['r_squared']}, so treat the fit strength accordingly).")

    if chart_type == "scatter_plot":
        a, b = columns
        n_out = len(built["outlier_indices"])
        return (f"This scatter plot compares '{a}' and '{b}' across {stats_['n']} row(s), with a Pearson "
                f"correlation of {stats_['correlation']}. "
                + (f"{n_out} point(s) are highlighted because one of their two values is a 1.5×IQR outlier "
                   "in its own column." if n_out else "No points were flagged as outliers in either column."))

    if chart_type == "box_plot":
        cat_col, num_col = columns
        n_groups = stats_["groups_shown"]
        total_outliers = sum(g["outlier_count"] for g in data["groups"])
        return (f"This box plot compares '{num_col}' across {n_groups} group(s) of '{cat_col}'. "
                f"{total_outliers} row(s) across all groups fall outside their own group's 1.5×IQR range."
                + (f" {data['note']}" if data.get("note") else ""))

    if chart_type == "heatmap":
        top_pairs = stats_["top_pairs"]
        strongest = top_pairs[0] if top_pairs else None
        return (f"This heatmap shows pairwise Pearson correlation across {len(data['columns'])} numeric column(s). "
                + (f"The strongest relationship is between '{strongest['column_a']}' and "
                   f"'{strongest['column_b']}' at {strongest['correlation']} ({strongest['strength']})."
                   if strongest else "No numeric pairs were available to correlate.")
                + f" {data['note']}")

    raise ValueError(f"Unknown chart_type: {chart_type}")
