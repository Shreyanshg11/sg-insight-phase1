"""
Analysis Recommendation Engine (#14).

Considers all seven techniques named in the brief - trend analysis,
regression, clustering, forecasting, hypothesis testing, PCA, anomaly
detection - and returns an entry for EVERY one of them, every time, whether
or not it's currently viable. A `recommended=False` entry with a concrete
`not_recommended_because` is not a hidden failure case, it's the explicit
"when NOT recommended" the brief calls for - the same "surfaced, not
silently skipped" principle already used for cleaning/feature-engineering
recommendations and the ComingSoonPanel.

`run_analysis` executes a confirmed technique and returns real computed
numbers - it never mutates the dataset or the version/undo stack (unlike
cleaning/feature-engineering actions), since analysis is read-only.
"""
import uuid
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from app.engines.feature_engineering_engine import (
    _usable_numeric_columns, _usable_categorical_columns, _usable_date_columns,
)
from app.core.confidence import capped_linear

MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR = 10
MIN_ROWS_FOR_CLUSTERING = 30
MIN_ROWS_FOR_PCA = 30
MIN_ROWS_FOR_ANOMALY = 20
MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST = 5
MIN_SPAN_DAYS_FOR_TREND = 14
MIN_PERIODS_FOR_FORECASTING = 8


def _is_goal_relevant(columns: list[str], goal_keywords: list[str]) -> bool:
    if not goal_keywords:
        return False
    return any(any(kw in c.lower() or c.lower() in kw for kw in goal_keywords) for c in columns)


def _rec(analysis_type, title, columns, recommended, reason, confidence, required_characteristics,
          not_recommended_because=None, how_detected="", risks=None, alternatives=None,
          goal_relevant=False, preview=None):
    return {
        "id": str(uuid.uuid4()), "analysis_type": analysis_type, "title": title, "columns": columns,
        "recommended": recommended, "reason": reason, "confidence": confidence if recommended else 0.0,
        "required_characteristics": required_characteristics,
        "not_recommended_because": not_recommended_because, "how_detected": how_detected,
        "risks": risks or [], "alternative_actions": alternatives or [],
        "goal_relevant": goal_relevant, "preview": preview or {},
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend_analyses(df: pd.DataFrame, goal_keywords: list[str] | None = None) -> list[dict]:
    goal_keywords = goal_keywords or []
    n_rows = len(df)
    numeric_cols = _usable_numeric_columns(df)
    categorical_cols = _usable_categorical_columns(df)
    date_cols = _usable_date_columns(df)
    recs = []

    # -- Trend analysis (one entry per usable date column) -----------------
    if not date_cols:
        recs.append(_rec(
            "trend_analysis", "Trend Analysis", [],
            False, "No parseable date column was found in this dataset.", 0.0,
            "Requires at least one column that parses as a date, with a span of "
            f"{MIN_SPAN_DAYS_FOR_TREND}+ days.",
            not_recommended_because="No date-like column detected during profiling.",
        ))
    else:
        for col in date_cols:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            span_days = (parsed.max() - parsed.min()).days if len(parsed) > 1 else 0
            if span_days < MIN_SPAN_DAYS_FOR_TREND:
                recs.append(_rec(
                    "trend_analysis", f"Trend Analysis over '{col}'", [col],
                    False, f"'{col}' only spans {span_days} day(s).", 0.0,
                    f"Requires a {MIN_SPAN_DAYS_FOR_TREND}+ day span to say anything meaningful about a trend.",
                    not_recommended_because=f"Span ({span_days} days) is below the {MIN_SPAN_DAYS_FOR_TREND}-day minimum.",
                ))
                continue
            confidence = capped_linear(
                0.5, 0.35 / 365, min(span_days, 365), cap=0.85,
                value_label="span_days (capped at 365)",
            ).score
            recs.append(_rec(
                "trend_analysis", f"Trend Analysis over '{col}'", [col],
                True, f"'{col}' spans {span_days} days with {len(parsed)} dated row(s) - enough to "
                      "check whether row volume (or a chosen numeric column) moves consistently over time.",
                confidence, f"Column parses as a date with a {span_days}-day span (>= {MIN_SPAN_DAYS_FOR_TREND}).",
                how_detected=f"pd.to_datetime() on '{col}' parsed {len(parsed)} value(s) spanning {span_days} days.",
                risks=["A trend fitted on a short or gappy time range can look far more confident than it is.",
                       "Correlation with time isn't causation - a trend doesn't explain why."],
                alternatives=["forecasting (if you want to project the trend forward)"],
                goal_relevant=_is_goal_relevant([col], goal_keywords),
            ))

    # -- Regression ----------------------------------------------------------
    if len(numeric_cols) < 3:
        recs.append(_rec(
            "regression", "Regression", numeric_cols,
            False, f"Only {len(numeric_cols)} usable continuous numeric column(s) available; "
                   "regression needs one target plus at least two predictors.", 0.0,
            "At least 3 usable numeric columns (1 target + 2+ predictors) and "
            f"{MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR}+ rows per predictor.",
            not_recommended_because=f"Only {len(numeric_cols)} usable numeric column(s), need >= 3.",
        ))
    else:
        max_predictors = (len(numeric_cols) - 1)
        min_rows_needed = MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR * min(max_predictors, 5)
        if n_rows < min_rows_needed:
            recs.append(_rec(
                "regression", "Regression", numeric_cols,
                False, f"{n_rows} rows is too few for a reliable fit with up to {max_predictors} predictor(s) "
                       f"(rule of thumb: {MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR}+ rows per predictor).", 0.0,
                f"{MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR}+ rows per predictor column used.",
                not_recommended_because=f"{n_rows} rows < {min_rows_needed} rows needed for a stable fit.",
            ))
        else:
            confidence = round(min(0.8, 0.5 + n_rows / (min_rows_needed * 4)), 2)
            recs.append(_rec(
                "regression", "Regression", numeric_cols,
                True, f"{len(numeric_cols)} usable numeric columns and {n_rows} rows support fitting a "
                      "linear regression - pick one column as the target at run time.",
                confidence, f"{MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR}+ rows per predictor "
                            f"({n_rows} rows available, up to {max_predictors} predictor(s)).",
                how_detected=f"{len(numeric_cols)} usable numeric columns found; {n_rows} rows vs "
                             f"{min_rows_needed} rows required.",
                risks=["A linear fit assumes a roughly linear relationship - it will silently understate "
                       "or miss non-linear effects.",
                       "In-sample R² can look better than the model actually generalizes."],
                alternatives=["Choose a different target column", "clustering (if there's no clear target)"],
                goal_relevant=_is_goal_relevant(numeric_cols, goal_keywords),
                preview={"candidate_targets": numeric_cols},
            ))

    # -- Clustering -----------------------------------------------------------
    if len(numeric_cols) < 2:
        recs.append(_rec(
            "clustering", "Clustering / Segmentation", numeric_cols,
            False, f"Only {len(numeric_cols)} usable numeric column(s); clustering needs at least 2 "
                   "to define a feature space.", 0.0,
            f"At least 2 usable numeric columns and {MIN_ROWS_FOR_CLUSTERING}+ rows.",
            not_recommended_because=f"Only {len(numeric_cols)} usable numeric column(s), need >= 2.",
        ))
    elif n_rows < MIN_ROWS_FOR_CLUSTERING:
        recs.append(_rec(
            "clustering", "Clustering / Segmentation", numeric_cols,
            False, f"{n_rows} rows is too few to form stable clusters.", 0.0,
            f"At least 2 usable numeric columns and {MIN_ROWS_FOR_CLUSTERING}+ rows.",
            not_recommended_because=f"{n_rows} rows < {MIN_ROWS_FOR_CLUSTERING} rows needed.",
        ))
    else:
        confidence = 0.75 if not categorical_cols else 0.65
        recs.append(_rec(
            "clustering", "Clustering / Segmentation", numeric_cols,
            True, f"{len(numeric_cols)} usable numeric columns and {n_rows} rows are enough to look for "
                  "natural groupings without needing a labeled target.",
            confidence, f"At least 2 usable numeric columns and {MIN_ROWS_FOR_CLUSTERING}+ rows.",
            how_detected=f"{len(numeric_cols)} usable numeric columns, {n_rows} rows.",
            risks=["K-means assumes roughly round, similarly-sized clusters - it can force a "
                   "structure onto data that doesn't actually have distinct groups.",
                   "Cluster count is chosen by silhouette score, which is a heuristic, not ground truth."],
            alternatives=["regression or hypothesis_test (if there's actually a known target/grouping)"],
            goal_relevant=_is_goal_relevant(numeric_cols, goal_keywords),
        ))

    # -- Forecasting (one entry per usable date column) ------------------------
    if not date_cols:
        recs.append(_rec(
            "forecasting", "Forecasting", [],
            False, "No parseable date column was found in this dataset.", 0.0,
            f"Requires a date column with {MIN_PERIODS_FOR_FORECASTING}+ distinct time periods.",
            not_recommended_because="No date-like column detected during profiling.",
        ))
    else:
        for col in date_cols:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            n_periods = parsed.dt.to_period("M").nunique() if len(parsed) else 0
            if n_periods < MIN_PERIODS_FOR_FORECASTING:
                recs.append(_rec(
                    "forecasting", f"Forecasting from '{col}'", [col],
                    False, f"'{col}' only covers {n_periods} distinct month(s).", 0.0,
                    f"Requires {MIN_PERIODS_FOR_FORECASTING}+ distinct monthly periods for a naive projection.",
                    not_recommended_because=f"{n_periods} period(s) < {MIN_PERIODS_FOR_FORECASTING} required.",
                ))
                continue
            recs.append(_rec(
                "forecasting", f"Forecasting from '{col}'", [col],
                True, f"'{col}' covers {n_periods} distinct month(s) - enough for a naive linear "
                      "projection of future periods.",
                0.55,  # capped low - this is intentionally a simple baseline, not a real forecast model
                f"{MIN_PERIODS_FOR_FORECASTING}+ distinct monthly periods.",
                how_detected=f"{n_periods} distinct monthly period(s) found in '{col}'.",
                risks=["This is a naive linear extrapolation of row-count/value trend, not a seasonal "
                       "or ARIMA-style forecast - treat it as a rough baseline, not a production forecast.",
                       "Assumes the historical trend continues; doesn't account for seasonality or shocks."],
                alternatives=["trend_analysis (to just describe the past, without projecting forward)"],
                goal_relevant=_is_goal_relevant([col], goal_keywords),
            ))

    # -- Hypothesis testing (one entry per usable categorical column, 2-5 groups) --
    hyp_candidates = []
    for col in categorical_cols:
        counts = df[col].dropna().value_counts()
        groups = counts[counts >= MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST]
        if 2 <= len(groups) <= 5:
            hyp_candidates.append((col, groups))
    if not hyp_candidates or not numeric_cols:
        reason = ("No categorical column has 2-5 groups with "
                   f"{MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST}+ rows each." if not hyp_candidates
                   else "No usable continuous numeric outcome column available.")
        recs.append(_rec(
            "hypothesis_test", "Hypothesis Test", [],
            False, reason, 0.0,
            f"A categorical column with 2-5 groups ({MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST}+ rows each) "
            "plus a continuous numeric outcome column.",
            not_recommended_because=reason,
        ))
    else:
        for col, groups in hyp_candidates:
            outcome = numeric_cols[0]
            recs.append(_rec(
                "hypothesis_test", f"Hypothesis Test: '{col}' groups", [col, outcome],
                True, f"'{col}' splits into {len(groups)} group(s) each with "
                      f"{MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST}+ rows - enough to test whether '{outcome}' "
                      "differs meaningfully between them (pick a different outcome column at run time if needed).",
                0.7, f"2-5 groups with {MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST}+ rows each, plus a numeric outcome.",
                how_detected=f"value_counts() on '{col}' found {len(groups)} group(s) meeting the "
                             f"{MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST}-row minimum.",
                risks=["A statistically significant difference isn't automatically a practically "
                       "important one - check the effect size, not just the p-value.",
                       "Multiple hypothesis tests across many columns inflate the false-positive rate."],
                alternatives=[f"Use a different numeric outcome column instead of '{outcome}'"],
                goal_relevant=_is_goal_relevant([col, outcome], goal_keywords),
                preview={"groups": groups.index.tolist(), "candidate_outcomes": numeric_cols},
            ))

    # -- PCA --------------------------------------------------------------
    if len(numeric_cols) < 4:
        recs.append(_rec(
            "pca", "Principal Component Analysis", numeric_cols,
            False, f"Only {len(numeric_cols)} usable numeric column(s); PCA is only useful for "
                   "reducing a meaningfully higher-dimensional space.", 0.0,
            f"At least 4 usable numeric columns and {MIN_ROWS_FOR_PCA}+ rows.",
            not_recommended_because=f"Only {len(numeric_cols)} usable numeric column(s), need >= 4.",
        ))
    elif n_rows < MIN_ROWS_FOR_PCA:
        recs.append(_rec(
            "pca", "Principal Component Analysis", numeric_cols,
            False, f"{n_rows} rows is too few for stable principal components.", 0.0,
            f"At least 4 usable numeric columns and {MIN_ROWS_FOR_PCA}+ rows.",
            not_recommended_because=f"{n_rows} rows < {MIN_ROWS_FOR_PCA} required.",
        ))
    else:
        recs.append(_rec(
            "pca", "Principal Component Analysis", numeric_cols,
            True, f"{len(numeric_cols)} usable numeric columns is enough dimensionality for PCA to "
                  "meaningfully compress into fewer components.",
            0.65, f"At least 4 usable numeric columns and {MIN_ROWS_FOR_PCA}+ rows.",
            how_detected=f"{len(numeric_cols)} usable numeric columns, {n_rows} rows.",
            risks=["Principal components are linear combinations of the original columns and lose "
                   "direct interpretability - always check the loadings before naming a component."],
            alternatives=["clustering (to find groups rather than compress dimensions)"],
            goal_relevant=_is_goal_relevant(numeric_cols, goal_keywords),
        ))

    # -- Anomaly detection ---------------------------------------------------
    if len(numeric_cols) < 2:
        recs.append(_rec(
            "anomaly_detection", "Anomaly Detection", numeric_cols,
            False, f"Only {len(numeric_cols)} usable numeric column(s); multivariate anomaly detection "
                   "needs at least 2 (a single column is already covered by the Cleaning Center's "
                   "per-column outlier check).", 0.0,
            f"At least 2 usable numeric columns and {MIN_ROWS_FOR_ANOMALY}+ rows.",
            not_recommended_because=f"Only {len(numeric_cols)} usable numeric column(s), need >= 2.",
        ))
    elif n_rows < MIN_ROWS_FOR_ANOMALY:
        recs.append(_rec(
            "anomaly_detection", "Anomaly Detection", numeric_cols,
            False, f"{n_rows} rows is too few to reliably separate anomalies from noise.", 0.0,
            f"At least 2 usable numeric columns and {MIN_ROWS_FOR_ANOMALY}+ rows.",
            not_recommended_because=f"{n_rows} rows < {MIN_ROWS_FOR_ANOMALY} required.",
        ))
    else:
        recs.append(_rec(
            "anomaly_detection", "Anomaly Detection", numeric_cols,
            True, f"{len(numeric_cols)} usable numeric columns and {n_rows} rows support looking for "
                  "rows that are jointly unusual across several columns at once - different from the "
                  "single-column IQR check already in the Cleaning Center's Outlier Center.",
            0.65, f"At least 2 usable numeric columns and {MIN_ROWS_FOR_ANOMALY}+ rows.",
            how_detected=f"{len(numeric_cols)} usable numeric columns, {n_rows} rows.",
            risks=["IsolationForest's contamination rate is an assumption, not a measurement - it sets "
                   "how many rows get flagged regardless of whether that many are truly anomalous."],
            alternatives=["Cleaning Center's per-column Outlier Center (for single-column outliers)"],
            goal_relevant=_is_goal_relevant(numeric_cols, goal_keywords),
        ))

    recs.sort(key=lambda r: (not r["recommended"], not r["goal_relevant"]))
    return recs


# ---------------------------------------------------------------------------
# Execution - read-only, never mutates the dataset or version stack
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, analysis_type: str, columns: list[str], params: dict) -> dict:
    if analysis_type == "trend_analysis":
        return _run_trend_analysis(df, columns, params)
    if analysis_type == "regression":
        return _run_regression(df, columns, params)
    if analysis_type == "clustering":
        return _run_clustering(df, columns, params)
    if analysis_type == "forecasting":
        return _run_forecasting(df, columns, params)
    if analysis_type == "hypothesis_test":
        return _run_hypothesis_test(df, columns, params)
    if analysis_type == "pca":
        return _run_pca(df, columns, params)
    if analysis_type == "anomaly_detection":
        return _run_anomaly_detection(df, columns, params)
    raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _run_trend_analysis(df, columns, params):
    date_col = columns[0]
    value_col = params.get("value_column")
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    frame = pd.DataFrame({"date": parsed, "value": df[value_col] if value_col else 1}).dropna(subset=["date"])
    monthly = frame.groupby(frame["date"].dt.to_period("M"))["value"].sum() if value_col else \
        frame.groupby(frame["date"].dt.to_period("M")).size()
    periods = [str(p) for p in monthly.index]
    values = monthly.values.astype(float)
    x = np.arange(len(values))
    if len(values) >= 2:
        slope, intercept, r, p, se = stats.linregress(x, values)
    else:
        slope = intercept = r = p = se = 0.0
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    result = {
        "periods": periods, "values": values.round(3).tolist(),
        "slope_per_period": round(float(slope), 4), "r_squared": round(float(r ** 2), 3),
        "p_value": round(float(p), 4), "trend_direction": direction,
        "metric": value_col or "row_count",
    }
    summary = (f"{result['metric']} is {direction} at ~{round(float(slope), 3)} per period over "
               f"{len(periods)} periods (R²={result['r_squared']}, p={result['p_value']}).")
    limitations = ["A single linear slope over the whole range can miss turning points within it.",
                   "Statistical significance (p-value) doesn't confirm the cause of the trend."]
    return {"result": result, "summary": summary, "limitations": limitations}


def _run_regression(df, columns, params):
    target = params.get("target")
    if not target or target not in df.columns:
        raise ValueError("params.target must name a valid numeric column")
    predictors = [c for c in (params.get("predictors") or columns) if c != target and c in df.columns]
    if len(predictors) < 1:
        raise ValueError("At least one predictor column is required")
    frame = df[[target] + predictors].dropna()
    X, y = frame[predictors].values, frame[target].values

    holdout_r2 = None
    if len(frame) >= 50:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        holdout_r2 = round(float(model.score(X_test, y_test)), 3)
    model_full = LinearRegression().fit(X, y)
    in_sample_r2 = round(float(model_full.score(X, y)), 3)
    coefficients = {p: round(float(c), 4) for p, c in zip(predictors, model_full.coef_)}

    result = {
        "target": target, "predictors": predictors, "n_rows": len(frame),
        "intercept": round(float(model_full.intercept_), 4), "coefficients": coefficients,
        "in_sample_r_squared": in_sample_r2, "holdout_r_squared": holdout_r2,
    }
    summary = (f"Linear regression on '{target}' using {len(predictors)} predictor(s) achieved "
               f"R²={in_sample_r2} (in-sample)" +
               (f", R²={holdout_r2} on a held-out 20% split." if holdout_r2 is not None else
                " (dataset too small for a held-out split, so this is in-sample only)."))
    limitations = ["This is a descriptive linear fit, not a validated causal model - no significance "
                   "testing (p-values per coefficient) was performed.",
                   "Assumes a linear relationship between predictors and target."]
    return {"result": result, "summary": summary, "limitations": limitations}


def _run_clustering(df, columns, params):
    frame = df[columns].dropna()
    X = StandardScaler().fit_transform(frame.values)
    k_range = range(2, min(7, len(frame) // 5 + 1))
    best_k, best_score, best_labels = None, -1, None
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_k is None:
        raise ValueError("Could not find a stable cluster count for this data")

    sizes = pd.Series(best_labels).value_counts().sort_index()
    centers_scaled = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X).cluster_centers_
    scaler = StandardScaler().fit(frame.values)
    centers = scaler.inverse_transform(centers_scaled)
    result = {
        "k": best_k, "silhouette_score": round(float(best_score), 3),
        "cluster_sizes": {str(i): int(sizes.get(i, 0)) for i in range(best_k)},
        "cluster_centers": [{col: round(float(v), 3) for col, v in zip(columns, row)} for row in centers],
        "columns": columns,
    }
    summary = (f"K-means found {best_k} clusters (silhouette={result['silhouette_score']}) across "
               f"{len(columns)} column(s); sizes: {result['cluster_sizes']}.")
    limitations = ["Cluster count was chosen by silhouette score, a heuristic - domain judgment may "
                   "suggest a different, more interpretable number of segments.",
                   "K-means assumes roughly spherical, similarly-sized clusters."]
    return {"result": result, "summary": summary, "limitations": limitations}


def _run_forecasting(df, columns, params):
    date_col = columns[0]
    value_col = params.get("value_column")
    horizon = int(params.get("horizon_periods", 3))
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    frame = pd.DataFrame({"date": parsed, "value": df[value_col] if value_col else 1}).dropna(subset=["date"])
    monthly = frame.groupby(frame["date"].dt.to_period("M"))["value"].sum() if value_col else \
        frame.groupby(frame["date"].dt.to_period("M")).size()
    values = monthly.values.astype(float)
    x = np.arange(len(values))
    slope, intercept, r, p, se = stats.linregress(x, values) if len(values) >= 2 else (0, values.mean() if len(values) else 0, 0, 1, 0)
    future_x = np.arange(len(values), len(values) + horizon)
    forecast_values = (slope * future_x + intercept).round(3).tolist()
    last_period = monthly.index[-1]
    future_periods = [str(last_period + i) for i in range(1, horizon + 1)]
    result = {
        "history_periods": [str(p_) for p_ in monthly.index], "history_values": values.round(3).tolist(),
        "forecast_periods": future_periods, "forecast_values": forecast_values,
        "slope_per_period": round(float(slope), 4), "r_squared": round(float(r ** 2), 3),
        "metric": value_col or "row_count",
    }
    summary = (f"Naive linear projection of {result['metric']} for the next {horizon} period(s): "
               f"{list(zip(future_periods, forecast_values))} (fit R²={result['r_squared']}).")
    limitations = ["This is a naive linear extrapolation, not a seasonal/ARIMA-style forecast - treat "
                   "it as a rough baseline only.",
                   "Assumes the historical linear trend continues unchanged and ignores seasonality."]
    return {"result": result, "summary": summary, "limitations": limitations}


def _run_hypothesis_test(df, columns, params):
    group_col, outcome_col = columns[0], params.get("outcome_column") or columns[1]
    frame = df[[group_col, outcome_col]].dropna()
    groups = [g[outcome_col].values for _, g in frame.groupby(group_col) if len(g) >= MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST]
    group_names = [name for name, g in frame.groupby(group_col) if len(g) >= MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST]
    if len(groups) < 2:
        raise ValueError(f"Fewer than 2 groups in '{group_col}' have {MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST}+ rows")

    group_stats = {str(name): {"n": len(g), "mean": round(float(np.mean(g)), 3), "std": round(float(np.std(g)), 3)}
                   for name, g in zip(group_names, groups)}
    # Same |skew| > 1 heuristic used elsewhere in this app to flag non-normal-ish data.
    max_abs_skew = max(abs(float(pd.Series(g).skew())) if len(g) > 2 else 0 for g in groups)
    use_nonparametric = max_abs_skew > 1

    if len(groups) == 2:
        if use_nonparametric:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1])
            test_name = "Mann-Whitney U"
        else:
            stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            test_name = "Welch's t-test"
    else:
        if use_nonparametric:
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        else:
            stat, p_value = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"

    result = {
        "group_column": group_col, "outcome_column": outcome_col, "groups": group_stats,
        "test": test_name, "statistic": round(float(stat), 4), "p_value": round(float(p_value), 5),
        "max_abs_skew": round(max_abs_skew, 3), "used_nonparametric": use_nonparametric,
    }
    significant = p_value < 0.05
    summary = (f"{test_name} on '{outcome_col}' across {len(groups)} group(s) of '{group_col}': "
               f"p={result['p_value']} ({'a significant' if significant else 'no significant'} "
               "difference at the 0.05 level).")
    limitations = ["p<0.05 flags a difference is unlikely to be random noise - it says nothing about "
                   "how large or practically meaningful that difference is.",
                   f"Test choice ({test_name}) was based on skew as a normality proxy, not a formal "
                   "normality test."]
    return {"result": result, "summary": summary, "limitations": limitations}


def _run_pca(df, columns, params):
    frame = df[columns].dropna()
    X = StandardScaler().fit_transform(frame.values)
    n_components = min(len(columns), 5)
    pca = PCA(n_components=n_components).fit(X)
    explained = pca.explained_variance_ratio_.round(4).tolist()
    cumulative = np.cumsum(explained).round(4).tolist()
    loadings = {
        f"PC{i+1}": {col: round(float(v), 3) for col, v in zip(columns, pca.components_[i])}
        for i in range(min(2, n_components))
    }
    result = {
        "n_components": n_components, "explained_variance_ratio": explained,
        "cumulative_variance_ratio": cumulative, "top_loadings": loadings, "columns": columns,
    }
    summary = (f"First {min(2, n_components)} principal component(s) explain "
               f"{round(cumulative[min(1, n_components-1)]*100, 1)}% of variance across {len(columns)} column(s).")
    limitations = ["Components are linear combinations of the original columns - check `top_loadings` "
                   "before assuming a component means what its top-weighted column suggests."]
    return {"result": result, "summary": summary, "limitations": limitations}


def _run_anomaly_detection(df, columns, params):
    frame = df[columns].dropna()
    contamination = float(params.get("contamination", 0.05))
    X = StandardScaler().fit_transform(frame.values)
    model = IsolationForest(contamination=contamination, random_state=42).fit(X)
    scores = model.decision_function(X)
    is_anomaly = model.predict(X) == -1
    anomaly_indices = frame.index[is_anomaly].tolist()
    top_anomalies = frame.loc[anomaly_indices].assign(anomaly_score=scores[is_anomaly]) \
        .sort_values("anomaly_score").head(10)
    result = {
        "contamination": contamination, "n_flagged": int(is_anomaly.sum()), "n_total": len(frame),
        "flagged_pct": round(float(is_anomaly.mean() * 100), 2),
        "top_anomalies": top_anomalies.round(3).reset_index().rename(columns={"index": "row_index"}).to_dict(orient="records"),
        "columns": columns,
    }
    summary = (f"IsolationForest flagged {result['n_flagged']} of {result['n_total']} rows "
               f"({result['flagged_pct']}%) as jointly anomalous across {len(columns)} column(s).")
    limitations = [f"The {int(contamination*100)}% contamination rate is an assumption, not a measurement - "
                   "it fixes roughly how many rows get flagged regardless of the data's actual anomaly rate.",
                   "Flags multivariate outliers only; a row can be flagged even if every individual "
                   "column value looks normal on its own."]
    return {"result": result, "summary": summary, "limitations": limitations}
