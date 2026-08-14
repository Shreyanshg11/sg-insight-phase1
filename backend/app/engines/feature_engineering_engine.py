"""
Feature Engineering Engine (#12).

Same interaction contract as the Cleaning Engine: every transform below is a
FeatureRecommendation with a preview/reason/confidence/how_detected/risks/
alternative_actions payload, and nothing is applied to the dataset until the
user confirms via the frontend's RecommendationCard. `apply_feature_action`
is only ever called after that confirmation.

Categories implemented (per the brief): ratio, date_features, bucketing,
scaling_normalization, encoding (one-hot / label), log_transform.
"""
import uuid
import numpy as np
import pandas as pd

from app.models.schemas import FeatureRecommendation
from app.core.dtype_utils import stringy_columns
from app.engines.profiling_engine import ID_NAME_HINTS, DATE_NAME_HINTS, _is_id_column, _is_date_column

MAX_RATIO_CANDIDATE_COLUMNS = 6   # bounds ratio pair combinations to <= 15
MAX_ONE_HOT_CARDINALITY = 15      # above this, one-hot would blow up column count
MAX_LABEL_ENCODE_CARDINALITY = 50 # above this, label encoding is unlikely to be meaningful

# Value tiers for FeatureRecommendation.value_tier — distinct from `confidence`
# (confidence = "did the detection rule fire correctly"; value_tier = "is the
# resulting feature likely to actually be worth having"). See per-category
# classifiers below for the concrete rule behind each tier assignment.
VALUE_TIERS = ("high_value", "situational", "risky", "redundant")

REDUNDANCY_CORRELATION_THRESHOLD = 0.97  # near-duplicate of an existing column


def _usable_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are actually continuous quantities — excludes ID-like
    columns and binary 0/1 flag columns (e.g. a churn target), since ratios,
    scaling, bucketing, and log transforms aren't meaningful on a 2-value flag."""
    cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if _is_id_column(df[col], col):
            continue
        series = df[col].dropna()
        if series.empty or series.nunique() <= 2:
            continue
        cols.append(col)
    return cols


def _usable_date_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if _is_date_column(df[col], col)]


def _usable_categorical_columns(df: pd.DataFrame) -> list[str]:
    date_cols = set(_usable_date_columns(df))
    cols = []
    for col in stringy_columns(df):
        if col in date_cols or ID_NAME_HINTS.search(col):
            continue
        if df[col].dropna().empty:
            continue
        cols.append(col)
    return cols


def _is_goal_relevant(columns: list[str], goal_keywords: list[str]) -> bool:
    if not goal_keywords:
        return False
    return any(
        any(kw in col.lower() or col.lower() in kw for kw in goal_keywords)
        for col in columns
    )


def _iqr_outlier_ratio(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 8:
        return 0.0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((s < lower) | (s > upper)).mean())


def _max_abs_correlation(new_series: pd.Series, df: pd.DataFrame, exclude_cols: list[str]) -> float:
    """Highest |correlation| between a candidate new numeric feature and any
    EXISTING numeric column not among its own source columns — used to flag a
    suggestion as redundant when it would just duplicate information already
    in the dataset."""
    best = 0.0
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in exclude_cols:
            continue
        other = df[col]
        if other.dropna().nunique() <= 1:
            continue
        corr = new_series.corr(other)
        if corr is not None and not pd.isna(corr):
            best = max(best, abs(float(corr)))
    return best


# ---------------------------------------------------------------------------
# Value classification — per category, a stated rule for high_value /
# situational / risky / redundant, mirroring the "cite the numbers" convention
# used for confidence and how_detected elsewhere in this app.
# ---------------------------------------------------------------------------

def _classify_date_features_value(valid_pct: float, parsed: pd.Series) -> tuple[str, str]:
    distinct_dates = parsed.dropna().dt.date.nunique()
    if valid_pct < 0.6:
        return "risky", f"Only {round(valid_pct*100,1)}% of values parsed as dates — the majority of rows " \
                         "would get missing date-part values."
    if distinct_dates <= 1:
        return "redundant", "Every parsed value falls on the same calendar date, so none of the extracted " \
                             "parts (year/month/day/weekday) would vary — no discriminating signal added."
    if valid_pct >= 0.9:
        return "high_value", f"{round(valid_pct*100,1)}% parse rate and {distinct_dates} distinct dates " \
                              "give the extracted parts real variation to work with."
    return "situational", f"{round(valid_pct*100,1)}% parse rate is usable but not clean — worth trying, " \
                           "review how the unparsed rows are handled downstream."


def _classify_ratio_value(ratio_series: pd.Series, zero_pct: float, max_corr: float) -> tuple[str, str]:
    if zero_pct > 0.15:
        return "risky", f"{round(zero_pct*100,1)}% of rows have a zero denominator and would get a " \
                         "missing ratio value — a meaningfully unstable feature at this rate."
    dropped = ratio_series.dropna()
    if dropped.empty or dropped.nunique() <= 1:
        return "redundant", "The computed ratio is constant across all rows — it can't discriminate " \
                             "between observations, so it adds no information."
    if max_corr >= REDUNDANCY_CORRELATION_THRESHOLD:
        return "redundant", f"This ratio correlates at |r|={round(max_corr,3)} with an existing column — " \
                             "close to duplicating information already in the dataset."
    if zero_pct < 0.05:
        return "high_value", f"Only {round(zero_pct*100,1)}% undefined-denominator rows and no strong " \
                              "correlation with an existing column — a genuinely new, mostly-complete signal."
    return "situational", f"{round(zero_pct*100,1)}% undefined-denominator rate is moderate — usable, but " \
                           "review whether the two columns are conceptually related before relying on it."


def _classify_bucketing_value(distinct_buckets: int, outlier_ratio: float) -> tuple[str, str]:
    if distinct_buckets < 4:
        return "redundant", f"Quartile edges collapsed to only {distinct_buckets} distinct band(s) (heavy " \
                             "ties in the data) — limited new information over the raw column."
    if outlier_ratio > 0.05:
        return "high_value", f"{round(outlier_ratio*100,1)}% IQR-outlier rate in the source column — " \
                              "bucketing into equal-count bands reduces how much those extreme values can " \
                              "distort tables/charts built on this feature."
    return "situational", "A clean quartile split with no notable outlier pressure — a reasonable " \
                           "discretization, but not clearly more useful than the raw numeric column."


def _classify_scaling_value(outlier_ratio: float) -> tuple[str, str]:
    if outlier_ratio > 0.15:
        return "risky", f"{round(outlier_ratio*100,1)}% IQR-outlier rate — scaling (even the robust method " \
                         "chosen here) reduces but doesn't remove the influence of that many outliers; " \
                         "consider addressing them directly before relying on this feature."
    return "high_value", "Numeric columns generally benefit from a consistent scale before distance- or " \
                          "gradient-based methods; no unusually high outlier rate complicating that here."


def _classify_encoding_value(action: str, cardinality: int) -> tuple[str, str]:
    if action == "label_encode":
        return "risky", f"Label encoding assigns an arbitrary 0-{cardinality-1} ordering to {cardinality} " \
                         "categories with no inherent order — safe for tree-based methods, misleading for " \
                         "linear/distance-based ones that would read the codes as a real numeric scale."
    if cardinality <= 5:
        return "high_value", f"Only {cardinality} distinct categories — one-hot encoding stays compact and " \
                              "gives a clean, unambiguous numeric representation."
    if cardinality <= 10:
        return "situational", f"{cardinality} distinct categories is a moderate one-hot expansion — usable, " \
                               "but check whether that many new columns are actually needed downstream."
    return "risky", f"{cardinality} distinct categories means one-hot adds that many new, mostly-sparse " \
                     "columns — reconsider whether this column needs encoding at all before applying."


def _classify_log_transform_value(skew: float, transformed_skew: float) -> tuple[str, str]:
    skew_reduction = skew - transformed_skew
    if skew_reduction < 0.1:
        return "redundant", f"log1p barely changed the skew ({round(skew,2)} -> {round(transformed_skew,2)}) " \
                             "— the distortion likely comes from a few extreme values log1p doesn't fix, " \
                             "not the overall shape."
    if skew > 2:
        return "high_value", f"Skew dropped from {round(skew,2)} to {round(transformed_skew,2)} — a " \
                              "meaningful correction to a strongly distorted distribution."
    return "situational", f"Skew dropped from {round(skew,2)} to {round(transformed_skew,2)} — a real but " \
                           "modest improvement on a borderline-skewed column."


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def generate_feature_recommendations(df: pd.DataFrame, goal_keywords: list[str] | None = None) -> list[FeatureRecommendation]:
    goal_keywords = goal_keywords or []
    recs: list[FeatureRecommendation] = []
    n_rows = len(df)

    numeric_cols = _usable_numeric_columns(df)
    date_cols = _usable_date_columns(df)
    categorical_cols = _usable_categorical_columns(df)

    # -- Date features --------------------------------------------------
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid_pct = float(parsed.notna().mean()) if n_rows else 0.0
        if valid_pct < 0.5:
            continue
        new_col = f"{col}_year"
        sample_before = df[col].dropna().astype(str).head(5).tolist()
        sample_after = parsed.dropna().dt.year.head(5).tolist()
        value_tier, value_tier_reason = _classify_date_features_value(valid_pct, parsed)
        recs.append(FeatureRecommendation(
            id=str(uuid.uuid4()),
            category="date_features",
            columns=[col],
            new_column_name=new_col,
            issue=f"'{col}' parses as a date ({round(valid_pct*100,1)}% of values) but its year/month/day/"
                  "weekday components aren't available as separate features yet.",
            recommended_action="extract_date_parts",
            reason="Date columns are rarely useful to models in raw string/timestamp form — splitting "
                   "into year, month, day, day-of-week, and is_weekend lets downstream analysis pick up "
                   "seasonality and calendar effects directly.",
            confidence=round(min(0.9, 0.5 + valid_pct * 0.4), 2),
            preview={"before_sample": sample_before, "after_sample": sample_after,
                     "rows_affected": n_rows,
                     "new_columns": [f"{col}_year", f"{col}_month", f"{col}_day",
                                     f"{col}_dayofweek", f"{col}_is_weekend"],
                     "estimated_impact": f"Adds 5 new columns derived from '{col}'; "
                                         f"{round((1-valid_pct)*100,1)}% of rows will get missing parts "
                                         "where the date failed to parse."},
            how_detected=f"pd.to_datetime() parsed {round(valid_pct*100,1)}% of non-null values in '{col}'.",
            risks=[f"{round((1-valid_pct)*100,1)}% of values didn't parse as dates and will produce "
                   "missing part-columns.", "Adds column-count overhead if the date isn't actually "
                   "predictive on its own."],
            alternative_actions=["extract_year_month_only", "extract_is_weekend_only"],
            goal_relevant=_is_goal_relevant([col], goal_keywords),
            value_tier=value_tier, value_tier_reason=value_tier_reason,
        ))

    # -- Ratios (numeric x numeric) --------------------------------------
    # Exclude columns this same engine already derived (e.g. a prior "x_per_y"
    # ratio, "_bucket", "_scaled", "_encoded", "_log" column) from being ratio
    # candidates again - otherwise a ratio column becomes a new ratio input on
    # the next recommendation pass, producing nonsensical ratio-of-ratio names
    # like "revenue_per_quantity_per_cost".
    _derived_suffixes = ("_per_", "_bucket", "_scaled", "_encoded", "_onehot", "_log")
    ratio_candidates = [c for c in numeric_cols if not any(s in c for s in _derived_suffixes)]
    ratio_candidates = ratio_candidates[:MAX_RATIO_CANDIDATE_COLUMNS]
    for i in range(len(ratio_candidates)):
        for j in range(len(ratio_candidates)):
            if i == j:
                continue
            num_col, den_col = ratio_candidates[i], ratio_candidates[j]
            denom = df[den_col]
            zero_pct = float((denom == 0).mean()) if n_rows else 0.0
            if zero_pct > 0.3:
                continue  # too many undefined ratios to be a useful feature
            ratio_series = df[num_col] / denom.replace(0, np.nan)
            if ratio_series.dropna().empty:
                continue
            new_col = f"{num_col}_per_{den_col}"
            confidence = round(max(0.5, 0.85 - zero_pct), 2)
            max_corr = _max_abs_correlation(ratio_series, df, [num_col, den_col])
            value_tier, value_tier_reason = _classify_ratio_value(ratio_series, zero_pct, max_corr)
            recs.append(FeatureRecommendation(
                id=str(uuid.uuid4()),
                category="ratio",
                columns=[num_col, den_col],
                new_column_name=new_col,
                issue=f"'{num_col}' and '{den_col}' are both numeric but their per-unit relationship "
                      f"('{new_col}') isn't captured as its own column.",
                recommended_action="create_ratio",
                reason=f"Dividing '{num_col}' by '{den_col}' surfaces a per-unit rate (e.g. price-per-item, "
                       "rate-per-count) that raw totals hide, and is a standard feature-engineering move "
                       "whenever two numeric quantities are conceptually related.",
                confidence=confidence,
                preview={"before_sample": None, "after_sample": ratio_series.dropna().round(3).head(5).tolist(),
                         "rows_affected": n_rows,
                         "new_columns": [new_col],
                         "estimated_impact": f"Adds '{new_col}'; {round(zero_pct*100,1)}% of rows have "
                                             f"'{den_col}' == 0 and will get a missing ratio value there."},
                how_detected=f"Both columns are usable numeric quantities; {round(zero_pct*100,1)}% "
                             f"zero-denominator rate in '{den_col}' checked against a 30% cutoff.",
                risks=["A ratio between two columns that aren't actually conceptually related can be "
                       "meaningless — review whether the relationship makes real-world sense.",
                       "Zero-denominator rows become missing rather than infinite."],
                alternative_actions=[f"create_ratio({den_col}_per_{num_col})", "skip — columns unrelated"],
                goal_relevant=_is_goal_relevant([num_col, den_col], goal_keywords),
                value_tier=value_tier, value_tier_reason=value_tier_reason,
            ))

    # -- Bucketing (numeric -> quartile categories) ----------------------
    for col in numeric_cols:
        series = df[col].dropna()
        if series.nunique() < 5 or len(series) < 8:
            continue
        try:
            bucketed = pd.qcut(series, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            continue
        distinct_buckets = bucketed.nunique()
        if distinct_buckets < 2:
            continue
        new_col = f"{col}_bucket"
        outlier_ratio_for_col = _iqr_outlier_ratio(series)
        value_tier, value_tier_reason = _classify_bucketing_value(distinct_buckets, outlier_ratio_for_col)
        recs.append(FeatureRecommendation(
            id=str(uuid.uuid4()),
            category="bucketing",
            columns=[col],
            new_column_name=new_col,
            issue=f"'{col}' is continuous; grouping it into quartile bands could make patterns "
                  "easier to read in tables/charts and less sensitive to exact values.",
            recommended_action="quartile_bucket",
            reason="Quartile bucketing (Q1-Q4, equal row counts per band) is a simple, explainable "
                   "discretization that works for any numeric distribution without picking arbitrary "
                   "cutoffs by hand.",
            confidence=0.7,
            preview={"before_sample": series.head(5).tolist(),
                     "after_sample": bucketed.astype(str).head(5).tolist(),
                     "rows_affected": len(series),
                     "new_columns": [new_col],
                     "estimated_impact": f"Adds '{new_col}' with {distinct_buckets} bands "
                                         f"(~{round(len(series)/distinct_buckets)} rows each)."},
            how_detected=f"pandas.qcut(4) on '{col}' produced {distinct_buckets} distinct quartile bands.",
            risks=["Quartile boundaries shift if the dataset changes (e.g. after later cleaning), so "
                   "re-run this after later pipeline steps rather than treating bands as fixed."],
            alternative_actions=["custom_bin_edges", "equal_width_bins"],
            goal_relevant=_is_goal_relevant([col], goal_keywords),
            value_tier=value_tier, value_tier_reason=value_tier_reason,
        ))

    # -- Scaling / normalization ------------------------------------------
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5 or series.std() == 0:
            continue
        outlier_ratio = _iqr_outlier_ratio(series)
        if outlier_ratio > 0.05:
            action, method_reason, confidence = "robust_scale", (
                f"{round(outlier_ratio*100,1)}% of values are IQR outliers, so scaling by median/IQR "
                "(robust scaling) resists their influence better than min-max or z-score would."
            ), 0.75
        else:
            action, method_reason, confidence = "z_score_standardize", (
                f"Only {round(outlier_ratio*100,1)}% of values are IQR outliers, so standard z-score "
                "scaling (mean 0, std 1) is appropriate and keeps the distribution's shape."
            ), 0.75

        if action == "robust_scale":
            median, iqr = series.median(), series.quantile(0.75) - series.quantile(0.25)
            scaled = (series - median) / iqr if iqr != 0 else series - median
        else:
            scaled = (series - series.mean()) / series.std()

        new_col = f"{col}_scaled"
        all_alts = ["min_max_scale", "z_score_standardize", "robust_scale"]
        value_tier, value_tier_reason = _classify_scaling_value(outlier_ratio)
        recs.append(FeatureRecommendation(
            id=str(uuid.uuid4()),
            category="scaling_normalization",
            columns=[col],
            new_column_name=new_col,
            issue=f"'{col}' is on its own raw scale, which can dominate distance- or gradient-based "
                  "downstream methods if left unscaled alongside other numeric columns.",
            recommended_action=action,
            reason=method_reason,
            confidence=confidence,
            preview={"before_sample": series.head(5).tolist(), "after_sample": scaled.round(3).head(5).tolist(),
                     "rows_affected": len(series),
                     "new_columns": [new_col],
                     "estimated_impact": f"Adds '{new_col}' scaled via {action}; original '{col}' is kept "
                                         "unchanged."},
            how_detected=f"1.5×IQR outlier ratio on '{col}' = {round(outlier_ratio*100,1)}%, checked "
                         "against a 5% cutoff to choose robust vs standard scaling.",
            risks=["Scaling parameters (mean/std or median/IQR) are computed from the current dataset — "
                   "if new rows arrive later with a different distribution, re-fit rather than reuse "
                   "these numbers blindly."],
            alternative_actions=[a for a in all_alts if a != action],
            goal_relevant=_is_goal_relevant([col], goal_keywords),
            value_tier=value_tier, value_tier_reason=value_tier_reason,
        ))

    # -- Encoding (categorical -> numeric) ---------------------------------
    for col in categorical_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        cardinality = int(series.nunique())
        if cardinality < 2:
            continue
        if cardinality <= MAX_ONE_HOT_CARDINALITY:
            action, reason, confidence, how = "one_hot_encode", (
                f"'{col}' has {cardinality} distinct values, low enough that one-hot encoding "
                f"(one 0/1 column per category) stays manageable without exploding column count."
            ), round(max(0.6, 0.9 - cardinality * 0.02), 2), (
                f"Distinct-value count ({cardinality}) checked against a {MAX_ONE_HOT_CARDINALITY}-category cutoff."
            )
            risks = [f"Adds {cardinality} new columns — reconsider if this column isn't actually needed "
                     "as a model input."]
            new_col = f"{col}_onehot"
            preview_after = pd.get_dummies(series.head(5), prefix=col).to_dict(orient="records")
        elif cardinality <= MAX_LABEL_ENCODE_CARDINALITY:
            action, reason, confidence, how = "label_encode", (
                f"'{col}' has {cardinality} distinct values — too many for one-hot encoding to stay "
                "compact, so a single integer-coded column (label encoding) is the more practical choice."
            ), 0.6, (
                f"Distinct-value count ({cardinality}) falls between the one-hot cutoff "
                f"({MAX_ONE_HOT_CARDINALITY}) and the label-encode cutoff ({MAX_LABEL_ENCODE_CARDINALITY})."
            )
            risks = ["Label encoding implies a numeric ordering between categories that may not exist in "
                     "reality — fine for tree-based methods, risky for linear/distance-based ones."]
            new_col = f"{col}_encoded"
            codes = series.astype("category").cat.codes
            preview_after = codes.head(5).tolist()
        else:
            continue  # too high-cardinality for either method to be a sensible default

        value_tier, value_tier_reason = _classify_encoding_value(action, cardinality)
        recs.append(FeatureRecommendation(
            id=str(uuid.uuid4()),
            category="encoding",
            columns=[col],
            new_column_name=new_col,
            issue=f"'{col}' is categorical/text ({cardinality} distinct values) and has no numeric "
                  "representation yet for methods that require numeric input.",
            recommended_action=action,
            reason=reason,
            confidence=confidence,
            preview={"before_sample": series.head(5).tolist(), "after_sample": preview_after,
                     "rows_affected": len(series),
                     "new_columns": [new_col] if action == "label_encode" else None,
                     "estimated_impact": (
                         f"Adds {cardinality} one-hot column(s) prefixed '{col}_'." if action == "one_hot_encode"
                         else f"Adds '{new_col}' with integer codes 0-{cardinality-1}."
                     )},
            how_detected=how,
            risks=risks,
            alternative_actions=["one_hot_encode", "label_encode"],
            goal_relevant=_is_goal_relevant([col], goal_keywords),
            value_tier=value_tier, value_tier_reason=value_tier_reason,
        ))

    # -- Log transform (right-skewed, non-negative numeric) ---------------
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 8 or (series < 0).any():
            continue
        skew = float(series.skew())
        if skew <= 1:
            continue
        new_col = f"{col}_log"
        transformed = np.log1p(series)
        value_tier, value_tier_reason = _classify_log_transform_value(skew, float(transformed.skew()))
        recs.append(FeatureRecommendation(
            id=str(uuid.uuid4()),
            category="log_transform",
            columns=[col],
            new_column_name=new_col,
            issue=f"'{col}' is strongly right-skewed (skew={round(skew,2)}), which can distort methods "
                  "that assume roughly symmetric numeric inputs.",
            recommended_action="log1p_transform",
            reason=f"skew={round(skew,2)} is above the |skew|>1 threshold used elsewhere in this app; "
                   "log1p (log(1+x), safe for zeros) compresses the long right tail and brings the "
                   "distribution closer to symmetric.",
            confidence=round(min(0.85, 0.5 + skew * 0.1), 2),
            preview={"before_sample": series.head(5).tolist(), "after_sample": transformed.round(3).head(5).tolist(),
                     "rows_affected": len(series),
                     "new_columns": [new_col],
                     "estimated_impact": f"Adds '{new_col}'; resulting skew drops from {round(skew,2)} to "
                                         f"{round(float(transformed.skew()),2)}."},
            how_detected=f"pandas .skew() on non-null, non-negative values of '{col}' = {round(skew,2)}.",
            risks=["Log-transformed values are harder to interpret directly in the original units — keep "
                   "the original column alongside this one for reporting."],
            alternative_actions=["sqrt_transform", "leave_untransformed"],
            goal_relevant=_is_goal_relevant([col], goal_keywords),
            value_tier=value_tier, value_tier_reason=value_tier_reason,
        ))

    recs.sort(key=lambda r: not r.goal_relevant)
    return recs


# ---------------------------------------------------------------------------
# Action executors — only called after user confirmation
# ---------------------------------------------------------------------------

def apply_feature_action(df: pd.DataFrame, category: str, action: str, columns: list[str],
                          new_column_name: str | None, params: dict) -> tuple[pd.DataFrame, dict]:
    """Returns (new_df, preview_of_change). Never mutates df in place — every
    transform ADDS new column(s); it never overwrites or drops the source
    column, so the original data is always still there to compare against."""
    out = df.copy()
    preview: dict = {}

    if category == "date_features" and action == "extract_date_parts":
        col = columns[0]
        parsed = pd.to_datetime(out[col], errors="coerce")
        parts = params.get("parts") or ["year", "month", "day", "dayofweek", "is_weekend"]
        new_cols = []
        if "year" in parts:
            out[f"{col}_year"] = parsed.dt.year
            new_cols.append(f"{col}_year")
        if "month" in parts:
            out[f"{col}_month"] = parsed.dt.month
            new_cols.append(f"{col}_month")
        if "day" in parts:
            out[f"{col}_day"] = parsed.dt.day
            new_cols.append(f"{col}_day")
        if "dayofweek" in parts:
            out[f"{col}_dayofweek"] = parsed.dt.dayofweek
            new_cols.append(f"{col}_dayofweek")
        if "is_weekend" in parts:
            out[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6])
            new_cols.append(f"{col}_is_weekend")
        preview = {"new_columns": new_cols, "unparsed_rows": int(parsed.isna().sum())}

    elif category == "ratio" and action == "create_ratio":
        num_col, den_col = columns[0], columns[1]
        new_col = new_column_name or f"{num_col}_per_{den_col}"
        out[new_col] = out[num_col] / out[den_col].replace(0, np.nan)
        preview = {"new_columns": [new_col],
                   "undefined_rows": int((out[den_col] == 0).sum())}

    elif category == "bucketing" and action == "quartile_bucket":
        col = columns[0]
        new_col = new_column_name or f"{col}_bucket"
        try:
            out[new_col] = pd.qcut(out[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError as e:
            raise ValueError(f"Could not bucket '{col}': {e}")
        preview = {"new_columns": [new_col], "bucket_counts": out[new_col].value_counts().to_dict()}

    elif category == "scaling_normalization":
        col = columns[0]
        new_col = new_column_name or f"{col}_scaled"
        series = out[col]
        if action == "min_max_scale":
            lo, hi = series.min(), series.max()
            out[new_col] = (series - lo) / (hi - lo) if hi != lo else 0.0
        elif action == "z_score_standardize":
            out[new_col] = (series - series.mean()) / series.std()
        elif action == "robust_scale":
            median = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            out[new_col] = (series - median) / iqr if iqr != 0 else series - median
        else:
            raise ValueError(f"Unknown scaling_normalization action: {action}")
        preview = {"new_columns": [new_col]}

    elif category == "encoding":
        col = columns[0]
        if action == "one_hot_encode":
            dummies = pd.get_dummies(out[col], prefix=col)
            new_cols = dummies.columns.tolist()
            out = pd.concat([out, dummies], axis=1)
            preview = {"new_columns": new_cols}
        elif action == "label_encode":
            new_col = new_column_name or f"{col}_encoded"
            out[new_col] = out[col].astype("category").cat.codes
            preview = {"new_columns": [new_col],
                       "category_map": dict(enumerate(out[col].astype("category").cat.categories.tolist()))}
        else:
            raise ValueError(f"Unknown encoding action: {action}")

    elif category == "log_transform" and action == "log1p_transform":
        col = columns[0]
        new_col = new_column_name or f"{col}_log"
        if (out[col] < 0).any():
            raise ValueError(f"'{col}' contains negative values - log1p is undefined there.")
        out[new_col] = np.log1p(out[col])
        preview = {"new_columns": [new_col]}

    else:
        raise ValueError(f"Unknown category/action combination: {category}/{action}")

    return out, preview
