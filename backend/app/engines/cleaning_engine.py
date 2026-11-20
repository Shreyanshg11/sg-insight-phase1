"""
Data quality scoring + cleaning recommendations + action executors.

Design constraint from the spec: confidence scores must map to a real,
explainable heuristic — never an arbitrary-looking number. Every
recommendation below documents in `reason` exactly which heuristic produced it.
"""
import re
import uuid
import pandas as pd
import numpy as np
from app.models.schemas import QualityScore, QualityBreakdownItem, CleaningRecommendation
from app.core.dtype_utils import stringy_columns
from app.engines.missingness_engine import classify_missingness
from app.engines.outlier_engine import classify_outliers_for_column
from app.core.iqr import FAR_OUT_MULTIPLIER, MIN_SAMPLE_SIZE, finite_values, iqr_bounds, outlier_mask
from app.core.action_policy import is_review_only
from app.core.recommendation_identity import assign_stable_ids

# Fuzzy category merge (#7): a small built-in alias table for well-known
# abbreviation cases (USA/US/United States, UK, UAE) plus a deliberately
# STRICT spelling-variant check for typos. Similar spelling is not evidence of
# the same meaning (Australia/Austria, Male/Female), so the spelling tier only
# accepts a single-character edit on real words, and every fuzzy group is a
# review-only suggestion that needs explicit user confirmation — never auto-merged.
FUZZY_SIMILARITY_THRESHOLD = 0.72  # DEPRECATED: no longer used for matching; kept so existing imports don't break
FUZZY_MAX_DISTINCT_VALUES = 150  # skip high-cardinality columns; not really "categories"
FUZZY_MIN_VALUE_LENGTH = 5       # shorter values are never fuzzy-matched (Gray/Grey, Jon/John, US/UK)
FUZZY_MAX_VALUE_LENGTH = 64      # longer values are free text, not categories (also bounds edit-distance cost)
FUZZY_MAX_EDIT_DISTANCE = 1      # Damerau-Levenshtein distance allowed between two spelling variants
FUZZY_MIN_DIFFERING_TOKEN_LENGTH = 4  # a word that differs must be a real word, not a 1-3 char label ("Box"/"Fox", "A"/"B")

_ALIAS_GROUPS = [
    {"usa", "us", "u.s.", "u.s.a.", "united states", "united states of america", "america"},
    {"uk", "u.k.", "united kingdom", "great britain", "britain"},
    {"uae", "u.a.e.", "united arab emirates"},
]


# ---------------------------------------------------------------------------
# Data Quality scoring
# ---------------------------------------------------------------------------

def compute_quality_score(df: pd.DataFrame) -> QualityScore:
    n_rows = len(df)
    n_cols = len(df.columns) or 1

    # Completeness: 1 - overall missing ratio
    total_cells = n_rows * n_cols if n_rows else 1
    missing_cells = int(df.isna().sum().sum())
    completeness = 1 - (missing_cells / total_cells) if total_cells else 1.0

    # Uniqueness: 1 - duplicate row ratio
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / n_rows) if n_rows else 0.0
    uniqueness = 1 - dup_pct

    # Consistency: fraction of object columns whose values don't show obvious
    # casing/whitespace inconsistency (proxy: trimmed+lowercased nunique vs raw nunique)
    consistency_scores = []
    for col in stringy_columns(df):
        raw_nunique = df[col].nunique(dropna=True)
        if raw_nunique == 0:
            continue
        normalized = df[col].dropna().astype(str).str.strip().str.lower()
        norm_nunique = normalized.nunique()
        consistency_scores.append(norm_nunique / raw_nunique if raw_nunique else 1.0)
    consistency = float(np.mean(consistency_scores)) if consistency_scores else 1.0

    # Validity: fraction of numeric columns without extreme outliers (proxy via IQR)
    validity_scores = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = finite_values(df[col])
        if len(series) < MIN_SAMPLE_SIZE:
            continue
        bounds = iqr_bounds(series, FAR_OUT_MULTIPLIER)  # shared policy; 3.0x = "extreme value" proxy
        if bounds is None:
            validity_scores.append(1.0)
            continue
        outlier_ratio = outlier_mask(series, bounds).mean()
        validity_scores.append(1 - outlier_ratio)
    validity = float(np.mean(validity_scores)) if validity_scores else 1.0

    # --- Explainable point-deduction breakdown ---
    # Each dimension's shortfall (100 - dimension%) is converted into a capped
    # point deduction, so `overall` is always traceable to a specific, named
    # cause rather than being an opaque average. Weights/caps below are the
    # tunable knobs; the *reason* string always states the exact numbers used.
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0
    missing_deduction = round(min(40.0, missing_pct * 0.3), 1)

    dup_pct_100 = dup_pct * 100
    duplicate_deduction = round(min(25.0, dup_pct_100 * 0.5), 1)

    consistency_shortfall = (1 - consistency) * 100
    consistency_deduction = round(min(20.0, consistency_shortfall * 0.3), 1)

    validity_shortfall = (1 - validity) * 100
    formatting_deduction = round(min(20.0, validity_shortfall * 0.3), 1)

    breakdown = [
        QualityBreakdownItem(
            label="Missing Values", points_deducted=missing_deduction,
            reason=f"{round(missing_pct, 1)}% of all cells are missing; deduction = "
                   f"min(40, missing% × 0.3) = {missing_deduction}."
        ),
        QualityBreakdownItem(
            label="Duplicates", points_deducted=duplicate_deduction,
            reason=f"{dup_rows} duplicate row(s) = {round(dup_pct_100, 1)}% of rows; deduction = "
                   f"min(25, duplicate% × 0.5) = {duplicate_deduction}."
        ),
        QualityBreakdownItem(
            label="Formatting Issues", points_deducted=formatting_deduction,
            reason=f"Numeric columns average {round(validity_shortfall, 1)}% extreme-value (3×IQR) rate. This is a "
                   "statistical proxy for validity, not a count of formatting errors; "
                   f"deduction = min(20, shortfall% × 0.3) = {formatting_deduction}."
        ),
        QualityBreakdownItem(
            label="Consistency Issues", points_deducted=consistency_deduction,
            reason=f"Text columns average {round(consistency_shortfall, 1)}% category-variant rate "
                   f"(e.g. casing/whitespace duplicates); deduction = min(20, shortfall% × 0.3) = "
                   f"{consistency_deduction}."
        ),
    ]
    total_deduction = sum(item.points_deducted for item in breakdown)
    overall = max(0.0, 100.0 - total_deduction)

    return QualityScore(
        overall=round(overall, 1),
        completeness=round(completeness * 100, 1),
        consistency=round(consistency * 100, 1),
        validity=round(validity * 100, 1),
        uniqueness=round(uniqueness * 100, 1),
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Recommendations (preview + reason + confidence, nothing applied yet)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fuzzy category grouping (#7)
# ---------------------------------------------------------------------------

_ALIAS_INDEX = {entry: idx for idx, group in enumerate(_ALIAS_GROUPS) for entry in group}
_TOKEN_RE = re.compile(r"[^\W_]+")
_DIGIT_RUN_RE = re.compile(r"\d+")


def _alias_match(a: str, b: str) -> bool:
    return any(a in group and b in group for group in _ALIAS_GROUPS)


def _fuzzy_key(value) -> str:
    """The single definition of "the same value" for fuzzy matching and for
    fuzzy_merge execution: strip + collapse inner whitespace + casefold.
    Returns "" for None/NaN/blank so callers can skip them."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass  # list-likes etc. — fall through to str()
    return " ".join(str(value).split()).casefold()


def _damerau_levenshtein(a: str, b: str) -> int:
    """Edit distance counting insertion, deletion, substitution and adjacent
    transposition as one edit each (optimal-string-alignment form; identical to
    full Damerau-Levenshtein for the distance <= 1 checks made here)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _is_spelling_variant(a: str, b: str) -> bool:
    """True only when two *normalized keys* (see _fuzzy_key) look like one
    value mistyped. Deliberately strict — every rule below exists because
    string similarity alone merges different categories (Australia/Austria,
    Male/Female, Region 1/Region 2, Product A/Product B)."""
    if a == b:
        return False
    if min(len(a), len(b)) < FUZZY_MIN_VALUE_LENGTH or max(len(a), len(b)) > FUZZY_MAX_VALUE_LENGTH:
        return False
    if abs(len(a) - len(b)) > FUZZY_MAX_EDIT_DISTANCE:
        return False
    tokens_a, tokens_b = _TOKEN_RE.findall(a), _TOKEN_RE.findall(b)
    if not tokens_a or len(tokens_a) != len(tokens_b):
        return False
    # Never match digit differences (Region 1 vs Region 2, Tier 1 vs Tier 2, Plan A1 vs A2).
    if _DIGIT_RUN_RE.findall(a) != _DIGIT_RUN_RE.findall(b):
        return False
    if _damerau_levenshtein(a, b) > FUZZY_MAX_EDIT_DISTANCE:
        return False
    # A differing word must be a real word, not a short label or a different short word.
    for ta, tb in zip(tokens_a, tokens_b):
        if ta != tb and min(len(ta), len(tb)) < FUZZY_MIN_DIFFERING_TOKEN_LENGTH:
            return False
    return True


def find_fuzzy_category_groups(values: list[str], counts: dict | None = None) -> list[dict]:
    """Groups distinct category values that MAY be the same category written
    differently. These are review-only suggestions — similar spelling does not
    prove identical meaning, and nothing here is ever applied automatically.

    Two tiers (every group carries ``match_type``):
    - ``alias``: values that belong to the same built-in alias group
      (USA/US/United States...). similarity is 1.0.
    - ``spelling_variant``: a strict typo check (see _is_spelling_variant).
      Groups are built as cliques — a value joins only if it qualifies against
      EVERY existing member, so there is no transitive chaining (A~B, B~C does
      not put A and C together). ``similarity`` is the weakest pair in the group.

    ``counts`` (optional, keyed by the strings in ``values``) is used to pick
    the suggested canonical for spelling variants: the most frequent spelling.
    Without counts the choice is deterministic (alphabetical) but not
    frequency-based. None/NaN/blank values are ignored; duplicates and
    case/whitespace-only variants collapse into one value. Columns with more
    than FUZZY_MAX_DISTINCT_VALUES distinct values return [].
    Output does not depend on the order of ``values``."""
    counts = counts or {}

    # 1. Normalize + dedupe. key -> {spelling: count}
    spellings: dict[str, dict[str, int]] = {}
    for raw in values or []:
        key = _fuzzy_key(raw)
        if not key:
            continue
        spelling = str(raw).strip()
        count = counts.get(raw, counts.get(spelling, 0))
        spellings.setdefault(key, {}).setdefault(spelling, count if isinstance(count, (int, float)) else 0)

    if len(spellings) < 2 or len(spellings) > FUZZY_MAX_DISTINCT_VALUES:
        return []

    total = {k: sum(v.values()) for k, v in spellings.items()}
    rep = {k: min(v, key=lambda sp: (-v[sp], sp)) for k, v in spellings.items()}  # most frequent raw spelling
    order = sorted(spellings, key=lambda k: (-total[k], k))  # deterministic, input-order independent

    results: list[dict] = []

    # 2. Alias tier (existing verified groups, unchanged semantics).
    alias_members: dict[int, list[str]] = {}
    for key in order:
        idx = _ALIAS_INDEX.get(key)
        if idx is not None:
            alias_members.setdefault(idx, []).append(key)
    consumed: set[str] = set()
    for idx in sorted(alias_members):
        keys = alias_members[idx]
        if len(keys) < 2:
            continue
        consumed.update(keys)
        group_values = [rep[k] for k in keys]
        results.append({
            "values": group_values,
            # Alias groups keep the long-standing "fullest spelling" suggestion.
            "suggested_canonical": max(group_values, key=len),
            "similarity": 1.0,
            "match_type": "alias",
            "review_required": True,
        })

    # 3. Spelling-variant tier: greedy clique partition, no transitive chaining.
    pool = [k for k in order if k not in consumed]
    assigned: set[str] = set()
    for anchor in pool:
        if anchor in assigned:
            continue
        members = [anchor]
        for cand in pool:
            if cand == anchor or cand in assigned:
                continue
            if all(_is_spelling_variant(m, cand) for m in members):
                members.append(cand)
        if len(members) < 2:
            continue
        assigned.update(members)
        weakest = min(
            1 - FUZZY_MAX_EDIT_DISTANCE / max(len(x), len(y))
            for i, x in enumerate(members) for y in members[i + 1:]
        )
        group_values = [rep[k] for k in members]
        results.append({
            "values": group_values,
            "suggested_canonical": group_values[0],  # members are ordered most-frequent first
            "similarity": round(weakest, 2),
            "match_type": "spelling_variant",
            "review_required": True,
            "max_edit_distance": FUZZY_MAX_EDIT_DISTANCE,
        })

    return results


# ---------------------------------------------------------------------------
# Missing-value method comparison (#6 "Compare Methods")
# ---------------------------------------------------------------------------

def compare_missing_value_methods(df: pd.DataFrame, column: str) -> dict:
    """Computes what each candidate imputation/drop method would actually do
    to a column's summary stats, so a user can pick a method by its effect
    rather than trusting the single recommended one blind."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    series = df[column]
    missing = int(series.isna().sum())
    is_numeric = pd.api.types.is_numeric_dtype(series)

    def _stats(s: pd.Series) -> dict:
        if is_numeric:
            s = s.dropna()
            if len(s) == 0:
                return {"mean": None, "median": None, "std": None}
            return {"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
                    "std": round(float(s.std()), 3) if len(s) > 1 else 0.0}
        top = s.value_counts().head(3)
        return {"top_values": {str(k): int(v) for k, v in top.items()}}

    before_stats = _stats(series)
    methods = []

    if is_numeric:
        mean_val, median_val = series.mean(), series.median()
        methods.append({"method": "impute_mean", "fill_value": round(float(mean_val), 3),
                         "after_stats": _stats(series.fillna(mean_val)), "rows_affected": missing})
        methods.append({"method": "impute_median", "fill_value": round(float(median_val), 3),
                         "after_stats": _stats(series.fillna(median_val)), "rows_affected": missing})
        methods.append({"method": "drop_rows", "fill_value": None,
                         "after_stats": _stats(series.dropna()), "rows_affected": missing})
    else:
        mode_series = series.mode(dropna=True)
        mode_val = mode_series.iloc[0] if not mode_series.empty else None
        methods.append({"method": "impute_mode", "fill_value": mode_val,
                         "after_stats": _stats(series.fillna(mode_val)), "rows_affected": missing})
        methods.append({"method": "drop_rows", "fill_value": None,
                         "after_stats": _stats(series.dropna()), "rows_affected": missing})

    return {"column": column, "missing_count": missing, "before_stats": before_stats, "methods": methods}


def generate_recommendations(df: pd.DataFrame, goal_keywords: list[str] | None = None) -> list[CleaningRecommendation]:
    goal_keywords = goal_keywords or []
    recs: list[CleaningRecommendation] = []
    n_rows = len(df)
    missingness_map = classify_missingness(df)  # column -> MissingnessClassification, only for columns with any missing

    def _is_goal_relevant(column: str | None) -> bool:
        if not column or not goal_keywords:
            return False
        col_lower = column.lower()
        return any(kw in col_lower or col_lower in kw for kw in goal_keywords)

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        recs.append(CleaningRecommendation(
            id=str(uuid.uuid4()),
            step="duplicate_manager",
            column=None,
            issue=f"{dup_count} duplicate rows found ({round(dup_count/n_rows*100,1)}% of dataset)",
            recommended_action="drop_duplicates",
            reason="Exact full-row duplicates detected via pandas.duplicated(); dropping is safe "
                   "because every column value matches another row exactly.",
            confidence=0.95,
            preview={"duplicate_rows": dup_count, "example_index": df[df.duplicated()].index[:5].tolist(),
                     "rows_affected": dup_count,
                     "estimated_impact": f"Dataset shrinks from {n_rows} to {n_rows - dup_count} rows "
                                         f"({round(dup_count/n_rows*100,1)}% removed)."},
            how_detected="pandas.DataFrame.duplicated() — flags rows that are a byte-for-byte match "
                         "of an earlier row across every column.",
            risks=["If two genuinely different real-world events happen to share identical values in "
                   "every recorded column, dropping would lose one of them."],
            alternative_actions=["Keep duplicates and flag them instead of dropping",
                                  "Review duplicate rows manually before deciding"],
        ))

    # Missing values per column
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        missing_pct = missing / n_rows if n_rows else 0
        alternatives = ["impute_mean", "impute_median", "impute_mode", "drop_rows", "drop_column"]
        classification = missingness_map.get(col)
        is_structural = classification is not None and classification.classification == "structural"

        if is_structural:
            # Missing-by-design: the old ">60% -> drop_column" rule used to fire
            # here regardless of *why* the column was missing, which is exactly
            # the bug this classification exists to prevent (see missingness_engine).
            action, confidence = "ignore", classification.confidence
            reason = ("Missingness looks structural (missing by design), not a data-quality defect: " +
                      " ".join(classification.evidence) + " " + classification.recommendation_hint)
            how = f"missingness_engine classified this column as 'structural' — see reason for the evidence."
            risks = ["If this assumption is wrong and the values are actually just lost/uncollected, "
                     "leaving them as NaN understates real missingness elsewhere (e.g. in aggregates)."]
            alternatives = ["impute_constant", "drop_rows", "drop_column"]
        elif missing_pct > 0.6:
            action, reason, confidence, how = "drop_column", (
                f"{round(missing_pct*100,1)}% of values are missing — above the ~60% threshold "
                "where imputation typically introduces more noise than signal."
            ), 0.7, "Missing-value ratio compared against a fixed 60% threshold."
            risks = ["Discards the entire column, including whatever signal the non-missing 40% held."]
            if classification is not None and classification.classification == "conditional":
                confidence = round(confidence * 0.75, 2)
                reason += (" Caveat: " + " ".join(classification.evidence) +
                           " Dropping is still an option but may lose more signal than a purely random-missing "
                           "column would, since the missingness itself correlates with other data.")
                risks.append(f"Missingness correlates with '{classification.predictor_column}', so this isn't "
                              "a clean MCAR case — consider group-aware imputation as an alternative.")
        elif pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].dropna().skew() if df[col].notna().sum() > 2 else 0
            if abs(skew) > 1:
                action, reason, confidence, how = "impute_median", (
                    f"Column is numeric with skew={round(float(skew),2)} (|skew|>1), so median is more "
                    "robust to the skew than mean."
                ), 0.8, f"pandas .skew() on non-null values = {round(float(skew),2)}; |skew| > 1 threshold."
                risks = ["Imputed value doesn't reflect any real observation and slightly reduces variance."]
            else:
                action, reason, confidence, how = "impute_mean", (
                    f"Column is numeric and roughly symmetric (skew={round(float(skew),2)}), "
                    "so mean imputation won't distort the distribution much."
                ), 0.75, f"pandas .skew() on non-null values = {round(float(skew),2)}; |skew| ≤ 1 threshold."
                risks = ["Imputed value doesn't reflect any real observation and slightly reduces variance."]
        else:
            action, reason, confidence, how = "impute_mode", (
                "Column is categorical/text; mode (most frequent value) imputation preserves "
                "the existing category distribution better than any numeric substitute."
            ), 0.7, "Column failed pandas numeric-dtype check, so treated as categorical/text."
            risks = ["Over-represents the already-most-common category."]

        rec_goal_relevant = _is_goal_relevant(col)

        # Rich preview (#4): a real before/after sample and an estimated-impact
        # sentence, computed directly rather than by re-running apply_action —
        # some recommended actions here (e.g. drop_column) don't have a single
        # simple "after value" to show per-row, so each case is spelled out.
        before_sample = df[col].dropna().head(5).tolist()
        if action == "ignore":
            after_sample = df[col].head(5).tolist()
            rows_affected = 0
            impact = f"No change applied — '{col}' is left as-is with its {missing} missing value(s) intact."
        elif action == "drop_column":
            after_sample = []
            rows_affected = n_rows
            impact = f"Removes the '{col}' column entirely, discarding {n_rows - missing} non-missing value(s)."
        elif action == "drop_rows":
            after_sample = df[col].dropna().head(5).tolist()
            rows_affected = missing
            impact = f"Removes {missing} row(s) missing '{col}' ({round(missing_pct*100,1)}% of dataset)."
        else:
            if action == "impute_mean":
                fill_value = df[col].mean()
            elif action == "impute_median":
                fill_value = df[col].median()
            else:  # impute_mode
                mode_series = df[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else None
            after_sample = df[col].fillna(fill_value).head(5).tolist()
            rows_affected = missing
            fill_display = round(float(fill_value), 3) if isinstance(fill_value, (int, float, np.floating, np.integer)) else fill_value
            impact = f"Fills {missing} missing value(s) in '{col}' with {fill_display} (via {action})."

        recs.append(CleaningRecommendation(
            id=str(uuid.uuid4()),
            step="missing_value_center",
            column=col,
            issue=f"{missing} missing values ({round(missing_pct*100,1)}%)",
            recommended_action=action,
            reason=reason,
            confidence=confidence,
            preview={"missing_count": missing, "missing_pct": round(missing_pct * 100, 1),
                     "before_sample": before_sample, "after_sample": after_sample,
                     "rows_affected": rows_affected, "estimated_impact": impact,
                     "missingness_classification": classification.classification if classification else None,
                     "missingness_predictor_column": classification.predictor_column if classification else None},
            how_detected=how,
            risks=risks,
            alternative_actions=[a for a in alternatives if a != action],
            goal_relevant=rec_goal_relevant,
        ))

    # Outliers (IQR method, tiered via outlier_engine) on numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        summary = classify_outliers_for_column(df, col)
        if summary is None:
            continue
        outlier_pct = summary.total_outliers / max(summary.n_values, 1)
        counts = summary.tier_counts
        tier_summary = (f"{counts['likely_data_entry_error']} potential data-quality issue(s), "
                         f"{counts['legitimate_rare_value']} recurring value(s), "
                         f"{counts['statistically_unusual']} unclassified statistical outlier(s)")

        # Column-level confidence is the MEAN evidence strength over ALL flagged points
        # (see outlier_engine). It used to be the maximum error-point confidence, which let
        # one coincidental point set the headline number for the whole column.
        confidence = summary.column_confidence

        recs.append(CleaningRecommendation(
            id=str(uuid.uuid4()),
            step="outlier_center",
            column=col,
            issue=f"{summary.total_outliers} statistical outlier(s) outside IQR fences [{summary.lower_fence}, "
                  f"{summary.upper_fence}] — {tier_summary}",
            recommended_action="flag_for_review",
            reason=f"Column '{col}': values outside the Tukey 1.5×IQR fences (Q1={summary.q1}, Q3={summary.q3}, "
                   f"IQR={summary.iqr}, n={summary.n_values}) are statistically unusual, not automatically wrong. "
                   "Each was additionally checked for concrete patterns: a far-out (3.0×IQR) position, an isolated "
                   "~10×/~100× decimal-slip magnitude that a long-tailed distribution does not explain, a "
                   f"plausible sign flip, and whether the exact value repeats — {tier_summary}. "
                   + " ".join(summary.limitations) +
                   " Flagged for review only; nothing is changed unless you choose an action and confirm.",
            confidence=confidence,
            preview={"outlier_count": summary.total_outliers, "lower_fence": summary.lower_fence,
                     "upper_fence": summary.upper_fence, "far_out_lower": summary.far_out_lower,
                     "far_out_upper": summary.far_out_upper, "tier_counts": counts,
                     "n_values": summary.n_values, "skewness": summary.skewness, "long_tailed": summary.long_tailed,
                     "method": summary.method, "limitations": summary.limitations,
                     "column_confidence": summary.column_confidence, "review_required": True,
                     "sample_outliers": [p.model_dump() for p in summary.points[:5]],
                     "rows_affected": summary.total_outliers,
                     "estimated_impact": f"{round(outlier_pct*100,1)}% of finite values in '{col}' fall "
                                         "outside the fence. No change is applied until you pick ignore/"
                                         "convert-to-missing/clip/drop below — and those act on ALL statistical "
                                         "outliers, not only the potential data-quality issues."},
            how_detected=f"Tukey 1.5×IQR fence on finite numeric values: outside [{summary.lower_fence}, {summary.upper_fence}] "
                         f"(Q1={summary.q1}, Q3={summary.q3}, IQR={summary.iqr}); each point is then labelled by "
                         "evidence (far-out distance, isolated magnitude pattern, sign pattern, repetition). "
                         "Labels are hypotheses; confidence is an evidence score, not a probability.",
            risks=["A legitimate extreme value (not an error) could be clipped or dropped if you act on "
                   "this without checking context. This applies even to values labelled as potential data-quality "
                   "issues: the labels are heuristics, and the engine assumes no business rules or valid ranges."],
            alternative_actions=["ignore", "convert_to_missing", "clip", "drop_rows"],
            goal_relevant=_is_goal_relevant(col),
        ))

    # Category merge candidates (case/whitespace variants of the same category)
    for col in stringy_columns(df):
        raw = df[col].dropna().astype(str)
        if raw.empty:
            continue
        normalized_key = raw.str.strip().str.lower()
        normalized_groups = normalized_key.value_counts()
        raw_groups = raw.value_counts()
        if len(raw_groups) > len(normalized_groups):
            merge_savings = len(raw_groups) - len(normalized_groups)
            rows_affected = int((raw.str.strip() != raw).sum() + (raw.str.strip().str.lower() != raw.str.strip()).sum())
            recs.append(CleaningRecommendation(
                id=str(uuid.uuid4()),
                step="category_merge_center",
                column=col,
                issue=f"{merge_savings} category variant(s) collapse when case/whitespace is normalized",
                recommended_action="normalize_case_whitespace",
                reason="Detected values that are identical after trim+lowercase but distinct in raw form "
                       "(e.g. 'USA' vs 'usa ' vs ' USA'). This is a mechanical, low-risk normalization.",
                confidence=0.85,
                preview={"raw_category_count": int(len(raw_groups)), "normalized_category_count": int(len(normalized_groups)),
                         "before_sample": raw_groups.index[:5].tolist(),
                         "after_sample": normalized_groups.index[:5].tolist(),
                         "rows_affected": rows_affected,
                         "estimated_impact": f"{merge_savings} raw categories collapse into "
                                             f"{len(normalized_groups)}; affects {rows_affected} row(s)."},
                how_detected="Compared value_counts() before vs after trim+lowercase; a drop in distinct "
                             "count means two raw values collapsed to the same normalized form.",
                risks=["Only catches casing/whitespace variants, not spelling variants like 'USA' vs "
                       "'United States' — those need the fuzzy-matching category merge tool."],
                alternative_actions=["Ignore (values are intentionally distinct)",
                                      "Rename categories individually"],
                goal_relevant=_is_goal_relevant(col),
            ))

        # Fuzzy category merge (#7): strict, review-only suggestions. Case/whitespace
        # variants are already handled by the normalize recommendation above, so
        # find_fuzzy_category_groups() collapses them to one value per normalized
        # key and uses row counts to suggest the most frequent spelling.
        stripped = raw.str.strip()
        fuzzy_keys = stripped.map(_fuzzy_key)
        spelling_counts = stripped[fuzzy_keys != ""].value_counts()
        fuzzy_groups = find_fuzzy_category_groups(
            spelling_counts.index.tolist(), counts=spelling_counts.to_dict()
        )
        for group in fuzzy_groups:
            group_values = group["values"]
            similarity = group["similarity"]
            is_alias_match = group["match_type"] == "alias"
            member_keys = {_fuzzy_key(v) for v in group_values}
            rows_affected = int(fuzzy_keys.isin(member_keys).sum())
            recs.append(CleaningRecommendation(
                id=str(uuid.uuid4()),
                step="category_merge_center",
                column=col,
                issue=(f"{len(group_values)} values match a known alias group: {group_values}"
                       if is_alias_match else
                       f"{len(group_values)} values differ by only one character and may be "
                       f"spelling variants: {group_values}"),
                recommended_action="fuzzy_merge",
                reason=("These values match a built-in alias group (e.g. country abbreviations such as "
                        "USA / US / United States). This is a lookup against a fixed list, not a judgement "
                        "about your data — confirm they mean the same thing in this column. "
                        if is_alias_match else
                        "These values differ by a single character edit (case and whitespace ignored). "
                        "Similar spelling does not prove the same meaning — different real-world categories "
                        "can be spelled almost identically. The suggested canonical is the most frequent "
                        "spelling in this column. ") +
                       "Review-only suggestion: nothing is merged unless you confirm.",
                # Capped below exact-merge's 0.85; spelling matches are capped lower still because
                # similar spelling is weaker evidence than a known alias.
                confidence=(round(min(0.75, similarity * 0.8), 2) if is_alias_match
                            else round(min(0.6, similarity * 0.65), 2)),
                preview={"values": group_values, "suggested_canonical": group["suggested_canonical"],
                         "similarity": similarity, "rows_affected": rows_affected,
                         "match_type": group["match_type"], "review_required": True,
                         "estimated_impact": f"Merging would affect {rows_affected} row(s) across "
                                             f"{len(group_values)} distinct value(s)."},
                how_detected=("Alias table match (known abbreviations)." if is_alias_match else
                              "Case/whitespace-insensitive comparison: at most 1 character edit "
                              "(insertion, deletion, substitution or adjacent swap), both values at least "
                              f"{FUZZY_MIN_VALUE_LENGTH} characters, same word count, no digit differences, "
                              "and every pair in the group qualifies independently (no chaining)."),
                risks=["Fuzzy matches can be wrong for values that are similar-looking but genuinely "
                       "different (e.g. a US state name vs. a same-named country) — always confirm the "
                       "grouping before merging."],
                alternative_actions=["ignore", "rename_category"],
                goal_relevant=_is_goal_relevant(col),
            ))

    # Goal-relevant recommendations surface first — this is the concrete
    # mechanism behind "quality prioritization by investigation goal" (#1):
    # everything is still shown, just reordered so the columns the user
    # actually cares about aren't buried under generic findings.
    recs.sort(key=lambda r: not r.goal_relevant)

    assign_stable_ids(recs)
    for rec in recs:
        rec.review_only = is_review_only(rec.recommended_action)
        rec.executable = is_executable_recommendation(rec.step, rec.recommended_action)

    return recs


# Recommended actions that apply_action() can execute for the card's Apply/Merge button.
# Kept next to the generators so adding a recommendation type forces a decision here;
# tests/test_recommendation_lifecycle.py executes every generated recommendation to
# prove this table matches apply_action's real behaviour.
EXECUTABLE_RECOMMENDED_ACTIONS = {
    "duplicate_manager": {"drop_duplicates"},
    "missing_value_center": {"ignore", "drop_column", "drop_rows", "impute_mean", "impute_median", "impute_mode"},
    "outlier_center": set(),  # outlier recommendations are review-only (flag_for_review); explicit actions are never "recommended"
    "category_merge_center": {"normalize_case_whitespace", "fuzzy_merge"},
}


def is_executable_recommendation(step: str, action: str) -> bool:
    return not is_review_only(action) and action in EXECUTABLE_RECOMMENDED_ACTIONS.get(step, set())


# ---------------------------------------------------------------------------
# Action executors — every one of these is only called AFTER user confirmation
# ---------------------------------------------------------------------------

def apply_action(df: pd.DataFrame, step: str, action: str, column: str | None, params: dict) -> tuple[pd.DataFrame, dict]:
    """Returns (new_df, preview_of_change). Never mutates df in place."""
    out = df.copy()
    preview: dict = {}

    if step == "duplicate_manager" and action == "drop_duplicates":
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        preview = {"rows_before": before, "rows_after": len(out), "rows_removed": before - len(out)}

    elif step == "missing_value_center":
        if column is None:
            raise ValueError("column is required for missing_value_center actions")
        missing_before = int(out[column].isna().sum())
        if action == "ignore":
            pass  # explicit no-op — used when missingness looks structural (missing by design)
        elif action == "impute_mean":
            out[column] = out[column].fillna(out[column].mean())
        elif action == "impute_median":
            out[column] = out[column].fillna(out[column].median())
        elif action == "impute_mode":
            mode_val = out[column].mode(dropna=True)
            out[column] = out[column].fillna(mode_val.iloc[0] if not mode_val.empty else None)
        elif action == "impute_constant":
            out[column] = out[column].fillna(params.get("value"))
        elif action == "drop_column":
            out = out.drop(columns=[column])
        elif action == "drop_rows":
            out = out.dropna(subset=[column]).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown missing_value_center action: {action}")
        preview = {"missing_before": missing_before,
                   "missing_after": int(out[column].isna().sum()) if column in out.columns else 0}

    elif step == "outlier_center":
        if column is None:
            raise ValueError("column is required for outlier_center actions")
        series = out[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"outlier_center requires a numeric column; '{column}' is not numeric")
        bounds = iqr_bounds(series)  # shared policy: finite values only, >= MIN_SAMPLE_SIZE, non-zero IQR
        if bounds is None and action != "ignore":
            raise ValueError(f"IQR fences are not reliable for '{column}' (needs at least {MIN_SAMPLE_SIZE} "
                             "finite numeric values and a non-zero IQR), so no outlier action was applied")
        mask = outlier_mask(series, bounds)
        count_before = int(mask.sum())
        if action == "ignore":
            pass
        elif action == "convert_to_missing":
            out.loc[mask, column] = np.nan
        elif action == "clip":
            # clip only finite values; +/-inf stays as-is (non-finite is a data-quality matter, not an outlier)
            out[column] = series.clip(lower=bounds.lower, upper=bounds.upper).where(
                np.isfinite(series.astype(float)) | series.isna(), series)
        elif action == "drop_rows":
            out = out[~mask].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown outlier_center action: {action}")
        preview = {"outliers_affected": count_before}

    elif step == "category_merge_center":
        if column is None:
            raise ValueError("column is required for category_merge_center actions")
        if action == "normalize_case_whitespace":
            before_unique = int(out[column].nunique(dropna=True))
            out[column] = out[column].astype(str).str.strip()
            mapping = out[column].dropna().drop_duplicates()
            lower_map = {}
            for val in mapping:
                lower_map.setdefault(val.lower(), val)
            out[column] = out[column].apply(lambda v: lower_map.get(v.lower(), v) if isinstance(v, str) else v)
            preview = {"categories_before": before_unique, "categories_after": int(out[column].nunique(dropna=True))}
        elif action == "fuzzy_merge":
            if column not in out.columns:
                raise ValueError(f"Column '{column}' not found")
            values_to_merge = params.get("values")
            canonical = params.get("canonical")
            if not values_to_merge or canonical is None or canonical == "":
                raise ValueError("fuzzy_merge requires 'values' and 'canonical' params")
            if (not isinstance(values_to_merge, (list, tuple))
                    or not all(isinstance(v, str) and _fuzzy_key(v) for v in values_to_merge)):
                raise ValueError("fuzzy_merge 'values' must be a list of non-empty strings")
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("fuzzy_merge 'canonical' must be a non-empty string")
            merge_keys: list[str] = []
            for v in values_to_merge:
                if _fuzzy_key(v) not in merge_keys:
                    merge_keys.append(_fuzzy_key(v))
            if len(merge_keys) < 2:
                raise ValueError("fuzzy_merge needs at least two distinct values to merge "
                                 "(case/whitespace-only differences are handled by normalize_case_whitespace)")
            canonical = canonical.strip()

            # strip + casefold on BOTH sides, so 'usa' / '  USA ' follow 'USA'.
            present = out[column].notna()
            cell_keys = out[column].astype(str).map(_fuzzy_key)
            existing_keys = set(cell_keys[present])
            missing_values = [v for v in values_to_merge if _fuzzy_key(v) not in existing_keys]
            if missing_values:
                raise ValueError(f"fuzzy_merge values not found in column '{column}': {missing_values}")

            mask = present & cell_keys.isin(merge_keys)
            rows_affected = int(mask.sum())
            changed = mask & (out[column].astype(str) != canonical).fillna(False).astype(bool)
            rows_changed = int(changed.sum())
            if rows_changed == 0:
                raise ValueError("fuzzy_merge would not change any rows")

            canonical_key = _fuzzy_key(canonical)
            collides = canonical_key in existing_keys and canonical_key not in merge_keys
            before_unique = int(out[column].nunique(dropna=True))
            try:
                if isinstance(out[column].dtype, pd.CategoricalDtype) and canonical not in out[column].cat.categories:
                    out[column] = out[column].cat.add_categories([canonical])
                out.loc[mask, column] = canonical
            except (TypeError, ValueError) as exc:
                raise ValueError(f"fuzzy_merge could not write '{canonical}' to column '{column}': {exc}") from exc
            preview = {"categories_before": before_unique, "categories_after": int(out[column].nunique(dropna=True)),
                       "rows_affected": rows_affected, "rows_changed": rows_changed, "canonical": canonical,
                       "merged_values": list(values_to_merge),
                       "canonical_collides_with_other_category": bool(collides)}
        elif action == "rename_category":
            old_value = params.get("old_value")
            new_value = params.get("new_value")
            if old_value is None or new_value is None:
                raise ValueError("rename_category requires 'old_value' and 'new_value' params")
            mask = out[column].astype(str).str.strip() == str(old_value).strip()
            rows_affected = int(mask.sum())
            out.loc[mask, column] = new_value
            preview = {"rows_affected": rows_affected, "old_value": old_value, "new_value": new_value}
        else:
            raise ValueError(f"Unknown category_merge_center action: {action}")

    elif step == "text_cleaning":
        if column is None:
            raise ValueError("column is required for text_cleaning actions")
        if action == "trim":
            out[column] = out[column].astype(str).str.strip()
        elif action == "lowercase":
            out[column] = out[column].astype(str).str.lower()
        elif action == "uppercase":
            out[column] = out[column].astype(str).str.upper()
        elif action == "remove_special_chars":
            out[column] = out[column].astype(str).str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        else:
            raise ValueError(f"Unknown text_cleaning action: {action}")
        preview = {"column": column, "action": action}

    else:
        raise ValueError(f"Unknown step/action combination: {step}/{action}")

    return out, preview
