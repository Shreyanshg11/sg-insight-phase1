"""
Data quality scoring + cleaning recommendations + action executors.

Design constraint from the spec: confidence scores must map to a real,
explainable heuristic — never an arbitrary-looking number. Every
recommendation below documents in `reason` exactly which heuristic produced it.
"""
import uuid
import difflib
import pandas as pd
import numpy as np
from app.models.schemas import QualityScore, QualityBreakdownItem, CleaningRecommendation
from app.core.dtype_utils import stringy_columns
from app.engines.missingness_engine import classify_missingness
from app.engines.outlier_engine import classify_outliers_for_column

# Fuzzy category merge (#7): a small built-in alias table for well-known
# abbreviation cases called out explicitly in the brief (USA/US/United States),
# plus generic difflib string-similarity for typos/spelling variants that
# aren't in the table. Deliberately conservative — never auto-merges;
# every fuzzy group is just another recommendation requiring confirmation.
FUZZY_SIMILARITY_THRESHOLD = 0.72
FUZZY_MAX_DISTINCT_VALUES = 150  # skip high-cardinality columns; not really "categories"

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
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            validity_scores.append(1.0)
            continue
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outlier_ratio = ((series < lower) | (series > upper)).mean()
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
            reason=f"Numeric columns average {round(validity_shortfall, 1)}% IQR-outlier rate; "
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

def _alias_match(a: str, b: str) -> bool:
    return any(a in group and b in group for group in _ALIAS_GROUPS)


def find_fuzzy_category_groups(values: list[str]) -> list[dict]:
    """Groups distinct category values likely to be the same real-world
    category spelled differently — via the alias table (similarity 1.0) or
    generic difflib.SequenceMatcher ratio for everything else. Returns groups
    of size >= 2 only. Skips columns with more distinct values than
    FUZZY_MAX_DISTINCT_VALUES to avoid an O(n^2) blowup on high-cardinality
    (i.e. not really categorical) columns."""
    if len(values) < 2 or len(values) > FUZZY_MAX_DISTINCT_VALUES:
        return []

    parent = list(range(len(values)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    lowered = [v.lower().strip() for v in values]
    pair_scores: dict[tuple[int, int], float] = {}

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if _alias_match(lowered[i], lowered[j]):
                union(i, j)
                pair_scores[(i, j)] = 1.0
                continue
            ratio = difflib.SequenceMatcher(None, lowered[i], lowered[j]).ratio()
            if ratio >= FUZZY_SIMILARITY_THRESHOLD:
                union(i, j)
                pair_scores[(i, j)] = ratio

    groups_by_root: dict[int, list[int]] = {}
    for idx in range(len(values)):
        groups_by_root.setdefault(find(idx), []).append(idx)

    results = []
    for idxs in groups_by_root.values():
        if len(idxs) < 2:
            continue
        group_values = [values[i] for i in idxs]
        # Longest spelling as the default canonical suggestion — simple and
        # explainable; the user can always override via Rename before merging.
        canonical = max(group_values, key=len)
        relevant_scores = [s for (i, j), s in pair_scores.items() if i in idxs and j in idxs]
        avg_similarity = round(sum(relevant_scores) / len(relevant_scores), 2) if relevant_scores else FUZZY_SIMILARITY_THRESHOLD
        results.append({"values": group_values, "suggested_canonical": canonical, "similarity": avg_similarity})

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
        outlier_pct = summary.total_outliers / len(df[col].dropna())
        counts = summary.tier_counts
        tier_summary = (f"{counts['likely_data_entry_error']} likely data-entry error(s), "
                         f"{counts['legitimate_rare_value']} legitimate rare value(s), "
                         f"{counts['statistically_unusual']} statistically unusual but unclassified")

        # Column-level confidence: the highest per-point confidence among the
        # points actually flagged as errors (if any) wins, since that's the
        # single most actionable signal in the set; otherwise fall back to the
        # average confidence across whatever was found. Keeps a single number
        # for the recommendation card while the full per-point breakdown is
        # still available in `preview` for anyone who wants to drill in.
        error_points = [p for p in summary.points if p.tier == "likely_data_entry_error"]
        if error_points:
            confidence = max(p.confidence for p in error_points)
        else:
            confidence = round(float(np.mean([p.confidence for p in summary.points])), 2) if summary.points else 0.5

        recs.append(CleaningRecommendation(
            id=str(uuid.uuid4()),
            step="outlier_center",
            column=col,
            issue=f"{summary.total_outliers} values outside IQR fences [{summary.lower_fence}, {summary.upper_fence}] "
                  f"— {tier_summary}",
            recommended_action="flag_for_review",
            reason=f"Standard 1.5×IQR rule: Q1={summary.q1}, Q3={summary.q3}, IQR={summary.iqr}. Each flagged "
                   "value was further checked against a stricter 3.0×IQR 'far-out' fence, a decimal-point/"
                   "unit-slip magnitude pattern (~10x or ~100x the median), and whether it repeats elsewhere in "
                   f"the column — {tier_summary}. Flagged for review rather than auto-modified — outlier "
                   "handling is context-dependent even for the 'likely error' tier.",
            confidence=confidence,
            preview={"outlier_count": summary.total_outliers, "lower_fence": summary.lower_fence,
                     "upper_fence": summary.upper_fence, "far_out_lower": summary.far_out_lower,
                     "far_out_upper": summary.far_out_upper, "tier_counts": counts,
                     "sample_outliers": [p.model_dump() for p in summary.points[:5]],
                     "rows_affected": summary.total_outliers,
                     "estimated_impact": f"{round(outlier_pct*100,1)}% of non-null values in '{col}' fall "
                                         "outside the fence. No change is applied until you pick ignore/"
                                         "convert-to-missing/clip/drop below."},
            how_detected=f"1.5×IQR fence: values outside [{summary.lower_fence}, {summary.upper_fence}] "
                         f"(Q1={summary.q1}, Q3={summary.q3}, IQR={summary.iqr}), each point additionally "
                         "tiered by far-out distance, magnitude pattern, and repetition (see outlier_engine).",
            risks=["A legitimate extreme value (not an error) could be clipped or dropped if you act on "
                   "this without checking context — this applies even to values tiered as 'likely error', "
                   "since the tiering is still a heuristic, not a certainty."],
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

        # Fuzzy category merge (#7): run on one representative original-cased
        # value per normalized group, so pure case/whitespace variants (already
        # handled above) don't get double-flagged here.
        representative = raw.groupby(normalized_key).first()
        distinct_values = representative.tolist()
        fuzzy_groups = find_fuzzy_category_groups(distinct_values)
        for group in fuzzy_groups:
            group_values = group["values"]
            similarity = group["similarity"]
            rows_affected = int(raw.str.strip().isin(group_values).sum())
            is_alias_match = similarity == 1.0
            recs.append(CleaningRecommendation(
                id=str(uuid.uuid4()),
                step="category_merge_center",
                column=col,
                issue=f"{len(group_values)} values may refer to the same category: {group_values}",
                recommended_action="fuzzy_merge",
                reason=(f"These values matched a known alias group (e.g. country/region abbreviations)."
                        if is_alias_match else
                        f"These values are {int(similarity*100)}% similar by string comparison — likely "
                        "the same category spelled differently.") +
                       " Fuzzy matches are suggestions only; review before merging.",
                confidence=round(min(0.75, similarity * 0.8), 2),  # capped below exact-merge's 0.85 — fuzzy is inherently less certain
                preview={"values": group_values, "suggested_canonical": group["suggested_canonical"],
                         "similarity": similarity, "rows_affected": rows_affected,
                         "estimated_impact": f"Merging would affect {rows_affected} row(s) across "
                                             f"{len(group_values)} distinct value(s)."},
                how_detected="Alias table match (known abbreviations)" if is_alias_match else
                             f"difflib.SequenceMatcher ratio ≥ {FUZZY_SIMILARITY_THRESHOLD} between normalized values.",
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

    return recs


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
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count_before = int(mask.sum())
        if action == "ignore":
            pass
        elif action == "convert_to_missing":
            out.loc[mask, column] = np.nan
        elif action == "clip":
            out[column] = series.clip(lower=lower, upper=upper)
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
            values_to_merge = params.get("values", [])
            canonical = params.get("canonical")
            if not values_to_merge or not canonical:
                raise ValueError("fuzzy_merge requires 'values' and 'canonical' params")
            before_unique = int(out[column].nunique(dropna=True))
            mask = out[column].astype(str).str.strip().isin(values_to_merge)
            rows_affected = int(mask.sum())
            out.loc[mask, column] = canonical
            preview = {"categories_before": before_unique, "categories_after": int(out[column].nunique(dropna=True)),
                       "rows_affected": rows_affected, "canonical": canonical}
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
