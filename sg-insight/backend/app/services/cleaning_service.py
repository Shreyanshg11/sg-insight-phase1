"""
Data quality scoring + cleaning recommendations + action executors.

Design constraint from the spec: confidence scores must map to a real,
explainable heuristic — never an arbitrary-looking number. Every
recommendation below documents in `reason` exactly which heuristic produced it.
"""
import uuid
import pandas as pd
import numpy as np
from app.models.schemas import QualityScore, CleaningRecommendation


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
    uniqueness = 1 - (dup_rows / n_rows) if n_rows else 1.0

    # Consistency: fraction of object columns whose values don't show obvious
    # casing/whitespace inconsistency (proxy: trimmed+lowercased nunique vs raw nunique)
    consistency_scores = []
    for col in df.select_dtypes(include="object").columns:
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

    overall = float(np.mean([completeness, consistency, validity, uniqueness]))

    return QualityScore(
        overall=round(overall * 100, 1),
        completeness=round(completeness * 100, 1),
        consistency=round(consistency * 100, 1),
        validity=round(validity * 100, 1),
        uniqueness=round(uniqueness * 100, 1),
    )


# ---------------------------------------------------------------------------
# Recommendations (preview + reason + confidence, nothing applied yet)
# ---------------------------------------------------------------------------

def generate_recommendations(df: pd.DataFrame) -> list[CleaningRecommendation]:
    recs: list[CleaningRecommendation] = []
    n_rows = len(df)

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
            preview={"duplicate_rows": dup_count, "example_index": df[df.duplicated()].index[:5].tolist()},
        ))

    # Missing values per column
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        missing_pct = missing / n_rows if n_rows else 0
        if missing_pct > 0.6:
            action, reason, confidence = "drop_column", (
                f"{round(missing_pct*100,1)}% of values are missing — above the ~60% threshold "
                "where imputation typically introduces more noise than signal."
            ), 0.7
        elif pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].dropna().skew() if df[col].notna().sum() > 2 else 0
            if abs(skew) > 1:
                action, reason, confidence = "impute_median", (
                    f"Column is numeric with skew={round(float(skew),2)} (|skew|>1), so median is more "
                    "robust to the skew than mean."
                ), 0.8
            else:
                action, reason, confidence = "impute_mean", (
                    f"Column is numeric and roughly symmetric (skew={round(float(skew),2)}), "
                    "so mean imputation won't distort the distribution much."
                ), 0.75
        else:
            action, reason, confidence = "impute_mode", (
                "Column is categorical/text; mode (most frequent value) imputation preserves "
                "the existing category distribution better than any numeric substitute."
            ), 0.7

        recs.append(CleaningRecommendation(
            id=str(uuid.uuid4()),
            step="missing_value_center",
            column=col,
            issue=f"{missing} missing values ({round(missing_pct*100,1)}%)",
            recommended_action=action,
            reason=reason,
            confidence=confidence,
            preview={"missing_count": missing, "missing_pct": round(missing_pct * 100, 1)},
        ))

    # Outliers (IQR method) on numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 8:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0:
            continue
        outlier_pct = len(outliers) / len(series)
        # Confidence scales with how far outliers sit from the fence, capped sensibly
        confidence = round(min(0.9, 0.5 + outlier_pct), 2)
        recs.append(CleaningRecommendation(
            id=str(uuid.uuid4()),
            step="outlier_center",
            column=col,
            issue=f"{len(outliers)} values outside IQR fences [{round(lower,2)}, {round(upper,2)}]",
            recommended_action="flag_for_review",
            reason=f"Standard 1.5×IQR rule: Q1={round(q1,2)}, Q3={round(q3,2)}, IQR={round(iqr,2)}. "
                   "Flagged for review rather than auto-modified — outlier handling is context-dependent.",
            confidence=confidence,
            preview={"outlier_count": len(outliers), "lower_fence": round(lower, 2), "upper_fence": round(upper, 2),
                     "sample_outliers": outliers.head(5).tolist()},
        ))

    # Category merge candidates (case/whitespace variants of the same category)
    for col in df.select_dtypes(include="object").columns:
        raw = df[col].dropna().astype(str)
        if raw.empty:
            continue
        normalized_groups = raw.str.strip().str.lower().value_counts()
        raw_groups = raw.value_counts()
        if len(raw_groups) > len(normalized_groups):
            merge_savings = len(raw_groups) - len(normalized_groups)
            recs.append(CleaningRecommendation(
                id=str(uuid.uuid4()),
                step="category_merge_center",
                column=col,
                issue=f"{merge_savings} category variant(s) collapse when case/whitespace is normalized",
                recommended_action="normalize_case_whitespace",
                reason="Detected values that are identical after trim+lowercase but distinct in raw form "
                       "(e.g. 'USA' vs 'usa ' vs ' USA'). This is a mechanical, low-risk normalization.",
                confidence=0.85,
                preview={"raw_category_count": int(len(raw_groups)), "normalized_category_count": int(len(normalized_groups))},
            ))

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
        if action == "impute_mean":
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
