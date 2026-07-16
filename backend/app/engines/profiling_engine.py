"""
Dataset profiling logic.

Heuristics here are deliberately simple and explainable (per the "no arbitrary
confidence numbers" constraint from the project spec) — every classification
can be traced back to a concrete rule, not a black-box score.
"""
import re
import pandas as pd
import numpy as np
from app.models.schemas import ColumnProfile, ProfileResponse
from app.core.dtype_utils import _is_stringy

DATE_NAME_HINTS = re.compile(r"(date|_at$|_on$|timestamp|time)", re.IGNORECASE)
GEO_NAME_HINTS = re.compile(r"(country|state|province|city|region|zip|postal|latitude|longitude|^lat$|^lon$|^lng$|address)", re.IGNORECASE)
ID_NAME_HINTS = re.compile(r"(^id$|_id$|^uuid$|^index$|^key$)", re.IGNORECASE)


def _is_id_column(series: pd.Series, name: str) -> bool:
    if ID_NAME_HINTS.search(name):
        return True
    # High-cardinality unique column with no repeats often functions as an ID -
    # but ONLY when the values are whole numbers (or non-numeric unique codes).
    # A continuous measurement like revenue/price is naturally almost-all-unique
    # too, and fractional values are virtually never IDs, so require
    # integer-valued data before treating high uniqueness as ID-like.
    n = len(series)
    if n == 0:
        return False
    unique_ratio = series.nunique(dropna=True) / n
    if not (unique_ratio > 0.98 and n > 20):
        return False
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        return bool(np.allclose(non_null, non_null.round(), equal_nan=False))
    return True


def _is_date_column(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if DATE_NAME_HINTS.search(name) and _is_stringy(series):
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.7
    return False


def _is_geo_column(name: str) -> bool:
    return bool(GEO_NAME_HINTS.search(name))


def _infer_role(series: pd.Series, name: str, is_target_candidate: bool) -> str:
    # Date check comes first: a column with a distinct date/timestamp per row
    # (e.g. signup_date, order_date) is naturally close to 100% unique, which
    # would otherwise trip the generic high-uniqueness ID fallback below.
    if _is_date_column(series, name):
        return "date"
    if _is_id_column(series, name):
        return "id"
    if _is_geo_column(name):
        return "geographic"
    if is_target_candidate:
        return "target_candidate"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if _is_stringy(series):
        avg_len = series.dropna().astype(str).str.len().mean() if series.notna().any() else 0
        if avg_len and avg_len > 40:
            return "text"
        return "categorical"
    return "unknown"


def _target_candidates(df: pd.DataFrame) -> list[str]:
    """A very lightweight heuristic: binary/low-cardinality columns near the end
    of the dataframe, or columns literally named target/label/outcome/churn/etc."""
    name_hint = re.compile(r"(target|label|outcome|churn|default|fraud|status|result|class$)", re.IGNORECASE)
    candidates = []
    for col in df.columns:
        if name_hint.search(col):
            candidates.append(col)
            continue
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 5 and not pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
    return candidates[:5]


def profile_dataset(dataset_id: str, df: pd.DataFrame, investigation_goal: str | None = None,
                     goal_keywords: list[str] | None = None) -> ProfileResponse:
    goal_keywords = goal_keywords or []
    n_rows = len(df)
    n_cols = len(df.columns)
    memory_kb = df.memory_usage(deep=True).sum() / 1024

    dup_rows = int(df.duplicated().sum())
    dup_pct = round((dup_rows / n_rows * 100), 2) if n_rows else 0.0

    target_candidates = _target_candidates(df)

    columns_detail: list[ColumnProfile] = []
    id_cols, date_cols, geo_cols, goal_cols = [], [], [], []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows * 100), 2) if n_rows else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_pct = round((unique_count / n_rows * 100), 2) if n_rows else 0.0

        role = _infer_role(series, col, col in target_candidates)
        if role == "id":
            id_cols.append(col)
        elif role == "date":
            date_cols.append(col)
        elif role == "geographic":
            geo_cols.append(col)

        col_lower = col.lower()
        is_goal_relevant = any(kw in col_lower or col_lower in kw for kw in goal_keywords)
        if is_goal_relevant:
            goal_cols.append(col)

        sample_values = series.dropna().head(3).tolist()
        sample_values = [v if not isinstance(v, (np.integer, np.floating)) else v.item() for v in sample_values]

        columns_detail.append(ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            inferred_role=role,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            sample_values=sample_values,
            goal_relevant=is_goal_relevant,
        ))

    return ProfileResponse(
        dataset_id=dataset_id,
        rows=n_rows,
        columns=n_cols,
        memory_kb=round(memory_kb, 2),
        duplicate_rows=dup_rows,
        duplicate_pct=dup_pct,
        id_column_candidates=id_cols,
        target_candidates=target_candidates,
        date_columns=date_cols,
        geographic_columns=geo_cols,
        columns_detail=columns_detail,
        investigation_goal=investigation_goal,
        goal_relevant_columns=goal_cols,
    )
