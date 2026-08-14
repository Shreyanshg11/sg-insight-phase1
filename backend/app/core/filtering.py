"""
Filtering — the mechanism behind Interactive Investigation (session 12):
picking a subset of rows to view, either by hand or by clicking a chart
element (drill-down), and having every filter-aware endpoint compute
against that subset instead of the whole dataset.

A filter is a VIEW, not a transformation: applying one never touches
session.log or session.versions, and there's no "undo" for it — it's
recomputed fresh against session.current on every request, the same
"always live off the current version" principle the Dashboard's pinned
charts already follow. This is deliberately different from cleaning/feature
engineering, which do push versions.
"""
import json
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import HTTPException, Query

from app.models.schemas import FilterCondition


def _json_safe(value):
    """Same numpy-scalar-to-native pattern profiling_engine.py already uses
    for sample_values, extended to cover NaN/NaT and Timestamps too, since
    filtered-rows returns whole raw rows rather than a handful of samples."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # non-scalar cell (rare) — fall through and return as-is
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def rows_to_json_safe_records(df: pd.DataFrame) -> list[dict]:
    return [{col: _json_safe(v) for col, v in row.items()} for row in df.to_dict(orient="records")]


def apply_filter(df: pd.DataFrame, filters: list[FilterCondition]) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for f in filters:
        if f.column not in df.columns:
            raise ValueError(f"Filter references column '{f.column}', which isn't in this dataset "
                              f"(it may have been dropped by a cleaning step since the filter was set).")
        s = df[f.column]
        if f.op == "eq":
            mask &= (s == f.value)
        elif f.op == "in":
            mask &= s.isin(f.value)
        elif f.op == "gte":
            mask &= (s >= f.value)
        elif f.op == "lte":
            mask &= (s <= f.value)
        elif f.op == "between":
            lo, hi = f.value
            mask &= (s >= lo) & (s <= hi)
        else:
            raise ValueError(f"Unknown filter op: {f.op!r} (must be eq/in/gte/lte/between)")
    return df[mask]


def parse_filters_query(filters: Optional[str] = Query(
    None, description="JSON-encoded list of {column, op, value} objects")) -> list[FilterCondition]:
    """
    FastAPI dependency for GET routes (EDA endpoints) — filters arrive as a
    JSON-encoded string query param rather than a request body, since GET
    requests here don't have one. POST routes (filtered-rows, build-chart)
    just take `filters: List[FilterCondition]` directly in their body instead
    of using this.
    """
    if not filters:
        return []
    try:
        raw = json.loads(filters)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`filters` query param must be valid JSON.")
    try:
        return [FilterCondition(**f) for f in raw]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter condition: {e}")
