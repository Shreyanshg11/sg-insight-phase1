"""
Dataset Versioning engine (#10).

Comparison is positional/index-based within the overlapping row range of the
two versions being compared. This is an explicit, documented limitation: once
a cleaning step removes rows (dedupe, drop_rows, outlier drop), row N in a
later version isn't guaranteed to be "the same real row" as row N in an
earlier version — pandas' reset_index(drop=True) in apply_action makes the
positions shift. For a linear cleaning pipeline (rows are only ever removed,
never reordered or re-inserted) this positional comparison is still a
reasonable, cheap approximation of "what changed", and is stated plainly in
the response rather than presented as more precise than it is.
"""
import pandas as pd
import numpy as np


def _safe(v):
    """Converts a single cell value to a JSON-safe native Python type."""
    if pd.isna(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def compare_versions(df_a: pd.DataFrame, df_b: pd.DataFrame, max_columns_detailed: int = 30,
                      max_samples_per_column: int = 3) -> dict:
    cols_a, cols_b = set(df_a.columns), set(df_b.columns)
    columns_added = sorted(cols_b - cols_a)
    columns_removed = sorted(cols_a - cols_b)
    common_columns = sorted(cols_a & cols_b)

    min_rows = min(len(df_a), len(df_b))
    column_changes = []

    # Cap detailed cell-level comparison to avoid pathological cost on very
    # wide datasets — still reports added/removed columns and row counts either way.
    for col in common_columns[:max_columns_detailed]:
        a_vals = df_a[col].reset_index(drop=True).iloc[:min_rows]
        b_vals = df_b[col].reset_index(drop=True).iloc[:min_rows]

        both_na = a_vals.isna() & b_vals.isna()
        # Compare as strings to sidestep dtype-mismatch edge cases (e.g. int64 vs float64
        # after an imputation) — this is a "did the displayed value change" comparison,
        # not a strict type-aware equality check.
        changed_mask = ~(both_na | (a_vals.astype(str) == b_vals.astype(str)))
        changed_count = int(changed_mask.sum())

        if changed_count > 0:
            changed_idx = a_vals.index[changed_mask][:max_samples_per_column]
            samples = [
                {"row": int(idx), "before": _safe(a_vals.loc[idx]), "after": _safe(b_vals.loc[idx])}
                for idx in changed_idx
            ]
            column_changes.append({"column": col, "cells_changed": changed_count, "samples": samples})

    return {
        "rows_a": len(df_a), "rows_b": len(df_b), "row_delta": len(df_b) - len(df_a),
        "columns_a": len(df_a.columns), "columns_b": len(df_b.columns),
        "columns_added": columns_added, "columns_removed": columns_removed,
        "column_changes": column_changes,
        "compared_rows": min_rows,
        "note": "Cell comparison is positional over the first min(rows_a, rows_b) rows — accurate for "
                "a purely linear cleaning pipeline (rows only ever removed, never reordered), but not a "
                "guaranteed same-row match once rows have been dropped.",
    }
