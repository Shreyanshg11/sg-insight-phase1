"""
Notebook Engine — turns the session's Analyst Notebook (session.log, the
exact record of every cleaning/feature-engineering action a user confirmed)
into an actual, executable Jupyter notebook. One markdown+code cell pair per
logged action, in the exact order applied, each markdown cell carrying the
same reason/confidence the user saw in the UI before confirming it — the
notebook IS the trust/explainability story this whole app is built around,
just handed to the user as a portable, re-runnable artifact instead of a
one-way report.

This deliberately does NOT go through build_report_data() (report_engine.py):
that function summarizes the log into display text and drops the raw
`params` dict, but exact code generation needs `params` verbatim (e.g. the
literal imputed constant, the exact fuzzy-merge canonical value) plus the
raw session.versions DataFrames themselves (to embed the original data and
to detect exactly which column name a feature-engineering step produced,
covered below). Reads the session directly instead.

Code generation mirrors cleaning_engine.apply_action and
feature_engineering_engine.apply_feature_action line-for-line — if either of
those functions changes, this needs to change with it. For feature
engineering specifically, `new_column_name` (if the user overrode the
default) isn't stored in the log at all, only the resulting DataFrame is —
so rather than reconstruct the default-naming logic (a whole extra place to
get subtly wrong), the actual new column name(s) are recovered by diffing
session.versions[k].columns against session.versions[k+1].columns. This is
robust by construction: whatever name ended up in the real dataframe is
exactly what the generated code will produce too.
"""
import json
import uuid

# Above this, the original dataset is NOT embedded inline (keeps the .ipynb
# file from becoming enormous) — the load cell instead asks the user to
# supply their own copy of the file.
_MAX_EMBED_CSV_BYTES = 500_000


def _cell_id() -> str:
    # nbformat_minor >= 5 requires a unique `id` on every cell.
    return uuid.uuid4().hex[:8]


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": source}


def _new_columns_for_step(session, log_index: int) -> list[str]:
    """Columns present after this action that weren't present before it,
    in the order they appear in the resulting DataFrame."""
    prev_cols = set(session.versions[log_index].columns)
    return [c for c in session.versions[log_index + 1].columns if c not in prev_cols]


def _generate_cleaning_code(entry) -> str:
    col, action, params = entry.column, entry.action, entry.params

    if entry.step == "duplicate_manager":
        return "df = df.drop_duplicates().reset_index(drop=True)"

    if entry.step == "missing_value_center":
        if action == "impute_mean":
            return f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].mean())"
        if action == "impute_median":
            return f"df[{col!r}] = df[{col!r}].fillna(df[{col!r}].median())"
        if action == "impute_mode":
            return (f"_mode_val = df[{col!r}].mode(dropna=True)\n"
                     f"df[{col!r}] = df[{col!r}].fillna(_mode_val.iloc[0] if not _mode_val.empty else None)")
        if action == "impute_constant":
            return f"df[{col!r}] = df[{col!r}].fillna({params.get('value')!r})"
        if action == "drop_column":
            return f"df = df.drop(columns=[{col!r}])"
        if action == "drop_rows":
            return f"df = df.dropna(subset=[{col!r}]).reset_index(drop=True)"

    if entry.step == "outlier_center":
        bounds = (f"_q1, _q3 = df[{col!r}].quantile(0.25), df[{col!r}].quantile(0.75)\n"
                  f"_iqr = _q3 - _q1\n"
                  f"_lower, _upper = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr")
        if action == "ignore":
            return "# Outliers reviewed and intentionally left unchanged."
        if action == "convert_to_missing":
            return f"{bounds}\n_mask = (df[{col!r}] < _lower) | (df[{col!r}] > _upper)\ndf.loc[_mask, {col!r}] = np.nan"
        if action == "clip":
            return f"{bounds}\ndf[{col!r}] = df[{col!r}].clip(lower=_lower, upper=_upper)"
        if action == "drop_rows":
            return (f"{bounds}\n_mask = (df[{col!r}] < _lower) | (df[{col!r}] > _upper)\n"
                     f"df = df[~_mask].reset_index(drop=True)")

    if entry.step == "category_merge_center":
        if action == "normalize_case_whitespace":
            return (f"df[{col!r}] = df[{col!r}].astype(str).str.strip()\n"
                     f"_mapping = df[{col!r}].dropna().drop_duplicates()\n"
                     f"_lower_map = {{}}\n"
                     f"for _val in _mapping:\n"
                     f"    _lower_map.setdefault(_val.lower(), _val)\n"
                     f"df[{col!r}] = df[{col!r}].apply(lambda v: _lower_map.get(v.lower(), v) if isinstance(v, str) else v)")
        if action == "fuzzy_merge":
            return (f"_mask = df[{col!r}].astype(str).str.strip().isin({params.get('values')!r})\n"
                     f"df.loc[_mask, {col!r}] = {params.get('canonical')!r}")
        if action == "rename_category":
            return (f"_mask = df[{col!r}].astype(str).str.strip() == str({params.get('old_value')!r}).strip()\n"
                     f"df.loc[_mask, {col!r}] = {params.get('new_value')!r}")

    if entry.step == "text_cleaning":
        if action == "trim":
            return f"df[{col!r}] = df[{col!r}].astype(str).str.strip()"
        if action == "lowercase":
            return f"df[{col!r}] = df[{col!r}].astype(str).str.lower()"
        if action == "uppercase":
            return f"df[{col!r}] = df[{col!r}].astype(str).str.upper()"
        if action == "remove_special_chars":
            return f"df[{col!r}] = df[{col!r}].astype(str).str.replace(r\"[^a-zA-Z0-9\\s]\", \"\", regex=True)"

    return f"# Could not regenerate code for {entry.step}/{action} — apply manually."


def _generate_feature_code(entry, new_cols: list[str]) -> str:
    category = entry.step.split(":", 1)[1]
    columns = entry.column.split(", ")
    action, params = entry.action, entry.params

    if category == "date_features" and action == "extract_date_parts":
        col = columns[0]
        parts = params.get("parts") or ["year", "month", "day", "dayofweek", "is_weekend"]
        lines = [f"_parsed = pd.to_datetime(df[{col!r}], errors=\"coerce\")"]
        part_exprs = {
            "year": f"df[{col!r} + '_year'] = _parsed.dt.year",
            "month": f"df[{col!r} + '_month'] = _parsed.dt.month",
            "day": f"df[{col!r} + '_day'] = _parsed.dt.day",
            "dayofweek": f"df[{col!r} + '_dayofweek'] = _parsed.dt.day_of_week",
            "is_weekend": f"df[{col!r} + '_is_weekend'] = _parsed.dt.day_of_week.isin([5, 6])",
        }
        lines += [part_exprs[p] for p in parts if p in part_exprs]
        return "\n".join(lines)

    if category == "ratio" and action == "create_ratio":
        num_col, den_col = columns[0], columns[1]
        new_col = new_cols[0] if new_cols else f"{num_col}_per_{den_col}"
        return f"df[{new_col!r}] = df[{num_col!r}] / df[{den_col!r}].replace(0, np.nan)"

    if category == "bucketing" and action == "quartile_bucket":
        col = columns[0]
        new_col = new_cols[0] if new_cols else f"{col}_bucket"
        return f"df[{new_col!r}] = pd.qcut(df[{col!r}], 4, labels=[\"Q1\", \"Q2\", \"Q3\", \"Q4\"], duplicates=\"drop\")"

    if category == "scaling_normalization":
        col = columns[0]
        new_col = new_cols[0] if new_cols else f"{col}_scaled"
        if action == "min_max_scale":
            return (f"_lo, _hi = df[{col!r}].min(), df[{col!r}].max()\n"
                     f"df[{new_col!r}] = (df[{col!r}] - _lo) / (_hi - _lo) if _hi != _lo else 0.0")
        if action == "z_score_standardize":
            return f"df[{new_col!r}] = (df[{col!r}] - df[{col!r}].mean()) / df[{col!r}].std()"
        if action == "robust_scale":
            return (f"_median = df[{col!r}].median()\n"
                     f"_iqr = df[{col!r}].quantile(0.75) - df[{col!r}].quantile(0.25)\n"
                     f"df[{new_col!r}] = (df[{col!r}] - _median) / _iqr if _iqr != 0 else df[{col!r}] - _median")

    if category == "encoding":
        col = columns[0]
        if action == "one_hot_encode":
            return (f"_dummies = pd.get_dummies(df[{col!r}], prefix={col!r})\n"
                     f"df = pd.concat([df, _dummies], axis=1)")
        if action == "label_encode":
            new_col = new_cols[0] if new_cols else f"{col}_encoded"
            return f"df[{new_col!r}] = df[{col!r}].astype(\"category\").cat.codes"

    if category == "log_transform" and action == "log1p_transform":
        col = columns[0]
        new_col = new_cols[0] if new_cols else f"{col}_log"
        return f"df[{new_col!r}] = np.log1p(df[{col!r}])"

    return f"# Could not regenerate code for {category}/{action} — apply manually."


def build_notebook(session) -> str:
    """Returns the notebook as a JSON string (nbformat v4), ready to write
    straight to a .ipynb file — Jupyter's on-disk format IS this JSON."""
    cells = [_md_cell(
        f"# {session.filename} — reproducible pipeline\n\n"
        f"Generated from an SG-Insight session. Investigation goal: "
        f"**{session.investigation_goal or '(none set)'}**.\n\n"
        f"Runs the exact {len(session.log)} cleaning/feature-engineering action(s) "
        f"confirmed in that session, in order, each with the reason and confidence "
        f"shown at the time. This covers the deterministic pipeline only — EDA, "
        f"Analysis, and Visualization were exploratory in the app and aren't replayed here."
    )]
    cells.append(_code_cell("import pandas as pd\nimport numpy as np"))

    original_df = session.versions[0]
    csv_text = original_df.to_csv(index=False)
    if len(csv_text.encode("utf-8")) <= _MAX_EMBED_CSV_BYTES:
        cells.append(_md_cell("## Load data\nOriginal dataset, embedded inline so this notebook runs standalone."))
        cells.append(_code_cell(
            "import io\n\n_csv_text = \"\"\"\\\n" + csv_text.replace("\\", "\\\\") + "\"\"\"\n"
            "df = pd.read_csv(io.StringIO(_csv_text))"
        ))
    else:
        cells.append(_md_cell(
            f"## Load data\nOriginal dataset was {len(csv_text.encode('utf-8')):,} bytes as CSV — too large to "
            f"embed inline. Replace the path below with your copy of **{session.filename}**."
        ))
        cells.append(_code_cell(f"df = pd.read_csv({session.filename!r})  # <-- update this path"))

    for i, entry in enumerate(session.log):
        header = f"## Step {i + 1}: {entry.action}"
        if entry.column:
            header += f" — `{entry.column}`"
        meta = f"**Reason:** {entry.reason}"
        if entry.confidence is not None:
            meta += f"  \n**Confidence:** {entry.confidence}"
        if entry.result_summary:
            meta += f"  \n**Result at the time:** {entry.result_summary}"
        cells.append(_md_cell(f"{header}\n\n{meta}"))

        if entry.step.startswith("feature_engineering:"):
            code = _generate_feature_code(entry, _new_columns_for_step(session, i))
        else:
            code = _generate_cleaning_code(entry)
        cells.append(_code_cell(code))

    cells.append(_md_cell("## Result"))
    cells.append(_code_cell("print(df.shape)\ndf.head()"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1)
