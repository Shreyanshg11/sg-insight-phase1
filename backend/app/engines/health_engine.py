"""
Dataset Health Dashboard engine.

This is deliberately a *detection* layer, not a remediation one — it answers
"what's wrong and how much of it is there", using the same explainable-heuristic
principle as everywhere else in the app. Fixing currency/unit/date-format issues
gets its own dedicated cleaning module later (matching the original Phase 2 plan);
this only has to flag their presence clearly enough that a user knows to look.

Each HealthCategory maps 1:1 to a dashboard card. `status` drives the card's
color; `summary` is the one-line metric; `detail` is the exact rule that
produced the status, so nothing on the dashboard is an unexplained traffic light.
"""
import re
import pandas as pd
import numpy as np
from app.core.iqr import finite_values, iqr_bounds, outlier_mask
from pydantic import BaseModel
from typing import List
from app.core.dtype_utils import stringy_columns

MISSING_WARNING_THRESHOLD = 5.0    # % of all cells
MISSING_CRITICAL_THRESHOLD = 20.0

CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹"]

UNIT_PATTERN = re.compile(r'\d+(?:\.\d+)?\s*(kg|kgs|g|lb|lbs|oz|km|cm|mm|m|mi|ft|in|l|ml|gal)\b', re.IGNORECASE)
_UNIT_ALIASES = {"kgs": "kg", "lbs": "lb"}

# Named date-format patterns checked against string values to detect a column
# mixing more than one literal format (as opposed to just being unparsed dates).
DATE_FORMAT_PATTERNS = {
    "YYYY-MM-DD": re.compile(r'^\d{4}-\d{1,2}-\d{1,2}$'),
    "YYYY/MM/DD": re.compile(r'^\d{4}/\d{1,2}/\d{1,2}$'),
    "MM/DD/YYYY": re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$'),
    "DD-MM-YYYY": re.compile(r'^\d{1,2}-\d{1,2}-\d{4}$'),
    "DD.MM.YYYY": re.compile(r'^\d{1,2}\.\d{1,2}\.\d{4}$'),
    "Month D, YYYY": re.compile(r'^[A-Za-z]+\s+\d{1,2},?\s+\d{4}$'),
}

DATE_NAME_HINT = re.compile(r"(date|_at$|_on$|time)", re.IGNORECASE)


class HealthCategory(BaseModel):
    key: str
    label: str
    status: str  # "good" | "warning" | "critical"
    summary: str
    detail: str
    affected_columns: List[str] = []


def _missing_values_health(df: pd.DataFrame) -> HealthCategory:
    n_rows, n_cols = len(df), len(df.columns) or 1
    total_cells = n_rows * n_cols if n_rows else 1
    missing_cells = int(df.isna().sum().sum())
    pct = round(missing_cells / total_cells * 100, 1) if total_cells else 0.0

    per_col = df.isna().mean() * 100
    affected = per_col[per_col > 0].sort_values(ascending=False)

    status = "good" if pct == 0 else "critical" if pct >= MISSING_CRITICAL_THRESHOLD else \
             "warning" if pct >= MISSING_WARNING_THRESHOLD else "good"

    return HealthCategory(
        key="missing_values", label="Missing Values", status=status,
        summary=f"{len(affected)} column(s), {pct}% of all cells" if len(affected) else "No missing values",
        detail=f"Dataset-wide missing rate {pct}% — good <{MISSING_WARNING_THRESHOLD}%, "
               f"warning <{MISSING_CRITICAL_THRESHOLD}%, critical ≥{MISSING_CRITICAL_THRESHOLD}%.",
        affected_columns=affected.index.tolist()[:10],
    )


def _duplicate_health(df: pd.DataFrame) -> HealthCategory:
    n_rows = len(df)
    dup_rows = int(df.duplicated().sum())
    pct = round(dup_rows / n_rows * 100, 1) if n_rows else 0.0
    status = "good" if dup_rows == 0 else "critical" if pct >= 10 else "warning"
    return HealthCategory(
        key="duplicates", label="Duplicate Status", status=status,
        summary=f"{dup_rows} duplicate row(s) ({pct}%)" if dup_rows else "No exact duplicate rows",
        detail="Exact full-row duplicates via pandas.duplicated() — warning under 10% of rows, "
               "critical at or above 10%.",
    )


def _outlier_health(df: pd.DataFrame) -> HealthCategory:
    affected = []
    total_outliers = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        series = finite_values(df[col])
        bounds = iqr_bounds(series)  # shared IQR policy
        if bounds is None:
            continue
        count = int(outlier_mask(series, bounds).sum())
        if count > 0:
            affected.append(col)
            total_outliers += count

    status = "good" if not affected else "warning" if len(affected) <= 2 else "critical"
    return HealthCategory(
        key="outliers", label="Outliers", status=status,
        summary=f"{total_outliers} value(s) across {len(affected)} column(s)" if affected else "No IQR outliers detected",
        detail="Standard 1.5×IQR fence per numeric column — warning at 1-2 affected columns, "
               "critical at 3+.",
        affected_columns=affected,
    )


def _data_type_issues_health(df: pd.DataFrame) -> HealthCategory:
    """Flags text columns that are actually numeric-looking (stored as the wrong
    dtype) and object columns holding a genuine mix of Python types."""
    affected = []
    reasons = []
    for col in stringy_columns(df):
        series = df[col].dropna()
        if series.empty:
            continue
        # Looks-numeric-but-stored-as-text check
        coerced = pd.to_numeric(series, errors="coerce")
        numeric_ratio = coerced.notna().mean()
        if numeric_ratio > 0.9:
            affected.append(col)
            reasons.append(f"'{col}': {round(numeric_ratio*100,1)}% of values parse as numeric but "
                            "column is stored as text")
            continue
        # Mixed literal Python types within an object column (rare but real)
        if df[col].dtype == object:
            type_count = series.map(type).nunique()
            if type_count > 1:
                affected.append(col)
                reasons.append(f"'{col}': column holds {type_count} distinct Python types")

    status = "good" if not affected else "warning"
    return HealthCategory(
        key="data_type_issues", label="Data Type Issues", status=status,
        summary=f"{len(affected)} column(s) with type mismatches" if affected else "No type mismatches detected",
        detail="; ".join(reasons) if reasons else "Checked text columns for numeric-parseable content "
               "(>90% threshold) and object columns for mixed Python types.",
        affected_columns=affected,
    )


def _mixed_categories_health(df: pd.DataFrame) -> HealthCategory:
    affected = []
    for col in stringy_columns(df):
        raw = df[col].dropna().astype(str)
        if raw.empty:
            continue
        raw_count = raw.value_counts().shape[0]
        norm_count = raw.str.strip().str.lower().value_counts().shape[0]
        if raw_count > norm_count:
            affected.append(col)

    status = "good" if not affected else "warning"
    return HealthCategory(
        key="mixed_categories", label="Mixed Categories", status=status,
        summary=f"{len(affected)} column(s) with casing/whitespace variants" if affected else "No category variants detected",
        detail="Compared value_counts() before vs after trim+lowercase per text column — a drop in "
               "distinct count means variants like 'USA' / 'usa ' collapse to one category.",
        affected_columns=affected,
    )


def _mixed_date_formats_health(df: pd.DataFrame) -> HealthCategory:
    affected = []
    details = []
    for col in stringy_columns(df):
        if not DATE_NAME_HINT.search(col):
            continue
        values = df[col].dropna().astype(str).head(200)  # sample for cost control
        if values.empty:
            continue
        formats_seen = set()
        for val in values:
            for fmt_name, pattern in DATE_FORMAT_PATTERNS.items():
                if pattern.match(val.strip()):
                    formats_seen.add(fmt_name)
                    break
        if len(formats_seen) > 1:
            affected.append(col)
            details.append(f"'{col}': formats {sorted(formats_seen)}")

    status = "good" if not affected else "warning"
    return HealthCategory(
        key="mixed_date_formats", label="Mixed Date Formats", status=status,
        summary=f"{len(affected)} column(s) mixing date formats" if affected else "No mixed date formats detected",
        detail="; ".join(details) if details else "Matched sampled values in date-hinted columns "
               "against known literal formats (ISO, US-slash, EU-dash, dotted, written-month).",
        affected_columns=affected,
    )


def _currency_issues_health(df: pd.DataFrame) -> HealthCategory:
    affected = []
    details = []
    for col in stringy_columns(df):
        values = df[col].dropna().astype(str).head(200)
        if values.empty:
            continue
        symbols_found = set()
        has_plain_number = False
        for val in values:
            v = val.strip()
            matched_symbol = next((s for s in CURRENCY_SYMBOLS if s in v), None)
            if matched_symbol:
                symbols_found.add(matched_symbol)
            elif re.match(r'^-?\d+(\.\d+)?$', v):
                has_plain_number = True
        if len(symbols_found) > 1 or (symbols_found and has_plain_number):
            affected.append(col)
            details.append(f"'{col}': symbols found {sorted(symbols_found)}"
                            f"{' + unmarked numbers' if has_plain_number else ''}")

    status = "good" if not affected else "warning"
    return HealthCategory(
        key="currency_issues", label="Currency Issues", status=status,
        summary=f"{len(affected)} column(s) with inconsistent currency marking" if affected else "No currency inconsistencies detected",
        detail="; ".join(details) if details else "Scanned sampled values for currency symbols "
               f"({', '.join(CURRENCY_SYMBOLS)}) and flagged columns mixing symbols or mixing "
               "marked/unmarked numbers. Best-effort — verify before relying on this.",
        affected_columns=affected,
    )


def _unit_issues_health(df: pd.DataFrame) -> HealthCategory:
    affected = []
    details = []
    for col in stringy_columns(df):
        values = df[col].dropna().astype(str).head(200)
        if values.empty:
            continue
        units_found = set()
        for val in values:
            m = UNIT_PATTERN.search(val.strip())
            if m:
                unit = m.group(1).lower()
                unit = _UNIT_ALIASES.get(unit, unit)
                units_found.add(unit)
        if len(units_found) > 1:
            affected.append(col)
            details.append(f"'{col}': units found {sorted(units_found)}")

    status = "good" if not affected else "warning"
    return HealthCategory(
        key="unit_issues", label="Unit Issues", status=status,
        summary=f"{len(affected)} column(s) mixing units" if affected else "No mixed units detected",
        detail="; ".join(details) if details else "Scanned sampled values for common unit suffixes "
               "(kg/lb, km/mi, cm/in, l/gal, etc.) and flagged columns using more than one. "
               "Best-effort — verify before relying on this.",
        affected_columns=affected,
    )


def compute_health_dashboard(df: pd.DataFrame) -> List[HealthCategory]:
    return [
        _missing_values_health(df),
        _duplicate_health(df),
        _outlier_health(df),
        _data_type_issues_health(df),
        _mixed_categories_health(df),
        _mixed_date_formats_health(df),
        _currency_issues_health(df),
        _unit_issues_health(df),
    ]
