"""
Trust & Warnings engine.

A centralized place for the "hold on, be careful with this" signals the
project brief calls out (small sample size, high missingness, class
imbalance, etc). Only warnings that are actually computable from profiling +
cleaning data belong here in Phase 1 — correlation-vs-causation, time
leakage, and sampling bias reminders wait for the EDA/Deep Analysis modules
(Phase 2/3) where there's real correlation/temporal data to reason about,
since a canned warning with nothing behind it isn't trustworthy, it's noise.

Every warning documents the exact rule that produced it in `reason`, in
keeping with the "no arbitrary claims" principle used for confidence scores.
"""
import uuid
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional, List

SMALL_SAMPLE_THRESHOLD = 100
HIGH_MISSING_THRESHOLD = 20.0  # percent, dataset-wide
CLASS_IMBALANCE_THRESHOLD = 0.90  # majority class share


class TrustWarning(BaseModel):
    id: str
    severity: str  # "info" | "warning" | "critical"
    title: str
    description: str
    reason: str
    affected_columns: List[str] = []


def generate_warnings(df: pd.DataFrame, target_candidates: Optional[List[str]] = None) -> List[TrustWarning]:
    warnings: List[TrustWarning] = []
    n_rows = len(df)
    target_candidates = target_candidates or []

    # Small sample size
    if n_rows < SMALL_SAMPLE_THRESHOLD:
        warnings.append(TrustWarning(
            id=str(uuid.uuid4()),
            severity="warning",
            title="Small sample size",
            description=f"This dataset has only {n_rows} rows. Statistics, correlations, and model "
                        "fits computed on this few rows can be unstable and sensitive to individual outliers.",
            reason=f"Row count ({n_rows}) is below the {SMALL_SAMPLE_THRESHOLD}-row threshold used here "
                   "as a rough stability floor.",
        ))

    # High overall missingness
    total_cells = n_rows * len(df.columns) if n_rows and len(df.columns) else 1
    missing_pct = (df.isna().sum().sum() / total_cells) * 100
    if missing_pct > HIGH_MISSING_THRESHOLD:
        worst_cols = df.isna().mean().sort_values(ascending=False)
        worst_cols = worst_cols[worst_cols > 0].head(3).index.tolist()
        warnings.append(TrustWarning(
            id=str(uuid.uuid4()),
            severity="warning",
            title="High missing-value percentage",
            description=f"{round(missing_pct, 1)}% of all cells in the dataset are missing. Imputation "
                        "at this scale can introduce bias, especially if values aren't missing at random.",
            reason=f"Dataset-wide missing rate ({round(missing_pct,1)}%) exceeds the "
                   f"{HIGH_MISSING_THRESHOLD}% threshold.",
            affected_columns=worst_cols,
        ))

    # Class imbalance on target candidates
    for col in target_candidates:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(normalize=True, dropna=True)
        if counts.empty:
            continue
        majority_share = float(counts.iloc[0])
        if majority_share >= CLASS_IMBALANCE_THRESHOLD:
            warnings.append(TrustWarning(
                id=str(uuid.uuid4()),
                severity="warning",
                title=f"Class imbalance in '{col}'",
                description=f"The most common value in '{col}' makes up {round(majority_share*100,1)}% "
                            "of rows. Models trained against this as a target will tend to just predict "
                            "the majority class unless this is addressed.",
                reason=f"Majority class share ({round(majority_share*100,1)}%) exceeds the "
                       f"{int(CLASS_IMBALANCE_THRESHOLD*100)}% imbalance threshold.",
                affected_columns=[col],
            ))

    # Near-constant columns (zero or near-zero variance -> little analytical value, risk in encoding)
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1 and n_rows > 1:
            warnings.append(TrustWarning(
                id=str(uuid.uuid4()),
                severity="info",
                title=f"'{col}' has a single distinct value",
                description=f"Every non-missing value in '{col}' is identical. This column carries no "
                            "discriminating information for analysis or modeling as-is.",
                reason=f"nunique(dropna=True) == {nunique} across {n_rows} rows.",
                affected_columns=[col],
            ))

    # Potential target leakage (very simple heuristic): a numeric non-target column
    # that is near-perfectly correlated with a numeric target candidate.
    numeric_targets = [c for c in target_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_targets:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for target in numeric_targets:
            for col in numeric_cols:
                if col == target:
                    continue
                pair = df[[col, target]].dropna()
                if len(pair) < 10:
                    continue
                # A constant column after the dropna-aligned subset (e.g. a
                # numeric column that's only non-null within one category of
                # a boolean/near-constant target) has zero variance, which
                # makes the correlation formula divide by zero. Skip rather
                # than let pandas/numpy emit a RuntimeWarning for a
                # correlation that's undefined anyway.
                if pair[col].nunique() <= 1 or pair[target].nunique() <= 1:
                    continue
                corr = pair[col].corr(pair[target])
                if corr is not None and abs(corr) > 0.95:
                    warnings.append(TrustWarning(
                        id=str(uuid.uuid4()),
                        severity="critical",
                        title=f"Possible target leakage: '{col}' vs '{target}'",
                        description=f"'{col}' correlates with candidate target '{target}' at "
                                    f"r={round(float(corr), 3)}. If '{col}' wouldn't be known at "
                                    "prediction time, this likely won't generalize.",
                        reason=f"Pearson correlation |r|={round(abs(float(corr)),3)} exceeds the 0.95 "
                               "leakage-suspicion threshold.",
                        affected_columns=[col, target],
                    ))

    return warnings
