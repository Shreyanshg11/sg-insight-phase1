"""
Missingness classification engine.

Problem this fixes: the cleaning engine used to treat "% missing" as the only
signal for a column, so any column missing above ~60% got a blanket
drop_column suggestion — including columns that are missing *by design*
(e.g. "dismissal_type" is only ever populated on rows where a wicket fell).
That's structural missingness, not a data-quality defect, and dropping the
column throws away real signal.

This module distinguishes three patterns, using the same explainable-heuristic
principle as the rest of the app — every classification traces to a stated
rule against real numbers, never a vibe:

- structural:  missingness is (almost) perfectly predicted by another column
               — e.g. 98%+ missing in one group, <5% missing in another.
               Usually means "not applicable here", not "data lost".
- conditional: missingness correlates with another column or with the
               dataset's own values, but not as a clean on/off split.
               Closer to MAR (missing-at-random-given-other-data) — mean/
               median imputation on the whole column is likely biased.
- random:      no predictor column or numeric correlation explains the
               missingness pattern well enough to clear the thresholds below.
               This is an absence-of-evidence heuristic, not a formal MCAR
               test — stated honestly in the evidence text rather than
               implied as certainty.
- insufficient_data: too few missing or too few candidate predictor columns
               to test anything meaningfully.

Kept deliberately dependency-free (no scipy) to match the rest of the
codebase's plain pandas/numpy style.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel
from app.core.confidence import from_group_gap, capped_linear

# A predictor column must have at least this many distinct values to be
# worth grouping by, and no more than this many (past this it's closer to
# an ID column than a category, and any split there is likely spurious).
MIN_PREDICTOR_CARDINALITY = 2
MAX_PREDICTOR_CARDINALITY = 20

# Each group compared for a predictor must have at least this many rows,
# otherwise a "100% missing" group could just be 2 unlucky rows.
MIN_GROUP_SIZE = 5

# Gap between the highest- and lowest-missing-rate groups, on a 0-1 scale.
STRUCTURAL_GAP_THRESHOLD = 0.85
CONDITIONAL_GAP_THRESHOLD = 0.30

# Point-biserial-style correlation (missing-indicator vs numeric column)
# strong enough to call "conditional" even without a clean group split.
CONDITIONAL_CORRELATION_THRESHOLD = 0.40

MIN_MISSING_FOR_CLASSIFICATION = 3  # below this, not enough to test anything


class MissingnessClassification(BaseModel):
    column: str
    classification: str  # "structural" | "conditional" | "random" | "insufficient_data"
    confidence: float    # 0-1, ties to the gap/correlation that produced it
    evidence: List[str]
    predictor_column: Optional[str] = None
    recommendation_hint: str


def _candidate_predictor_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    candidates = []
    for col in df.columns:
        if col == target_col:
            continue
        nunique = df[col].nunique(dropna=True)
        if MIN_PREDICTOR_CARDINALITY <= nunique <= MAX_PREDICTOR_CARDINALITY:
            candidates.append(col)
    return candidates


def _best_group_split(df: pd.DataFrame, target_col: str, predictor_col: str):
    """Groups rows by predictor_col and returns the gap between the highest-
    and lowest missing-rate groups among groups with enough rows, plus the
    two groups' rates and sizes for the evidence text. Returns None if no
    qualifying pair of groups exists."""
    is_missing = df[target_col].isna()
    grouped = is_missing.groupby(df[predictor_col], dropna=True)
    rates = grouped.mean()
    sizes = grouped.size()
    qualifying = rates[sizes >= MIN_GROUP_SIZE]
    if len(qualifying) < 2:
        return None
    max_group, min_group = qualifying.idxmax(), qualifying.idxmin()
    gap = float(qualifying[max_group] - qualifying[min_group])
    return {
        "gap": gap,
        "high_group": max_group, "high_rate": float(qualifying[max_group]), "high_n": int(sizes[max_group]),
        "low_group": min_group, "low_rate": float(qualifying[min_group]), "low_n": int(sizes[min_group]),
    }


def _numeric_correlation(df: pd.DataFrame, target_col: str, predictor_col: str) -> Optional[float]:
    """Point-biserial correlation between the missing-indicator (0/1) for
    target_col and a numeric predictor column's values, computed with plain
    pandas .corr() (Pearson on a binary variable == point-biserial)."""
    if not pd.api.types.is_numeric_dtype(df[predictor_col]):
        return None
    is_missing = df[target_col].isna().astype(int)
    mask = df[predictor_col].notna()
    if mask.sum() < MIN_GROUP_SIZE * 2:
        return None
    corr = is_missing[mask].corr(df[predictor_col][mask])
    if pd.isna(corr):
        return None
    return float(corr)


def classify_column_missingness(df: pd.DataFrame, col: str) -> MissingnessClassification:
    n_rows = len(df)
    missing = int(df[col].isna().sum())

    if missing == 0:
        return MissingnessClassification(
            column=col, classification="insufficient_data", confidence=1.0,
            evidence=["Column has no missing values."],
            recommendation_hint="Not applicable — nothing missing.",
        )

    if missing < MIN_MISSING_FOR_CLASSIFICATION or n_rows - missing < MIN_GROUP_SIZE:
        return MissingnessClassification(
            column=col, classification="insufficient_data", confidence=0.4,
            evidence=[f"Only {missing} missing value(s) out of {n_rows} rows — too few to test a pattern "
                      f"against other columns reliably (need at least {MIN_MISSING_FOR_CLASSIFICATION})."],
            recommendation_hint="Too little data to classify — fall back to the standard missing-value heuristics.",
        )

    best_split = None
    best_predictor = None
    for predictor in _candidate_predictor_columns(df, col):
        split = _best_group_split(df, col, predictor)
        if split is None:
            continue
        if best_split is None or split["gap"] > best_split["gap"]:
            best_split, best_predictor = split, predictor

    if best_split is not None and best_split["gap"] >= STRUCTURAL_GAP_THRESHOLD:
        conf_result = from_group_gap(best_split["gap"], base=0.5, cap=0.95, gap_label="group-rate gap")
        confidence = conf_result.score
        evidence = [
            f"When '{best_predictor}' = {best_split['high_group']!r} (n={best_split['high_n']}), "
            f"'{col}' is missing {round(best_split['high_rate']*100,1)}% of the time.",
            f"When '{best_predictor}' = {best_split['low_group']!r} (n={best_split['low_n']}), "
            f"'{col}' is missing only {round(best_split['low_rate']*100,1)}% of the time.",
            f"That {round(best_split['gap']*100,1)}-point gap (≥{int(STRUCTURAL_GAP_THRESHOLD*100)} pts triggers "
            "'structural') suggests the missingness is explained by '{}', not by data loss.".format(best_predictor),
            f"Confidence formula: {conf_result.formula}.",
        ]
        return MissingnessClassification(
            column=col, classification="structural", confidence=confidence, evidence=evidence,
            predictor_column=best_predictor,
            recommendation_hint=(
                "Likely missing by design (e.g. 'not applicable' for this row's category) rather than a data "
                "quality problem. Dropping or blanket-imputing this column would discard that signal — consider "
                "leaving it as-is, or encoding missingness itself as a feature."
            ),
        )

    # Not a clean structural split — check for a softer group correlation,
    # and separately check numeric predictors for a linear relationship.
    conditional_evidence = []
    conditional_predictor = None
    conditional_confidence = 0.0

    if best_split is not None and best_split["gap"] >= CONDITIONAL_GAP_THRESHOLD:
        conditional_predictor = best_predictor
        conf_result = from_group_gap(best_split["gap"], base=0.35, cap=0.75, gap_label="group-rate gap")
        conditional_confidence = conf_result.score
        conditional_evidence = [
            f"Missing rate for '{col}' varies with '{best_predictor}': "
            f"{round(best_split['high_rate']*100,1)}% missing when '{best_predictor}' = {best_split['high_group']!r} "
            f"vs {round(best_split['low_rate']*100,1)}% when = {best_split['low_group']!r} "
            f"(n={best_split['high_n']} and n={best_split['low_n']}).",
            f"That {round(best_split['gap']*100,1)}-point gap is meaningful but not clean enough (< "
            f"{int(STRUCTURAL_GAP_THRESHOLD*100)} pts) to call fully structural.",
            f"Confidence formula: {conf_result.formula}.",
        ]
    else:
        best_corr, best_corr_col = 0.0, None
        for predictor in df.columns:
            if predictor == col:
                continue
            corr = _numeric_correlation(df, col, predictor)
            if corr is not None and abs(corr) > abs(best_corr):
                best_corr, best_corr_col = corr, predictor
        if best_corr_col is not None and abs(best_corr) >= CONDITIONAL_CORRELATION_THRESHOLD:
            conditional_predictor = best_corr_col
            conf_result = capped_linear(0.3, 1.0, abs(best_corr), cap=0.7, value_label="|correlation|")
            conditional_confidence = conf_result.score
            conditional_evidence = [
                f"Whether '{col}' is missing correlates with '{best_corr_col}' values "
                f"(point-biserial r={round(best_corr,2)}, threshold ±{CONDITIONAL_CORRELATION_THRESHOLD}).",
                "A correlation this size means the missing rows aren't a random subset of the dataset — "
                "imputing with a single dataset-wide mean/median would bias toward whichever end of "
                f"'{best_corr_col}' the missing rows cluster on.",
                f"Confidence formula: {conf_result.formula}.",
            ]

    if conditional_predictor is not None:
        return MissingnessClassification(
            column=col, classification="conditional", confidence=conditional_confidence,
            evidence=conditional_evidence, predictor_column=conditional_predictor,
            recommendation_hint=(
                f"Missingness correlates with '{conditional_predictor}' rather than looking random. "
                "Group-aware imputation (e.g. impute per-group instead of one dataset-wide value) or a "
                "missing-indicator flag will preserve more signal than a single blanket fill or drop."
            ),
        )

    gap_for_confidence = best_split["gap"] if best_split else 0.15
    conf_result = capped_linear(1.0, -1.0, gap_for_confidence, cap=1.0, floor=0.35,
                                 value_label="best candidate gap/correlation found")
    confidence = conf_result.score
    return MissingnessClassification(
        column=col, classification="random", confidence=confidence,
        evidence=[
            f"Checked {len(_candidate_predictor_columns(df, col))} candidate categorical/low-cardinality "
            f"predictor column(s) and all numeric columns for a relationship with '{col}' missingness; "
            f"none showed a gap ≥{int(CONDITIONAL_GAP_THRESHOLD*100)} pts or correlation "
            f"≥{CONDITIONAL_CORRELATION_THRESHOLD}.",
            "This is an absence-of-evidence heuristic, not a statistical proof of missing-completely-at-random — "
            "treat it as 'no obvious pattern found' rather than certainty.",
            f"Confidence formula: {conf_result.formula}.",
        ],
        recommendation_hint="No predictor explains the missingness — standard imputation heuristics "
                             "(mean/median/mode, or drop above a fixed threshold) are reasonable defaults here.",
    )


def classify_missingness(df: pd.DataFrame) -> Dict[str, MissingnessClassification]:
    """Classifies every column with at least one missing value. Columns with
    zero missing values are omitted from the result (nothing to classify)."""
    result = {}
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        result[col] = classify_column_missingness(df, col)
    return result
