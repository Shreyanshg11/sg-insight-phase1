"""
Analytical Risk & Readiness scoring.

Problem this fixes: "quality" used to mean one blended number
(cleaning_engine.compute_quality_score) covering completeness, duplicates,
consistency, and outlier-driven validity. That's a real, useful construct —
it answers "how clean is this data?" — but it's a different question from
"how risky would analysis on this data be?" (small sample, class imbalance,
target leakage — already computed by warnings_engine, just never packaged
as a *score*) and from "is this dataset structurally ready for the analyses
the pipeline offers?" (enough rows for the techniques in analysis_engine,
enough variety of column types, an identifiable target).

This module keeps Data Quality exactly as cleaning_engine already computes
it — that construct wasn't broken — and adds the other two as genuine,
explainable scores of their own, each with a breakdown in the same
additive, cited-formula style as the rest of the app.

Score directions are deliberately NOT all "higher is better": Analytical
Risk reads like a risk meter (0 = no risk found, 100 = severe), matching
how risk is normally communicated and how warnings_engine already frames
severity. Readiness reads like Data Quality does (0 = not ready, 100 =
fully ready). Each score's docstring/field states its own direction so
there's no ambiguity displaying it.
"""
import pandas as pd
from typing import List

from app.models.schemas import ScoreBreakdownItem, AnalyticalRiskScore, ReadinessScore
from app.engines.warnings_engine import generate_warnings
from app.engines.feature_engineering_engine import (
    _usable_numeric_columns, _usable_categorical_columns, _usable_date_columns,
)
from app.engines.analysis_engine import (
    MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR, MIN_ROWS_FOR_CLUSTERING, MIN_ROWS_FOR_PCA,
    MIN_ROWS_FOR_ANOMALY, MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST, MIN_PERIODS_FOR_FORECASTING,
)

# Points added to the risk score per warning severity, capped per-category so
# e.g. ten near-constant "info" columns can't alone drown out everything else.
RISK_POINTS_PER_CRITICAL = 35
RISK_POINTS_PER_WARNING = 15
RISK_POINTS_PER_INFO = 5
RISK_CATEGORY_CAP = {"critical": 70, "warning": 45, "info": 15}

# ScoreBreakdownItem, AnalyticalRiskScore, ReadinessScore are defined in
# app.models.schemas (imported above) — see their docstrings there for the
# score-direction convention (risk: 0=none/100=severe, readiness: 0=not
# ready/100=fully ready).


def compute_analytical_risk(df: pd.DataFrame, target_candidates: List[str] | None = None) -> AnalyticalRiskScore:
    warnings = generate_warnings(df, target_candidates or [])
    by_severity = {"critical": [], "warning": [], "info": []}
    for w in warnings:
        by_severity.setdefault(w.severity, []).append(w)

    breakdown = []
    total = 0.0
    for severity, per_item_points in (
        ("critical", RISK_POINTS_PER_CRITICAL), ("warning", RISK_POINTS_PER_WARNING), ("info", RISK_POINTS_PER_INFO),
    ):
        items = by_severity.get(severity, [])
        if not items:
            continue
        raw_points = len(items) * per_item_points
        capped_points = round(min(RISK_CATEGORY_CAP[severity], raw_points), 1)
        total += capped_points
        breakdown.append(ScoreBreakdownItem(
            label=f"{severity.capitalize()} signals ({len(items)})",
            points=capped_points,
            reason=f"{len(items)} {severity}-severity warning(s) ({', '.join(w.title for w in items)}) "
                   f"× {per_item_points} pts each, capped at {RISK_CATEGORY_CAP[severity]} for this category "
                   f"= {capped_points}.",
        ))

    overall = round(min(100.0, total), 1)
    return AnalyticalRiskScore(
        overall=overall, breakdown=breakdown,
        warning_count=len(by_severity.get("warning", [])), critical_count=len(by_severity.get("critical", [])),
    )


def compute_readiness(df: pd.DataFrame, target_candidates: List[str] | None = None) -> ReadinessScore:
    target_candidates = target_candidates or []
    n_rows = len(df)
    breakdown = []
    earned = 0.0

    # --- Row-count sufficiency (40 pts): fraction of the app's own analysis
    # techniques whose minimum-row gate this dataset actually clears.
    technique_minimums = {
        "regression (per predictor)": MIN_ROWS_FOR_REGRESSION_PER_PREDICTOR,
        "clustering": MIN_ROWS_FOR_CLUSTERING,
        "pca": MIN_ROWS_FOR_PCA,
        "anomaly detection": MIN_ROWS_FOR_ANOMALY,
        "hypothesis test (per group)": MIN_ROWS_PER_GROUP_FOR_HYPOTHESIS_TEST,
        "forecasting (periods)": MIN_PERIODS_FOR_FORECASTING,
    }
    cleared = [name for name, minimum in technique_minimums.items() if n_rows >= minimum]
    row_fraction = len(cleared) / len(technique_minimums)
    row_points = round(row_fraction * 40, 1)
    earned += row_points
    breakdown.append(ScoreBreakdownItem(
        label="Row-count sufficiency", points=row_points,
        reason=f"{n_rows} rows clears the minimum row threshold for {len(cleared)}/{len(technique_minimums)} "
               f"analysis techniques ({', '.join(cleared) if cleared else 'none'}); "
               f"points = {len(cleared)}/{len(technique_minimums)} × 40 = {row_points}.",
    ))

    # --- Column-type breadth (30 pts): having numeric + categorical + date
    # columns available unlocks more of the analysis catalog than any one type alone.
    numeric_present = len(_usable_numeric_columns(df)) > 0
    categorical_present = len(_usable_categorical_columns(df)) > 0
    date_present = len(_usable_date_columns(df)) > 0
    present_types = [t for t, present in
                      [("numeric", numeric_present), ("categorical", categorical_present), ("date", date_present)]
                      if present]
    breadth_points = round(len(present_types) / 3 * 30, 1)
    earned += breadth_points
    breakdown.append(ScoreBreakdownItem(
        label="Column-type breadth", points=breadth_points,
        reason=f"Usable column types present: {', '.join(present_types) if present_types else 'none'} "
               f"({len(present_types)}/3); points = {len(present_types)}/3 × 30 = {breadth_points}.",
    ))

    # --- Target identifiability (30 pts): at least one target candidate,
    # and whether it's usably distributed (not a single-value column).
    if not target_candidates:
        breakdown.append(ScoreBreakdownItem(
            label="Target identifiability", points=0.0,
            reason="No target-like column was identified (see profiling's target_candidates) — "
                   "supervised analyses have nothing to predict.",
        ))
    else:
        usable_targets = [c for c in target_candidates if c in df.columns and df[c].nunique(dropna=True) > 1]
        target_points = round(len(usable_targets) / len(target_candidates) * 30, 1) if target_candidates else 0.0
        earned += target_points
        breakdown.append(ScoreBreakdownItem(
            label="Target identifiability", points=target_points,
            reason=f"{len(usable_targets)}/{len(target_candidates)} target candidate(s) "
                   f"({', '.join(target_candidates)}) have more than one distinct value; "
                   f"points = {len(usable_targets)}/{len(target_candidates)} × 30 = {target_points}.",
        ))

    overall = round(min(100.0, earned), 1)
    return ReadinessScore(overall=overall, breakdown=breakdown)
