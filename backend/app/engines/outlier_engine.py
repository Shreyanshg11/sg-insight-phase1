"""
Outlier tiering engine.

Problem this fixes: outlier detection used to be single-tier — a value was
either inside the 1.5×IQR fence or "an outlier", with no distinction between
a value that's probably a typo (an age of 999, a price that's 100× every
other price in the column — a decimal point slipped) and a value that's
just a genuinely rare but real observation (a whale customer's order size,
a record-breaking score). Both got the exact same "flag_for_review" treatment,
so a user reviewing 40 flagged values had no way to triage which ones to
look at first.

This module classifies each outlier POINT (not just the column) into one of
three tiers, each backed by a stated, checkable rule:

- likely_data_entry_error: far outside the fence (beyond a wider 3.0×IQR
  "far out" fence — the standard Tukey convention for a second, stricter
  tier) AND shows a concrete error signature: the value sits at ~10x/100x
  the column's median (classic decimal-point/unit slip), or it breaks an
  obvious hard bound the rest of the column respects (e.g. negative in an
  otherwise-nonnegative column).
- legitimate_rare_value: outside the standard fence but not far-out, AND
  the same value (or a value very close to it) appears more than once in
  the column — a one-off keystroke error doesn't usually repeat.
- statistically_unusual: outside the fence but doesn't clear either bar
  above — the honest "flagged, but we can't tell you more" default. This
  is deliberately the largest bucket in ambiguous cases; the whole point is
  not to overclaim certainty a plain distance-from-fence check can't support.

Confidence per point is computed via the shared explainable-confidence
framework (app.core.confidence) so it's directly comparable to every other
engine's confidence numbers.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel
from app.core.confidence import capped_linear

IQR_FENCE_MULTIPLIER = 1.5      # standard Tukey fence — same as the rest of the app
FAR_OUT_MULTIPLIER = 3.0        # standard Tukey "far out" fence — a stricter second tier
MIN_SAMPLE_SIZE = 8             # same floor used elsewhere in the app before trusting an IQR at all

# A value within this fraction of an exact 10x/100x/0.1x/0.01x multiple of the
# column median is treated as showing a "decimal point slipped" signature.
MAGNITUDE_RATIO_TOLERANCE = 0.15
MAGNITUDE_RATIOS = [10, 100]

MAX_POINTS_RETURNED_PER_COLUMN = 25  # cap payload size; counts are still exact


class OutlierPoint(BaseModel):
    row_index: int
    value: float
    tier: str  # "likely_data_entry_error" | "legitimate_rare_value" | "statistically_unusual"
    confidence: float
    evidence: List[str]


class OutlierColumnSummary(BaseModel):
    column: str
    q1: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
    far_out_lower: float
    far_out_upper: float
    tier_counts: Dict[str, int]
    points: List[OutlierPoint]  # capped at MAX_POINTS_RETURNED_PER_COLUMN, worst tier first
    total_outliers: int


def _magnitude_signature(value: float, median: float) -> Optional[str]:
    if median == 0 or value == 0:
        return None
    ratio = abs(value / median)
    for m in MAGNITUDE_RATIOS:
        if abs(ratio - m) / m < MAGNITUDE_RATIO_TOLERANCE:
            return f"Value is ~{m}× the column median ({round(value,2)} vs median {round(median,2)}) — " \
                   "consistent with a decimal-point or unit-entry slip."
        if abs(ratio - 1 / m) / (1 / m) < MAGNITUDE_RATIO_TOLERANCE:
            return f"Value is ~1/{m}th the column median ({round(value,2)} vs median {round(median,2)}) — " \
                   "consistent with a decimal-point or unit-entry slip."
    return None


def _hard_bound_signature(value: float, series: pd.Series) -> Optional[str]:
    nonneg_ratio = (series >= 0).mean()
    if nonneg_ratio >= 0.95 and value < 0:
        return f"Value is negative but {round(nonneg_ratio*100,1)}% of this column's other values are " \
               "non-negative — a sign flip is a common entry error."
    return None


def classify_outliers_for_column(df: pd.DataFrame, col: str) -> Optional[OutlierColumnSummary]:
    series_full = df[col]
    series = series_full.dropna()
    if len(series) < MIN_SAMPLE_SIZE:
        return None
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower, upper = q1 - IQR_FENCE_MULTIPLIER * iqr, q3 + IQR_FENCE_MULTIPLIER * iqr
    far_lower, far_upper = q1 - FAR_OUT_MULTIPLIER * iqr, q3 + FAR_OUT_MULTIPLIER * iqr
    median = series.median()

    outlier_mask = (series < lower) | (series > upper)
    outliers = series[outlier_mask]
    if len(outliers) == 0:
        return None

    value_counts = series.value_counts()

    points: List[OutlierPoint] = []
    tier_counts = {"likely_data_entry_error": 0, "legitimate_rare_value": 0, "statistically_unusual": 0}

    for idx, value in outliers.items():
        is_far_out = value < far_lower or value > far_upper
        distance_iqr = abs((lower - value) / iqr) if value < lower else abs((value - upper) / iqr)

        magnitude_evidence = _magnitude_signature(float(value), float(median))
        bound_evidence = _hard_bound_signature(float(value), series)
        error_signature = magnitude_evidence or bound_evidence
        is_repeated = value_counts.get(value, 0) > 1

        evidence = [f"Distance from nearest fence: {round(distance_iqr, 2)}× the column's IQR "
                    f"({'beyond the 3.0×IQR far-out fence' if is_far_out else 'within the far-out fence'})."]

        if is_far_out and error_signature:
            tier = "likely_data_entry_error"
            evidence.append(error_signature)
            conf = capped_linear(0.55, 0.1, distance_iqr, cap=0.9, value_label="IQR multiples beyond fence").score
        elif is_repeated and not error_signature:
            # Repetition is checked regardless of far/not-far: an identical value
            # occurring more than once is unlikely to be a one-off keystroke,
            # whether it sits just past the near fence or well past the far one —
            # unless it ALSO carries an error signature (e.g. a batch import bug
            # that repeated the same bad decimal-shift multiple times), in which
            # case the error tier above still wins.
            tier = "legitimate_rare_value"
            evidence.append(f"This exact value appears {value_counts.get(value)} times in the column — "
                             "unlikely to be a one-off keystroke error.")
            conf = capped_linear(0.4, 0.05, distance_iqr, cap=0.6, value_label="IQR multiples beyond fence").score
        else:
            tier = "statistically_unusual"
            evidence.append("No repeat occurrence or clear error signature found — flagged on distance alone.")
            conf = capped_linear(0.45, 0.08, distance_iqr, cap=0.75, value_label="IQR multiples beyond fence").score

        tier_counts[tier] += 1
        points.append(OutlierPoint(
            row_index=int(idx), value=float(value), tier=tier, confidence=conf, evidence=evidence,
        ))

    tier_priority = {"likely_data_entry_error": 0, "statistically_unusual": 1, "legitimate_rare_value": 2}
    points.sort(key=lambda p: (tier_priority[p.tier], -p.confidence))

    return OutlierColumnSummary(
        column=col, q1=round(float(q1), 4), q3=round(float(q3), 4), iqr=round(float(iqr), 4),
        lower_fence=round(float(lower), 4), upper_fence=round(float(upper), 4),
        far_out_lower=round(float(far_lower), 4), far_out_upper=round(float(far_upper), 4),
        tier_counts=tier_counts, points=points[:MAX_POINTS_RETURNED_PER_COLUMN],
        total_outliers=len(outliers),
    )


def classify_outliers(df: pd.DataFrame) -> Dict[str, OutlierColumnSummary]:
    """Classifies outliers for every numeric column that has any. Columns
    with too little data, zero IQR, or no fence-breaching values are omitted."""
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        summary = classify_outliers_for_column(df, col)
        if summary is not None:
            result[col] = summary
    return result
