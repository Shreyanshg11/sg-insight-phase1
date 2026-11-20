"""
Outlier classification engine.

Principle: an IQR outlier is STATISTICALLY UNUSUAL, not automatically wrong.
Every value outside the 1.5xIQR fences is first and foremost a *statistical
outlier*; this module then looks for concrete, checkable evidence for one of
three labels. Nothing here proves a value is invalid, and the engine never
invents business constraints (valid ranges, whether negatives are allowed...).

Classification vocabulary (the first three are what the engine can assign; the
tier keys are kept stable for API/test compatibility):

  statistical outlier .... every flagged point (outside the fences) - umbrella term
  potential data-quality issue  [tier key: likely_data_entry_error]
        far outside the fence AND one concrete, checkable pattern:
        (a) ~10x/100x (or 1/10, 1/100) of the column median AND NOT
            consistent with a long-tailed distribution (the value is also
            outside the 1.5xIQR fences on a log scale, for all-positive
            columns), AND separated by a gap wider than 1xIQR from the next value
            toward the bulk. Systematic batches of such values (a unit/decimal
            bug repeated by a sensor or an import) qualify; natural tails do not, or
        (b) negative in a column that is >=95% non-negative AND its magnitude is
            within the observed range of the positive values (a sign flip is
            plausible, e.g. -12 among quantities of 1-20). A -900 loss next to
            profits of ~500 is NOT matched.
        It is a hypothesis to verify against the source, not a finding.
  recurring observation  [tier key: legitimate_rare_value]
        the exact value repeats; a one-off keystroke error rarely does. This
        makes it more plausibly genuine, it does not prove it.
  unclassified  [tier key: statistically_unusual]
        flagged on distance alone - the honest default for ambiguous cases.
  confirmed invalid value
        NEVER assigned by this engine. Only the user can confirm a value is
        invalid (human-in-the-loop); outlier actions stay review-only until then.

Skewed / long-tailed data (revenue, order size, durations) routinely produces
legitimate values beyond 1.5xIQR. Columns with |skewness| > 1 are reported as
long-tailed with an explicit limitation, and a log-scale check prevents natural
tail values from being labelled as potential data-quality issues.

Confidence: per point = strength of the evidence for THAT point's label, built
with the shared explainable framework (app.core.confidence) and deliberately
capped (potential issue <= 0.70; recurring/unclassified <= 0.50). Column-level
confidence is the MEAN over all flagged points, never the maximum, so a single
coincidence cannot make a column look more certain than its evidence. These are
heuristic evidence scores, not calibrated probabilities.

All quartile/fence/missing/infinity/small-sample handling is shared with every
other IQR consumer via app.core.iqr.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pydantic import BaseModel
from app.core.confidence import capped_linear
from app.core.iqr import (  # noqa: F401  (re-exported: other modules/tests import these names from here)
    FAR_OUT_MULTIPLIER, IQR_FENCE_MULTIPLIER, MIN_SAMPLE_SIZE, finite_values, iqr_bounds, outlier_mask,
)

# A value within this fraction of an exact 10x/100x/0.1x/0.01x multiple of the
# column median shows a "decimal point slipped" pattern (necessary, not sufficient).
MAGNITUDE_RATIO_TOLERANCE = 0.10
MAGNITUDE_RATIOS = [10, 100]
ISOLATION_GAP_IQR = 1.0          # nearest value toward the bulk must be > this many IQRs away
LONG_TAIL_SKEWNESS = 1.0         # |skewness| above this => long-tailed warning
HARD_BOUND_NONNEG_SHARE = 0.95
SMALL_SAMPLE_NOTE_BELOW = 30

MAX_POINTS_RETURNED_PER_COLUMN = 25  # cap payload size; counts are still exact

TIER_ISSUE = "likely_data_entry_error"      # key kept for compatibility; meaning: potential data-quality issue
TIER_RECURRING = "legitimate_rare_value"    # key kept for compatibility; meaning: recurring observation
TIER_UNCLASSIFIED = "statistically_unusual"

CLASSIFICATION = {
    TIER_ISSUE: ("potential_data_quality_issue", "Potential data-quality issue (evidence-based hypothesis)"),
    TIER_RECURRING: ("recurring_observation", "Recurring value (more plausibly genuine than a one-off typo)"),
    TIER_UNCLASSIFIED: ("unclassified_statistical_outlier", "Unclassified statistical outlier"),
}

METHOD = ("Tukey 1.5×IQR fences on finite numeric values (far-out tier at 3.0×IQR); "
          f"requires at least {MIN_SAMPLE_SIZE} values and a non-zero IQR")


class OutlierPoint(BaseModel):
    row_index: int
    value: float
    tier: str  # "likely_data_entry_error" | "legitimate_rare_value" | "statistically_unusual"
    confidence: float
    evidence: List[str]
    classification: str = "unclassified_statistical_outlier"
    classification_label: str = ""
    # Structured, numeric facts behind the label (so reports/validators never have to parse evidence prose).
    distance_iqr: float = 0.0          # how many IQRs beyond the nearest 1.5xIQR fence
    far_out: bool = False              # beyond the 3.0xIQR far-out fence
    signature: Optional[str] = None    # "decimal_or_unit_slip" | "possible_sign_error" | None
    repeat_count: int = 1              # how many times this exact value occurs in the column


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
    # Additive explainability fields
    n_values: int = 0
    skewness: Optional[float] = None
    long_tailed: bool = False
    column_confidence: float = 0.0   # mean over ALL flagged points (not the max)
    tail_consistent_count: int = 0   # flagged values inside the log-scale fences (consistent with a long tail)
    method: str = METHOD
    limitations: List[str] = []


def _magnitude_signature(value: float, median: float) -> Optional[str]:
    """The ratio test only. Callers must also require a gap to the bulk (isolation) and rule out a
    long-tailed explanation before treating this as evidence."""
    if median == 0 or value == 0:
        return None
    ratio = abs(value / median)
    for m in MAGNITUDE_RATIOS:
        if abs(ratio - m) / m < MAGNITUDE_RATIO_TOLERANCE:
            return f"Value is ~{m}× the column median ({round(value,2)} vs median {round(median,2)})"
        if abs(ratio - 1 / m) / (1 / m) < MAGNITUDE_RATIO_TOLERANCE:
            return f"Value is ~1/{m}th the column median ({round(value,2)} vs median {round(median,2)})"
    return None


def _hard_bound_signature(value: float, series: pd.Series) -> Optional[str]:
    if value >= 0:
        return None
    nonneg = series[series >= 0]
    nonneg_share = len(nonneg) / len(series) if len(series) else 0.0
    if nonneg_share < HARD_BOUND_NONNEG_SHARE or nonneg.empty:
        return None
    if abs(value) > float(nonneg.max()):
        return None  # larger in magnitude than anything positive: a sign flip would not explain it
    return (f"Value is negative but {round(nonneg_share*100,1)}% of this column's values are non-negative, and its "
            "magnitude is within the observed positive range - consistent with a possible sign error "
            "(losses or adjustments can also be legitimate in some columns).")


def _neighbour_gap(sorted_values: np.ndarray, value: float, above: bool) -> float:
    """Distance from `value` to the nearest OTHER observation on the side toward the bulk.

    `sorted_values` must be sorted ascending (the caller passes np.sort of finite values). O(log n) per call via
    binary search; the previous implementation built a boolean mask over the whole column for every outlier,
    which made classification quadratic. Results are identical: "nearest other observation" means the largest
    value strictly below (above=True) / the smallest value strictly above (above=False), else infinity."""
    if above:
        i = int(np.searchsorted(sorted_values, value, side="left"))      # first index with sorted >= value
        return float(value - sorted_values[i - 1]) if i > 0 else float("inf")
    i = int(np.searchsorted(sorted_values, value, side="right"))         # first index with sorted > value
    return float(sorted_values[i] - value) if i < len(sorted_values) else float("inf")

def classify_outliers_for_column(df: pd.DataFrame, col: str) -> Optional[OutlierColumnSummary]:
    series = finite_values(df[col])
    bounds = iqr_bounds(series, IQR_FENCE_MULTIPLIER, MIN_SAMPLE_SIZE)
    if bounds is None:
        return None
    q1, q3, iqr, lower, upper = bounds.q1, bounds.q3, bounds.iqr, bounds.lower, bounds.upper
    far_lower, far_upper = q1 - FAR_OUT_MULTIPLIER * iqr, q3 + FAR_OUT_MULTIPLIER * iqr
    median = float(series.median())
    sorted_values = np.sort(series.to_numpy())

    outliers = series[outlier_mask(series, bounds)]
    if len(outliers) == 0:
        return None

    value_counts = series.value_counts()
    skewness = float(series.skew()) if len(series) >= 3 and np.isfinite(series.skew()) else None
    long_tailed = skewness is not None and abs(skewness) > LONG_TAIL_SKEWNESS
    log_bounds = iqr_bounds(np.log(series), IQR_FENCE_MULTIPLIER, MIN_SAMPLE_SIZE) if bool((series > 0).all()) else None

    points: List[OutlierPoint] = []
    tier_counts = {TIER_ISSUE: 0, TIER_RECURRING: 0, TIER_UNCLASSIFIED: 0}
    tail_consistent = 0

    for idx, value in outliers.items():
        value = float(value)
        is_far_out = value < far_lower or value > far_upper
        above = value > upper
        distance_iqr = abs((value - upper) / iqr) if above else abs((lower - value) / iqr)
        gap = _neighbour_gap(sorted_values, value, above)
        isolated = gap > ISOLATION_GAP_IQR * iqr
        in_log_fences = bool(log_bounds is not None and value > 0 and log_bounds.lower <= np.log(value) <= log_bounds.upper)
        tail_consistent += int(in_log_fences)
        is_repeated = value_counts.get(value, 0) > 1

        evidence = [f"Statistical outlier: {round(distance_iqr, 2)}× the column's IQR beyond the nearest 1.5×IQR fence "
                    f"({'beyond' if is_far_out else 'within'} the 3.0×IQR far-out fence)."]

        magnitude = _magnitude_signature(value, median) if (is_far_out and isolated and not in_log_fences) else None
        bound = _hard_bound_signature(value, series) if is_far_out else None
        if magnitude:
            signature = (f"{magnitude}, separated from the next value toward the bulk by a gap of {round(gap, 2)} "
                         "and not explained by a long-tailed distribution - a pattern a decimal-point or unit slip "
                         "can produce. "
                         "This is a hypothesis to verify against the source, not proof.")
        else:
            signature = bound

        if signature:
            tier = TIER_ISSUE
            evidence.append(signature)
            conf = capped_linear(0.40, 0.03, distance_iqr, cap=0.70, value_label="IQR multiples beyond fence").score
        elif is_repeated:
            tier = TIER_RECURRING
            evidence.append(f"This exact value appears {value_counts.get(value)} times in the column - a one-off "
                            "keystroke error rarely repeats, so it is more plausibly genuine (not proof).")
            conf = capped_linear(0.30, 0.03, distance_iqr, cap=0.50, value_label="IQR multiples beyond fence").score
        else:
            tier = TIER_UNCLASSIFIED
            evidence.append("No repeat occurrence or concrete error pattern found - flagged on statistical distance only.")
            if in_log_fences:
                evidence.append("It lies within the fences on a log scale, which is consistent with a naturally long-tailed distribution.")
            conf = capped_linear(0.30, 0.03, distance_iqr, cap=0.50, value_label="IQR multiples beyond fence").score

        key, label = CLASSIFICATION[tier]
        tier_counts[tier] += 1
        points.append(OutlierPoint(row_index=int(idx), value=value, tier=tier, confidence=conf, evidence=evidence,
                                   classification=key, classification_label=label,
                                   distance_iqr=round(float(distance_iqr), 2), far_out=bool(is_far_out),
                                   signature=("decimal_or_unit_slip" if magnitude else ("possible_sign_error" if bound and tier == TIER_ISSUE else None)),
                                   repeat_count=int(value_counts.get(value, 1))))

    tier_priority = {TIER_ISSUE: 0, TIER_UNCLASSIFIED: 1, TIER_RECURRING: 2}
    points.sort(key=lambda p: (tier_priority[p.tier], -p.confidence))
    column_confidence = round(float(np.mean([p.confidence for p in points])), 2)

    limitations = ["Values outside the fences are statistically unusual relative to this column, not necessarily wrong.",
                   "No business rules or domain limits were assumed (valid ranges, whether negatives are allowed)."]
    if long_tailed:
        limitations.append(f"The column is strongly skewed (skewness {round(skewness, 3)}); IQR fences flag many "
                           "legitimate values in long-tailed business data.")
    if tail_consistent:
        limitations.append(f"{tail_consistent} flagged value(s) lie within the fences on a log scale, which is "
                           "consistent with a naturally long-tailed distribution.")
    if bounds.n < SMALL_SAMPLE_NOTE_BELOW:
        limitations.append(f"Only {bounds.n} finite values - quartile estimates are noisy.")

    return OutlierColumnSummary(
        column=col, q1=round(q1, 4), q3=round(q3, 4), iqr=round(iqr, 4),
        lower_fence=round(lower, 4), upper_fence=round(upper, 4),
        far_out_lower=round(far_lower, 4), far_out_upper=round(far_upper, 4),
        tier_counts=tier_counts, points=points[:MAX_POINTS_RETURNED_PER_COLUMN], total_outliers=len(outliers),
        n_values=bounds.n, skewness=round(skewness, 3) if skewness is not None else None, long_tailed=long_tailed,
        column_confidence=column_confidence, limitations=limitations, tail_consistent_count=int(tail_consistent),
    )


def classify_outliers(df: pd.DataFrame) -> Dict[str, OutlierColumnSummary]:
    """Classifies outliers for every numeric column that has any. Columns with too
    little data, zero IQR, or no fence-breaching values are omitted."""
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        summary = classify_outliers_for_column(df, col)
        if summary is not None:
            result[col] = summary
    return result
