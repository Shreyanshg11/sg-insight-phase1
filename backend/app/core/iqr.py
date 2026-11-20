"""
The one shared IQR (Tukey fence) implementation.

Every outlier/IQR calculation in the backend goes through here so quartiles,
fences, missing values, infinities and small samples are handled identically.

Policy (single source of truth):
- Only finite numeric values count. NaN is missing; +/-inf is non-finite and is
  never used for quartiles nor reported as an "outlier" (it is a data-quality
  matter, not a distribution one). Booleans are not treated as numeric.
- Fewer than MIN_SAMPLE_SIZE finite values -> no fences (quartiles of a handful
  of points are not a trustworthy basis for calling anything unusual).
- IQR == 0 (constant / near-constant middle) -> no fences; every value that
  differs would otherwise be flagged.
- Quartiles use pandas' default linear interpolation.

Documented, intentional differences (they are chart/scoring conventions, not
detection policy): the box-plot builder keeps the conventional per-group whiskers
for any group size, and the 3.0x "far-out" multiplier is used only for the
quality-score proxy and the far-out tier of outlier_engine.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

MIN_SAMPLE_SIZE = 8
IQR_FENCE_MULTIPLIER = 1.5
FAR_OUT_MULTIPLIER = 3.0


@dataclass(frozen=True)
class IQRBounds:
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float
    multiplier: float
    n: int  # finite values the quartiles were computed from


def finite_values(series) -> pd.Series:
    """Finite numeric values only (index preserved). Booleans and non-numeric
    text yield nothing / NaN-dropped values; +/-inf is excluded."""
    if series is None or pd.api.types.is_bool_dtype(series):
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values[np.isfinite(values)]


def iqr_bounds(series, multiplier: float = IQR_FENCE_MULTIPLIER,
               min_samples: int = MIN_SAMPLE_SIZE) -> Optional[IQRBounds]:
    values = finite_values(series)
    if len(values) < max(min_samples, 1):
        return None
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return None
    return IQRBounds(q1=q1, q3=q3, iqr=iqr, lower=q1 - multiplier * iqr, upper=q3 + multiplier * iqr,
                     multiplier=multiplier, n=len(values))


def outlier_mask(series, bounds: Optional[IQRBounds]) -> pd.Series:
    """Boolean mask aligned to `series`: True where a FINITE value lies outside the fences."""
    index = series.index if hasattr(series, "index") else None
    if bounds is None:
        return pd.Series(False, index=index)
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return ((values < bounds.lower) | (values > bounds.upper)) & np.isfinite(values)
