"""
Shared dtype-detection helpers.

Why this exists: raw comparisons like `series.dtype == object` are fragile
across pandas/numpy versions. As of pandas 2.2+ (the versions that actually
ship Python 3.13 wheels), text columns aren't guaranteed to land on plain
numpy `object` dtype anymore — they can come back as pandas' nullable
`StringDtype` ("string"), or a pyarrow-backed string dtype ("string[pyarrow]",
"large_string[pyarrow]") depending on how the file was read or whether
`pd.options.future.infer_string` is set. A raw `== object` check silently
misses those and miscategorizes the column (e.g. a text column getting
treated as "unknown" instead of "categorical"/"text" during profiling).

`_is_stringy()` is the single source of truth for "is this column text-like"
and should be used everywhere instead of comparing dtypes directly.
"""
import pandas as pd


def _is_stringy(obj) -> bool:
    """
    Returns True if `obj` (a pandas Series, or a dtype) represents
    string/text data, regardless of which concrete dtype backs it.

    True for text-like columns under both pandas 2.x (plain numpy `object`
    dtype) and pandas 3.x (default `str` dtype introduced by PDEP-14), plus
    the nullable/pyarrow-backed StringDtype variants in between. A raw
    `series.dtype == object` comparison alone silently misclassifies string
    columns once any of those other dtypes are in play, so every "is this a
    text column" check in the app goes through this helper instead.
    """
    dtype = getattr(obj, "dtype", obj)
    try:
        return bool(pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype))
    except TypeError:
        # obj was a bare Python type / something pandas can't introspect as a dtype
        return dtype is object


def stringy_columns(df: pd.DataFrame) -> list[str]:
    """Column-name equivalent of df.select_dtypes(include='object'), but
    version-robust — use this instead of select_dtypes(include='object')."""
    return [col for col in df.columns if _is_stringy(df[col])]
