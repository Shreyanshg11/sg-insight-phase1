"""
Shared fixtures for the backend test suite. Fixture dataframes are built to
deliberately exercise the specific conditions each engine looks for (missing
values, IQR outliers, duplicate rows, mixed-case categories, a real date
range) rather than being arbitrary sample data — a test asserting "found 2
outliers" only means something if the fixture provably contains exactly 2.
"""
import os
import tempfile

# session_store.py has a module-level singleton (`store = SessionStore()`)
# that calls persistence.init_db() at IMPORT time — the first test file that
# imports anything from session_store would otherwise create a real
# backend/data/ directory before any per-test monkeypatching of
# SG_INSIGHT_DATA_DIR gets a chance to run. Setting this here, at the very
# top of conftest.py (always imported before any test module), makes sure
# even that one-time import-time side effect lands in a throwaway directory
# instead of the real one.
os.environ.setdefault("SG_INSIGHT_DATA_DIR", tempfile.mkdtemp(prefix="sg_insight_test_"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_df():
    """50 rows with: 4 missing 'cost' values, 2 clear IQR outliers in
    'revenue', 1 exact duplicate row, mixed-case/whitespace 'region' values,
    and a parseable 'signup_date' column — covers most cleaning/EDA/health
    checks in one fixture."""
    np.random.seed(0)
    n = 48
    df = pd.DataFrame({
        "revenue": np.random.normal(1000, 100, n).tolist() + [50000.0, -500.0],  # 2 outliers appended
        "cost": np.random.normal(500, 50, n + 2),
        "region": (["North", " north ", "South", "south", "East"] * 10)[: n + 2],
        "signup_date": pd.date_range("2024-01-01", periods=n + 2, freq="D").astype(str),
    })
    df.loc[[5, 6, 7, 8], "cost"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # exact duplicate of row 0
    return df


@pytest.fixture
def clean_numeric_df():
    """No missing values, no outliers, no duplicates — a baseline for
    asserting engines correctly report 'nothing to flag' rather than always
    finding *something*."""
    np.random.seed(1)
    return pd.DataFrame({
        "x": np.random.normal(0, 1, 100),
        "y": np.random.normal(0, 1, 100),
        "z": np.random.normal(0, 1, 100),
    })


@pytest.fixture
def categorical_df():
    np.random.seed(2)
    return pd.DataFrame({
        "group": np.random.choice(["A", "B", "C"], 60),
        "value": np.random.normal(50, 10, 60),
    })
