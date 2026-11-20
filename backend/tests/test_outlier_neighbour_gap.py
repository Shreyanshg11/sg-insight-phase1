"""_neighbour_gap(): the O(log n) binary-search implementation must return EXACTLY what the old O(n) mask version did.

No wall-clock assertions: the complexity guard below is structural (a column that refuses whole-column comparisons)."""
import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from app.engines import outlier_engine as oe
from app.engines.outlier_engine import _neighbour_gap, classify_outliers_for_column


def reference_gap(sorted_values, value, above):
    """The previous implementation, kept verbatim as the oracle."""
    if above:
        lower = sorted_values[sorted_values < value]
        return float(value - lower[-1]) if len(lower) else float("inf")
    higher = sorted_values[sorted_values > value]
    return float(higher[0] - value) if len(higher) else float("inf")


def arrays():
    rng = np.random.default_rng(11)
    return {
        "continuous": rng.lognormal(4.6, 0.9, 1500),
        "ties_rounded": np.round(rng.lognormal(4.6, 0.9, 3000)),
        "few_distinct": rng.integers(0, 6, 400).astype(float),
        "all_equal": np.full(50, 7.0),
        "pair": np.array([3.0, 9.0]),
        "single": np.array([4.0]),
        "negative_and_zero": np.concatenate([rng.normal(0, 5, 300), np.zeros(25), -np.abs(rng.normal(50, 20, 10))]),
        "integers_as_float": np.round(rng.normal(1000, 300, 800)),
        "huge_magnitudes": np.array([-1.7e308, -1e300, 0.0, 1e300, 1.7e308]),
        "tiny_gaps": np.array([1.0, np.nextafter(1.0, 2.0), np.nextafter(np.nextafter(1.0, 2.0), 2.0), 2.0]),
    }


def probe_values(arr):
    """Members (incl. both ends), values between members, and values outside the array on both sides."""
    members = np.unique(arr)
    step = max(1, len(members) // 300)
    chosen = list(members[::step]) + [members[0], members[-1]]
    between = [(a + b) / 2 for a, b in zip(members[:-1:step], members[1::step])]
    return chosen + between + [members[0] - 1.0, members[-1] + 1.0]


class TestIdenticalResults:
    @pytest.mark.parametrize("name", list(arrays()))
    def test_matches_the_old_implementation_for_every_probe(self, name):
        s = np.sort(arrays()[name])
        probes = probe_values(s)
        assert len(probes) >= 3
        with np.errstate(over="ignore"):                                   # huge_magnitudes overflows to inf in BOTH versions
            for v in probes:
                for above in (True, False):
                    assert _neighbour_gap(s, v, above) == reference_gap(s, v, above), (name, v, above)
                    assert _neighbour_gap(s, np.float64(v), above) == reference_gap(s, np.float64(v), above)

    def test_known_values_and_infinity_at_the_ends(self):
        s = np.array([1.0, 2.0, 2.0, 5.0, 9.0, 20.0])
        assert _neighbour_gap(s, 20.0, True) == 11.0 and _neighbour_gap(s, 1.0, False) == 1.0
        assert _neighbour_gap(s, 2.0, True) == 1.0 and _neighbour_gap(s, 2.0, False) == 3.0     # ties: nearest OTHER value
        assert _neighbour_gap(s, 1.0, True) == float("inf") and _neighbour_gap(s, 20.0, False) == float("inf")
        assert _neighbour_gap(np.full(5, 3.0), 3.0, True) == float("inf") == _neighbour_gap(np.full(5, 3.0), 3.0, False)
        assert _neighbour_gap(np.array([4.0]), 4.0, True) == float("inf")

    def test_signature_and_return_type_are_unchanged(self):
        assert list(inspect.signature(_neighbour_gap).parameters) == ["sorted_values", "value", "above"]
        for above in (True, False):
            assert type(_neighbour_gap(np.array([1.0, 2.0, 3.0]), 2.0, above)) is float
            assert type(_neighbour_gap(np.array([1.0]), 1.0, above)) is float                # the inf case too


def frames():
    rng = np.random.default_rng(5)
    lognormal = pd.DataFrame({"v": np.round(rng.lognormal(4.6, 0.9, 20000))})
    lognormal.loc[[10, 20, 30], "v"] = [46000.0, 69000.0, 92000.0]                               # decimal-slip style outliers
    return {
        "lognormal_with_slips": lognormal,
        "heavy_ties": pd.DataFrame({"v": np.concatenate([rng.poisson(3, 5000).astype(float), [400.0, 400.0, 900.0, -50.0]])}),
        "low_spread_with_ties": pd.DataFrame({"v": np.concatenate([np.round(rng.normal(10, 0.5, 300), 1), [10.5, 80.0, 81.0, -60.0, 79.5]])}),
        "small_n": pd.DataFrame({"v": rng.normal(100, 10, 40).tolist() + [500.0]}),
        "non_default_index": pd.DataFrame({"v": np.round(rng.lognormal(4, 1, 2000))}, index=np.arange(2000) + 5000),
    }


def dump(summary):
    return summary.model_dump() if hasattr(summary, "model_dump") else dataclasses.asdict(summary)


class TestClassificationIsUnchanged:
    @pytest.mark.parametrize("name", list(frames()))
    def test_whole_column_summary_equals_the_one_built_with_the_old_implementation(self, name, monkeypatch):
        df = frames()[name]
        new = classify_outliers_for_column(df, "v")
        assert new is not None and new.total_outliers > 0, "fixture must contain outliers"
        monkeypatch.setattr(oe, "_neighbour_gap", reference_gap)
        old = classify_outliers_for_column(df, "v")
        assert dump(new) == dump(old)                                                         # every point, field, tier and count
        assert new.total_outliers == old.total_outliers


class TestComplexityGuard:
    def test_never_compares_against_the_whole_column(self):
        """Machine-independent: the column refuses < <= > >=, so only a binary search can answer."""
        class NoMask(np.ndarray):
            def _refuse(self, other):
                raise AssertionError("whole-column comparison: quadratic implementation")
            __lt__ = __le__ = __gt__ = __ge__ = _refuse

        s = np.sort(np.random.default_rng(2).lognormal(4.6, 0.9, 5000)).view(NoMask)
        for v in (s[0], s[2500], s[-1]):
            for above in (True, False):
                _neighbour_gap(s, v, above)
