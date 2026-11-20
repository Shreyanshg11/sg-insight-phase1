"""Phase 1 — fuzzy category matching safety (F1) and fuzzy_merge execution (F2).

Principle under test: similar spelling is NOT evidence of the same meaning.
Fuzzy results are review-only suggestions; merging happens only through an
explicit apply_action(..., "fuzzy_merge", ...) call.
"""
import itertools
import time

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.session_store import store
from app.engines.cleaning_engine import (
    FUZZY_MAX_DISTINCT_VALUES,
    _damerau_levenshtein,
    _is_spelling_variant,
    apply_action,
    find_fuzzy_category_groups,
    generate_recommendations,
)
from app.routers.cleaning import router as cleaning_router


def _grouped(values, counts=None):
    return [sorted(g["values"]) for g in find_fuzzy_category_groups(values, counts)]


# ---------------------------------------------------------------------------
# Edit distance primitive
# ---------------------------------------------------------------------------
class TestDamerauLevenshtein:
    @pytest.mark.parametrize("a,b,expected", [
        ("color", "color", 0),
        ("colour", "color", 1),        # deletion
        ("color", "colxr", 1),         # substitution
        ("receive", "recieve", 1),     # adjacent transposition counts as ONE edit
        ("australia", "austria", 2),
        ("male", "female", 2),
        ("", "abc", 3),
    ])
    def test_distances(self, a, b, expected):
        assert _damerau_levenshtein(a, b) == expected
        assert _damerau_levenshtein(b, a) == expected  # symmetric


# ---------------------------------------------------------------------------
# F1 — values that must NEVER be suggested as the same category
# ---------------------------------------------------------------------------
class TestFalsePositivesAreRejected:
    @pytest.mark.parametrize("a,b", [
        ("Australia", "Austria"),
        ("Male", "Female"),
        ("East", "West"),
        ("Active", "Inactive"),
        ("Agree", "Disagree"),
        ("Satisfied", "Dissatisfied"),
        ("Paid", "Unpaid"),
        ("Married", "Unmarried"),
        ("Niger", "Nigeria"),
        ("India", "Indiana"),
        ("Oman", "Romania"),
        ("Low", "Lower"),
        ("Region 1", "Region 2"),
        ("Tier 1", "Tier 2"),
        ("Plan A1", "Plan A2"),
        ("Product A", "Product B"),
        ("Grade A", "Grade B"),
        ("Red Box", "Red Fox"),         # differing word is a short real word
        ("New York", "Newyork"),        # token count differs
        ("Gray", "Grey"),               # below the 5-character floor
        ("Jon", "John"),
        ("US", "UK"),
    ])
    def test_pair_not_grouped(self, a, b):
        assert find_fuzzy_category_groups([a, b]) == []
        assert not _is_spelling_variant(a.casefold(), b.casefold())

    def test_multi_value_column_with_distinct_categories(self):
        values = ["Region 1", "Region 2", "Region 3", "Male", "Female", "Australia", "Austria"]
        assert find_fuzzy_category_groups(values) == []


# ---------------------------------------------------------------------------
# F1 — genuine variants and verified aliases are still found
# ---------------------------------------------------------------------------
class TestGenuineVariantsAndAliases:
    @pytest.mark.parametrize("a,b", [
        ("Colour", "Color"),
        ("Organisation", "Organization"),
        ("Recieve", "Receive"),            # transposition
        ("Version 1", "Versoin 1"),        # digits identical, typo in the word
        ("St. Louis", "St Louis"),
    ])
    def test_spelling_variant_grouped(self, a, b):
        groups = find_fuzzy_category_groups([a, b])
        assert len(groups) == 1
        g = groups[0]
        assert g["match_type"] == "spelling_variant"
        assert g["review_required"] is True
        assert sorted(g["values"]) == sorted([a, b])
        assert 0 < g["similarity"] < 1

    def test_usa_aliases_preserved(self):
        groups = find_fuzzy_category_groups(["USA", "US", "United States", "U.S.A.", "America"])
        assert len(groups) == 1
        assert groups[0]["match_type"] == "alias" and groups[0]["similarity"] == 1.0
        assert sorted(groups[0]["values"]) == sorted(["USA", "US", "United States", "U.S.A.", "America"])
        assert groups[0]["suggested_canonical"] == "United States"  # existing "fullest spelling" behaviour

    def test_us_vs_usa(self):
        assert _grouped(["USA", "US"]) == [["US", "USA"]]

    def test_uk_vs_dotted_uk(self):
        assert _grouped(["UK", "U.K."]) == [["U.K.", "UK"]]
        assert _grouped(["UK", "U.K.", "United Kingdom", "Britain"]) == [["Britain", "U.K.", "UK", "United Kingdom"]]

    def test_uae_aliases(self):
        assert _grouped(["UAE", "United Arab Emirates"]) == [["UAE", "United Arab Emirates"]]

    def test_different_alias_groups_never_merge(self):
        assert find_fuzzy_category_groups(["USA", "UK", "UAE"]) == []

    def test_alias_matching_ignores_case_and_whitespace(self):
        assert _grouped(["  usa ", "United STATES"]) == [["United STATES", "usa"]]


# ---------------------------------------------------------------------------
# F1 — no transitive chaining
# ---------------------------------------------------------------------------
class TestNoTransitiveGrouping:
    # a~b and b~c (one edit each) but a!~c (two edits)
    A, B, C = "abcdef", "abcdeg", "abcdgg"

    def test_precondition_chain_is_real(self):
        assert _is_spelling_variant(self.A, self.B)
        assert _is_spelling_variant(self.B, self.C)
        assert not _is_spelling_variant(self.A, self.C)

    def test_chain_is_not_collapsed(self):
        groups = find_fuzzy_category_groups([self.A, self.B, self.C])
        for g in groups:
            assert not ({self.A, self.C} <= set(g["values"]))
        assert len(groups) == 1 and len(groups[0]["values"]) == 2

    def test_every_pair_in_every_group_independently_qualifies(self):
        chain = ["abcdef", "abcdeg", "abcdgg", "abcdgh", "abcdhh", "abcdhi", "abcdii"]
        for g in find_fuzzy_category_groups(chain):
            for x, y in itertools.combinations(g["values"], 2):
                assert _is_spelling_variant(x.casefold(), y.casefold()), (x, y)

    def test_long_chain_of_digit_labels_never_collapses(self):
        chain = ["Plan A1", "Plan A2", "Plan B2", "Plan B3", "Plan C3"]
        assert find_fuzzy_category_groups(chain) == []

    def test_frequency_decides_which_end_of_the_chain_joins(self):
        counts = {self.A: 5, self.B: 50, self.C: 7}
        (g,) = find_fuzzy_category_groups([self.A, self.B, self.C], counts)
        assert g["suggested_canonical"] == self.B
        assert set(g["values"]) == {self.B, self.C}  # C outranks A; A cannot join because A!~C

    def test_result_does_not_depend_on_input_order(self):
        values = ["abcdef", "abcdeg", "abcdgg", "Colour", "Color", "USA", "US", "Austria", "Australia"]
        counts = {"abcdef": 3, "abcdeg": 9, "abcdgg": 4, "Colour": 12, "Color": 2}
        expected = find_fuzzy_category_groups(values, counts)
        for perm in itertools.islice(itertools.permutations(values), 200):
            assert find_fuzzy_category_groups(list(perm), counts) == expected


# ---------------------------------------------------------------------------
# F1 — canonical is data-grounded, not arbitrary
# ---------------------------------------------------------------------------
class TestCanonicalSelection:
    def test_most_frequent_spelling_wins(self):
        (g,) = find_fuzzy_category_groups(["Colour", "Color"], {"Colour": 3, "Color": 50})
        assert g["suggested_canonical"] == "Color"
        (g,) = find_fuzzy_category_groups(["Colour", "Color"], {"Colour": 50, "Color": 3})
        assert g["suggested_canonical"] == "Colour"

    def test_longest_string_is_no_longer_used_for_spelling_variants(self):
        (g,) = find_fuzzy_category_groups(["Organisation", "Organization"], {"Organisation": 1, "Organization": 9})
        assert g["suggested_canonical"] == "Organization"

    def test_case_variants_pool_their_counts(self):
        counts = {"Colour": 30, "colour": 25, "Color": 40}
        (g,) = find_fuzzy_category_groups(["Colour", "colour", "Color"], counts)
        assert g["suggested_canonical"] == "Colour"      # 55 rows vs 40; most frequent raw spelling of that value
        assert sorted(g["values"]) == ["Color", "Colour"]  # case variants collapse into one entry

    def test_tie_is_deterministic(self):
        a = find_fuzzy_category_groups(["Colour", "Color"], {"Colour": 5, "Color": 5})
        b = find_fuzzy_category_groups(["Color", "Colour"], {"Colour": 5, "Color": 5})
        assert a == b


# ---------------------------------------------------------------------------
# F1 — empty / duplicate / None / NaN / blank / cardinality
# ---------------------------------------------------------------------------
class TestInputHandling:
    @pytest.mark.parametrize("values", [[], ["Only"], [None], [None, None], [""], ["   "], [float("nan")],
                                        [None, "", "  ", np.nan, pd.NA]])
    def test_degenerate_inputs_return_empty(self, values):
        assert find_fuzzy_category_groups(values) == []

    def test_none_values_argument(self):
        assert find_fuzzy_category_groups(None) == []

    def test_duplicates_are_not_a_group(self):
        assert find_fuzzy_category_groups(["USA", "USA"]) == []
        assert find_fuzzy_category_groups(["Colour", "Colour"]) == []

    def test_duplicates_do_not_inflate_a_real_group(self):
        (g,) = find_fuzzy_category_groups(["Colour", "Colour", "Color", "Color", "Color"])
        assert sorted(g["values"]) == ["Color", "Colour"]

    def test_blank_and_missing_are_ignored_alongside_real_values(self):
        (g,) = find_fuzzy_category_groups([None, "", "  ", np.nan, "Colour", "Color", pd.NA])
        assert sorted(g["values"]) == ["Color", "Colour"]

    def test_whitespace_padded_values_are_stripped(self):
        (g,) = find_fuzzy_category_groups(["  Colour ", "Color  "])
        assert sorted(g["values"]) == ["Color", "Colour"]

    def test_input_list_is_not_mutated(self):
        values = ["Colour", "Color", None, "USA", "US"]
        snapshot = list(values)
        find_fuzzy_category_groups(values)
        assert values == snapshot

    def test_non_string_values_do_not_crash(self):
        assert isinstance(find_fuzzy_category_groups([12345, 12346, "Colour", "Color"]), list)

    def test_overlong_values_are_free_text_not_categories(self):
        a, b = "x" * 70, "x" * 69 + "y"
        assert find_fuzzy_category_groups([a, b]) == []


class TestHighCardinality:
    def test_above_limit_is_skipped(self):
        values = [f"label{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}x" for i in range(FUZZY_MAX_DISTINCT_VALUES + 1)]
        assert len(set(values)) == FUZZY_MAX_DISTINCT_VALUES + 1
        assert find_fuzzy_category_groups(values + ["Colour", "Color"]) == []

    def test_at_limit_is_still_analysed_and_fast(self):
        filler = [f"term{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}qz" for i in range(FUZZY_MAX_DISTINCT_VALUES - 2)]
        start = time.perf_counter()
        groups = find_fuzzy_category_groups(filler + ["Colour", "Color"])
        assert time.perf_counter() - start < 3
        assert any(sorted(g["values"]) == ["Color", "Colour"] for g in groups)

    def test_150_near_identical_numbered_labels_never_group(self):
        values = [f"category number {i:03d}" for i in range(150)]
        start = time.perf_counter()
        assert find_fuzzy_category_groups(values) == []
        assert time.perf_counter() - start < 3


# ---------------------------------------------------------------------------
# F1 — recommendation wording / flags (generate_recommendations integration)
# ---------------------------------------------------------------------------
class TestFuzzyRecommendations:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "country": ["Australia"] * 6 + ["Austria"] * 5 + ["USA"] * 4 + ["usa"] * 2 + ["US"] * 2 + ["United States"] * 3,
            "spelling": ["Colour"] * 5 + ["Color"] * 10 + ["color"] * 2 + ["Blue"] * 3 + ["Pale"] * 2,
        })

    @staticmethod
    def _fuzzy(recs, column):
        return [r for r in recs if r.recommended_action == "fuzzy_merge" and r.column == column]

    def test_australia_austria_never_suggested(self, df):
        for r in self._fuzzy(generate_recommendations(df), "country"):
            assert not ({"Australia", "Austria"} & set(r.preview["values"]))

    def test_alias_recommendation_is_labelled_review_only(self, df):
        (r,) = self._fuzzy(generate_recommendations(df), "country")
        assert r.preview["match_type"] == "alias" and r.preview["review_required"] is True
        assert "Review-only" in r.reason
        assert "alias group" in r.issue
        assert "likely the same category" not in r.reason
        # rows_affected counts case variants too (USA 4 + usa 2 + US 2 + United States 3)
        assert r.preview["rows_affected"] == 11

    def test_spelling_recommendation_never_claims_identical_meaning(self, df):
        (r,) = self._fuzzy(generate_recommendations(df), "spelling")
        assert r.preview["match_type"] == "spelling_variant" and r.preview["review_required"] is True
        assert "does not prove the same meaning" in r.reason
        assert "Review-only" in r.reason
        assert "may be spelling variants" in r.issue
        assert "likely the same category" not in r.reason.lower()
        assert r.confidence <= 0.6
        # 'color' value = 12 rows (10 Color + 2 color) vs Colour = 5
        assert r.preview["suggested_canonical"] in {"Color", "color"}
        assert sorted(v.casefold() for v in r.preview["values"]) == ["color", "colour"]
        assert r.preview["rows_affected"] == 17
        assert r.recommended_action == "fuzzy_merge"          # API compatibility
        assert r.alternative_actions == ["ignore", "rename_category"]

    def test_recommendation_alone_never_mutates_data(self, df):
        before = df.copy(deep=True)
        generate_recommendations(df)
        pd.testing.assert_frame_equal(df, before)

    def test_gender_male_female_not_suggested(self):
        df = pd.DataFrame({"gender": ["Male"] * 40 + ["Female"] * 55 + ["Other"] * 5})
        assert self._fuzzy(generate_recommendations(df), "gender") == []

    def test_numbered_labels_not_suggested(self):
        df = pd.DataFrame({"region": ["Region 1"] * 10 + ["Region 2"] * 10 + ["Region 3"] * 10})
        assert self._fuzzy(generate_recommendations(df), "region") == []


# ---------------------------------------------------------------------------
# F2 — fuzzy_merge execution
# ---------------------------------------------------------------------------
@pytest.fixture
def country_df():
    return pd.DataFrame({"country": ["USA", "usa", "US", "United States", "  USA  ", "Canada", None, "Mexico"]})


def _merge(df, values, canonical, column="country"):
    return apply_action(df, "category_merge_center", "fuzzy_merge", column, {"values": values, "canonical": canonical})


class TestFuzzyMergeExecution:
    def test_case_and_whitespace_variants_follow_listed_values(self, country_df):
        out, pv = _merge(country_df, ["USA", "US", "United States"], "United States")
        assert out["country"].tolist()[:5] == ["United States"] * 5   # incl. 'usa' and '  USA  '
        assert out["country"].iloc[5] == "Canada" and out["country"].iloc[7] == "Mexico"
        assert pd.isna(out["country"].iloc[6])                          # missing stays missing
        assert pv["rows_affected"] == 5 and pv["rows_changed"] == 4
        assert pv["canonical"] == "United States"
        assert pv["categories_after"] < pv["categories_before"]
        assert pv["canonical_collides_with_other_category"] is False

    def test_listed_values_may_use_any_casing(self, country_df):
        out, pv = _merge(country_df, ["usa", "us"], "USA")
        assert out["country"].tolist()[:5] == ["USA", "USA", "USA", "United States", "USA"]
        assert pv["rows_affected"] == 4

    def test_canonical_is_stripped(self, country_df):
        out, pv = _merge(country_df, ["USA", "US"], "  United States  ")
        assert pv["canonical"] == "United States"
        assert "  United States  " not in out["country"].tolist()

    def test_input_frame_is_not_mutated(self, country_df):
        before = country_df.copy(deep=True)
        _merge(country_df, ["USA", "US"], "USA")
        pd.testing.assert_frame_equal(country_df, before)

    def test_collision_with_existing_other_category_is_flagged_not_blocked(self):
        df = pd.DataFrame({"g": ["Male"] * 4 + ["Female"] * 5 + ["Other"] * 2})
        out, pv = apply_action(df, "category_merge_center", "fuzzy_merge", "g", {"values": ["Male", "Other"], "canonical": "Female"})
        assert pv["canonical_collides_with_other_category"] is True      # surfaced in the preview for the user
        assert out["g"].value_counts().to_dict() == {"Female": 11}

    def test_categorical_dtype_column(self):
        df = pd.DataFrame({"c": pd.Categorical(["Colour", "Color", "Color", "Blue"])})
        out, pv = apply_action(df, "category_merge_center", "fuzzy_merge", "c", {"values": ["Colour", "Color"], "canonical": "Color"})
        assert out["c"].astype(str).tolist() == ["Color", "Color", "Color", "Blue"]
        out2, _ = apply_action(df, "category_merge_center", "fuzzy_merge", "c", {"values": ["Colour", "Color"], "canonical": "Hue"})
        assert out2["c"].astype(str).tolist() == ["Hue", "Hue", "Hue", "Blue"]

    def test_literal_nan_string_is_not_confused_with_missing(self):
        df = pd.DataFrame({"c": ["nan", "alpha", None, "beta"]})
        out, pv = apply_action(df, "category_merge_center", "fuzzy_merge", "c", {"values": ["nan", "alpha"], "canonical": "x"})
        assert out["c"].tolist()[:2] == ["x", "x"] and pd.isna(out["c"].iloc[2]) and out["c"].iloc[3] == "beta"

    def test_rename_and_normalize_actions_still_work(self, country_df):
        out, _ = apply_action(country_df, "category_merge_center", "rename_category", "country", {"old_value": "Canada", "new_value": "CA"})
        assert "CA" in out["country"].values
        out, _ = apply_action(country_df, "category_merge_center", "normalize_case_whitespace", "country", {})
        assert out["country"].dropna().str.strip().eq(out["country"].dropna()).all()


class TestFuzzyMergeValidation:
    @pytest.mark.parametrize("params", [
        {}, {"values": ["USA", "US"]}, {"canonical": "USA"},
        {"values": [], "canonical": "USA"}, {"values": ["USA", "US"], "canonical": ""},
        {"values": None, "canonical": "USA"}, {"values": ["USA", "US"], "canonical": None},
    ])
    def test_missing_params_raise_requires(self, country_df, params):
        with pytest.raises(ValueError, match="requires"):
            apply_action(country_df, "category_merge_center", "fuzzy_merge", "country", params)

    @pytest.mark.parametrize("values", ["USA", {"a": 1}, 5, [1, 2], ["USA", 2], ["USA", None], ["USA", ""], ["USA", "   "], [["USA"], ["US"]]])
    def test_values_must_be_list_of_non_empty_strings(self, country_df, values):
        with pytest.raises(ValueError, match="list of non-empty strings"):
            _merge(country_df, values, "USA")  # str / dict / int -> ValueError, never TypeError (HTTP 500)

    @pytest.mark.parametrize("canonical", ["   ", 5, ["USA"], {"a": 1}, True])
    def test_canonical_must_be_non_empty_string(self, country_df, canonical):
        with pytest.raises(ValueError, match="canonical"):
            _merge(country_df, ["USA", "US"], canonical)

    @pytest.mark.parametrize("values", [["USA", "usa"], ["USA", "  usa "], ["US", "US"], ["US"]])
    def test_needs_two_distinct_values(self, country_df, values):
        with pytest.raises(ValueError, match="at least two distinct"):
            _merge(country_df, values, "USA")

    def test_values_must_exist_in_column(self, country_df):
        with pytest.raises(ValueError, match="not found"):
            _merge(country_df, ["USA", "Atlantis"], "USA")

    def test_zero_matching_rows_rejected(self, country_df):
        with pytest.raises(ValueError, match="not found"):
            _merge(country_df, ["Atlantis", "Narnia"], "Atlantis")

    def test_missing_values_do_not_count_as_existing(self):
        df = pd.DataFrame({"c": ["a-value", None, None]})
        with pytest.raises(ValueError, match="not found"):
            apply_action(df, "category_merge_center", "fuzzy_merge", "c", {"values": ["a-value", "nan"], "canonical": "x"})

    def test_unknown_column(self, country_df):
        with pytest.raises(ValueError, match="not found"):
            _merge(country_df, ["USA", "US"], "USA", column="nope")

    def test_column_required(self, country_df):
        with pytest.raises(ValueError, match="column is required"):
            apply_action(country_df, "category_merge_center", "fuzzy_merge", None, {"values": ["USA", "US"], "canonical": "USA"})


# ---------------------------------------------------------------------------
# F2 — HTTP behaviour: 400 and NO new dataset version for invalid / no-op merges
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(cleaning_router)
    return TestClient(app)


def _post(client, session, params, **extra):
    return client.post("/api/dataset/apply-action", json={
        "dataset_id": session.dataset_id, "step": "category_merge_center", "action": "fuzzy_merge",
        "column": "country", "params": params, "reason": "test", "confidence": 0.5, **extra,
    })


class TestFuzzyMergeHttp:
    @pytest.fixture
    def session(self, country_df):
        return store.create("fuzzy_http_test.csv", country_df)

    def test_valid_merge_creates_exactly_one_version(self, client, session):
        r = _post(client, session, {"values": ["USA", "US", "United States"], "canonical": "United States"})
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 2
        assert r.json()["preview"]["rows_affected"] == 5
        s = store.get(session.dataset_id)
        assert s.version_number == 2 and len(s.log) == 1
        assert "merged 5 row(s)" in s.log[0].result_summary

    @pytest.mark.parametrize("params", [
        {},
        {"values": "USA", "canonical": "USA"},                        # was an unhandled TypeError -> 500
        {"values": ["USA"], "canonical": "USA"},                      # fewer than two
        {"values": ["USA", "usa"], "canonical": "USA"},               # one distinct value
        {"values": ["USA", "Atlantis"], "canonical": "USA"},          # non-existent value
        {"values": ["Atlantis", "Narnia"], "canonical": "X"},         # zero rows
        {"values": ["USA", "US"], "canonical": "   "},                # blank canonical
        {"values": ["USA", "US"], "canonical": 7},                    # non-string canonical
        {"values": [1, 2], "canonical": "X"},
    ])
    def test_invalid_or_noop_returns_400_and_creates_no_version(self, client, session, params):
        r = _post(client, session, params)
        assert r.status_code == 400, (r.status_code, r.text)
        assert isinstance(r.json()["detail"], str) and r.json()["detail"]
        s = store.get(session.dataset_id)
        assert s.version_number == 1 and s.log == []
        pd.testing.assert_frame_equal(s.current, session.versions[0])

    def test_unknown_column_returns_400_without_version(self, client, session):
        r = client.post("/api/dataset/apply-action", json={
            "dataset_id": session.dataset_id, "step": "category_merge_center", "action": "fuzzy_merge",
            "column": "nope", "params": {"values": ["USA", "US"], "canonical": "USA"}})
        assert r.status_code == 400
        assert store.get(session.dataset_id).version_number == 1

    def test_failed_merge_then_valid_merge_still_works(self, client, session):
        assert _post(client, session, {"values": ["USA", "Atlantis"], "canonical": "USA"}).status_code == 400
        assert _post(client, session, {"values": ["USA", "US"], "canonical": "USA"}).status_code == 200
        assert store.get(session.dataset_id).version_number == 2
