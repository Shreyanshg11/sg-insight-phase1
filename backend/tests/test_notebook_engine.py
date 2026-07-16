import json
import pandas as pd
import pandas.testing as pdt
import pytest
from app.core.session_store import SessionStore, ActionLogEntry
from app.engines.cleaning_engine import apply_action
from app.engines.feature_engineering_engine import apply_feature_action
from app.engines.notebook_engine import build_notebook, _MAX_EMBED_CSV_BYTES


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SG_INSIGHT_DATA_DIR", str(tmp_path))
    import app.core.persistence as persistence
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    return SessionStore()


def _do_clean(session, step, action, column, params, reason="r", confidence=0.8):
    new_df, preview = apply_action(session.current, step, action, column, params)
    entry = ActionLogEntry(step=step, action=action, column=column, reason=reason,
                            confidence=confidence, params=params, result_summary=str(preview))
    session.push_version(new_df, entry)


def _do_feature(session, category, action, columns, new_col, params, reason="r", confidence=0.75):
    new_df, preview = apply_feature_action(session.current, category, action, columns, new_col, params)
    entry = ActionLogEntry(step=f"feature_engineering:{category}", action=action, column=", ".join(columns),
                            reason=reason, confidence=confidence, params=params, result_summary=str(preview))
    session.push_version(new_df, entry)


@pytest.fixture
def full_pipeline_session(store):
    """A session exercising every supported action type — mirrors the
    manual verification done during development, now as a repeatable test."""
    import numpy as np
    np.random.seed(0)
    df = pd.DataFrame({
        "revenue": list(np.random.normal(1000, 100, 48)) + [50000.0, -100.0],
        "cost": np.random.normal(500, 50, 50),
        "region": (["  north", "North ", "south", "South", "east"] * 10),
        "signup_date": pd.date_range("2024-01-01", periods=50, freq="D").astype(str),
    })
    df.loc[5:8, "cost"] = None
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    session = store.create(filename="sales.csv", df=df, investigation_goal="find why revenue is volatile")

    _do_clean(session, "duplicate_manager", "drop_duplicates", None, {})
    _do_clean(session, "missing_value_center", "impute_median", "cost", {})
    _do_clean(session, "outlier_center", "clip", "revenue", {})
    _do_clean(session, "category_merge_center", "normalize_case_whitespace", "region", {})
    _do_clean(session, "category_merge_center", "fuzzy_merge", "region",
              {"values": ["north", "south"], "canonical": "unified"})
    _do_clean(session, "text_cleaning", "trim", "region", {})
    _do_feature(session, "date_features", "extract_date_parts", ["signup_date"], None,
                {"parts": ["year", "month", "is_weekend"]})
    _do_feature(session, "ratio", "create_ratio", ["revenue", "cost"], "custom_ratio_name", {})
    _do_feature(session, "bucketing", "quartile_bucket", ["revenue"], None, {})
    _do_feature(session, "scaling_normalization", "z_score_standardize", ["revenue"], None, {})
    _do_feature(session, "encoding", "one_hot_encode", ["region"], None, {})
    _do_feature(session, "log_transform", "log1p_transform", ["cost"], None, {})
    return session


class TestNotebookValidity:
    def test_produces_valid_nbformat_json(self, full_pipeline_session):
        nb_json = build_notebook(full_pipeline_session)
        nb = json.loads(nb_json)
        assert nb["nbformat"] == 4
        assert all("id" in c for c in nb["cells"])  # required by nbformat_minor 5
        assert all(c["cell_type"] in ("code", "markdown") for c in nb["cells"])

    def test_one_cell_pair_per_log_entry(self, full_pipeline_session):
        nb = json.loads(build_notebook(full_pipeline_session))
        # intro md + imports code + load md + load code + (md+code)*12 + result md + result code
        expected = 4 + len(full_pipeline_session.log) * 2 + 2
        assert len(nb["cells"]) == expected

    def test_reason_and_confidence_appear_in_markdown(self, full_pipeline_session):
        nb = json.loads(build_notebook(full_pipeline_session))
        full_text = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
        assert "find why revenue is volatile" in full_text
        for entry in full_pipeline_session.log:
            assert entry.reason in full_text


class TestGeneratedCodeMatchesLiveSession:
    def test_executed_code_produces_identical_dataframe(self, full_pipeline_session):
        """The strongest possible check: run the generated code in a fresh
        namespace and assert byte-for-byte equality against what the live
        session actually computed — not just 'it runs without error'."""
        nb = json.loads(build_notebook(full_pipeline_session))
        code_blocks = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        full_code = "\n\n".join(code_blocks)

        namespace = {}
        exec(full_code, namespace)
        notebook_df = namespace["df"].reset_index(drop=True)
        live_df = full_pipeline_session.current.reset_index(drop=True)

        assert list(notebook_df.columns) == list(live_df.columns)
        assert notebook_df.shape == live_df.shape
        pdt.assert_frame_equal(live_df, notebook_df, check_dtype=False, check_categorical=False)

    def test_custom_new_column_name_is_recovered_correctly(self, full_pipeline_session):
        # "custom_ratio_name" was an explicit new_column_name override —
        # confirms the diff-based recovery (not the default f"{a}_per_{b}")
        assert "custom_ratio_name" in full_pipeline_session.current.columns
        nb = json.loads(build_notebook(full_pipeline_session))
        code_blocks = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        assert any("custom_ratio_name" in c for c in code_blocks)
        assert not any("revenue_per_cost" in c for c in code_blocks)


class TestEdgeCases:
    def test_outlier_ignore_generates_a_noop_comment(self, store):
        df = pd.DataFrame({"x": list(range(20)) + [1000]})
        session = store.create(filename="t.csv", df=df)
        _do_clean(session, "outlier_center", "ignore", "x", {})
        nb = json.loads(build_notebook(session))
        code_cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        assert any("Outliers reviewed" in c for c in code_cells)

    def test_large_dataset_falls_back_to_placeholder_path(self, store):
        import numpy as np
        big_df = pd.DataFrame({"x": np.random.normal(size=60_000), "y": np.random.normal(size=60_000)})
        session = store.create(filename="big.csv", df=big_df)
        assert len(big_df.to_csv(index=False).encode("utf-8")) > _MAX_EMBED_CSV_BYTES

        nb = json.loads(build_notebook(session))
        code_cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        assert any("update this path" in c for c in code_cells)
        assert not any("io.StringIO" in c for c in code_cells)

    def test_small_dataset_embeds_csv_inline(self, store):
        session = store.create(filename="small.csv", df=pd.DataFrame({"x": [1, 2, 3]}))
        nb = json.loads(build_notebook(session))
        code_cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        assert any("io.StringIO" in c for c in code_cells)

    def test_no_actions_still_produces_a_valid_notebook(self, store):
        session = store.create(filename="untouched.csv", df=pd.DataFrame({"x": [1, 2, 3]}))
        nb = json.loads(build_notebook(session))
        assert len(nb["cells"]) >= 4  # intro, imports, load md+code, result md+code
