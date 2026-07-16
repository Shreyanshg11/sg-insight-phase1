import pandas as pd
import pytest
from app.core.session_store import SessionStore, ActionLogEntry, _extract_goal_keywords


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A SessionStore backed by a throwaway temp dir, so tests never touch
    the real backend/data/ directory or leak state between test runs."""
    monkeypatch.setenv("SG_INSIGHT_DATA_DIR", str(tmp_path))
    import app.core.persistence as persistence
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    return SessionStore()


class TestGoalKeywordExtraction:
    def test_extracts_meaningful_words(self):
        kws = _extract_goal_keywords("find why revenue dropped in Q2")
        assert "revenue" in kws
        assert "dropped" in kws

    def test_filters_stopwords(self):
        kws = _extract_goal_keywords("find why the revenue is low")
        assert "why" not in kws
        assert "the" not in kws
        assert "is" not in kws

    def test_none_goal_returns_empty(self):
        assert _extract_goal_keywords(None) == []

    def test_empty_string_returns_empty(self):
        assert _extract_goal_keywords("") == []


class TestDatasetSessionBasics:
    def test_create_sets_version_1(self, store):
        df = pd.DataFrame({"a": [1, 2, 3]})
        session = store.create(filename="test.csv", df=df)
        assert session.version_number == 1
        assert session.current.equals(df)

    def test_push_version_increments(self, store):
        df = pd.DataFrame({"a": [1, 2, 3]})
        session = store.create(filename="test.csv", df=df)
        df2 = df.copy()
        df2["a"] = [10, 20, 30]
        entry = ActionLogEntry(step="s", action="a", column="a", reason="r", confidence=0.9)
        session.push_version(df2, entry)
        assert session.version_number == 2
        assert session.current["a"].tolist() == [10, 20, 30]
        assert len(session.log) == 1

    def test_is_goal_relevant(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"revenue": [1]}),
                                investigation_goal="find why revenue dropped")
        assert session.is_goal_relevant("revenue") is True
        assert session.is_goal_relevant("unrelated_column") is False

    def test_no_goal_means_nothing_is_relevant(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"revenue": [1]}))
        assert session.is_goal_relevant("revenue") is False


class TestUndo:
    def test_undo_reverts_to_previous_version(self, store):
        df = pd.DataFrame({"a": [1, 2, 3]})
        session = store.create(filename="t.csv", df=df)
        df2 = df.copy(); df2["a"] = [9, 9, 9]
        session.push_version(df2, ActionLogEntry(step="s", action="a", column="a", reason="r", confidence=0.9))
        assert session.undo() is True
        assert session.version_number == 1
        assert session.current["a"].tolist() == [1, 2, 3]
        assert len(session.log) == 0

    def test_cannot_undo_past_version_1(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        assert session.undo() is False
        assert session.version_number == 1


class TestRestoreTo:
    def test_restore_to_earlier_version_truncates_forward_history(self, store):
        df = pd.DataFrame({"a": [1]})
        session = store.create(filename="t.csv", df=df)
        for i in range(2, 5):
            new_df = pd.DataFrame({"a": [i]})
            session.push_version(new_df, ActionLogEntry(step="s", action="a", column="a", reason="r", confidence=0.9))
        assert session.version_number == 4
        assert session.restore_to(2) is True
        assert session.version_number == 2
        assert session.current["a"].tolist() == [2]
        assert len(session.log) == 1

    def test_restore_to_invalid_version_returns_false(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        assert session.restore_to(0) is False
        assert session.restore_to(99) is False
        assert session.version_number == 1  # unchanged


class TestPinnedCharts:
    def test_add_and_remove_pinned_chart(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        session.add_pinned_chart({"id": "c1", "chart_type": "bar_chart", "columns": ["a"]})
        assert len(session.pinned_charts) == 1
        session.remove_pinned_chart("c1")
        assert len(session.pinned_charts) == 0

    def test_remove_nonexistent_chart_is_a_noop(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        session.add_pinned_chart({"id": "c1", "chart_type": "bar_chart", "columns": ["a"]})
        session.remove_pinned_chart("does-not-exist")
        assert len(session.pinned_charts) == 1


class TestSessionStoreLookup:
    def test_get_unknown_id_returns_none(self, store):
        assert store.get("not-a-real-id") is None

    def test_require_unknown_id_raises(self, store):
        with pytest.raises(KeyError):
            store.require("not-a-real-id")

    def test_get_known_id_returns_session(self, store):
        session = store.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        assert store.get(session.dataset_id) is session
