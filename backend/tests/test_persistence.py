import pandas as pd
import pytest
from app.core.session_store import SessionStore, ActionLogEntry


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SG_INSIGHT_DATA_DIR", str(tmp_path))
    import app.core.persistence as persistence
    monkeypatch.setattr(persistence, "DATA_DIR", tmp_path)
    monkeypatch.setattr(persistence, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", tmp_path / "versions")
    return tmp_path


class TestPersistenceRoundTrip:
    def test_survives_a_simulated_restart(self, isolated_data_dir):
        """The core guarantee session 5 added: a brand-new SessionStore (no
        in-memory cache — simulating a fresh backend process) can still load
        a session that a *previous* SessionStore instance created."""
        store1 = SessionStore()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
        session = store1.create(filename="test.csv", df=df, investigation_goal="find why b is missing")
        did = session.dataset_id

        df2 = df.copy()
        df2["b"] = df2["b"].fillna("unknown")
        entry = ActionLogEntry(step="missing_value_center", action="impute_constant", column="b",
                                reason="test reason", confidence=0.9, params={"value": "unknown"},
                                rows_affected=1, result_summary="filled 1 value")
        session.push_version(df2, entry)
        session.add_pinned_chart({"id": "chart-1", "chart_type": "bar_chart", "columns": ["a"]})

        # Simulate a restart: a fresh store, no in-memory cache at all.
        store2 = SessionStore()
        assert did not in store2._sessions
        loaded = store2.get(did)

        assert loaded is not None
        assert loaded.filename == "test.csv"
        assert loaded.investigation_goal == "find why b is missing"
        assert loaded.version_number == 2
        assert loaded.current["b"].tolist() == ["x", "y", "unknown"]
        assert len(loaded.log) == 1
        assert loaded.log[0].step == "missing_value_center"
        assert loaded.log[0].params == {"value": "unknown"}
        assert loaded.pinned_charts[0]["id"] == "chart-1"

    def test_undo_after_reload_frees_the_orphaned_version_file(self, isolated_data_dir):
        store1 = SessionStore()
        session = store1.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        did = session.dataset_id
        session.push_version(pd.DataFrame({"a": [2]}),
                              ActionLogEntry(step="s", action="a", column="a", reason="r", confidence=0.9))

        version_dir = isolated_data_dir / "versions" / did
        assert sorted(p.name for p in version_dir.glob("*.pkl")) == ["v1.pkl", "v2.pkl"]

        store2 = SessionStore()
        loaded = store2.get(did)
        assert loaded.undo() is True
        assert sorted(p.name for p in version_dir.glob("*.pkl")) == ["v1.pkl"]

        # And the truncation itself persisted — a third fresh store agrees.
        store3 = SessionStore()
        loaded3 = store3.get(did)
        assert loaded3.version_number == 1

    def test_unknown_dataset_id_returns_none_not_an_error(self, isolated_data_dir):
        store = SessionStore()
        assert store.get("00000000-0000-0000-0000-000000000000") is None

    def test_missing_version_file_on_disk_raises_clear_error(self, isolated_data_dir):
        """If the data dir is ever partially deleted (e.g. manual tampering),
        loading should fail loudly rather than silently return wrong data."""
        store1 = SessionStore()
        session = store1.create(filename="t.csv", df=pd.DataFrame({"a": [1]}))
        did = session.dataset_id

        version_file = isolated_data_dir / "versions" / did / "v1.pkl"
        version_file.unlink()

        store2 = SessionStore()
        with pytest.raises(RuntimeError, match="missing its version"):
            store2.get(did)
