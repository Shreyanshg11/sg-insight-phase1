import pandas as pd
import pytest
import json
import zipfile

from app.core import persistence
from app.core.backup import create_backup, restore_backup
from app.core.session_store import ActionLogEntry, SessionStore


def test_backup_restore_preserves_versions_actions_and_pinned_charts(tmp_path, monkeypatch):
    data_dir = tmp_path / "live"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "DB_PATH", data_dir / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", data_dir / "versions")
    store = SessionStore()
    session = store.create("sales.csv", pd.DataFrame({"sales": [1, 2, 3]}))
    session.push_version(pd.DataFrame({"sales": [1, 3, 4]}), ActionLogEntry(
        step="test", action="increment", column="sales", reason="test", confidence=1.0,
    ))
    session.add_pinned_chart({"id": "chart-1", "type": "bar"})
    archive = create_backup(tmp_path / "snapshot.zip")
    restored = restore_backup(archive, tmp_path / "restored")

    monkeypatch.setattr(persistence, "DATA_DIR", restored)
    monkeypatch.setattr(persistence, "DB_PATH", restored / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", restored / "versions")
    persistence.check_storage()
    loaded = persistence.load_session_data(session.dataset_id)
    assert [frame["sales"].tolist() for frame in loaded["versions"]] == [[1, 2, 3], [1, 3, 4]]
    assert loaded["log"][0]["action"] == "increment"
    assert loaded["pinned_charts"] == [{"id": "chart-1", "type": "bar"}]


def test_restore_refuses_to_overwrite_nonempty_directory(tmp_path):
    target = tmp_path / "active"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        restore_backup(tmp_path / "missing.zip", target)
    assert (target / "keep.txt").read_text() == "keep"


def test_backup_detects_corrupt_or_missing_version(tmp_path, monkeypatch):
    data_dir = tmp_path / "live"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "DB_PATH", data_dir / "sessions.db")
    monkeypatch.setattr(persistence, "VERSIONS_DIR", data_dir / "versions")
    session = SessionStore().create("one.csv", pd.DataFrame({"x": [1]}))
    (data_dir / "versions" / session.dataset_id / "v1.pkl").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="corrupt"):
        persistence.load_session_data(session.dataset_id)


def test_restore_rejects_windows_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": 1, "files": {"..\\outside": "0" * 64}}))
        zf.writestr("..\\outside", "bad")
    with pytest.raises(ValueError, match="unsafe file path"):
        restore_backup(archive, tmp_path / "restore")
