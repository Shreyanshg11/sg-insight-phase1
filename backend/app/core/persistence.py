"""
Disk-backed persistence for dataset sessions.

`SessionStore` (in session_store.py) still keeps everything in an in-memory
dict for the hot path — every request reads/writes a plain DatasetSession
object with no serialization overhead. This module is the write-through /
read-through layer underneath it, so restarting the backend process doesn't
lose datasets:

- Session metadata (filename, investigation goal, pinned charts) and the
  full action log live in a small SQLite database — structured and cheap to
  rewrite on every mutation.
- Each dataset VERSION's actual DataFrame is stored as its own pickle file
  on disk (`versions/<dataset_id>/v<N>.pkl`) rather than as a SQLite BLOB.
  That keeps the DB itself small regardless of how large the datasets are,
  and means undo/restore can free a version's storage immediately (delete
  its file) instead of leaving dead rows behind.

Security note: pickle can execute arbitrary code on load. That's safe here
ONLY because this backend ever unpickles files it wrote itself — never a
user-supplied pickle (uploads go through pandas' CSV/Excel readers, never
straight to pickle.load). If this ever becomes a shared/multi-tenant
deployment, swap `to_pickle`/`read_pickle` below for `to_parquet`/
`read_parquet` — same exact-dtype round trip, no arbitrary-code risk.
"""
import json
import os
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Defaults to <repo>/backend/data — override with SG_INSIGHT_DATA_DIR for a
# different disk location (e.g. a mounted volume in a container).
DATA_DIR = Path(os.environ.get("SG_INSIGHT_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
DB_PATH = DATA_DIR / "sessions.db"
VERSIONS_DIR = DATA_DIR / "versions"


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect():
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet. Safe to call every startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                dataset_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                investigation_goal TEXT,
                goal_keywords TEXT NOT NULL,   -- JSON list
                pinned_charts TEXT NOT NULL,   -- JSON list of chart specs
                version_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS log_entries (
                dataset_id TEXT NOT NULL,
                version_index INTEGER NOT NULL,  -- 0-based: log[0] produced version 2
                step TEXT NOT NULL,
                action TEXT NOT NULL,
                column_name TEXT,
                reason TEXT NOT NULL,
                confidence REAL,
                params TEXT NOT NULL,            -- JSON object
                timestamp TEXT NOT NULL,
                rows_affected INTEGER,
                result_summary TEXT,
                PRIMARY KEY (dataset_id, version_index)
            )
        """)


def _version_path(dataset_id: str, version_number: int) -> Path:
    return VERSIONS_DIR / dataset_id / f"v{version_number}.pkl"


def save_session(session) -> None:
    """
    Upserts everything about `session`: metadata, the full action log
    (small — cheap to rewrite wholesale each time), and any version
    DataFrames not yet on disk. Also deletes version files beyond the
    session's current version count, so undo/restore actually frees that
    version's storage rather than leaving orphaned files behind.
    """
    _ensure_dirs()
    session_dir = VERSIONS_DIR / session.dataset_id
    session_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = {int(p.stem[1:]) for p in session_dir.glob("v*.pkl")}
    current_numbers = set(range(1, session.version_number + 1))

    for n in current_numbers - existing_numbers:
        with open(_version_path(session.dataset_id, n), "wb") as f:
            pickle.dump(session.versions[n - 1], f, protocol=pickle.HIGHEST_PROTOCOL)
    for n in existing_numbers - current_numbers:
        _version_path(session.dataset_id, n).unlink(missing_ok=True)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions
                (dataset_id, filename, investigation_goal, goal_keywords,
                 pinned_charts, version_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT created_at FROM sessions WHERE dataset_id = ?), ?))
            ON CONFLICT(dataset_id) DO UPDATE SET
                filename = excluded.filename,
                investigation_goal = excluded.investigation_goal,
                goal_keywords = excluded.goal_keywords,
                pinned_charts = excluded.pinned_charts,
                version_count = excluded.version_count
            """,
            (
                session.dataset_id, session.filename, session.investigation_goal,
                json.dumps(session.goal_keywords), json.dumps(session.pinned_charts),
                session.version_number,
                session.dataset_id, datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.execute("DELETE FROM log_entries WHERE dataset_id = ?", (session.dataset_id,))
        for i, entry in enumerate(session.log):
            conn.execute(
                """
                INSERT INTO log_entries
                    (dataset_id, version_index, step, action, column_name, reason,
                     confidence, params, timestamp, rows_affected, result_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.dataset_id, i, entry.step, entry.action, entry.column, entry.reason,
                    entry.confidence, json.dumps(entry.params), entry.timestamp,
                    entry.rows_affected, entry.result_summary,
                ),
            )


def load_session_data(dataset_id: str) -> Optional[dict]:
    """
    Returns a plain dict shaped for `session_store.py` to rebuild a
    DatasetSession + its ActionLogEntry list from, or None if this
    dataset_id was never persisted (e.g. genuinely doesn't exist).
    """
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        log_rows = conn.execute(
            "SELECT * FROM log_entries WHERE dataset_id = ? ORDER BY version_index ASC",
            (dataset_id,),
        ).fetchall()

    session_dir = VERSIONS_DIR / dataset_id
    versions = []
    for n in range(1, row["version_count"] + 1):
        path = _version_path(dataset_id, n)
        if not path.exists():
            raise RuntimeError(
                f"Session {dataset_id} is missing its version {n} file on disk ({path}) — "
                f"the data/ directory may have been partially deleted."
            )
        with open(path, "rb") as f:
            versions.append(pickle.load(f))

    log = [
        {
            "step": r["step"], "action": r["action"], "column": r["column_name"],
            "reason": r["reason"], "confidence": r["confidence"],
            "params": json.loads(r["params"]), "timestamp": r["timestamp"],
            "rows_affected": r["rows_affected"], "result_summary": r["result_summary"],
        }
        for r in log_rows
    ]

    return {
        "filename": row["filename"],
        "investigation_goal": row["investigation_goal"],
        "goal_keywords": json.loads(row["goal_keywords"]),
        "pinned_charts": json.loads(row["pinned_charts"]),
        "versions": versions,
        "log": log,
    }
