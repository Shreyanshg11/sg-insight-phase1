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
import shutil
import tempfile
import threading
import uuid
from functools import wraps
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
_STORAGE_LOCK = threading.RLock()
MIN_FREE_DISK_BYTES = int(os.getenv("SG_INSIGHT_MIN_FREE_DISK_MB", "100")) * 1024 * 1024


class StorageCapacityError(RuntimeError):
    """Persistent storage is too close to its configured safety floor."""


def _ensure_capacity(required_bytes: int = 0) -> None:
    free_bytes = shutil.disk_usage(DATA_DIR).free
    if free_bytes < MIN_FREE_DISK_BYTES + max(0, required_bytes):
        raise StorageCapacityError("Persistent storage is low. Create a backup or free disk space before continuing.")


def storage_locked(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _STORAGE_LOCK:
            return function(*args, **kwargs)
    return wrapped


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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revoked_sessions (
                session_id TEXT PRIMARY KEY,
                expires_at INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_revoked_sessions_expiry ON revoked_sessions(expires_at)")
    ensure_schema()  # additive migrations (last-activity column) for databases created by any earlier version


def _validated_dataset_id(dataset_id: str) -> str:
    """Accept only canonical UUID strings before using IDs in filesystem paths."""
    if not isinstance(dataset_id, str):
        raise ValueError("Invalid dataset ID")
    try:
        normalized = str(uuid.UUID(dataset_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid dataset ID") from exc
    if normalized != dataset_id:
        raise ValueError("Dataset ID must be a canonical UUID")
    return normalized


def _version_path(dataset_id: str, version_number: int) -> Path:
    dataset_id = _validated_dataset_id(dataset_id)
    if isinstance(version_number, bool) or not isinstance(version_number, int) or version_number < 1:
        raise ValueError("Invalid version number")
    return VERSIONS_DIR / dataset_id / f"v{version_number}.pkl"


@storage_locked
def save_session(session, *, write_current_version: bool = False) -> None:
    """
    Upserts everything about `session`: metadata, the full action log
    (small — cheap to rewrite wholesale each time), and any version
    DataFrames not yet on disk. Also deletes version files beyond the
    session's current version count, so undo/restore actually frees that
    version's storage rather than leaving orphaned files behind.
    """
    dataset_id = _validated_dataset_id(session.dataset_id)
    _ensure_dirs()
    session_dir = VERSIONS_DIR / dataset_id
    session_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = {int(p.stem[1:]) for p in session_dir.glob("v*.pkl")}
    current_numbers = set(range(1, session.version_number + 1))

    # Publish complete version files atomically before metadata can reference them.
    versions_to_write = current_numbers - existing_numbers
    if write_current_version:
        versions_to_write.add(session.version_number)
    if versions_to_write:
        required = sum(int(session.versions[n - 1].memory_usage(index=True, deep=True).sum()) for n in versions_to_write)
        _ensure_capacity(required)
    for n in versions_to_write:
        destination = _version_path(session.dataset_id, n)
        with tempfile.NamedTemporaryFile(dir=session_dir, prefix=f".v{n}-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            pickle.dump(session.versions[n - 1], f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, destination)

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

    # Remove versions only after the new authoritative count is committed.
    for n in existing_numbers - current_numbers:
        try:
            _version_path(session.dataset_id, n).unlink(missing_ok=True)
        except OSError:
            # Stale versions are safe orphans; readiness and future backups can report them.
            pass


@storage_locked
def load_session_data(dataset_id: str) -> Optional[dict]:
    """
    Returns a plain dict shaped for `session_store.py` to rebuild a
    DatasetSession + its ActionLogEntry list from, or None if this
    dataset_id was never persisted (e.g. genuinely doesn't exist).
    """
    dataset_id = _validated_dataset_id(dataset_id)
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
        try:
            with open(path, "rb") as f:
                versions.append(pickle.load(f))
        except Exception as exc:
            raise RuntimeError(f"Stored dataset version {n} is corrupt and could not be loaded.") from exc

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


@storage_locked
def check_storage() -> None:
    """Check SQLite integrity and confirm every indexed version is present."""
    with _connect() as conn:
        result = conn.execute("PRAGMA quick_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError("SQLite storage failed its integrity check")
        rows = conn.execute("SELECT dataset_id, version_count FROM sessions").fetchall()
    for row in rows:
        dataset_id = _validated_dataset_id(row["dataset_id"])
        count = int(row["version_count"])
        if count < 1:
            raise RuntimeError("Stored session has an invalid version count")
        for number in range(1, count + 1):
            path = _version_path(dataset_id, number)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Stored dataset version {number} is missing or empty")
    if shutil.disk_usage(DATA_DIR).free < MIN_FREE_DISK_BYTES:
        raise RuntimeError("Persistent storage is below its configured free-space reserve")


_DATASET_SCOPED_TABLES = ("copilot_proposals", "recommendation_status", "recommendation_events")


@storage_locked
def ensure_schema() -> None:
    """Additive migration: last-activity column used by idle cleanup. Idempotent; never touches revoked_sessions."""
    with _connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
        if cols and "touched_at" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN touched_at TEXT")


@storage_locked
def touch(dataset_id: str) -> None:
    dataset_id = _validated_dataset_id(dataset_id)
    ensure_schema()
    with _connect() as conn:
        conn.execute("UPDATE sessions SET touched_at = ? WHERE dataset_id = ?",
                     (datetime.now(timezone.utc).isoformat(), dataset_id))


@storage_locked
def list_activity() -> list[tuple[str, str]]:
    """[(dataset_id, last_activity_iso)] oldest first; last activity falls back to creation time."""
    ensure_schema()
    with _connect() as conn:
        rows = conn.execute("SELECT dataset_id, COALESCE(touched_at, created_at) AS last FROM sessions ORDER BY last").fetchall()
    return [(r["dataset_id"], r["last"]) for r in rows]


@storage_locked
def delete_session(dataset_id: str) -> bool:
    """Remove one dataset's rows (session, log, proposals, recommendation history) and every version file.
    Authentication state (revoked_sessions) is deliberately never touched. True if anything existed."""
    dataset_id = _validated_dataset_id(dataset_id)
    session_dir = VERSIONS_DIR / dataset_id
    existed = session_dir.exists()
    with _connect() as conn:
        removed = conn.execute("DELETE FROM sessions WHERE dataset_id = ?", (dataset_id,)).rowcount
        conn.execute("DELETE FROM log_entries WHERE dataset_id = ?", (dataset_id,))
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table in _DATASET_SCOPED_TABLES:
            if table in present:
                conn.execute(f"DELETE FROM {table} WHERE dataset_id = ?", (dataset_id,))
    if existed:
        shutil.rmtree(session_dir, ignore_errors=True)
    return existed or removed > 0
