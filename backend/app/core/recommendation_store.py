"""
Persistent status of cleaning recommendations (dismissed / open).

Same SQLite file and connection helper as dataset sessions and Copilot
proposals (app.core.persistence._connect). Tables are additive and created
lazily with CREATE TABLE IF NOT EXISTS, exactly like proposal_store, so
existing databases need no migration step and an older build simply ignores
them.

  recommendation_status  one row per (dataset_id, rec_key): current status
  recommendation_events  append-only audit: dismissed / restored

Dismissal and restoration NEVER touch session data: no push_version, no
save_session, no log_entries row, no dataframe access. They only write here.

Concurrency: each operation is one transaction whose first statement is a
write (an upsert / conditional UPDATE guarded by the current status), so
SQLite serialises racing callers; the loser sees "no change" and writes no
event. This is safe across worker processes too.
"""
from datetime import datetime, timezone
from typing import Optional

from app.core import persistence
from app.core.recommendation_identity import evidence_fingerprint

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS recommendation_status (
        dataset_id TEXT NOT NULL,
        rec_key TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('cleaning')),
        status TEXT NOT NULL CHECK (status IN ('open', 'dismissed')),
        step TEXT NOT NULL,
        column_name TEXT,
        action TEXT,
        evidence_fingerprint TEXT,
        status_version INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (dataset_id, rec_key)
    )""",
    """CREATE TABLE IF NOT EXISTS recommendation_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT NOT NULL,
        rec_key TEXT NOT NULL,
        event TEXT NOT NULL CHECK (event IN ('dismissed', 'restored')),
        dataset_version INTEGER NOT NULL,
        state_token TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_recommendation_events_dataset ON recommendation_events(dataset_id, rec_key)",
)


def _ensure(conn) -> None:
    for statement in _SCHEMA:
        conn.execute(statement)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_statuses(dataset_id: str) -> dict:
    """rec_key -> row (dict) for every recommendation this dataset has a status for."""
    with persistence._connect() as conn:
        _ensure(conn)
        rows = conn.execute("SELECT * FROM recommendation_status WHERE dataset_id = ?", (dataset_id,)).fetchall()
    return {r["rec_key"]: dict(r) for r in rows}


def dismiss(dataset_id: str, rec, dataset_version: int, state_token: str, note: Optional[str] = None) -> bool:
    """Marks `rec` dismissed. Returns True if this call changed the status,
    False if it was already dismissed (idempotent; no duplicate row or event)."""
    now = _now()
    with persistence._connect() as conn:
        _ensure(conn)
        cur = conn.execute(
            """INSERT INTO recommendation_status
                   (dataset_id, rec_key, kind, status, step, column_name, action,
                    evidence_fingerprint, status_version, created_at, updated_at)
               VALUES (?, ?, 'cleaning', 'dismissed', ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(dataset_id, rec_key) DO UPDATE SET
                   status = 'dismissed',
                   action = excluded.action,
                   evidence_fingerprint = excluded.evidence_fingerprint,
                   status_version = excluded.status_version,
                   updated_at = excluded.updated_at
               WHERE recommendation_status.status != 'dismissed'""",
            (dataset_id, rec.id, rec.step, rec.column, rec.recommended_action,
             evidence_fingerprint(rec), dataset_version, now, now))
        changed = cur.rowcount == 1
        if changed:
            conn.execute(
                "INSERT INTO recommendation_events (dataset_id, rec_key, event, dataset_version, state_token, note, created_at) "
                "VALUES (?, ?, 'dismissed', ?, ?, ?, ?)", (dataset_id, rec.id, dataset_version, state_token, note, now))
    return changed


def restore(dataset_id: str, rec_key: str, dataset_version: int, state_token: str,
            note: Optional[str] = None) -> Optional[bool]:
    """Re-opens a dismissed recommendation. True = changed, False = it was
    already open, None = this dataset has no status record for that key. The
    row is kept (status 'open') and history stays in recommendation_events —
    nothing is deleted."""
    now = _now()
    with persistence._connect() as conn:
        _ensure(conn)
        cur = conn.execute(
            "UPDATE recommendation_status SET status = 'open', updated_at = ? "
            "WHERE dataset_id = ? AND rec_key = ? AND status = 'dismissed'", (now, dataset_id, rec_key))
        if cur.rowcount == 1:
            conn.execute(
                "INSERT INTO recommendation_events (dataset_id, rec_key, event, dataset_version, state_token, note, created_at) "
                "VALUES (?, ?, 'restored', ?, ?, ?, ?)", (dataset_id, rec_key, dataset_version, state_token, note, now))
            return True
        exists = conn.execute("SELECT 1 FROM recommendation_status WHERE dataset_id = ? AND rec_key = ?",
                              (dataset_id, rec_key)).fetchone()
    return False if exists else None


def events(dataset_id: str, rec_key: Optional[str] = None) -> list:
    with persistence._connect() as conn:
        _ensure(conn)
        if rec_key:
            rows = conn.execute("SELECT * FROM recommendation_events WHERE dataset_id = ? AND rec_key = ? ORDER BY event_id",
                                (dataset_id, rec_key)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM recommendation_events WHERE dataset_id = ? ORDER BY event_id",
                                (dataset_id,)).fetchall()
    return [dict(r) for r in rows]


def apply_status_overlay(dataset_id: str, recs) -> dict:
    """Sets status / dismissed_at / changed_since_dismissed on each freshly
    generated recommendation (in place) and returns {open, dismissed, total}.
    Recommendations with no record are 'open'."""
    statuses = get_statuses(dataset_id)
    for rec in recs:
        row = statuses.get(rec.id)
        if row and row["status"] == "dismissed":
            rec.status = "dismissed"
            rec.dismissed_at = row["updated_at"]
            rec.changed_since_dismissed = row["evidence_fingerprint"] != evidence_fingerprint(rec)
        else:
            rec.status, rec.dismissed_at, rec.changed_since_dismissed = "open", None, False
    dismissed = sum(1 for r in recs if r.status == "dismissed")
    return {"open": len(recs) - dismissed, "dismissed": dismissed, "total": len(recs)}
