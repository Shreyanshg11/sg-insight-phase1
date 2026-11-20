"""
Server-side registry of Copilot proposals.

Why: the browser must never be the source of truth for "what will be executed".
A proposal is created by the server from a genuine backend recommendation,
stored here, and confirmation only ever executes the STORED, server-built
action — never parameters the client or the LLM sent back.

Storage: the same SQLite file as dataset sessions (app.core.persistence), so
proposals survive a backend restart. Status changes are single-statement
compare-and-set UPDATEs (`... WHERE status = 'pending'`), which SQLite
serialises, so two simultaneous confirms can't both win even across worker
processes. A per-dataset in-process lock additionally serialises the actual
dataframe mutation (the session cache itself is per-process).

Statuses: pending -> confirming -> confirmed | rejected
          pending -> dismissed | expired
"confirming" exists so a crash between claim and apply leaves an honest,
non-confirmable record instead of a "confirmed" proposal that never ran.

Limitations (documented, see README): the dataframe cache is per-process, so
multi-worker deployments still need a shared session cache; a proposal stuck in
'confirming' after a hard crash must be re-requested from Copilot.
"""
import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core import persistence

PROPOSAL_TTL_SECONDS = int(os.environ.get("SG_INSIGHT_PROPOSAL_TTL_SECONDS", "1800"))
STATUSES = ("pending", "confirming", "confirmed", "dismissed", "expired", "rejected")

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def dataset_lock(dataset_id: str) -> threading.RLock:
    """One re-entrant lock per dataset; serialises Copilot confirmations."""
    with _locks_guard:
        return _locks.setdefault(dataset_id, threading.RLock())


def state_token(session) -> str:
    """Identifies the exact dataset state a proposal was generated against.
    version_number alone is not enough: undo + a new action yields the same
    number with different data, so the last log entry's timestamp is included."""
    last = session.log[-1].timestamp if session.log else "original"
    return f"v{session.version_number}:{last}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS copilot_proposals (
            proposal_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            dataset_version INTEGER NOT NULL,
            state_token TEXT NOT NULL,
            payload TEXT NOT NULL,          -- JSON: the trusted, server-built action
            recommendation_ref TEXT NOT NULL,
            origin TEXT NOT NULL,           -- 'copilot' | 'client_action'
            status TEXT NOT NULL,
            status_reason TEXT,
            result_summary TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_copilot_proposals_dataset ON copilot_proposals(dataset_id)")


def _row(r) -> dict:
    d = dict(r)
    d["payload"] = json.loads(d["payload"])
    d["recommendation_ref"] = json.loads(d["recommendation_ref"])
    return d


def create_proposal(session, payload: dict, recommendation_ref: dict, origin: str = "copilot") -> dict:
    now = _now()
    record = {
        "proposal_id": str(uuid.uuid4()), "dataset_id": session.dataset_id,
        "dataset_version": session.version_number, "state_token": state_token(session),
        "payload": json.dumps(payload, default=str), "recommendation_ref": json.dumps(recommendation_ref, default=str),
        "origin": origin, "status": "pending", "status_reason": None, "result_summary": None,
        "created_at": now.isoformat(), "expires_at": (now + timedelta(seconds=PROPOSAL_TTL_SECONDS)).isoformat(),
        "updated_at": now.isoformat(),
    }
    with persistence._connect() as conn:
        _ensure(conn)
        conn.execute(f"INSERT INTO copilot_proposals ({', '.join(record)}) VALUES ({', '.join('?' * len(record))})",
                     list(record.values()))
    return get_proposal(record["proposal_id"])


def get_proposal(proposal_id: str) -> Optional[dict]:
    """Reads a proposal, lazily expiring it if its TTL has passed."""
    with persistence._connect() as conn:
        _ensure(conn)
        r = conn.execute("SELECT * FROM copilot_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        if r is None:
            return None
        if r["status"] == "pending" and datetime.fromisoformat(r["expires_at"]) <= _now():
            conn.execute("UPDATE copilot_proposals SET status='expired', status_reason='ttl', updated_at=? "
                         "WHERE proposal_id=? AND status='pending'", (_now().isoformat(), proposal_id))
            r = conn.execute("SELECT * FROM copilot_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
    return _row(r)


def transition(proposal_id: str, from_status: str, to_status: str, reason: Optional[str] = None,
               result_summary: Optional[str] = None) -> bool:
    """Atomic compare-and-set. True only for the single caller that wins."""
    assert to_status in STATUSES and from_status in STATUSES
    with persistence._connect() as conn:
        _ensure(conn)
        cur = conn.execute(
            "UPDATE copilot_proposals SET status=?, status_reason=?, result_summary=COALESCE(?, result_summary), "
            "updated_at=? WHERE proposal_id=? AND status=?",
            (to_status, reason, result_summary, _now().isoformat(), proposal_id, from_status))
        return cur.rowcount == 1
