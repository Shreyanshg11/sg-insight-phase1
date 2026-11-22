"""
Dataset session store.

Kept in-memory as a plain dict of DatasetSession objects for the request
hot-path (no serialization on every read), backed by app.core.persistence
underneath: every mutation (push_version, undo, restore_to, pinned chart
add/remove) writes straight through to disk, and a session missing from the
in-memory cache is lazily loaded from disk on first access — so restarting
the backend process doesn't lose datasets, only the in-memory cache of them.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import re
import pandas as pd
import uuid
import os


MAX_SESSION_VERSIONS = 100
MAX_SESSION_MEMORY_BYTES = int(os.getenv("SG_INSIGHT_MAX_SESSION_MEMORY_MB", "768")) * 1024 * 1024


class DatasetResourceLimitError(ValueError):
    """A dataset session would exceed configured memory/history limits."""

from app.core import persistence

# Small stopword list for pulling candidate "topic" words out of a free-text
# investigation goal (e.g. "find why revenue dropped" -> {"revenue", "dropped"}).
# Deliberately simple and explainable — no NLP model, just noun-ish token
# extraction, so goal-relevance can always be traced back to a literal word match.
_GOAL_STOPWORDS = {
    "find", "why", "the", "a", "an", "is", "are", "was", "were", "did", "does",
    "do", "to", "of", "in", "on", "for", "and", "or", "what", "how", "which",
    "understand", "explain", "investigate", "analyze", "analyse", "check",
    "see", "look", "at", "my", "our", "this", "that", "there", "has", "have",
}


def _extract_goal_keywords(investigation_goal: Optional[str]) -> List[str]:
    if not investigation_goal:
        return []
    tokens = re.findall(r"[a-zA-Z]+", investigation_goal.lower())
    return [t for t in tokens if t not in _GOAL_STOPWORDS and len(t) > 2]


@dataclass
class ActionLogEntry:
    """One entry in the Analyst Notebook — every cleaning action logs one of
    these. Timestamp/rows_affected/result_summary exist specifically so the
    notebook can show a reproducible, human-readable trail, not just a step name."""
    step: str            # e.g. "missing_value_center"
    action: str          # e.g. "impute_median"
    column: Optional[str]
    reason: str
    confidence: Optional[float]
    params: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rows_affected: Optional[int] = None
    result_summary: Optional[str] = None


@dataclass
class DatasetSession:
    dataset_id: str
    filename: str
    investigation_goal: Optional[str] = None
    goal_keywords: List[str] = field(default_factory=list)
    versions: List[pd.DataFrame] = field(default_factory=list)  # index 0 = original
    log: List[ActionLogEntry] = field(default_factory=list)
    pinned_charts: List[dict] = field(default_factory=list)  # Dashboard Builder (#17) chart specs

    def _persist(self, write_current_version: bool = False):
        persistence.save_session(self, write_current_version=write_current_version)

    @property
    def current(self) -> pd.DataFrame:
        return self.versions[-1]

    @property
    def version_number(self) -> int:
        return len(self.versions)  # V1 = original, V2 = first cleaning action, ...

    def push_version(self, df: pd.DataFrame, log_entry: ActionLogEntry):
        if self.version_number >= MAX_SESSION_VERSIONS:
            raise DatasetResourceLimitError(f"A dataset can retain at most {MAX_SESSION_VERSIONS} versions.")
        estimated = sum(int(version.memory_usage(index=True, deep=True).sum()) for version in self.versions)
        estimated += int(df.memory_usage(index=True, deep=True).sum())
        if estimated > MAX_SESSION_MEMORY_BYTES:
            limit_mb = MAX_SESSION_MEMORY_BYTES // (1024 * 1024)
            raise DatasetResourceLimitError(f"Dataset version history exceeds the {limit_mb}MB memory limit. Export or undo older work before continuing.")
        self.versions.append(df)
        self.log.append(log_entry)
        try:
            self._persist(write_current_version=True)
        except Exception:
            self.versions.pop()
            self.log.pop()
            raise

    def undo(self) -> bool:
        if len(self.versions) <= 1:
            return False
        removed_version = self.versions.pop()
        removed_log = self.log.pop() if self.log else None
        try:
            self._persist()
        except Exception:
            self.versions.append(removed_version)
            if removed_log is not None:
                self.log.append(removed_log)
            raise
        return True

    def restore_to(self, version_number: int) -> bool:
        """Jumps directly to an earlier version (#10 Dataset Versioning).
        Still linear, not branching: restoring to V2 while at V5 discards
        V3-V5 permanently, same as repeated undo would, just in one step.
        Full branching version trees are explicitly out of scope for this
        phase per the brief."""
        if version_number < 1 or version_number > self.version_number:
            return False
        old_versions, old_log = self.versions, self.log
        self.versions = self.versions[:version_number]
        self.log = self.log[:version_number - 1]
        try:
            self._persist()
        except Exception:
            self.versions, self.log = old_versions, old_log
            raise
        return True

    def add_pinned_chart(self, spec: dict):
        self.pinned_charts.append(spec)
        try:
            self._persist()
        except Exception:
            self.pinned_charts.pop()
            raise

    def remove_pinned_chart(self, chart_id: str):
        old_charts = self.pinned_charts
        self.pinned_charts = [c for c in old_charts if c["id"] != chart_id]
        try:
            self._persist()
        except Exception:
            self.pinned_charts = old_charts
            raise

    def is_goal_relevant(self, column_name: str) -> bool:
        """Cheap, explainable relevance check: does the column name literally
        contain (or get contained by) one of the goal keywords? This is the
        mechanism threading the investigation goal through recommendations,
        quality prioritization, etc. — not a semantic/LLM match, so it's
        always possible to say exactly why something was flagged as relevant."""
        if not self.goal_keywords:
            return False
        col_lower = column_name.lower()
        return any(kw in col_lower or col_lower in kw for kw in self.goal_keywords)


class SessionStore:
    TOUCH_INTERVAL_SECONDS = 60
    MIN_IDLE_HOURS = 1  # a session used within the last hour is never purged, whatever the configuration

    def __init__(self):
        self._sessions: Dict[str, DatasetSession] = {}
        self._last_touch: Dict[str, float] = {}
        persistence.init_db()
        persistence.ensure_schema()

    def _touch(self, dataset_id: str) -> None:
        now = time.monotonic()
        if now - self._last_touch.get(dataset_id, -1e9) >= self.TOUCH_INTERVAL_SECONDS:
            self._last_touch[dataset_id] = now
            try:
                persistence.touch(dataset_id)
            except Exception:  # activity bookkeeping must never break a request
                logging.getLogger("sg_insight.sessions").exception("touch failed")

    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset everywhere (memory and disk). True if it existed."""
        if self.get(dataset_id) is None:
            return False
        self._sessions.pop(dataset_id, None)
        self._last_touch.pop(dataset_id, None)
        persistence.delete_session(dataset_id)
        return True

    def purge(self, ttl_hours: float = 0, max_sessions: int = 0) -> int:
        """Delete datasets idle longer than ttl_hours and, beyond max_sessions, the least recently used.
        0 disables a rule (the default), so nothing is ever removed unless an operator opts in."""
        activity = persistence.list_activity()
        doomed: list[str] = []
        if ttl_hours and ttl_hours > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max(float(ttl_hours), self.MIN_IDLE_HOURS))
            doomed += [d for d, last in activity if datetime.fromisoformat(last) < cutoff]
        remaining = [d for d, _ in activity if d not in set(doomed)]
        if max_sessions and max_sessions > 0 and len(remaining) > max_sessions:
            floor = datetime.now(timezone.utc) - timedelta(hours=self.MIN_IDLE_HOURS)
            excess = remaining[: len(remaining) - int(max_sessions)]
            doomed += [d for d in excess if datetime.fromisoformat(dict(activity)[d]) < floor]
        removed = 0
        for d in doomed:
            self._sessions.pop(d, None)
            self._last_touch.pop(d, None)
            if persistence.delete_session(d):
                removed += 1
        return removed

    def create(self, filename: str, df: pd.DataFrame, investigation_goal: Optional[str] = None) -> DatasetSession:
        dataset_id = str(uuid.uuid4())
        session = DatasetSession(
            dataset_id=dataset_id, filename=filename,
            investigation_goal=investigation_goal,
            goal_keywords=_extract_goal_keywords(investigation_goal),
            versions=[df],
        )
        self._sessions[dataset_id] = session
        try:
            session._persist(write_current_version=True)  # write-through immediately, so even an un-touched upload survives a restart
        except Exception:
            self._sessions.pop(dataset_id, None)
            raise
        return session

    def get(self, dataset_id: str) -> Optional[DatasetSession]:
        if not isinstance(dataset_id, str):
            return None
        try:
            if str(uuid.UUID(dataset_id)) != dataset_id:
                return None
        except (ValueError, TypeError, AttributeError):
            return None

        if dataset_id in self._sessions:
            self._touch(dataset_id)
            return self._sessions[dataset_id]

        loaded = persistence.load_session_data(dataset_id)
        if loaded is None:
            return None

        session = DatasetSession(
            dataset_id=dataset_id,
            filename=loaded["filename"],
            investigation_goal=loaded["investigation_goal"],
            goal_keywords=loaded["goal_keywords"],
            versions=loaded["versions"],
            log=[ActionLogEntry(**entry) for entry in loaded["log"]],
            pinned_charts=loaded["pinned_charts"],
        )
        self._sessions[dataset_id] = session
        self._touch(dataset_id)
        return session

    def require(self, dataset_id: str) -> DatasetSession:
        session = self.get(dataset_id)
        if session is None:
            raise KeyError(f"No dataset session found for id={dataset_id}")
        return session


# Single process-wide store. In-memory cache is per-process (fine for local
# dev / a single uvicorn worker); the SQLite + pickle-file layer underneath
# is what actually survives a restart. A multi-worker deployment would need
# the cache itself shared (e.g. Redis) since SQLite alone doesn't coordinate
# in-memory state across processes — out of scope for this phase.
store = SessionStore()

