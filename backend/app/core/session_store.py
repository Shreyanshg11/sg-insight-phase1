"""
Dataset session store.

Kept in-memory as a plain dict of DatasetSession objects for the request
hot-path (no serialization on every read), backed by app.core.persistence
underneath: every mutation (push_version, undo, restore_to, pinned chart
add/remove) writes straight through to disk, and a session missing from the
in-memory cache is lazily loaded from disk on first access — so restarting
the backend process doesn't lose datasets, only the in-memory cache of them.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import re
import pandas as pd
import uuid

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

    def _persist(self):
        persistence.save_session(self)

    @property
    def current(self) -> pd.DataFrame:
        return self.versions[-1]

    @property
    def version_number(self) -> int:
        return len(self.versions)  # V1 = original, V2 = first cleaning action, ...

    def push_version(self, df: pd.DataFrame, log_entry: ActionLogEntry):
        self.versions.append(df)
        self.log.append(log_entry)
        self._persist()

    def undo(self) -> bool:
        if len(self.versions) <= 1:
            return False
        self.versions.pop()
        if self.log:
            self.log.pop()
        self._persist()
        return True

    def restore_to(self, version_number: int) -> bool:
        """Jumps directly to an earlier version (#10 Dataset Versioning).
        Still linear, not branching: restoring to V2 while at V5 discards
        V3-V5 permanently, same as repeated undo would, just in one step.
        Full branching version trees are explicitly out of scope for this
        phase per the brief."""
        if version_number < 1 or version_number > self.version_number:
            return False
        self.versions = self.versions[:version_number]
        self.log = self.log[:version_number - 1]
        self._persist()
        return True

    def add_pinned_chart(self, spec: dict):
        self.pinned_charts.append(spec)
        self._persist()

    def remove_pinned_chart(self, chart_id: str):
        self.pinned_charts = [c for c in self.pinned_charts if c["id"] != chart_id]
        self._persist()

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
    def __init__(self):
        self._sessions: Dict[str, DatasetSession] = {}
        persistence.init_db()

    def create(self, filename: str, df: pd.DataFrame, investigation_goal: Optional[str] = None) -> DatasetSession:
        dataset_id = str(uuid.uuid4())
        session = DatasetSession(
            dataset_id=dataset_id, filename=filename,
            investigation_goal=investigation_goal,
            goal_keywords=_extract_goal_keywords(investigation_goal),
            versions=[df],
        )
        self._sessions[dataset_id] = session
        session._persist()  # write-through immediately, so even an un-touched upload survives a restart
        return session

    def get(self, dataset_id: str) -> Optional[DatasetSession]:
        if dataset_id in self._sessions:
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

