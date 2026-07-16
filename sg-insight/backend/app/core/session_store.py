"""
In-memory dataset session store.

Phase 1 scope: simple linear undo (a list of DataFrame snapshots per dataset).
Phase 2 will replace this with real versioning (V1, V2, V3... + compare + restore),
but the interface below (get_current, push_version, undo) is written so that
swap-in later doesn't require touching the routers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import uuid


@dataclass
class ActionLogEntry:
    step: str            # e.g. "missing_value_center"
    action: str          # e.g. "impute_median"
    column: Optional[str]
    reason: str
    confidence: Optional[float]
    params: dict = field(default_factory=dict)


@dataclass
class DatasetSession:
    dataset_id: str
    filename: str
    investigation_goal: Optional[str] = None
    versions: List[pd.DataFrame] = field(default_factory=list)  # index 0 = original
    log: List[ActionLogEntry] = field(default_factory=list)

    @property
    def current(self) -> pd.DataFrame:
        return self.versions[-1]

    @property
    def version_number(self) -> int:
        return len(self.versions)  # V1 = original, V2 = first cleaning action, ...

    def push_version(self, df: pd.DataFrame, log_entry: ActionLogEntry):
        self.versions.append(df)
        self.log.append(log_entry)

    def undo(self) -> bool:
        if len(self.versions) <= 1:
            return False
        self.versions.pop()
        if self.log:
            self.log.pop()
        return True


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, DatasetSession] = {}

    def create(self, filename: str, df: pd.DataFrame, investigation_goal: Optional[str] = None) -> DatasetSession:
        dataset_id = str(uuid.uuid4())
        session = DatasetSession(dataset_id=dataset_id, filename=filename,
                                  investigation_goal=investigation_goal, versions=[df])
        self._sessions[dataset_id] = session
        return session

    def get(self, dataset_id: str) -> Optional[DatasetSession]:
        return self._sessions.get(dataset_id)

    def require(self, dataset_id: str) -> DatasetSession:
        session = self.get(dataset_id)
        if session is None:
            raise KeyError(f"No dataset session found for id={dataset_id}")
        return session


# Single process-wide store (fine for Phase 1 / local dev; swap for Redis-backed
# store or a real DB later if this needs to run multi-worker or persist restarts).
store = SessionStore()
