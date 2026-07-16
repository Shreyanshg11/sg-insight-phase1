from pydantic import BaseModel
from typing import Optional, List, Any, Dict


# ---------- Upload ----------

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    version: int


# ---------- Profiling ----------

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_role: str          # id / target_candidate / date / geographic / categorical / numeric / text / unknown
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    sample_values: List[Any]


class ProfileResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    memory_kb: float
    duplicate_rows: int
    duplicate_pct: float
    id_column_candidates: List[str]
    target_candidates: List[str]
    date_columns: List[str]
    geographic_columns: List[str]
    columns_detail: List[ColumnProfile]


# ---------- Data Quality ----------

class QualityScore(BaseModel):
    overall: float
    completeness: float
    consistency: float
    validity: float
    uniqueness: float


class CleaningRecommendation(BaseModel):
    id: str
    step: str                 # e.g. "missing_value_center", "outlier_center", "duplicate_manager"
    column: Optional[str]
    issue: str
    recommended_action: str
    reason: str
    confidence: float         # 0-1, must map to an explainable heuristic (see services/)
    preview: Dict[str, Any]   # small before/after style preview payload


class QualityResponse(BaseModel):
    dataset_id: str
    score: QualityScore
    recommendations: List[CleaningRecommendation]


# ---------- Cleaning actions ----------

class ApplyActionRequest(BaseModel):
    dataset_id: str
    step: str
    action: str
    column: Optional[str] = None
    params: Dict[str, Any] = {}
    reason: str = ""
    confidence: Optional[float] = None


class ApplyActionResponse(BaseModel):
    dataset_id: str
    version: int
    rows: int
    columns: int
    preview: Dict[str, Any]


class UndoResponse(BaseModel):
    dataset_id: str
    version: int
    success: bool


class HistoryEntry(BaseModel):
    version: int
    step: str
    action: str
    column: Optional[str]
    reason: str
    confidence: Optional[float]


class HistoryResponse(BaseModel):
    dataset_id: str
    current_version: int
    history: List[HistoryEntry]
