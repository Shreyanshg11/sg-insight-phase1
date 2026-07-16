from pydantic import BaseModel
from typing import Optional, List, Any, Dict


# ---------- Upload ----------

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    version: int
    investigation_goal: Optional[str] = None


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
    goal_relevant: bool = False  # True if column name matches a keyword from the investigation goal


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
    investigation_goal: Optional[str] = None
    goal_relevant_columns: List[str] = []


# ---------- Data Quality ----------

class QualityBreakdownItem(BaseModel):
    label: str            # e.g. "Missing Values"
    points_deducted: float
    reason: str            # explains exactly how the deduction was computed


class QualityScore(BaseModel):
    overall: float
    completeness: float
    consistency: float
    validity: float
    uniqueness: float
    breakdown: List[QualityBreakdownItem] = []  # additive deductions that explain `overall`


class CleaningRecommendation(BaseModel):
    id: str
    step: str                 # e.g. "missing_value_center", "outlier_center", "duplicate_manager"
    column: Optional[str]
    issue: str
    recommended_action: str
    reason: str
    confidence: float         # 0-1, must map to an explainable heuristic (see services/)
    preview: Dict[str, Any]   # small before/after style preview payload
    how_detected: str = ""            # the mechanical rule/check that surfaced this
    risks: List[str] = []             # what could go wrong if this action is applied
    alternative_actions: List[str] = []  # other valid choices, so "apply" isn't the only path shown
    goal_relevant: bool = False       # True if this recommendation touches a goal-relevant column


class QualityResponse(BaseModel):
    dataset_id: str
    score: QualityScore
    recommendations: List[CleaningRecommendation]


# ---------- Trust Warnings ----------

class TrustWarningResponse(BaseModel):
    id: str
    severity: str
    title: str
    description: str
    reason: str
    affected_columns: List[str] = []


class WarningsResponse(BaseModel):
    dataset_id: str
    warnings: List[TrustWarningResponse]


# ---------- Health Dashboard ----------

class HealthCategoryResponse(BaseModel):
    key: str
    label: str
    status: str
    summary: str
    detail: str
    affected_columns: List[str] = []


class HealthDashboardResponse(BaseModel):
    dataset_id: str
    categories: List[HealthCategoryResponse]


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
    timestamp: str
    rows_affected: Optional[int] = None
    result_summary: Optional[str] = None


class HistoryResponse(BaseModel):
    dataset_id: str
    current_version: int
    history: List[HistoryEntry]


# ---------- Compare Methods (#6) ----------

class MethodComparisonResult(BaseModel):
    method: str
    fill_value: Optional[Any] = None
    after_stats: Dict[str, Any]
    rows_affected: int


class CompareMethodsResponse(BaseModel):
    column: str
    missing_count: int
    before_stats: Dict[str, Any]
    methods: List[MethodComparisonResult]


# ---------- Dataset Versioning (#10) ----------

class VersionSummary(BaseModel):
    version: int
    label: str
    timestamp: Optional[str] = None
    step: Optional[str] = None
    action: Optional[str] = None
    column: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    rows: int
    columns: int
    result_summary: Optional[str] = None


class VersionListResponse(BaseModel):
    dataset_id: str
    current_version: int
    versions: List[VersionSummary]


class ColumnChange(BaseModel):
    column: str
    cells_changed: int
    samples: List[Dict[str, Any]]


class VersionCompareResponse(BaseModel):
    dataset_id: str
    version_a: int
    version_b: int
    rows_a: int
    rows_b: int
    row_delta: int
    columns_a: int
    columns_b: int
    columns_added: List[str]
    columns_removed: List[str]
    column_changes: List[ColumnChange]
    compared_rows: int
    note: str


class RestoreVersionResponse(BaseModel):
    dataset_id: str
    current_version: int
    success: bool


# ---------- Feature Engineering Center (#12) ----------

class FeatureRecommendation(BaseModel):
    id: str
    category: str              # ratio / date_features / bucketing / scaling_normalization / encoding / log_transform
    columns: List[str]         # one or more source columns involved
    new_column_name: str
    issue: str
    recommended_action: str
    reason: str
    confidence: float
    preview: Dict[str, Any]
    how_detected: str = ""
    risks: List[str] = []
    alternative_actions: List[str] = []
    goal_relevant: bool = False


class FeatureEngineeringResponse(BaseModel):
    dataset_id: str
    recommendations: List[FeatureRecommendation]


class ApplyFeatureActionRequest(BaseModel):
    dataset_id: str
    category: str
    action: str
    columns: List[str]
    new_column_name: Optional[str] = None
    params: Dict[str, Any] = {}
    reason: str = ""
    confidence: Optional[float] = None


class ApplyFeatureActionResponse(BaseModel):
    dataset_id: str
    version: int
    rows: int
    columns: int
    preview: Dict[str, Any]


# ---------- EDA module navigation (#13) ----------

class EDAOverviewResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]
    summary: List[Dict[str, Any]]  # one quick-stats row per column


class DistributionBin(BaseModel):
    label: str
    count: int


class DistributionResponse(BaseModel):
    dataset_id: str
    column: str
    kind: str          # "numeric" | "categorical"
    stats: Dict[str, Any]
    bins: List[DistributionBin]
    note: str = ""


class CorrelationPair(BaseModel):
    column_a: str
    column_b: str
    correlation: float
    strength: str      # weak / moderate / strong


class CorrelationResponse(BaseModel):
    dataset_id: str
    columns: List[str]
    matrix: List[List[Optional[float]]]
    top_pairs: List[CorrelationPair]
    note: str


class MissingPatternsResponse(BaseModel):
    dataset_id: str
    columns_with_missing: List[str]
    co_missing_pairs: List[Dict[str, Any]]
    missing_by_row_bucket: List[Dict[str, Any]]
    note: str


class RelationshipResponse(BaseModel):
    dataset_id: str
    column_x: str
    column_y: str
    kind: str           # numeric_numeric / numeric_categorical / categorical_categorical
    points: List[Dict[str, Any]]
    stats: Dict[str, Any]
    note: str


class CategoryAnalysisResponse(BaseModel):
    dataset_id: str
    column: str
    categories: List[Dict[str, Any]]
    unique_count: int
    top_n_coverage_pct: float
    note: str = ""


class TimeAnalysisResponse(BaseModel):
    dataset_id: str
    column: str
    granularity: str
    buckets: List[Dict[str, Any]]
    note: str = ""


# ---------- Analysis Recommendation Engine (#14) ----------

class AnalysisRecommendation(BaseModel):
    id: str
    analysis_type: str          # trend_analysis / regression / clustering / forecasting / hypothesis_test / pca / anomaly_detection
    title: str
    columns: List[str]          # candidate/relevant columns (final selection happens at run-analysis time)
    recommended: bool           # False entries are shown too - an honest "considered, not recommended" record
    reason: str
    confidence: float           # 0 when recommended=False
    required_characteristics: str
    not_recommended_because: Optional[str] = None
    how_detected: str = ""
    risks: List[str] = []
    alternative_actions: List[str] = []
    goal_relevant: bool = False
    preview: Dict[str, Any] = {}


class AnalysisRecommendationResponse(BaseModel):
    dataset_id: str
    recommendations: List[AnalysisRecommendation]


class RunAnalysisRequest(BaseModel):
    dataset_id: str
    analysis_type: str
    columns: List[str] = []
    params: Dict[str, Any] = {}


class RunAnalysisResponse(BaseModel):
    dataset_id: str
    analysis_type: str
    columns: List[str]
    result: Dict[str, Any]
    summary: str                # deterministic, non-AI summary computed directly from `result`
    limitations: List[str] = []


# ---------- AI Insight (#15) ----------

class AIInsightRequest(BaseModel):
    dataset_id: str
    analysis_type: str
    columns: List[str] = []
    result: Dict[str, Any]
    deterministic_summary: str = ""
    investigation_goal: Optional[str] = None


class UnverifiedFigure(BaseModel):
    figure: str
    reason: str


class AIInsightResponse(BaseModel):
    dataset_id: str
    observation: str
    evidence: str
    business_meaning: str
    possible_explanations: List[str]
    limitations: List[str]
    confidence: float             # the AI's own stated confidence - still just a claim, not verified
    unverified_figures: List[UnverifiedFigure] = []
    grounded: bool                # True only if every number in the AI's text traces back to `result`
    model: str

# ---------- Visualization + Dashboard Builder (#16 / #17) ----------

class ChartRecommendation(BaseModel):
    id: str
    chart_type: str          # histogram / bar_chart / line_chart / scatter_plot / box_plot / heatmap
    columns: List[str]
    title: str
    reason: str
    confidence: float
    how_detected: str = ""
    alternative_chart_types: List[str] = []
    goal_relevant: bool = False
    data: Dict[str, Any]              # ready-to-render payload; shape depends on chart_type
    summary_stats: Dict[str, Any] = {}
    outlier_indices: List[int] = []   # dataset row indices highlighted as outliers (histogram/scatter)
    trend: Optional[Dict[str, Any]] = None  # slope/direction/r_squared, line_chart only


class ChartRecommendationResponse(BaseModel):
    dataset_id: str
    recommendations: List[ChartRecommendation]


class ExplainChartRequest(BaseModel):
    dataset_id: str
    chart_type: str
    columns: List[str] = []


class ExplainChartResponse(BaseModel):
    dataset_id: str
    explanation: str


class PinChartRequest(BaseModel):
    dataset_id: str
    chart_type: str
    columns: List[str] = []
    title: str


class UnpinChartRequest(BaseModel):
    dataset_id: str
    chart_id: str


class DashboardChart(BaseModel):
    id: str
    chart_type: str
    columns: List[str]
    title: str
    pinned_at_version: int
    stale: bool               # True if the dataset has changed version since this was pinned
    data: Dict[str, Any]
    summary_stats: Dict[str, Any] = {}
    outlier_indices: List[int] = []
    trend: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # set instead of data if the pinned columns no longer exist/compute


class DashboardResponse(BaseModel):
    dataset_id: str
    current_version: int
    charts: List[DashboardChart]

# ---------- Report Generation (final pipeline stage) ----------

class ReportAnalysisInclude(BaseModel):
    """One analysis the user actually ran and chose to include in the report -
    the report never invents or re-runs an analysis on its own; it only
    composes what's handed to it here, same result/summary/limitations shape
    RunAnalysisResponse already returns, plus an optional already-generated
    AI insight (with its own grounding status carried through untouched)."""
    analysis_type: str
    columns: List[str] = []
    result: Dict[str, Any]
    summary: str
    limitations: List[str] = []
    ai_insight: Optional[Dict[str, Any]] = None


class GenerateReportRequest(BaseModel):
    dataset_id: str
    format: str = "markdown"    # markdown / html / pdf / notebook
    included_analyses: List[ReportAnalysisInclude] = []
