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


class ScoreBreakdownItem(BaseModel):
    label: str
    points: float
    reason: str


class AnalyticalRiskScore(BaseModel):
    """0 = no risk signals found, 100 = severe. Packages warnings_engine's
    output as a score instead of only a list."""
    overall: float
    breakdown: List[ScoreBreakdownItem] = []
    warning_count: int = 0
    critical_count: int = 0


class ReadinessScore(BaseModel):
    """0 = not structurally ready for the analyses this app offers, 100 =
    fully ready. Independent of Data Quality — a small but spotless dataset
    can still score low here if it's too small for clustering/PCA/trend
    analysis to be meaningful."""
    overall: float
    breakdown: List[ScoreBreakdownItem] = []


class QualityResponse(BaseModel):
    dataset_id: str
    score: QualityScore
    recommendations: List[CleaningRecommendation]
    # Additive fields — the composite "three constructs" view (data quality /
    # analytical risk / readiness). Optional so any existing consumer reading
    # only `score` is unaffected; new consumers can read these directly off
    # the same response instead of a second endpoint.
    analytical_risk: Optional[AnalyticalRiskScore] = None
    readiness: Optional[ReadinessScore] = None


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
    # Value classification (additive): how worthwhile the suggested feature
    # actually looks, distinct from confidence (confidence = "are we sure the
    # detection rule fired correctly"; value_tier = "is the resulting feature
    # actually likely to be useful"). One of: high_value, situational, risky,
    # redundant. value_tier_reason states the concrete check behind it.
    value_tier: str = "situational"
    value_tier_reason: str = ""


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


# ---------- Interactive filtering / drill-down ----------
# A filter is a view, never a dataset version — applying one never pushes to
# session.log or session.versions. It's recomputed on the fly against
# session.current every time, same "always live off the current version"
# principle the Dashboard's pinned charts already follow.

class FilterCondition(BaseModel):
    column: str
    op: str          # "eq" | "in" | "gte" | "lte" | "between"
    value: Any        # scalar for eq/gte/lte, list for in/between


class FilteredRowsRequest(BaseModel):
    dataset_id: str
    filters: List[FilterCondition] = []
    page: int = 1
    page_size: int = 25


class FilteredRowsResponse(BaseModel):
    dataset_id: str
    total_rows: int          # rows in the dataset overall (unfiltered)
    total_matching: int      # rows matching the filter
    page: int
    page_size: int
    columns: List[str]
    rows: List[Dict[str, Any]]


class BuildChartRequest(BaseModel):
    dataset_id: str
    chart_type: str
    columns: List[str] = []
    filters: List[FilterCondition] = []


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
    section_order: Optional[List[str]] = None   # session 13: which sections, and in what order — None = every section, default order


# ---------- Smart Report Builder (session 13) ----------
# AI curation in front of report_engine.py's renderers — decides WHICH
# sections/analyses go in, never renders anything itself.

class AvailableAnalysisSummary(BaseModel):
    """One already-completed analysis the builder can choose to include —
    the backend has no server-side memory of 'analyses run this session'
    (that's frontend UI state), so the caller passes these in rather than
    the engine fetching them via a tool."""
    index: int
    analysis_type: str
    columns: List[str] = []
    summary: str


class DraftReportPlanRequest(BaseModel):
    dataset_id: str
    available_analyses: List[AvailableAnalysisSummary] = []
    user_request: Optional[str] = None   # e.g. "focus this on regional performance"


class ReportSectionPlan(BaseModel):
    key: str
    label: str
    include: bool
    reason: str


class DraftReportPlanResponse(BaseModel):
    rationale: str
    sections: List[ReportSectionPlan]
    selected_analysis_indices: List[int] = []
    unverified_figures: List[Dict[str, Any]] = []
    grounded: bool = True


# ---------- Copilot (conversational layer, never mutates directly) ----------

class CopilotChatMessage(BaseModel):
    role: str            # "user" | "assistant"
    content: str


class CopilotMessageRequest(BaseModel):
    dataset_id: str
    message: str
    history: List[CopilotChatMessage] = []


class CopilotMessageResponse(BaseModel):
    reply: str
    proposed_action: Optional[Dict[str, Any]] = None
    has_more_suggestions: bool = False
    unverified_figures: List[Dict[str, Any]] = []
    grounded: bool = True


class CopilotConfirmActionRequest(BaseModel):
    dataset_id: str
    proposed_action: Dict[str, Any]


class CopilotConfirmActionResponse(BaseModel):
    dataset_id: str
    version: int
    rows: int
    columns: int
    result_summary: str
    has_more_suggestions: bool = False
