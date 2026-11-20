from fastapi import APIRouter, HTTPException, Request
from app.core.session_store import store
from app.core.rate_limit import limiter
from app.models.schemas import (
    AnalysisRecommendationResponse, RunAnalysisRequest, RunAnalysisResponse,
    AIInsightRequest, AIInsightResponse, DatasetIntelligenceResponse,
)
from app.engines.analysis_engine import recommend_analyses, run_analysis
from app.engines.ai_engine import generate_insight, generate_dataset_intelligence
from app.engines.eda_engine import profile_dataset

router = APIRouter(prefix="/api/dataset", tags=["analysis"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.get("/{dataset_id}/analysis-recommendations", response_model=AnalysisRecommendationResponse)
async def get_analysis_recommendations(dataset_id: str):
    session = _get_session(dataset_id)
    recs = recommend_analyses(session.current, session.goal_keywords)
    return AnalysisRecommendationResponse(dataset_id=dataset_id, recommendations=recs)


@router.post("/run-analysis", response_model=RunAnalysisResponse)
@limiter.limit("20/minute")
async def run_analysis_endpoint(request: Request, req: RunAnalysisRequest):
    """
    Read-only: computes and returns real results but never touches the
    dataset or the version/undo stack (unlike cleaning/feature-engineering
    actions, which mutate the dataset and therefore require confirmation
    first). Running an analysis doesn't change data, so there's nothing to
    confirm here — the confirm-before-apply gate applies to the AI Insight
    step next, since that's the point where an AI-authored claim gets shown
    to the user.
    """
    session = _get_session(req.dataset_id)
    try:
        outcome = run_analysis(session.current, req.analysis_type, req.columns, req.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RunAnalysisResponse(
        dataset_id=req.dataset_id, analysis_type=req.analysis_type, columns=req.columns,
        result=outcome["result"], summary=outcome["summary"], limitations=outcome["limitations"],
    )


@router.post("/ai-insight", response_model=AIInsightResponse)
@limiter.limit("5/minute")
async def ai_insight_endpoint(request: Request, req: AIInsightRequest):
    session = _get_session(req.dataset_id)
    goal = req.investigation_goal or session.investigation_goal
    try:
        insight = await generate_insight(req.analysis_type, req.columns, req.result, req.deterministic_summary, goal)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return AIInsightResponse(dataset_id=req.dataset_id, **insight)


@router.post(
    "/{dataset_id}/dataset-intelligence",
    response_model=DatasetIntelligenceResponse,
)
@limiter.limit("3/minute")
async def dataset_intelligence_endpoint(request: Request, dataset_id: str):
    """Profile the current dataset and generate grounded AI interpretation.

    This endpoint is read-only. It does not modify the dataset or its history.
    """
    session = _get_session(dataset_id)

    try:
        profile = profile_dataset(session.current)
        from app.engines.business_insight_engine import build_business_insights

        business_insights = build_business_insights(session.current)
        insight = await generate_dataset_intelligence(
            profile,
            session.investigation_goal,
            business_insights,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return DatasetIntelligenceResponse(
        dataset_id=dataset_id,
        dataset_profile=profile,
        **insight,
    )
