import uuid
from fastapi import APIRouter, HTTPException
from app.core.session_store import store
from app.models.schemas import (
    ChartRecommendationResponse, ExplainChartRequest, ExplainChartResponse,
    PinChartRequest, UnpinChartRequest, DashboardResponse, DashboardChart,
)
from app.engines.visualization_engine import recommend_charts, explain_chart, build_chart

router = APIRouter(prefix="/api/dataset", tags=["visualization"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.get("/{dataset_id}/chart-recommendations", response_model=ChartRecommendationResponse)
async def get_chart_recommendations(dataset_id: str):
    session = _get_session(dataset_id)
    recs = recommend_charts(session.current, session.goal_keywords)
    return ChartRecommendationResponse(dataset_id=dataset_id, recommendations=recs)


@router.post("/explain-chart", response_model=ExplainChartResponse)
async def explain_chart_endpoint(req: ExplainChartRequest):
    session = _get_session(req.dataset_id)
    try:
        explanation = explain_chart(session.current, req.chart_type, req.columns)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExplainChartResponse(dataset_id=req.dataset_id, explanation=explanation)


@router.post("/dashboard/pin", response_model=DashboardResponse)
async def pin_chart(req: PinChartRequest):
    """
    Stores the chart SPEC (chart_type + columns + title), not its data — the
    dashboard always recomputes live off the dataset's current version (see
    GET /{id}/dashboard), so a pinned chart never silently goes stale.
    """
    session = _get_session(req.dataset_id)
    try:
        build_chart(session.current, req.chart_type, req.columns)  # validate it actually computes
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    session.add_pinned_chart({
        "id": str(uuid.uuid4()), "chart_type": req.chart_type, "columns": req.columns,
        "title": req.title, "pinned_at_version": session.version_number,
    })
    return await get_dashboard(req.dataset_id)


@router.post("/dashboard/unpin", response_model=DashboardResponse)
async def unpin_chart(req: UnpinChartRequest):
    session = _get_session(req.dataset_id)
    session.remove_pinned_chart(req.chart_id)
    return await get_dashboard(req.dataset_id)


@router.get("/{dataset_id}/dashboard", response_model=DashboardResponse)
async def get_dashboard(dataset_id: str):
    session = _get_session(dataset_id)
    charts = []
    for spec in session.pinned_charts:
        stale = spec["pinned_at_version"] != session.version_number
        try:
            built = build_chart(session.current, spec["chart_type"], spec["columns"])
            charts.append(DashboardChart(
                id=spec["id"], chart_type=spec["chart_type"], columns=spec["columns"], title=spec["title"],
                pinned_at_version=spec["pinned_at_version"], stale=stale, **built,
            ))
        except (ValueError, KeyError) as e:
            # A pinned column was removed/renamed by a later cleaning/feature-engineering
            # action — surface that plainly rather than silently dropping the chart.
            charts.append(DashboardChart(
                id=spec["id"], chart_type=spec["chart_type"], columns=spec["columns"], title=spec["title"],
                pinned_at_version=spec["pinned_at_version"], stale=stale, data={}, error=str(e),
            ))
    return DashboardResponse(dataset_id=dataset_id, current_version=session.version_number, charts=charts)
