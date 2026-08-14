from fastapi import APIRouter, HTTPException
from app.core.session_store import store
from app.core.filtering import apply_filter, rows_to_json_safe_records
from app.models.schemas import FilteredRowsRequest, FilteredRowsResponse, BuildChartRequest
from app.engines.visualization_engine import build_chart

router = APIRouter(prefix="/api/dataset", tags=["filtering"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.post("/filtered-rows", response_model=FilteredRowsResponse)
async def get_filtered_rows(req: FilteredRowsRequest):
    """
    The drill-down result: given the active filter (set by hand or by
    clicking a chart element), return the actual matching rows as a real
    paginated table — not a chart, the underlying records themselves.
    """
    session = _get_session(req.dataset_id)
    try:
        filtered = apply_filter(session.current, req.filters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start = (req.page - 1) * req.page_size
    page_df = filtered.iloc[start:start + req.page_size]
    return FilteredRowsResponse(
        dataset_id=req.dataset_id, total_rows=len(session.current), total_matching=len(filtered),
        page=req.page, page_size=req.page_size,
        columns=list(session.current.columns), rows=rows_to_json_safe_records(page_df),
    )


@router.post("/build-chart", response_model=dict)
async def build_chart_endpoint(req: BuildChartRequest):
    """
    Ad-hoc chart preview against the active filter — deliberately separate
    from Dashboard pinning (POST /dashboard/pin): a pinned chart always
    recomputes off session.current unfiltered, by design, so it never goes
    stale in a way the user didn't cause. This endpoint is for browsing a
    filtered view before deciding whether anything's worth pinning.
    """
    session = _get_session(req.dataset_id)
    try:
        filtered = apply_filter(session.current, req.filters)
        result = build_chart(filtered, req.chart_type, req.columns)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"dataset_id": req.dataset_id, "chart_type": req.chart_type, "columns": req.columns, **result}
