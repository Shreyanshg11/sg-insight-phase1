from fastapi import APIRouter, HTTPException, Query, Depends
from app.core.session_store import store
from app.core.filtering import apply_filter, parse_filters_query
from app.models.schemas import (
    EDAOverviewResponse, DistributionResponse, CorrelationResponse,
    MissingPatternsResponse, RelationshipResponse, CategoryAnalysisResponse, TimeAnalysisResponse,
)
from app.engines.eda_engine import (
    compute_overview, compute_distributions, compute_correlation,
    compute_missing_patterns, compute_relationships, compute_category_analysis, compute_time_analysis,
)

router = APIRouter(prefix="/api/dataset", tags=["eda"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


def _filtered_df(session, filters):
    """Every EDA endpoint below is filter-aware (session 12, Interactive
    Investigation) — an active filter is a view, not a version, so this
    always recomputes against session.current rather than anything stored."""
    try:
        return apply_filter(session.current, filters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/eda/overview", response_model=EDAOverviewResponse)
async def get_eda_overview(dataset_id: str, filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    result = compute_overview(_filtered_df(session, filters))
    return EDAOverviewResponse(dataset_id=dataset_id, **result)


@router.get("/{dataset_id}/eda/distribution", response_model=DistributionResponse)
async def get_eda_distribution(dataset_id: str, column: str = Query(...), filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    try:
        result = compute_distributions(_filtered_df(session, filters), column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return DistributionResponse(dataset_id=dataset_id, **result)


@router.get("/{dataset_id}/eda/correlation", response_model=CorrelationResponse)
async def get_eda_correlation(dataset_id: str, filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    result = compute_correlation(_filtered_df(session, filters))
    return CorrelationResponse(dataset_id=dataset_id, **result)


@router.get("/{dataset_id}/eda/missing-patterns", response_model=MissingPatternsResponse)
async def get_eda_missing_patterns(dataset_id: str, filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    result = compute_missing_patterns(_filtered_df(session, filters))
    return MissingPatternsResponse(dataset_id=dataset_id, **result)


@router.get("/{dataset_id}/eda/relationships", response_model=RelationshipResponse)
async def get_eda_relationships(dataset_id: str, column_x: str = Query(...), column_y: str = Query(...),
                                 filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    try:
        result = compute_relationships(_filtered_df(session, filters), column_x, column_y)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RelationshipResponse(dataset_id=dataset_id, **result)


@router.get("/{dataset_id}/eda/category-analysis", response_model=CategoryAnalysisResponse)
async def get_eda_category_analysis(dataset_id: str, column: str = Query(...), filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    try:
        result = compute_category_analysis(_filtered_df(session, filters), column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CategoryAnalysisResponse(dataset_id=dataset_id, **result)


@router.get("/{dataset_id}/eda/time-analysis", response_model=TimeAnalysisResponse)
async def get_eda_time_analysis(dataset_id: str, column: str = Query(...), filters: list = Depends(parse_filters_query)):
    session = _get_session(dataset_id)
    try:
        result = compute_time_analysis(_filtered_df(session, filters), column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TimeAnalysisResponse(dataset_id=dataset_id, **result)
