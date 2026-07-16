from fastapi import APIRouter, HTTPException
from app.core.session_store import store
from app.models.schemas import HealthDashboardResponse, HealthCategoryResponse
from app.engines.health_engine import compute_health_dashboard

router = APIRouter(prefix="/api/dataset", tags=["health"])


@router.get("/{dataset_id}/health", response_model=HealthDashboardResponse)
async def get_health_dashboard(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    categories = compute_health_dashboard(session.current)
    return HealthDashboardResponse(
        dataset_id=dataset_id,
        categories=[HealthCategoryResponse(**c.model_dump()) for c in categories],
    )
