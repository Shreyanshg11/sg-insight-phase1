from fastapi import APIRouter, HTTPException
from app.core.session_store import store
from app.models.schemas import ProfileResponse, QualityResponse
from app.services.profiling_service import profile_dataset
from app.services.cleaning_service import compute_quality_score, generate_recommendations

router = APIRouter(prefix="/api/dataset", tags=["profiling"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.get("/{dataset_id}/profile", response_model=ProfileResponse)
async def get_profile(dataset_id: str):
    session = _get_session(dataset_id)
    return profile_dataset(dataset_id, session.current)


@router.get("/{dataset_id}/quality", response_model=QualityResponse)
async def get_quality(dataset_id: str):
    session = _get_session(dataset_id)
    df = session.current
    score = compute_quality_score(df)
    recs = generate_recommendations(df)
    return QualityResponse(dataset_id=dataset_id, score=score, recommendations=recs)
