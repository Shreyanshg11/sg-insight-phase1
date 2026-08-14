from fastapi import APIRouter, HTTPException
from app.core.session_store import store
from app.models.schemas import ProfileResponse, QualityResponse, WarningsResponse, TrustWarningResponse
from app.engines.profiling_engine import profile_dataset
from app.engines.cleaning_engine import compute_quality_score, generate_recommendations
from app.engines.warnings_engine import generate_warnings
from app.engines.quality_engine import compute_analytical_risk, compute_readiness

router = APIRouter(prefix="/api/dataset", tags=["profiling"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.get("/{dataset_id}/profile", response_model=ProfileResponse)
async def get_profile(dataset_id: str):
    session = _get_session(dataset_id)
    return profile_dataset(dataset_id, session.current, session.investigation_goal, session.goal_keywords)


@router.get("/{dataset_id}/quality", response_model=QualityResponse)
async def get_quality(dataset_id: str):
    session = _get_session(dataset_id)
    df = session.current
    score = compute_quality_score(df)
    recs = generate_recommendations(df, session.goal_keywords)
    profile = profile_dataset(dataset_id, df, session.investigation_goal, session.goal_keywords)
    analytical_risk = compute_analytical_risk(df, profile.target_candidates)
    readiness = compute_readiness(df, profile.target_candidates)
    return QualityResponse(dataset_id=dataset_id, score=score, recommendations=recs,
                            analytical_risk=analytical_risk, readiness=readiness)


@router.get("/{dataset_id}/warnings", response_model=WarningsResponse)
async def get_warnings(dataset_id: str):
    session = _get_session(dataset_id)
    profile = profile_dataset(dataset_id, session.current, session.investigation_goal, session.goal_keywords)
    warnings = generate_warnings(session.current, profile.target_candidates)
    return WarningsResponse(
        dataset_id=dataset_id,
        warnings=[TrustWarningResponse(**w.model_dump()) for w in warnings],
    )
