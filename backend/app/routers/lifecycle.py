from fastapi import APIRouter, HTTPException, Request

from app.core.rate_limit import limiter
from app.core.session_store import store

# No auth code here on purpose: every /api/ route except the explicit allow-list in
# app.main.require_operator_session is behind the signed-session middleware.
router = APIRouter(prefix="/api/dataset", tags=["lifecycle"])


@router.delete("/{dataset_id}")
@limiter.limit("30/minute")
def delete_dataset(request: Request, dataset_id: str):
    """Permanently delete a dataset: every version, the action log, proposals and recommendation history."""
    if not store.delete(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"deleted": True, "dataset_id": dataset_id}
