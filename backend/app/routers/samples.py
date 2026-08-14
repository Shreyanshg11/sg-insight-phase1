import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from app.core.session_store import store
from app.sample_data.registry import SAMPLES, SAMPLES_BY_ID, get_sample_path, get_example_report_path

router = APIRouter(prefix="/api/dataset/samples", tags=["samples"])


@router.get("")
async def list_samples():
    return {"samples": SAMPLES}


@router.post("/{sample_id}/load")
async def load_sample(sample_id: str):
    if sample_id not in SAMPLES_BY_ID:
        raise HTTPException(status_code=404, detail=f"No sample '{sample_id}'")
    meta = SAMPLES_BY_ID[sample_id]
    df = pd.read_csv(get_sample_path(sample_id))
    session = store.create(filename=meta["filename"], df=df, investigation_goal=meta["investigation_goal"])
    return {
        "dataset_id": session.dataset_id, "filename": session.filename,
        "rows": len(df), "columns": len(df.columns), "version": 1,
        "investigation_goal": session.investigation_goal,
    }


@router.get("/{sample_id}/example-report", response_class=HTMLResponse)
async def example_report(sample_id: str):
    if sample_id not in SAMPLES_BY_ID:
        raise HTTPException(status_code=404, detail=f"No sample '{sample_id}'")
    path = get_example_report_path(sample_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example report not generated yet.")
    return path.read_text()
