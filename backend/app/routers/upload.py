import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
from app.core.session_store import store
from app.models.schemas import UploadResponse

router = APIRouter(prefix="/api/dataset", tags=["upload"])

MAX_FILE_MB = 50


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), investigation_goal: str | None = Form(None)):
    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_MB}MB limit")

    filename = file.filename or "dataset"
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset has no rows")

    session = store.create(filename=filename, df=df, investigation_goal=investigation_goal)

    return UploadResponse(
        dataset_id=session.dataset_id,
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        version=session.version_number,
        investigation_goal=session.investigation_goal,
    )
