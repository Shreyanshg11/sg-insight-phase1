import io

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import pandas as pd

from app.core.session_store import store
from app.core.rate_limit import limiter
from app.models.schemas import UploadResponse

router = APIRouter(prefix="/api/dataset", tags=["upload"])

MAX_FILE_MB = 50
CHUNK_SIZE = 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    investigation_goal: str | None = Form(None),
):
    max_bytes = MAX_FILE_MB * 1024 * 1024
    contents = bytearray()

    while True:
        remaining = max_bytes - len(contents)
        chunk = await file.read(min(CHUNK_SIZE, remaining + 1))

        if not chunk:
            break

        contents.extend(chunk)

        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {MAX_FILE_MB}MB limit",
            )

    filename = file.filename or "dataset"

    try:
        file_buffer = io.BytesIO(contents)

        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_buffer)
        else:
            df = pd.read_csv(file_buffer)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not parse uploaded file. Please verify the file format and contents.",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded dataset has no rows",
        )

    session = store.create(
        filename=filename,
        df=df,
        investigation_goal=investigation_goal,
    )

    return UploadResponse(
        dataset_id=session.dataset_id,
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        version=session.version_number,
        investigation_goal=session.investigation_goal,
    )
