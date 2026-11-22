import io
import csv
import re
import zipfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import pandas as pd

from app.core.session_store import store
from app.core.persistence import StorageCapacityError
from app.core.rate_limit import limiter
from app.models.schemas import UploadResponse

router = APIRouter(prefix="/api/dataset", tags=["upload"])

MAX_FILE_MB = 50
CHUNK_SIZE = 1024 * 1024
MAX_ROWS = 1_000_000
MAX_COLUMNS = 500
MAX_XLSX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_FILENAME_LENGTH = 128
MAX_DATAFRAME_MEMORY_BYTES = 512 * 1024 * 1024


def _safe_filename(value: str | None) -> str:
    # Filenames are metadata only; discard both POSIX and Windows path parts.
    leaf = (value or "dataset").replace("\\", "/").split("/")[-1]
    leaf = re.sub(r"[\x00-\x1f\x7f]", "", leaf).strip(" .")
    return (leaf[:MAX_FILENAME_LENGTH] or "dataset")


def _validate_file_content(contents: bytes, suffix: str) -> None:
    if suffix == "csv":
        if b"\x00" in contents:
            raise HTTPException(status_code=400, detail="CSV content is not valid text.")
        try:
            sample = contents[:8192].decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV must use UTF-8 text encoding.")
        if not sample.strip():
            raise HTTPException(status_code=400, detail="CSV file contains no readable text.")
        if sum(ord(char) < 32 and char not in "\r\n\t" for char in sample) > max(2, len(sample) // 100):
            raise HTTPException(status_code=400, detail="CSV content contains unsupported binary control characters.")
    elif suffix == "xlsx":
        if not contents.startswith(b"PK\x03\x04"):
            raise HTTPException(status_code=400, detail="File content is not an XLSX workbook.")
    elif suffix == "xls" and not contents.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise HTTPException(status_code=400, detail="File content is not a legacy XLS workbook.")


def _preflight_csv(contents: bytes) -> None:
    try:
        text = contents.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""))
        rows = 0
        max_columns = 0
        for row in reader:
            if not row:
                continue
            rows += 1
            max_columns = max(max_columns, len(row))
            if rows > MAX_ROWS + 1:
                raise HTTPException(status_code=413, detail=f"Dataset exceeds the {MAX_ROWS:,}-row processing limit.")
            if max_columns > MAX_COLUMNS:
                raise HTTPException(status_code=413, detail=f"Dataset exceeds the {MAX_COLUMNS}-column processing limit.")
        estimated_bytes = max(0, rows - 1) * max_columns * 48
        if estimated_bytes > MAX_DATAFRAME_MEMORY_BYTES:
            raise HTTPException(status_code=413, detail="CSV dimensions exceed the 512MB in-memory processing limit.")
    except csv.Error:
        raise HTTPException(status_code=400, detail="CSV content is malformed or contains an oversized field.")


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

    filename = _safe_filename(file.filename)
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"csv", "xlsx", "xls"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a CSV or Excel workbook (.xlsx or .xls).")

    if suffix == "xlsx":
        try:
            with zipfile.ZipFile(io.BytesIO(contents)) as workbook:
                expanded_size = sum(info.file_size for info in workbook.infolist())
                if expanded_size > MAX_XLSX_UNCOMPRESSED_BYTES:
                    raise HTTPException(status_code=413, detail="Excel workbook expands beyond the 250MB processing limit.")
                if len(workbook.infolist()) > 10_000:
                    raise HTTPException(status_code=400, detail="Excel workbook contains too many internal files to process safely.")
                entries = set(workbook.namelist())
                if "[Content_Types].xml" not in entries or "xl/workbook.xml" not in entries:
                    raise HTTPException(status_code=400, detail="File content is not a valid XLSX workbook.")
                if any(name.lower().endswith("vbaproject.bin") for name in entries):
                    raise HTTPException(status_code=400, detail="Macro-enabled Excel workbooks are not supported.")
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Could not parse uploaded file. Please verify the file format and contents.")

    _validate_file_content(bytes(contents), suffix)
    if suffix == "csv":
        _preflight_csv(bytes(contents))
    if investigation_goal and len(investigation_goal) > 2000:
        raise HTTPException(status_code=422, detail="Investigation goal must be 2,000 characters or fewer.")

    try:
        file_buffer = io.BytesIO(contents)

        if suffix in {"xlsx", "xls"}:
            df = pd.read_excel(file_buffer, nrows=MAX_ROWS + 1)
        else:
            df = pd.read_csv(file_buffer, nrows=MAX_ROWS + 1, low_memory=False)

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

    if len(df) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Dataset exceeds the {MAX_ROWS:,}-row processing limit.")
    if len(df.columns) > MAX_COLUMNS:
        raise HTTPException(status_code=413, detail=f"Dataset exceeds the {MAX_COLUMNS}-column processing limit.")
    if int(df.memory_usage(index=True, deep=True).sum()) > MAX_DATAFRAME_MEMORY_BYTES:
        raise HTTPException(status_code=413, detail="Dataset exceeds the 512MB in-memory processing limit.")

    try:
        session = store.create(
            filename=filename,
            df=df,
            investigation_goal=investigation_goal,
        )
    except StorageCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc

    return UploadResponse(
        dataset_id=session.dataset_id,
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        version=session.version_number,
        investigation_goal=session.investigation_goal,
    )
