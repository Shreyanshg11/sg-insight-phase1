from io import BytesIO
import re
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from app.core.session_store import store
from app.models.schemas import (
    VersionListResponse, VersionSummary, VersionCompareResponse, RestoreVersionResponse
)
from app.engines.versioning_engine import compare_versions
from app.core.proposal_store import dataset_lock

router = APIRouter(prefix="/api/dataset", tags=["versioning"])


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.get("/{dataset_id}/versions", response_model=VersionListResponse)
async def list_versions(dataset_id: str):
    session = _get_session(dataset_id)
    versions = []
    for i, df in enumerate(session.versions):
        version_num = i + 1
        if i == 0:
            versions.append(VersionSummary(
                version=1, label="Original", rows=len(df), columns=len(df.columns),
            ))
        else:
            entry = session.log[i - 1]
            versions.append(VersionSummary(
                version=version_num, label=f"V{version_num}", timestamp=entry.timestamp,
                step=entry.step, action=entry.action, column=entry.column, reason=entry.reason,
                confidence=entry.confidence, rows=len(df), columns=len(df.columns),
                result_summary=entry.result_summary,
            ))
    return VersionListResponse(dataset_id=dataset_id, current_version=session.version_number, versions=versions)


@router.get("/{dataset_id}/versions/compare", response_model=VersionCompareResponse)
async def compare_dataset_versions(dataset_id: str, a: int = Query(...), b: int = Query(...)):
    session = _get_session(dataset_id)
    if not (1 <= a <= session.version_number) or not (1 <= b <= session.version_number):
        raise HTTPException(status_code=400,
                             detail=f"Version out of range — this dataset has versions 1..{session.version_number}.")
    result = compare_versions(session.versions[a - 1], session.versions[b - 1])
    return VersionCompareResponse(dataset_id=dataset_id, version_a=a, version_b=b, **result)


@router.post("/{dataset_id}/versions/{version_number}/restore", response_model=RestoreVersionResponse)
def restore_version(dataset_id: str, version_number: int):
    with dataset_lock(dataset_id):
        session = _get_session(dataset_id)
        success = session.restore_to(version_number)
        if not success:
            raise HTTPException(status_code=400,
                                 detail=f"Invalid version {version_number} — this dataset has versions "
                                        f"1..{session.version_number}.")
        return RestoreVersionResponse(dataset_id=dataset_id, current_version=session.version_number, success=True)


@router.get("/{dataset_id}/versions/{version_number}/export")
async def export_version(dataset_id: str, version_number: int, format: str = Query("csv")):
    session = _get_session(dataset_id)
    if not (1 <= version_number <= session.version_number):
        raise HTTPException(status_code=400,
                            detail=f"Invalid version {version_number} — this dataset has versions "
                                   f"1..{session.version_number}.")
    if format not in {"csv", "xlsx"}:
        raise HTTPException(status_code=400, detail="Export format must be csv or xlsx.")
    df = session.versions[version_number - 1]
    raw_base = session.filename.rsplit(".", 1)[0] if "." in session.filename else session.filename
    base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_base).strip("._") or "dataset"
    if format == "xlsx":
        output = BytesIO()
        try:
            df.to_excel(output, index=False, engine="openpyxl")
        except Exception:
            raise HTTPException(status_code=500, detail="Could not generate the Excel export for this dataset.")
        return Response(
            content=output.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{base_name}_v{version_number}.xlsx"'},
        )
    return Response(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{base_name}_v{version_number}.csv"'},
    )
