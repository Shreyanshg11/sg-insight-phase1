from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.core.session_store import store
from app.models.schemas import GenerateReportRequest, DraftReportPlanRequest, DraftReportPlanResponse
from app.engines.report_engine import generate_report
from app.engines.report_builder_engine import draft_report_plan

router = APIRouter(prefix="/api/dataset", tags=["report"])

_MEDIA_TYPES = {"markdown": "text/markdown", "html": "text/html", "pdf": "application/pdf",
                 "notebook": "application/x-ipynb+json"}
_EXTENSIONS = {"markdown": "md", "html": "html", "pdf": "pdf", "notebook": "ipynb"}


def _get_session(dataset_id: str):
    session = store.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a dataset first.")
    return session


@router.post("/report/draft-plan", response_model=DraftReportPlanResponse)
async def draft_report_plan_endpoint(req: DraftReportPlanRequest):
    """
    Smart Report Builder (session 13) — AI curation step. Returns a PLAN
    (which sections, which analyses, why), never a rendered report; the
    frontend shows this as an editable outline, and the actual report still
    comes from POST /report once the user is happy with (or has adjusted)
    this plan.
    """
    session = _get_session(req.dataset_id)
    try:
        plan = draft_report_plan(
            df=session.current, goal_keywords=session.goal_keywords,
            investigation_goal=session.investigation_goal,
            available_analyses=[a.model_dump() for a in req.available_analyses],
            user_request=req.user_request,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return DraftReportPlanResponse(**plan)


@router.post("/report")
async def generate_report_endpoint(req: GenerateReportRequest):
    """
    The last stage of the pipeline — composes a report from every other
    engine's already-computed output (profile, quality, health, the session's
    own Analyst Notebook action log, EDA findings, pinned Dashboard charts)
    plus whatever analyses the user actually ran and chose to include via
    `included_analyses`. Never re-runs or invents an analysis on its own.

    `section_order` (session 13): which sections to include and in what
    order — omit for every section in the original default order (fully
    backward compatible with report generation from before the Smart Report
    Builder existed).
    """
    session = _get_session(req.dataset_id)
    if req.format not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"format must be one of {list(_MEDIA_TYPES)}")

    included = [a.model_dump() for a in req.included_analyses]
    try:
        content = generate_report(session, req.format, included, req.section_order)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    base_name = session.filename.rsplit(".", 1)[0] if "." in session.filename else session.filename
    filename = f"{base_name}_report.{_EXTENSIONS[req.format]}"
    body = content if isinstance(content, bytes) else content.encode("utf-8")
    return Response(
        content=body, media_type=_MEDIA_TYPES[req.format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
