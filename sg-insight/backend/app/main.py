from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, profiling, cleaning

app = FastAPI(
    title="SG-Insight API",
    description="AI-assisted data analytics workbench — backend for upload, profiling, "
                "and human-confirmed cleaning actions (Phase 1).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(profiling.router)
app.include_router(cleaning.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "sg-insight-api"}
