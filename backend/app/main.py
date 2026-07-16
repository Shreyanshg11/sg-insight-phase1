import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, profiling, cleaning, health, versioning, feature_engineering, eda, analysis, visualization, report

app = FastAPI(
    title="SG-Insight API",
    description="AI-assisted data analytics workbench — backend for upload, profiling, "
                "and human-confirmed cleaning actions (Phase 1).",
    version="0.1.0",
)

# Comma-separated list, e.g. "https://sg-insight.onrender.com,https://my-domain.com".
# Falls back to the two local dev ports so `npm run dev` keeps working untouched.
_default_origins = "http://localhost:5173,http://localhost:3000"
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(profiling.router)
app.include_router(cleaning.router)
app.include_router(health.router)
app.include_router(versioning.router)
app.include_router(feature_engineering.router)
app.include_router(eda.router)
app.include_router(analysis.router)
app.include_router(visualization.router)
app.include_router(report.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "sg-insight-api"}
