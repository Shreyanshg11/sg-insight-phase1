# SG-Insight backend — FastAPI + pandas/scikit-learn/reportlab/matplotlib.
# Single-stage: nothing here needs a build toolchain beyond what pip pulls in.
FROM python:3.12-slim

WORKDIR /app

# System libs some wheels fall back to building against (matplotlib/scipy on
# platforms with no prebuilt wheel) — cheap to include, avoids a flaky build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Session data (SQLite + per-version pickle files, see app/core/persistence.py)
# lives here by default. Mount a volume at this path for real persistence —
# see the Deployment section in README.md.
ENV SG_INSIGHT_DATA_DIR=/app/data
RUN mkdir -p /app/data

# Render (and most PaaS hosts) inject $PORT at runtime; default 8000 keeps
# `docker run` / docker-compose working the same as local `uvicorn --reload`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
