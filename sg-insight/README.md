# SG-Insight

AI-assisted data analytics workbench. Upload a dataset, work through a structured
profiling → cleaning → EDA → analysis → report pipeline, with every AI recommendation
shown as a preview + reason + confidence reading that requires explicit confirmation
before anything is applied.

This repo currently implements **Phase 1**: upload, profiling, data quality scoring,
the cleaning center (missing values, duplicates, outliers, category merge, text
cleaning), and simple linear undo/history. Phases 2 and 3 (full versioning, session
memory, feature engineering, analysis recommendation engine, dashboard, analyst
notebook, report export) build on top of this foundation — see the phased plan in
the original project brief.

## Architecture

```
sg-insight/
├── backend/          FastAPI + pandas
│   └── app/
│       ├── main.py           app entrypoint, CORS
│       ├── routers/          upload, profiling, cleaning endpoints
│       ├── services/         profiling + cleaning heuristics (the actual logic)
│       ├── models/           pydantic request/response schemas
│       └── core/             in-memory session store (dataset + version history)
└── frontend/          React + Vite + Tailwind
    └── src/
        ├── api/client.js             thin fetch wrapper around the backend
        ├── components/
        │   ├── ConfidenceMeter.jsx       signal-meter confidence readout
        │   ├── RecommendationCard.jsx    THE reusable preview/reason/confidence/confirm pattern
        │   ├── ProfileSummary.jsx
        │   └── QualityScorePanel.jsx
        └── pages/
            ├── UploadPage.jsx
            └── WorkbenchPage.jsx
```

`RecommendationCard` is the one component every future cleaning/analysis module
(feature engineering suggestions, analysis recommendation engine, etc.) should
reuse — it's the interaction contract for the whole app: AI proposes, user confirms,
nothing mutates silently.

## Running locally

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend expects the API at `http://localhost:8000` by default — override with a
`VITE_API_URL` env var if needed.

## Design notes / known limitations (Phase 1)

- **Session store is in-memory and single-process.** Restarting the backend loses
  all uploaded datasets. Fine for local dev; swap for Redis or a DB before deploying
  anywhere with multiple workers or real persistence needs.
- **Undo is linear**, not branching — matches the Phase 1 scope in the brief.
  Full dataset versioning (V1/V2/V3 compare + restore) is Phase 2.
- **Confidence scores are heuristic-based and documented in-line** in
  `cleaning_service.py` — IQR distance for outliers, skew for imputation choice,
  exact-match ratio for duplicates, etc. None are arbitrary.
- **No auth, no persistence layer, no investigation-goal-aware AI insights yet** —
  those come with session memory and the Claude API integration in Phase 2/3.
- File size capped at 50MB in the upload endpoint; raise `MAX_FILE_MB` in
  `routers/upload.py` if needed.
