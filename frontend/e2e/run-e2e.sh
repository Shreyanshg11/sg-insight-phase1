#!/usr/bin/env bash
# Renders the real WorkbenchPage in jsdom against a live backend.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."

# Keep POSIX path for Bash commands and Windows path for Python/Node.
export E2E_WORK_POSIX="$(mktemp -d)"
export E2E_WORK="$(cygpath -w "$E2E_WORK_POSIX")"
export E2E_PORT="${E2E_PORT:-8124}"

export SG_INSIGHT_DATA_DIR="$E2E_WORK/data"
export PYTHONDONTWRITEBYTECODE=1

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

npm i --prefix "$E2E_WORK_POSIX" --silent esbuild jsdom

(
    cd "$ROOT/backend"
    python -c '
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from tests.test_recommendation_lifecycle import make_df

work = Path(os.environ["E2E_WORK"])
work.mkdir(parents=True, exist_ok=True)
make_df().to_csv(work / "data.csv", index=False)
'
)

(cd "$HERE" && node build.mjs)

(
    cd "$ROOT/backend"
    python -m uvicorn app.main:app --port "$E2E_PORT" --log-level warning \
        >"$E2E_WORK_POSIX/server.log" 2>&1
) &

SERVER_PID=$!
sleep 4

(cd "$HERE" && node run.mjs)