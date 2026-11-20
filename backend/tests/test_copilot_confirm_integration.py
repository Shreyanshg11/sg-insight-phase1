import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.core.session_store import store


def test_copilot_confirm_action_applies_cleaning():
    df = pd.DataFrame({
        "name": ["A", "B", "A"],
        "value": [10, 20, 10],
    })
    session = store.create("copilot_test.csv", df)

    client = TestClient(app)
    response = client.post(
        "/api/dataset/copilot/confirm-action",
        json={
            "dataset_id": session.dataset_id,
            "proposed_action": {
                "kind": "cleaning",
                "step": "duplicate_manager",
                "action": "drop_duplicates",
                "reason": "Integration test",
                "confidence": 0.95,
                "params": {},
            },
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()

    updated = store.get(session.dataset_id)
    assert data["version"] == 2
    assert data["rows"] == 2
    assert len(updated.current) == 2
    assert len(updated.log) == 1
    assert data["result_summary"]
