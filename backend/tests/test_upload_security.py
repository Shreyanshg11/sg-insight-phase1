import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import upload as upload_router

client = TestClient(app)


def test_valid_csv_upload():
    response = client.post(
        "/api/dataset/upload",
        files={
            "file": (
                "sample.csv",
                b"name,value\nA,10\nB,20\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["rows"] == 2
    assert data["columns"] == 2
    assert data["filename"] == "sample.csv"


def test_oversized_upload_rejected(monkeypatch):
    monkeypatch.setattr(upload_router, "MAX_FILE_MB", 0)

    response = client.post(
        "/api/dataset/upload",
        files={
            "file": (
                "large.csv",
                b"name,value\nA,10\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    ("filename", "parser_name"),
    [
        ("broken.csv", "read_csv"),
        ("broken.xlsx", "read_excel"),
    ],
)
def test_parser_error_does_not_expose_internal_details(
    monkeypatch,
    filename,
    parser_name,
):
    secret_error = "INTERNAL_SECRET_PATH_123"

    def failing_parser(*args, **kwargs):
        raise RuntimeError(secret_error)

    monkeypatch.setattr(upload_router.pd, parser_name, failing_parser)

    response = client.post(
        "/api/dataset/upload",
        files={
            "file": (
                filename,
                b"invalid,data\n",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert secret_error not in response.text
    assert response.json()["detail"] == (
        "Could not parse uploaded file. "
        "Please verify the file format and contents."
    )
