import io
import pandas as pd
from app.core.session_store import store
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


def test_valid_xlsx_upload():
    import io
    import pandas as pd

    buffer = io.BytesIO()
    pd.DataFrame({"region": ["north", "south"], "sales": [10, 20]}).to_excel(buffer, index=False)
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("sample.xlsx", buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    assert response.json()["rows"] == 2
    assert response.json()["columns"] == 2


def test_unsupported_upload_extension_is_rejected():
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("payload.json", b'{"x": 1}', "application/json")},
    )
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"].lower()


def test_csv_extension_does_not_bypass_content_validation():
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("payload.csv", b"\x00\x01\x02\xff", "text/csv")},
    )
    assert response.status_code == 400
    assert "text" in response.json()["detail"].lower() or "binary" in response.json()["detail"].lower()


def test_upload_filename_path_is_reduced_to_safe_basename():
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("../../..\\private\\sales.csv", "region,sales\nwest,3\n".encode(), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "sales.csv"


def test_upload_column_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(upload_router, "MAX_COLUMNS", 1)
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("wide.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 413
    assert "column" in response.json()["detail"].lower()


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


def test_oversized_request_is_rejected_before_route_body_parsing():
    response = client.post(
        "/api/dataset/upload",
        headers={"Content-Length": str(60 * 1024 * 1024), "Content-Type": "multipart/form-data; boundary=unused"},
        content=b"",
    )
    assert response.status_code == 413
    assert "size limit" in response.json()["detail"].lower()
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


def test_dataset_version_exports_csv_and_xlsx_with_safe_filenames():
    session = store.create('../sales"\r\n.csv', pd.DataFrame({"region": ["north", "south"], "sales": [10, 20]}))
    csv_response = client.get(f"/api/dataset/{session.dataset_id}/versions/1/export?format=csv")
    xlsx_response = client.get(f"/api/dataset/{session.dataset_id}/versions/1/export?format=xlsx")

    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "sales_v1.csv" in csv_response.headers["content-disposition"]
    assert xlsx_response.status_code == 200
    assert "spreadsheetml.sheet" in xlsx_response.headers["content-type"]
    exported = pd.read_excel(io.BytesIO(xlsx_response.content))
    assert exported.to_dict(orient="records") == [
        {"region": "north", "sales": 10}, {"region": "south", "sales": 20},
    ]
    invalid = client.get(f"/api/dataset/{session.dataset_id}/versions/1/export?format=html")
    assert invalid.status_code == 400
