from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_question() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_request",
        "message": "질문을 입력하세요.",
    }
