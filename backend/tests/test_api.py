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


def test_chat_returns_placeholder_for_valid_question() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"question": "청년수당 사용처는?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "아직 RAG 그래프가 연결되지 않았습니다.",
        "sources": [],
        "status": "insufficient_pdf_evidence",
        "needs_external_search": True,
    }
