from fastapi.testclient import TestClient

from app.api.schemas import ChatResponse, Source
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


def test_chat_returns_fallback_for_valid_question() -> None:
    client = TestClient(app)
    response = client.post("/chat", json={"question": "청년수당 사용처는?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "안내책자에서 해당 내용을 확인하지 못했습니다. 최신 공식 안내 확인이 필요할 수 있습니다.",
        "sources": [],
        "status": "insufficient_pdf_evidence",
        "needs_external_search": True,
    }


def test_chat_returns_graph_response(monkeypatch) -> None:
    def fake_run_chat_graph(question: str) -> ChatResponse:
        assert question == "청년수당 사용처는?"
        return ChatResponse(
            answer="사용처 답변",
            sources=[
                Source(
                    type="pdf",
                    title="청년수당 참여자 안내책자",
                    page=12,
                    excerpt="사용처 관련 문단",
                    chunk_id="pdf-page-12-chunk-2",
                )
            ],
            status="answered_from_pdf",
            needs_external_search=False,
        )

    monkeypatch.setattr("app.api.routes.run_chat_graph", fake_run_chat_graph)
    client = TestClient(app)

    response = client.post("/chat", json={"question": "청년수당 사용처는?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "사용처 답변"
    assert response.json()["sources"][0]["page"] == 12
