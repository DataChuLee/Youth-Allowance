import logging

from fastapi.testclient import TestClient

from app.api.schemas import ChatResponse, Source
from app.graph.workflow import PreparedChatStream
from app.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "dependencies" in body
    assert "faiss_index" in body["dependencies"]
    assert "redis_cache" in body["dependencies"]
    assert "config" in body
    assert "chat_model" in body["config"]
    assert "embedding_model" in body["config"]
    assert "retrieval_top_k" in body["config"]


def test_chat_rejects_empty_question() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/chat", json={"question": "   "})

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_request",
        "message": "질문을 입력하세요.",
    }


def test_chat_returns_fallback_for_valid_question() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/chat", json={"question": "청년수당 사용처는?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "안내책자에서 해당 내용을 확인하지 못했습니다. 최신 공식 안내 확인이 필요할 수 있습니다.",
        "sources": [],
        "status": "insufficient_pdf_evidence",
        "needs_external_search": True,
        "intent": "rag",
    }


def test_chat_returns_graph_response(monkeypatch) -> None:
    def fake_run_chat_graph(
        question: str,
        thread_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        assert question == "청년수당 사용처는?"
        assert thread_id == "tab-123"
        assert history == [{"role": "user", "content": "이전 질문"}]
        return ChatResponse(
            answer="사용처 답변",
            sources=[
                Source(
                    type="pdf",
                    title="청년수당 참여자 안내책자",
                    page=12,
                    excerpt="사용처 관련 문단",
                    chunk_id="pdf-page-12-chunk-2",
                    score=0.87,
                )
            ],
            status="answered_from_pdf",
            needs_external_search=False,
        )

    monkeypatch.setattr("app.api.routes.run_chat_graph", fake_run_chat_graph)
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "청년수당 사용처는?",
            "thread_id": "tab-123",
            "history": [{"role": "user", "content": "이전 질문"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "사용처 답변"
    assert response.json()["sources"][0]["page"] == 12
    assert response.json()["sources"][0]["score"] == 0.87


def test_chat_stream_returns_token_and_done_events(monkeypatch) -> None:
    def fake_prepare_chat_stream(
        question: str,
        thread_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> PreparedChatStream:
        assert question == "청년수당 사용처는?"
        assert thread_id == "tab-stream"
        assert history == [{"role": "assistant", "content": "이전 답변"}]
        return PreparedChatStream(
            chunks=iter(["사용처", " 답변"]),
            sources=[
                Source(
                    type="pdf",
                    title="청년수당 참여자 안내책자",
                    page=12,
                    excerpt="사용처 관련 문단",
                    chunk_id="pdf-page-12-chunk-2",
                    score=0.87,
                )
            ],
            status="answered_from_pdf",
            needs_external_search=False,
            intent="rag",
        )

    monkeypatch.setattr("app.api.routes.prepare_chat_stream", fake_prepare_chat_stream)
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/stream",
        json={
            "question": "청년수당 사용처는?",
            "thread_id": "tab-stream",
            "history": [{"role": "assistant", "content": "이전 답변"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: token\ndata: {"text": "사용처"}' in response.text
    assert 'event: token\ndata: {"text": " 답변"}' in response.text
    assert '"answer": "사용처 답변"' in response.text
    assert '"status": "answered_from_pdf"' in response.text
    assert '"chunk_id": "pdf-page-12-chunk-2"' in response.text
    assert '"score": 0.87' in response.text


def test_chat_stream_rejects_empty_question() -> None:
    client = TestClient(app)
    response = client.post("/api/v1/chat/stream", json={"question": "   "})

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_request",
        "message": "질문을 입력하세요.",
    }


def test_server_session_history_endpoint_is_not_available() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/sessions/nonexistent-thread-xyz/history")

    assert response.status_code == 404
    assert response.json() == {"code": "not_found", "message": "Not Found"}


def test_legacy_chat_route_is_deprecated_and_compatible(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.run_chat_graph",
        lambda *_args, **_kwargs: ChatResponse(
            answer="호환 답변",
            sources=[],
            status="answered_from_pdf",
            needs_external_search=False,
        ),
    )
    client = TestClient(app)

    response = client.post("/chat", json={"question": "기존 클라이언트 질문"})

    assert response.status_code == 200
    assert response.json()["answer"] == "호환 답변"
    assert response.headers["deprecation"] == "true"
    assert response.headers["link"] == '</api/v1/chat>; rel="successor-version"'
    assert "sunset" not in response.headers
    operation = app.openapi()["paths"]["/chat"]["post"]
    assert operation["deprecated"] is True


def test_chat_does_not_log_question_history_or_thread_id(
    caplog,
    monkeypatch,
) -> None:
    question = "민감한 사용자 질문 987654"
    history = "민감한 이전 답변 123456"
    thread_id = "private-thread-id-abcdef"
    monkeypatch.setattr(
        "app.api.routes.run_chat_graph",
        lambda *_args, **_kwargs: ChatResponse(
            answer="안전한 응답",
            sources=[],
            status="answered_from_pdf",
            needs_external_search=False,
        ),
    )
    client = TestClient(app)

    with caplog.at_level(logging.DEBUG):
        response = client.post(
            "/api/v1/chat",
            json={
                "question": question,
                "thread_id": thread_id,
                "history": [{"role": "assistant", "content": history}],
            },
        )

    assert response.status_code == 200
    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert question not in log_output
    assert history not in log_output
    assert thread_id not in log_output


def test_validation_error_uses_common_error_schema() -> None:
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={})

    assert response.status_code == 422
    assert response.json() == {
        "code": "invalid_request",
        "message": "요청 형식이 올바르지 않습니다.",
    }


def test_unexpected_error_uses_common_error_schema(monkeypatch) -> None:
    def fail_chat(*_args, **_kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr("app.api.routes.run_chat_graph", fail_chat)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/api/v1/chat", json={"question": "오류 테스트"})

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "서버 오류가 발생했습니다.",
    }


def test_chat_stream_error_event_uses_common_error_schema(monkeypatch) -> None:
    def fail_chunks():
        raise RuntimeError("stream failed")
        yield ""

    monkeypatch.setattr(
        "app.api.routes.prepare_chat_stream",
        lambda *_args, **_kwargs: PreparedChatStream(
            chunks=fail_chunks(),
            sources=[],
            status="answered_from_pdf",
            needs_external_search=False,
            intent="rag",
        ),
    )
    client = TestClient(app)

    response = client.post("/api/v1/chat/stream", json={"question": "스트림 오류"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"code": "generation_error"' in response.text
    assert '"message": "답변 생성 중 오류가 발생했습니다."' in response.text
