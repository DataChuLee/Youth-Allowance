from types import SimpleNamespace

from langchain_core.documents import Document

from app.rag import generation


def test_format_context_includes_chunk_and_page_metadata() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0"},
    )

    context = generation.format_context([document])

    assert "[chunk_id=pdf-page-3-chunk-0 page=3]" in context
    assert "청년수당 카드 사용처 기준 안내" in context


def test_generate_pdf_answer_invokes_chat_model_with_pdf_context(monkeypatch) -> None:
    captured = {}
    document = Document(
        page_content="청년수당 카드는 구직활동 비용에 사용할 수 있습니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0"},
    )
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_chat_model="test-chat-model",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="PDF 근거 기반 답변")

    monkeypatch.setattr(generation, "ChatOpenAI", FakeChatOpenAI)

    answer = generation.generate_pdf_answer("카드는 어디에 써?", [document], settings)

    assert answer == "PDF 근거 기반 답변"
    assert captured["kwargs"] == {
        "api_key": "test-key",
        "model": "test-chat-model",
        "temperature": 0,
    }
    assert captured["messages"][1][0] == "human"
    assert "카드는 어디에 써?" in captured["messages"][1][1]
    assert "pdf-page-12-chunk-0" in captured["messages"][1][1]
