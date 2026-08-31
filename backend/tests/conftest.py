from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss_rag_final"))
    monkeypatch.setenv("PDF_PATH", str(tmp_path / "booklet.pdf"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
