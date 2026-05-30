from app.core.config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_PDF_PATH,
    PROJECT_ROOT,
    Settings,
)


def test_default_paths_are_anchored_to_project_layout() -> None:
    assert DEFAULT_PDF_PATH.name == "청년수당 참여자 안내책자.pdf"
    assert DEFAULT_PDF_PATH.parent.name == "Data"
    assert DEFAULT_CHROMA_DIR.parts[-2:] == ("storage", "chroma")
    assert DEFAULT_ENV_FILE.name == ".env"
    assert DEFAULT_ENV_FILE.parent == PROJECT_ROOT


def test_openai_model_names_have_safe_defaults(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == DEFAULT_OPENAI_CHAT_MODEL
    assert settings.openai_embedding_model == DEFAULT_OPENAI_EMBEDDING_MODEL
