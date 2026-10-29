from app.core.config import (
    DEFAULT_ENV_FILE,
    DEFAULT_FAISS_INDEX_DIR,
    DEFAULT_OPENAI_CHAT_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OCR_JSONL_PATH,
    PROJECT_ROOT,
    Settings,
)


def test_default_paths_are_anchored_to_project_layout() -> None:
    assert DEFAULT_OCR_JSONL_PATH.name == "upstage_youth_allowance_parse.jsonl"
    assert DEFAULT_OCR_JSONL_PATH.parent.name == "Data"
    assert DEFAULT_FAISS_INDEX_DIR.name == "faiss_rag_final"
    assert DEFAULT_ENV_FILE.name == ".env"
    assert DEFAULT_ENV_FILE.parent == PROJECT_ROOT


def test_openai_model_names_have_safe_defaults(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_chat_model == DEFAULT_OPENAI_CHAT_MODEL
    assert settings.openai_embedding_model == DEFAULT_OPENAI_EMBEDDING_MODEL
    assert settings.retrieval_candidate_k == 20
    assert settings.retrieval_rrf_k == 60


def test_blocker_check_can_be_disabled_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BLOCKER_CHECK_ENABLED", "false")

    assert Settings(OPENAI_API_KEY="test").blocker_check_enabled is False
