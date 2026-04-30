from app.core.config import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_ENV_FILE,
    DEFAULT_PDF_PATH,
    PROJECT_ROOT,
)


def test_default_paths_are_anchored_to_project_layout() -> None:
    assert DEFAULT_PDF_PATH.name == "청년수당 참여자 안내책자.pdf"
    assert DEFAULT_PDF_PATH.parent.name == "Data"
    assert DEFAULT_CHROMA_DIR.parts[-2:] == ("storage", "chroma")
    assert DEFAULT_ENV_FILE.name == ".env"
    assert DEFAULT_ENV_FILE.parent == PROJECT_ROOT
