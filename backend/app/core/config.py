from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "Data" / "청년수당 참여자 안내책자.pdf"
DEFAULT_CHROMA_DIR = BACKEND_DIR / "storage" / "chroma"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(
        default=DEFAULT_OPENAI_CHAT_MODEL, alias="OPENAI_CHAT_MODEL"
    )
    openai_embedding_model: str = Field(
        default=DEFAULT_OPENAI_EMBEDDING_MODEL, alias="OPENAI_EMBEDDING_MODEL"
    )
    pdf_path: Path = Field(default=DEFAULT_PDF_PATH, alias="PDF_PATH")
    chroma_dir: Path = Field(default=DEFAULT_CHROMA_DIR, alias="CHROMA_DIR")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    min_similarity_score: float = Field(default=0.2, alias="MIN_SIMILARITY_SCORE")

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
