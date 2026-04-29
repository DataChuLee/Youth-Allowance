from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.core.errors import IndexMissingError

COLLECTION_NAME = "youth_allowance_booklet"


def assert_index_exists(chroma_dir: Path) -> None:
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        raise IndexMissingError("PDF 인덱스가 없습니다. 먼저 indexing 명령을 실행하세요.")


def create_vector_store(settings: Settings) -> Chroma:
    assert_index_exists(settings.chroma_dir)
    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
    )
