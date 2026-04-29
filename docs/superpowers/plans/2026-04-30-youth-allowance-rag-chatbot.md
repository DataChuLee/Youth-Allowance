# Youth Allowance RAG Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local MVP web app where 청년수당 participants can ask questions and receive PDF-grounded answers with page-level sources.

**Architecture:** Implement a FastAPI backend and Next.js frontend. The backend owns PDF indexing, Chroma persistence, LangChain retrieval, LangGraph orchestration, OpenAI calls, LangSmith tracing, and stable API contracts. The frontend owns browser-only chat state, quick FAQ buttons, loading/error states, and source rendering.

**Tech Stack:** Python 3.11, FastAPI, pytest, LangChain, LangGraph, Chroma, OpenAI API, LangSmith, Next.js, React, TypeScript, Vitest or Jest-compatible unit tests.

---

## File Structure

Create this structure:

```text
backend/
  pyproject.toml
  README.md
  app/
    __init__.py
    main.py
    api/
      __init__.py
      routes.py
      schemas.py
    core/
      __init__.py
      config.py
      errors.py
    graph/
      __init__.py
      state.py
      workflow.py
    indexing/
      __init__.py
      index_pdf.py
      quality.py
    rag/
      __init__.py
      prompts.py
      sources.py
      vector_store.py
  tests/
    conftest.py
    test_api.py
    test_indexing_quality.py
    test_sources.py
    test_workflow.py
frontend/
  package.json
  next.config.ts
  tsconfig.json
  vitest.config.ts
  test/
    setup.ts
  app/
    globals.css
    page.tsx
  components/
    ChatInput.tsx
    MessageList.tsx
    QuickQuestionBar.tsx
    SourceList.tsx
  lib/
    api.ts
    types.ts
  __tests__/
    api.test.ts
    chat-page.test.tsx
```

Generated local data:

```text
backend/storage/chroma/
```

This directory is rebuildable output from indexing and should stay out of Git.

## Task 1: Backend Project Skeleton and Configuration

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/README.md`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/errors.py`
- Create: `backend/tests/conftest.py`

- [ ] **Step 1: Write backend configuration tests**

Create `backend/tests/conftest.py`:

```python
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "test-chat-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("PDF_PATH", str(tmp_path / "booklet.pdf"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
```

Create `backend/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run the failing health test**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: fail because `backend/pyproject.toml`, FastAPI, or `app.main` is not implemented yet.

- [ ] **Step 3: Create backend package metadata**

Create `backend/pyproject.toml`:

```toml
[project]
name = "youth-allowance-rag-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.29",
  "pydantic-settings>=2.2",
  "python-dotenv>=1.0",
  "pytest>=8.0",
  "httpx>=0.27"
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Create `backend/README.md`:

```markdown
# Youth Allowance RAG Backend

Local FastAPI backend for the 청년수당 PDF-based RAG chatbot.

## Development

```powershell
venv\Scripts\python.exe -m pytest -v
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
```

- [ ] **Step 4: Create configuration and error primitives**

Create `backend/app/core/config.py`:

```python
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(alias="OPENAI_EMBEDDING_MODEL")
    pdf_path: Path = Field(default=Path("../Data/청년수당 참여자 안내책자.pdf"), alias="PDF_PATH")
    chroma_dir: Path = Field(default=Path("storage/chroma"), alias="CHROMA_DIR")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    min_similarity_score: float = Field(default=0.2, alias="MIN_SIMILARITY_SCORE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Create `backend/app/core/errors.py`:

```python
class AppError(Exception):
    error_code = "application_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(AppError):
    error_code = "invalid_request"
    status_code = 400


class IndexMissingError(AppError):
    error_code = "index_missing"
    status_code = 409


class GenerationError(AppError):
    error_code = "generation_error"
    status_code = 500


class IndexingError(AppError):
    error_code = "indexing_error"
    status_code = 500
```

- [ ] **Step 5: Create FastAPI app**

Create `backend/app/__init__.py`, `backend/app/core/__init__.py` as empty files.

Create `backend/app/main.py`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AppError

app = FastAPI(title="Youth Allowance RAG Chatbot API")


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": exc.message},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 6: Run backend test**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: `1 passed`.

- [ ] **Step 7: Commit backend skeleton**

```powershell
git add backend
git commit -m "feat: add backend skeleton"
```

## Task 2: API Schemas and Stable Error Contract

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/schemas.py`
- Create: `backend/app/api/routes.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_api.py`

- [ ] **Step 1: Add API contract tests**

Replace `backend/tests/test_api.py` with:

```python
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
```

- [ ] **Step 2: Run API contract test and confirm failure**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_api.py::test_chat_rejects_empty_question -v
```

Expected: fail with `404 Not Found` or missing route.

- [ ] **Step 3: Create schemas**

Create `backend/app/api/__init__.py` as an empty file.

Create `backend/app/api/schemas.py`:

```python
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    type: str
    title: str
    page: int
    excerpt: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    needs_external_search: bool
```

- [ ] **Step 4: Create chat route with empty-question validation**

Create `backend/app/api/routes.py`:

```python
from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.core.errors import InvalidRequestError

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise InvalidRequestError("질문을 입력하세요.")

    return ChatResponse(
        answer="아직 RAG 그래프가 연결되지 않았습니다.",
        sources=[],
        status="insufficient_pdf_evidence",
        needs_external_search=True,
    )
```

Modify `backend/app/main.py`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.errors import AppError

app = FastAPI(title="Youth Allowance RAG Chatbot API")
app.include_router(router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": exc.message},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 5: Run API tests**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit API contract**

```powershell
git add backend/app backend/tests
git commit -m "feat: add chat API contract"
```

## Task 3: PDF Indexing Quality Layer

**Files:**
- Create: `backend/app/indexing/__init__.py`
- Create: `backend/app/indexing/quality.py`
- Create: `backend/app/indexing/index_pdf.py`
- Create: `backend/tests/test_indexing_quality.py`
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add indexing quality tests**

Create `backend/tests/test_indexing_quality.py`:

```python
import pytest

from app.core.errors import IndexingError
from app.indexing.quality import IndexingStats, validate_indexing_stats


def test_validate_indexing_stats_accepts_text_rich_pdf() -> None:
    stats = IndexingStats(
        total_pages=10,
        extracted_pages=10,
        empty_pages=0,
        total_characters=25_000,
        chunk_count=80,
        sample_chunks=["page=1 chunk=pdf-page-1-chunk-0 text=hello"],
    )

    validate_indexing_stats(stats)


def test_validate_indexing_stats_rejects_low_text_pdf() -> None:
    stats = IndexingStats(
        total_pages=20,
        extracted_pages=1,
        empty_pages=19,
        total_characters=50,
        chunk_count=1,
        sample_chunks=[],
    )

    with pytest.raises(IndexingError, match="텍스트 추출 품질이 낮습니다"):
        validate_indexing_stats(stats)
```

- [ ] **Step 2: Run indexing quality test and confirm failure**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_indexing_quality.py -v
```

Expected: fail because `app.indexing.quality` does not exist.

- [ ] **Step 3: Add LangChain and Chroma dependencies**

Modify `backend/pyproject.toml` dependencies:

```toml
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.29",
  "pydantic-settings>=2.2",
  "python-dotenv>=1.0",
  "pytest>=8.0",
  "httpx>=0.27",
  "langchain>=0.2",
  "langchain-community>=0.2",
  "langchain-openai>=0.1",
  "langchain-chroma>=0.1",
  "chromadb>=0.5",
  "pypdf>=4.0"
]
```

- [ ] **Step 4: Implement quality checks**

Create `backend/app/indexing/__init__.py` as an empty file.

Create `backend/app/indexing/quality.py`:

```python
from dataclasses import dataclass

from app.core.errors import IndexingError


@dataclass(frozen=True)
class IndexingStats:
    total_pages: int
    extracted_pages: int
    empty_pages: int
    total_characters: int
    chunk_count: int
    sample_chunks: list[str]


def validate_indexing_stats(stats: IndexingStats) -> None:
    if stats.total_pages <= 0:
        raise IndexingError("PDF 페이지를 찾을 수 없습니다.")
    if stats.chunk_count <= 0:
        raise IndexingError("검색 청크가 생성되지 않았습니다.")
    empty_ratio = stats.empty_pages / stats.total_pages
    if stats.total_characters < 500 or empty_ratio > 0.5:
        raise IndexingError(
            "텍스트 추출 품질이 낮습니다. 이미지 기반 PDF일 수 있으므로 OCR 전처리가 필요합니다."
        )


def format_indexing_stats(stats: IndexingStats) -> str:
    samples = "\n".join(f"- {sample}" for sample in stats.sample_chunks)
    return (
        "Indexing quality report\n"
        f"total_pages={stats.total_pages}\n"
        f"extracted_pages={stats.extracted_pages}\n"
        f"empty_pages={stats.empty_pages}\n"
        f"total_characters={stats.total_characters}\n"
        f"chunk_count={stats.chunk_count}\n"
        f"sample_chunks:\n{samples}"
    )
```

- [ ] **Step 5: Implement indexing command**

Create `backend/app/indexing/index_pdf.py`:

```python
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.core.config import get_settings
from app.core.errors import IndexingError
from app.indexing.quality import IndexingStats, format_indexing_stats, validate_indexing_stats

BOOKLET_TITLE = "청년수당 참여자 안내책자"


def build_chunk_id(page: int, index: int) -> str:
    return f"pdf-page-{page}-chunk-{index}"


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    if not pdf_path.exists():
        raise IndexingError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks: list[Document] = []
    per_page_counts: dict[int, int] = {}
    for page in pages:
        page_number = int(page.metadata.get("page", 0)) + 1
        page_chunks = splitter.split_documents([page])
        for page_chunk in page_chunks:
            index = per_page_counts.get(page_number, 0)
            per_page_counts[page_number] = index + 1
            chunk_id = build_chunk_id(page_number, index)
            page_chunk.metadata.update(
                {
                    "source": str(page.metadata.get("source", "")),
                    "title": BOOKLET_TITLE,
                    "page": page_number,
                    "chunk_id": chunk_id,
                }
            )
            chunks.append(page_chunk)
    return chunks


def collect_stats(pages: list[Document], chunks: list[Document]) -> IndexingStats:
    total_pages = len(pages)
    empty_pages = sum(1 for page in pages if not page.page_content.strip())
    total_characters = sum(len(page.page_content.strip()) for page in pages)
    sample_chunks = [
        f"page={chunk.metadata['page']} chunk_id={chunk.metadata['chunk_id']} text={chunk.page_content[:120]}"
        for chunk in chunks[:3]
    ]
    return IndexingStats(
        total_pages=total_pages,
        extracted_pages=total_pages - empty_pages,
        empty_pages=empty_pages,
        total_characters=total_characters,
        chunk_count=len(chunks),
        sample_chunks=sample_chunks,
    )


def index_pdf() -> IndexingStats:
    settings = get_settings()
    pages = load_pdf_pages(settings.pdf_path)
    chunks = split_pages(pages)
    stats = collect_stats(pages, chunks)
    validate_indexing_stats(stats)

    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_name="youth_allowance_booklet",
    )
    return stats


def main() -> None:
    stats = index_pdf()
    print(format_indexing_stats(stats))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run indexing quality tests**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_indexing_quality.py -v
```

Expected: tests pass.

- [ ] **Step 7: Commit indexing quality layer**

```powershell
git add backend/app/indexing backend/tests/test_indexing_quality.py backend/pyproject.toml
git commit -m "feat: add PDF indexing quality checks"
```

## Task 4: Source Normalization and Retriever Factory

**Files:**
- Create: `backend/app/rag/__init__.py`
- Create: `backend/app/rag/sources.py`
- Create: `backend/app/rag/vector_store.py`
- Create: `backend/tests/test_sources.py`
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add source normalization tests**

Create `backend/tests/test_sources.py`:

```python
from langchain_core.documents import Document

from app.rag.sources import document_to_source


def test_document_to_source_preserves_required_metadata() -> None:
    document = Document(
        page_content="카드 사용 관련 안내 문단입니다.",
        metadata={
            "title": "청년수당 참여자 안내책자",
            "page": 12,
            "chunk_id": "pdf-page-12-chunk-2",
        },
    )

    source = document_to_source(document)

    assert source.type == "pdf"
    assert source.title == "청년수당 참여자 안내책자"
    assert source.page == 12
    assert source.chunk_id == "pdf-page-12-chunk-2"
    assert source.excerpt == "카드 사용 관련 안내 문단입니다."
```

- [ ] **Step 2: Run source test and confirm failure**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_sources.py -v
```

Expected: fail because `app.rag.sources` does not exist.

- [ ] **Step 3: Implement source normalization**

Create `backend/app/rag/__init__.py` as an empty file.

Create `backend/app/rag/sources.py`:

```python
from langchain_core.documents import Document

from app.api.schemas import Source

MAX_EXCERPT_LENGTH = 280


def compact_text(text: str) -> str:
    return " ".join(text.split())


def document_to_source(document: Document) -> Source:
    text = compact_text(document.page_content)
    excerpt = text[:MAX_EXCERPT_LENGTH]
    return Source(
        type="pdf",
        title=str(document.metadata.get("title", "청년수당 참여자 안내책자")),
        page=int(document.metadata["page"]),
        excerpt=excerpt,
        chunk_id=str(document.metadata["chunk_id"]),
    )
```

- [ ] **Step 4: Implement vector store factory**

Create `backend/app/rag/vector_store.py`:

```python
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
```

- [ ] **Step 5: Run source tests**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_sources.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit RAG source layer**

```powershell
git add backend/app/rag backend/tests/test_sources.py
git commit -m "feat: add RAG source normalization"
```

## Task 5: LangGraph Workflow with Evidence Grading

**Files:**
- Create: `backend/app/rag/prompts.py`
- Create: `backend/app/graph/__init__.py`
- Create: `backend/app/graph/state.py`
- Create: `backend/app/graph/workflow.py`
- Create: `backend/tests/test_workflow.py`
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add workflow tests using fake model and fake retriever**

Create `backend/tests/test_workflow.py`:

```python
from langchain_core.documents import Document

from app.graph.workflow import assess_evidence, fallback_no_answer
from app.graph.state import GraphState, RetrievedDocument


def test_assess_evidence_rejects_no_documents() -> None:
    state = GraphState(question="없는 내용인가요?", retrieved_documents=[])

    result = assess_evidence(state)

    assert result.evidence.is_sufficient is False
    assert result.status == "insufficient_pdf_evidence"


def test_assess_evidence_accepts_high_score_document() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0", "title": "청년수당 참여자 안내책자"},
    )
    state = GraphState(
        question="카드 사용처는?",
        retrieved_documents=[RetrievedDocument(document=document, score=0.9)],
    )

    result = assess_evidence(state)

    assert result.evidence.is_sufficient is True
    assert result.evidence.source_chunk_ids == ["pdf-page-3-chunk-0"]


def test_fallback_returns_external_search_signal() -> None:
    state = GraphState(question="최신 변경 사항은?")

    result = fallback_no_answer(state)

    assert result.answer.startswith("안내책자에서 해당 내용을 확인하지 못했습니다")
    assert result.needs_external_search is True
    assert result.status == "insufficient_pdf_evidence"
```

- [ ] **Step 2: Run workflow tests and confirm failure**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_workflow.py -v
```

Expected: fail because graph modules do not exist.

- [ ] **Step 3: Add LangGraph dependency**

Modify `backend/pyproject.toml` dependencies:

```toml
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.29",
  "pydantic-settings>=2.2",
  "python-dotenv>=1.0",
  "pytest>=8.0",
  "httpx>=0.27",
  "langchain>=0.2",
  "langchain-community>=0.2",
  "langchain-openai>=0.1",
  "langchain-chroma>=0.1",
  "langgraph>=0.2",
  "chromadb>=0.5",
  "pypdf>=4.0"
]
```

- [ ] **Step 4: Define prompts and graph state**

Create `backend/app/rag/prompts.py`:

```python
ANSWER_SYSTEM_PROMPT = """당신은 청년수당 참여자 안내책자 기반 FAQ 챗봇입니다.
검색된 근거에 있는 내용만 사용해서 한국어로 답변하세요.
근거에 없는 정책, 날짜, 금액, 절차를 추측하지 마세요.
답변은 간결하고 참여자가 바로 이해할 수 있게 작성하세요."""

EVIDENCE_GRADER_PROMPT = """질문과 검색된 안내책자 청크를 보고 답변 가능 여부를 판단하세요.
검색된 청크만으로 질문에 답할 수 있으면 is_sufficient=true를 반환하세요.
애매하거나 청크에 직접 근거가 없으면 is_sufficient=false를 반환하세요."""
```

Create `backend/app/graph/__init__.py` as an empty file.

Create `backend/app/graph/state.py`:

```python
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from app.api.schemas import Source


class RetrievedDocument(BaseModel):
    document: Document
    score: float

    model_config = {"arbitrary_types_allowed": True}


class EvidenceDecision(BaseModel):
    is_sufficient: bool = False
    reason: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class GraphState(BaseModel):
    question: str
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    evidence: EvidenceDecision = Field(default_factory=EvidenceDecision)
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    status: str = "insufficient_pdf_evidence"
    needs_external_search: bool = True

    model_config = {"arbitrary_types_allowed": True}
```

- [ ] **Step 5: Implement deterministic evidence and fallback nodes**

Create `backend/app/graph/workflow.py`:

```python
from langchain_core.documents import Document

from app.graph.state import EvidenceDecision, GraphState, RetrievedDocument
from app.rag.sources import document_to_source

DEFAULT_MIN_SIMILARITY_SCORE = 0.2


def assess_evidence(
    state: GraphState,
    min_similarity_score: float = DEFAULT_MIN_SIMILARITY_SCORE,
) -> GraphState:
    if not state.retrieved_documents:
        state.evidence = EvidenceDecision(
            is_sufficient=False,
            reason="검색된 PDF 청크가 없습니다.",
            source_chunk_ids=[],
        )
        state.status = "insufficient_pdf_evidence"
        state.needs_external_search = True
        return state

    best = max(state.retrieved_documents, key=lambda item: item.score)
    if best.score < min_similarity_score:
        state.evidence = EvidenceDecision(
            is_sufficient=False,
            reason="검색 점수가 최소 기준보다 낮습니다.",
            source_chunk_ids=[],
        )
        state.status = "insufficient_pdf_evidence"
        state.needs_external_search = True
        return state

    chunk_id = str(best.document.metadata["chunk_id"])
    state.evidence = EvidenceDecision(
        is_sufficient=True,
        reason="상위 검색 청크가 최소 검색 점수 기준을 통과했습니다.",
        source_chunk_ids=[chunk_id],
    )
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def generate_answer_from_documents(state: GraphState) -> GraphState:
    allowed = set(state.evidence.source_chunk_ids)
    documents = [
        item.document for item in state.retrieved_documents
        if str(item.document.metadata.get("chunk_id")) in allowed
    ]
    context = "\n\n".join(document.page_content for document in documents)
    state.answer = f"안내책자 기준으로 확인한 내용입니다.\n\n{context}"
    state.sources = [document_to_source(document) for document in documents]
    state.status = "answered_from_pdf"
    state.needs_external_search = False
    return state


def fallback_no_answer(state: GraphState) -> GraphState:
    state.answer = "안내책자에서 해당 내용을 확인하지 못했습니다. 최신 공식 안내 확인이 필요할 수 있습니다."
    state.sources = []
    state.status = "insufficient_pdf_evidence"
    state.needs_external_search = True
    return state


def retrieved_document_from_score(document: Document, score: float) -> RetrievedDocument:
    return RetrievedDocument(document=document, score=score)
```

- [ ] **Step 6: Run workflow tests**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_workflow.py -v
```

Expected: tests pass.

- [ ] **Step 7: Commit graph workflow primitives**

```powershell
git add backend/app/graph backend/app/rag/prompts.py backend/tests/test_workflow.py backend/pyproject.toml
git commit -m "feat: add RAG graph workflow primitives"
```

## Task 6: Connect Chat API to RAG Workflow

**Files:**
- Modify: `backend/app/api/routes.py`
- Modify: `backend/app/graph/workflow.py`
- Modify: `backend/tests/test_api.py`

- [ ] **Step 1: Add chat route behavior tests with monkeypatch**

Append to `backend/tests/test_api.py`:

```python
from app.api.schemas import ChatResponse, Source


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
```

- [ ] **Step 2: Run new API test and confirm failure**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_api.py::test_chat_returns_graph_response -v
```

Expected: fail because `run_chat_graph` is not wired into the route.

- [ ] **Step 3: Add graph runner**

Append to `backend/app/graph/workflow.py`:

```python
from app.api.schemas import ChatResponse


def run_chat_graph(question: str) -> ChatResponse:
    state = GraphState(question=question)
    state = fallback_no_answer(state)
    return ChatResponse(
        answer=state.answer,
        sources=state.sources,
        status=state.status,
        needs_external_search=state.needs_external_search,
    )
```

- [ ] **Step 4: Wire chat route to graph runner**

Replace `backend/app/api/routes.py` with:

```python
from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.core.errors import InvalidRequestError
from app.graph.workflow import run_chat_graph

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise InvalidRequestError("질문을 입력하세요.")

    return run_chat_graph(question)
```

- [ ] **Step 5: Run API tests**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Expected: tests pass.

- [ ] **Step 6: Commit API graph wiring**

```powershell
git add backend/app/api/routes.py backend/app/graph/workflow.py backend/tests/test_api.py
git commit -m "feat: connect chat API to graph runner"
```

## Task 7: Next.js Frontend Shell and API Client

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/test/setup.ts`
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/api.ts`
- Create: `frontend/__tests__/api.test.ts`

- [ ] **Step 1: Add API client test**

Create `frontend/__tests__/api.test.ts`:

```typescript
import { expect, test, vi } from "vitest";

import { sendChatMessage } from "../lib/api";

test("sendChatMessage posts question to backend", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      answer: "답변",
      sources: [],
      status: "insufficient_pdf_evidence",
      needs_external_search: true,
    }),
  });
  vi.stubGlobal("fetch", fetchMock);

  const result = await sendChatMessage("질문");

  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "질문" }),
  });
  expect(result.answer).toBe("답변");
});
```

- [ ] **Step 2: Create frontend project files**

Create `frontend/package.json`:

```json
{
  "name": "youth-allowance-rag-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "test": "vitest run"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "next": "^14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@testing-library/react": "^15.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "@types/node": "^20.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "jsdom": "^24.0.0",
    "typescript": "^5.0.0",
    "vitest": "^1.0.0"
  }
}
```

Create `frontend/next.config.ts`:

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {};

export default nextConfig;
```

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
```

Create `frontend/vitest.config.ts`:

```typescript
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: false,
    setupFiles: ["./test/setup.ts"],
  },
});
```

Create `frontend/test/setup.ts`:

```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 3: Create API types and client**

Create `frontend/lib/types.ts`:

```typescript
export type Source = {
  type: "pdf";
  title: string;
  page: number;
  excerpt: string;
  chunk_id: string;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  status: "answered_from_pdf" | "insufficient_pdf_evidence";
  needs_external_search: boolean;
};
```

Create `frontend/lib/api.ts`:

```typescript
import type { ChatResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function sendChatMessage(question: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      message: "서버 응답을 읽을 수 없습니다.",
    }));
    throw new Error(error.message ?? "요청에 실패했습니다.");
  }

  return response.json();
}
```

- [ ] **Step 4: Run frontend API test**

Run:

```powershell
cd frontend
npm test
```

Expected: test passes after dependencies are installed. If dependencies are missing, run `npm install` first.

- [ ] **Step 5: Commit frontend API client**

```powershell
git add frontend
git commit -m "feat: add frontend API client"
```

## Task 8: Chat UI Components

**Files:**
- Create: `frontend/app/globals.css`
- Create: `frontend/app/page.tsx`
- Create: `frontend/components/ChatInput.tsx`
- Create: `frontend/components/MessageList.tsx`
- Create: `frontend/components/QuickQuestionBar.tsx`
- Create: `frontend/components/SourceList.tsx`
- Create: `frontend/__tests__/chat-page.test.tsx`

- [ ] **Step 1: Add chat page rendering test**

Create `frontend/__tests__/chat-page.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import Page from "../app/page";

test("renders chatbot interface", () => {
  render(<Page />);

  expect(screen.getByText("청년수당 안내 챗봇")).toBeInTheDocument();
  expect(screen.getByPlaceholderText("청년수당에 대해 질문하세요")).toBeInTheDocument();
  expect(screen.getByText("청년수당 사용처")).toBeInTheDocument();
});
```

- [ ] **Step 2: Create UI components**

Create `frontend/components/SourceList.tsx`:

```typescript
import type { Source } from "../lib/types";

export function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;

  return (
    <div className="source-list">
      {sources.map((source) => (
        <div className="source-item" key={source.chunk_id}>
          <strong>{source.title} p.{source.page}</strong>
          <p>{source.excerpt}</p>
        </div>
      ))}
    </div>
  );
}
```

Create `frontend/components/MessageList.tsx`:

```typescript
import type { ChatResponse } from "../lib/types";
import { SourceList } from "./SourceList";

export type ChatMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; response: ChatResponse };

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
          <p>{message.content}</p>
          {message.role === "assistant" ? <SourceList sources={message.response.sources} /> : null}
        </div>
      ))}
    </div>
  );
}
```

Create `frontend/components/QuickQuestionBar.tsx`:

```typescript
const QUESTIONS = [
  "청년수당 사용처",
  "활동기록서 제출",
  "자격상실 또는 참여 중단",
  "현금 사용 가능 여부",
];

export function QuickQuestionBar({ onSelect }: { onSelect: (question: string) => void }) {
  return (
    <div className="quick-question-bar">
      {QUESTIONS.map((question) => (
        <button key={question} type="button" onClick={() => onSelect(question)}>
          {question}
        </button>
      ))}
    </div>
  );
}
```

Create `frontend/components/ChatInput.tsx`:

```typescript
import { FormEvent, useState } from "react";

export function ChatInput({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (question: string) => void;
}) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = value.trim();
    if (!question) return;
    onSubmit(question);
    setValue("");
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        value={value}
        disabled={disabled}
        placeholder="청년수당에 대해 질문하세요"
        onChange={(event) => setValue(event.target.value)}
      />
      <button disabled={disabled} type="submit">전송</button>
    </form>
  );
}
```

- [ ] **Step 3: Create page and CSS**

Create `frontend/app/page.tsx`:

```typescript
"use client";

import { useState } from "react";

import { ChatInput } from "../components/ChatInput";
import { MessageList, type ChatMessage } from "../components/MessageList";
import { QuickQuestionBar } from "../components/QuickQuestionBar";
import { sendChatMessage } from "../lib/api";

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(question: string) {
    setError(null);
    setIsLoading(true);
    setMessages((current) => [...current, { role: "user", content: question }]);
    try {
      const response = await sendChatMessage(question);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.answer, response },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "요청에 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="chat-shell">
        <header>
          <h1>청년수당 안내 챗봇</h1>
          <p>주민등록번호, 계좌번호 등 민감정보는 입력하지 마세요.</p>
        </header>
        <QuickQuestionBar onSelect={ask} />
        <MessageList messages={messages} />
        {isLoading ? <p className="status">답변을 생성하는 중입니다.</p> : null}
        {error ? <p className="error">{error}</p> : null}
        <ChatInput disabled={isLoading} onSubmit={ask} />
      </section>
    </main>
  );
}
```

Create `frontend/app/globals.css`:

```css
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #f6f7f9;
  color: #15171a;
}

.page {
  min-height: 100vh;
  padding: 32px 16px;
}

.chat-shell {
  width: min(920px, 100%);
  margin: 0 auto;
}

.quick-question-bar,
.chat-input {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.message-list {
  display: grid;
  gap: 12px;
  margin: 24px 0;
}

.message {
  padding: 12px 14px;
  border: 1px solid #d9dde3;
  background: #fff;
  border-radius: 8px;
}

.message.user {
  background: #e9f2ff;
}

.source-item {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #e3e6eb;
}

.chat-input input {
  flex: 1;
  min-width: 220px;
  padding: 10px;
}

button {
  padding: 10px 12px;
  cursor: pointer;
}

.error {
  color: #b42318;
}
```

- [ ] **Step 4: Run frontend tests**

Run:

```powershell
cd frontend
npm test
```

Expected: tests pass.

- [ ] **Step 5: Commit chat UI**

```powershell
git add frontend
git commit -m "feat: add participant chat UI"
```

## Task 9: Local Integration and Documentation

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Update `.gitignore`**

Ensure `.gitignore` contains:

```gitignore
.env
/venv
/Data
/.worktrees
backend/storage/
frontend/.next/
frontend/node_modules/
frontend/.env.local
```

- [ ] **Step 2: Update root README**

Replace `README.md` with:

```markdown
# 청년수당 RAG Chatbot Project

청년수당 참여자 안내책자 PDF를 기준으로 답변하는 로컬 MVP RAG 챗봇입니다.

## Structure

- `docs/PRD.md`: 제품 요구사항
- `docs/architecture.md`: 아키텍처
- `backend/`: FastAPI, LangChain, LangGraph, Chroma
- `frontend/`: Next.js chat UI
- `Data/청년수당 참여자 안내책자.pdf`: 기준 PDF

## Backend

```powershell
cd backend
venv\Scripts\python.exe -m pytest -v
venv\Scripts\python.exe -m app.indexing.index_pdf
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

## Frontend

```powershell
cd frontend
npm install
npm test
npm run dev
```

Open `http://localhost:3000`.
```

- [ ] **Step 3: Run backend tests**

Run:

```powershell
cd backend
venv\Scripts\python.exe -m pytest -v
```

Expected: all backend tests pass.

- [ ] **Step 4: Run frontend tests**

Run:

```powershell
cd frontend
npm test
```

Expected: all frontend tests pass.

- [ ] **Step 5: Run backend health check manually**

Start backend:

```powershell
cd backend
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected:

```text
status
------
ok
```

- [ ] **Step 6: Commit integration docs**

```powershell
git add README.md .gitignore
git commit -m "docs: add local development instructions"
```

## Final Verification

- [ ] Run backend tests:

```powershell
cd backend
venv\Scripts\python.exe -m pytest -v
```

- [ ] Run frontend tests:

```powershell
cd frontend
npm test
```

- [ ] Run indexing after dependencies and `.env` are ready:

```powershell
cd backend
venv\Scripts\python.exe -m app.indexing.index_pdf
```

Expected: quality report prints total pages, extracted pages, empty pages, total characters, chunk count, and sample chunks.

- [ ] Start backend:

```powershell
cd backend
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

- [ ] Start frontend:

```powershell
cd frontend
npm run dev
```

- [ ] Open `http://localhost:3000`, ask “청년수당 사용처”, and verify:

```text
User question appears.
Assistant answer appears.
If PDF evidence is sufficient, at least one source appears with title, page, excerpt, and chunk_id.
If PDF evidence is insufficient, the response says the booklet did not confirm the information and does not invent an answer.
```

## Self-Review Checklist

- PRD goals are covered by Tasks 1 through 9.
- Architecture modules map to concrete files.
- PDF indexing includes extraction quality checks.
- `/chat` separates HTTP 200 insufficient evidence from HTTP error responses.
- Evidence assessment has score and grader structure.
- Frontend keeps chat state in browser memory only.
- No account, database, public deployment, admin UI, or external official search implementation is included.
