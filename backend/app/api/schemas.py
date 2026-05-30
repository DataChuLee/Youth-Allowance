from typing import Literal

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
    status: Literal[
        "general_answer",
        "answered_from_pdf",
        "insufficient_pdf_evidence",
        "blocked_by_policy",
    ]
    needs_external_search: bool
    intent: Literal["general_answer", "rag"] = "rag"
