from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.documents import Document

from app.api.schemas import Source


class RetrievedDocument(BaseModel):
    document: Document
    score: float

    model_config = {"arbitrary_types_allowed": True}


class EvidenceDecision(BaseModel):
    is_sufficient: bool = False
    support_level: Literal["direct", "inferable", "insufficient"] = "insufficient"
    reason: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class IntentDecision(BaseModel):
    intent: Literal["general_answer", "rag"] = "rag"
    reason: str = ""


class PolicyBlocker(BaseModel):
    key: str
    label: str
    message: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class PolicyDecision(BaseModel):
    decision: Literal["blocked", "conditional", "insufficient"] = "insufficient"
    reason: str = ""
    matched_blockers: list[PolicyBlocker] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class GraphState(BaseModel):
    question: str
    intent: Literal["general_answer", "rag"] = "rag"
    search_queries: list[str] = Field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    evidence: EvidenceDecision = Field(default_factory=EvidenceDecision)
    policy: PolicyDecision = Field(default_factory=PolicyDecision)
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    status: Literal[
        "general_answer",
        "answered_from_pdf",
        "insufficient_pdf_evidence",
        "blocked_by_policy",
    ] = "insufficient_pdf_evidence"
    needs_external_search: bool = True

    model_config = {"arbitrary_types_allowed": True}
