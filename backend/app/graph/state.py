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
