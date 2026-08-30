from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.api.schemas import Source


class RetrievedDocument(BaseModel):
    document: Document
    score: float

    model_config = {"arbitrary_types_allowed": True}


class GraphPolicyResult(BaseModel):
    rule_id: str
    rule_name: str
    decision: Literal["allowed", "blocked", "conditional", "required", "restricted", "insufficient"]
    page: int | None = None
    chunk_id: str | None = None
    evidence: str = ""
    score: float = 0.0
    condition: str | None = None
    evidence_documents: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    deadlines: list[str] = Field(default_factory=list)


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
    graph_policy_results: list[GraphPolicyResult] = Field(default_factory=list)
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
    # 요청에 포함된 최근 대화 문맥이며 그래프 실행이 끝나면 폐기한다.
    messages: list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
