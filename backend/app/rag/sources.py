from langchain_core.documents import Document

from app.api.schemas import Source
from app.graph.state import GraphPolicyResult

MAX_EXCERPT_LENGTH = 280


def compact_text(text: str) -> str:
    return " ".join(text.split())


def document_to_source(document: Document, score: float = 0.0) -> Source:
    text = compact_text(document.page_content)
    excerpt = text[:MAX_EXCERPT_LENGTH]
    return Source(
        type="pdf",
        title=str(document.metadata.get("title", "청년수당 참여자 안내책자")),
        page=int(document.metadata["page"]),
        excerpt=excerpt,
        chunk_id=str(document.metadata["chunk_id"]),
        score=round(score, 4),
    )


def graph_policy_to_source(policy: GraphPolicyResult) -> Source:
    text = compact_text(policy.evidence)
    return Source(
        type="graph",
        title=policy.rule_name,
        page=int(policy.page or 0),
        excerpt=text[:MAX_EXCERPT_LENGTH],
        chunk_id=str(policy.chunk_id or policy.rule_id),
        score=round(policy.score, 4),
    )
