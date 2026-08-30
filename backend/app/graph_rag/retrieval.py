from __future__ import annotations

from typing import Any

from neo4j.exceptions import Neo4jError

from app.core.config import Settings, get_settings
from app.graph.state import GraphPolicyResult
from app.graph_rag.pipeline import neo4j_url_candidates

POLICY_RULE_VECTOR_INDEX = "policy_rule_search_embedding"
POLICY_RULE_FULLTEXT_INDEX = "policy_rule_search_fulltext"
ALLOWED_GRAPH_DECISIONS = {
    "allowed",
    "blocked",
    "conditional",
    "required",
    "restricted",
    "insufficient",
}

GRAPH_QUERY_EXPANSIONS = {
    "배민": "배달앱 배달의민족 음식 배달 식비 체크카드 간편결제 클린카드 사용 불가 업종",
    "배달": "배달앱 음식 배달 식비 체크카드 간편결제 클린카드",
    "키보드": "구직활동 장비 학습기기 노트북 태블릿 교육비 구매 고가 사치품",
    "쿠팡": "온라인 구매 쇼핑몰 구매 품목 체크카드 간편결제 클린카드",
    "월세": "월세 주거비 계좌이체 임대차계약서 이체확인증 현금 사용 증빙",
    "계좌이체": "현금 사용 계좌이체 주거비 생활공과금 교육비 증빙서류",
    "자기성장기록서": "자기성장기록서 미제출 제출 기한 지급 중단 보완",
    "안 내면": "미제출 제출 기한 지급 중단 보완",
    "취업": "취업 창업 자격상실신고 취업성공금 고용보험",
}

GRAPH_POLICY_RETRIEVAL_QUERY = """
MATCH (node:PolicyRule)-[:SUPPORTED_BY]->(chunk:SourceChunk)
OPTIONAL MATCH (node)-[:REQUIRES_EVIDENCE]->(doc:EvidenceDocument)
OPTIONAL MATCH (node)-[:VIOLATION_CAUSES]->(penalty:Penalty)
OPTIONAL MATCH (node)-[:HAS_DEADLINE]->(deadline:Deadline)
WITH node, score,
     collect(DISTINCT {
       chunk_id: chunk.key,
       page: chunk.page,
       section_title: chunk.section_title,
       text: chunk.text
     }) AS source_chunks,
     collect(DISTINCT doc.name) AS evidence_documents,
     collect(DISTINCT penalty.name) AS penalties,
     collect(DISTINCT deadline.name) AS deadlines
WITH node, score, source_chunks, evidence_documents, penalties, deadlines,
     head(source_chunks) AS primary_chunk
RETURN
  coalesce(node.evidence_text, primary_chunk.text, node.search_text, node.name) AS text,
  score AS score,
  {
    source: "neo4j_policy_rule_hybrid_cypher",
    rule_id: node.key,
    rule_name: node.name,
    decision: node.decision,
    condition: node.condition,
    evidence_text: node.evidence_text,
    chunk_id: primary_chunk.chunk_id,
    page: primary_chunk.page,
    section_title: primary_chunk.section_title,
    source_chunks: source_chunks,
    evidence_documents: evidence_documents,
    penalties: penalties,
    deadlines: deadlines
  } AS metadata
ORDER BY score DESC
"""


class GraphRetrievalUnavailable(RuntimeError):
    """Raised when Neo4j Graph RAG cannot be used for the current request."""


def expand_policy_question(question: str) -> str:
    expansions = [question]
    for keyword, expansion in GRAPH_QUERY_EXPANSIONS.items():
        if keyword in question:
            expansions.append(expansion)
    return "\n".join(dict.fromkeys(expansions))


def _require_neo4j_settings(settings: Settings) -> tuple[str, str, str, str | None]:
    missing = [
        name
        for name, value in {
            "NEO4J_URI": settings.neo4j_uri,
            "NEO4J_USERNAME": settings.neo4j_username,
            "NEO4J_PASSWORD": settings.neo4j_password,
        }.items()
        if not value
    ]
    if missing:
        raise GraphRetrievalUnavailable(
            f"Neo4j Graph RAG settings are missing: {', '.join(missing)}"
        )
    return (
        str(settings.neo4j_uri),
        str(settings.neo4j_username),
        str(settings.neo4j_password),
        settings.neo4j_database or None,
    )


def _result_from_item(item: Any) -> GraphPolicyResult:
    metadata = dict(getattr(item, "metadata", None) or {})
    evidence = str(metadata.get("evidence_text") or getattr(item, "content", "") or "")
    decision = str(metadata.get("decision") or "insufficient")
    if decision not in ALLOWED_GRAPH_DECISIONS:
        decision = "insufficient"
    return GraphPolicyResult(
        rule_id=str(metadata.get("rule_id") or ""),
        rule_name=str(metadata.get("rule_name") or ""),
        decision=decision,
        page=metadata.get("page"),
        chunk_id=metadata.get("chunk_id"),
        evidence=evidence,
        score=float(metadata.get("graph_score") or 0.0),
        condition=metadata.get("condition"),
        evidence_documents=list(metadata.get("evidence_documents") or []),
        penalties=list(metadata.get("penalties") or []),
        deadlines=list(metadata.get("deadlines") or []),
    )


def retrieve_graph_policy_rules(
    question: str,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
) -> list[GraphPolicyResult]:
    resolved_settings = settings or get_settings()
    uri, username, password, database = _require_neo4j_settings(resolved_settings)

    try:
        from neo4j import GraphDatabase
        from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
        from neo4j_graphrag.retrievers import HybridCypherRetriever
        from neo4j_graphrag.types import RetrieverResultItem
    except ImportError as exc:
        raise GraphRetrievalUnavailable("Graph RAG dependencies are not installed.") from exc

    def result_formatter(record: dict[str, Any]) -> Any:
        metadata = dict(record.get("metadata") or {})
        metadata["graph_score"] = float(record.get("score") or 0.0)
        return RetrieverResultItem(
            content=record.get("text") or "",
            metadata=metadata,
        )

    driver = GraphDatabase.driver(
        neo4j_url_candidates(uri)[0],
        auth=(username, password),
        connection_timeout=5,      # 5초 안에 연결 실패 시 즉시 포기
        max_connection_lifetime=60,
    )
    try:
        embedder = OpenAIEmbeddings(
            model=resolved_settings.openai_embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        retriever = HybridCypherRetriever(
            driver=driver,
            vector_index_name=POLICY_RULE_VECTOR_INDEX,
            fulltext_index_name=POLICY_RULE_FULLTEXT_INDEX,
            retrieval_query=GRAPH_POLICY_RETRIEVAL_QUERY,
            embedder=embedder,
            result_formatter=result_formatter,
            neo4j_database=database,
        )
        results = retriever.search(
            query_text=expand_policy_question(question),
            top_k=top_k or resolved_settings.retrieval_top_k,
        )
        return [_result_from_item(item) for item in results.items]
    except Exception as exc:
        raise GraphRetrievalUnavailable("Neo4j Graph RAG query failed.") from exc
    finally:
        driver.close()
