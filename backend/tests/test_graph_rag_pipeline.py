from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.graph_rag.pipeline import (
    GraphRelationship,
    ManualGraphItem,
    OcrPage,
    build_graph_nodes_edges,
    build_policy_blocks,
    clean_ocr_text,
    load_manual_graph_items,
    load_ocr_pages,
    neo4j_url_candidates,
    neo4j_reset_graph_queries,
    upsert_node_query,
    upsert_relationship_query,
    validate_graph_nodes_edges,
)


def test_load_ocr_pages_reads_page_content_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "ocr.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "page_content": "첫 페이지",
                        "metadata": {"page": 3, "source": "sample.pdf"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "page_content": "둘째 페이지",
                        "metadata": {"page": 4},
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    pages = load_ocr_pages(source)

    assert pages == [
        OcrPage(page=3, text="첫 페이지", metadata={"page": 3, "source": "sample.pdf"}),
        OcrPage(page=4, text="둘째 페이지", metadata={"page": 4}),
    ]


def test_clean_ocr_text_removes_placeholders_and_keeps_policy_text() -> None:
    raw = """
    # 카드 사용 안내 ![image](/image/placeholder)
    <table><tbody><tr><td>월세</td><td>이체확인증</td></tr></tbody></table>
    SE♡UL   M!   SOUL
    """

    cleaned = clean_ocr_text(raw)

    assert "placeholder" not in cleaned
    assert "<table>" not in cleaned
    assert "월세" in cleaned
    assert "이체확인증" in cleaned
    assert "SE♡UL" not in cleaned


def test_build_policy_blocks_uses_stable_page_block_ids() -> None:
    pages = [
        OcrPage(
            page=12,
            text=clean_ocr_text(
                """
                # 카드 사용 안내
                청년수당 지원금은 반드시 체크카드로 사용해야 합니다.
                ◆ 단 카카오페이, 네이버페이 등 간편 결제 불가
                | 사용 예 | 주거비 | 월세 |
                """
            ),
            metadata={},
        )
    ]

    blocks = build_policy_blocks(pages)

    assert [block.block_id for block in blocks] == [
        "ocr-page-12-block-0",
        "ocr-page-12-block-1",
        "ocr-page-12-block-2",
    ]
    assert blocks[0].section_title == "카드 사용 안내"
    assert "체크카드" in blocks[0].text
    assert "간편 결제 불가" in blocks[1].text
    assert "월세" in blocks[2].text


def test_build_policy_blocks_splits_long_unmarked_span_into_smaller_chunks() -> None:
    pages = [
        OcrPage(
            page=12,
            text=clean_ocr_text(
                """
                # Card guide
                Youth allowance must be spent with the check card.
                Online card payments are allowed when they match program purpose.
                Simple payments such as KakaoPay and NaverPay are not allowed.
                Housing costs such as monthly rent, utilities, and management fees are allowed.
                """
            ),
            metadata={},
        )
    ]

    blocks = build_policy_blocks(pages, target_chars=110)

    assert len(blocks) >= 3
    assert all(len(block.text) <= 154 for block in blocks)
    assert any(
        "Simple payments" in block.text and "Housing costs" not in block.text
        for block in blocks
    )
    assert any("Housing costs" in block.text for block in blocks)


def test_build_policy_blocks_splits_inline_policy_boundary_markers() -> None:
    pages = [
        OcrPage(
            page=12,
            text=(
                "Card guide. Check card use is required. "
                "Available items include housing, living, and education costs. "
                "◆ Simple payments such as KakaoPay and NaverPay are not allowed. "
                "Usage example | Housing | monthly rent and utilities |"
            ),
            metadata={},
        )
    ]

    blocks = build_policy_blocks(pages)

    assert any(block.text.startswith("◆ Simple payments") for block in blocks)
    assert any(block.text.startswith("Usage example") for block in blocks)
    assert not any(
        "Simple payments" in block.text and "Usage example" in block.text
        for block in blocks
    )


def test_build_graph_nodes_edges_adds_source_chunks_and_supported_by_edges() -> None:
    pages = [
        OcrPage(
            page=13,
            text="월세는 계좌이체 가능하나 임대차계약서와 이체확인증을 제출해야 합니다.",
            metadata={},
        )
    ]
    block = build_policy_blocks(pages)[0]
    item = ManualGraphItem(
        id="cash_transfer.rent.conditional",
        label="월세 계좌이체 조건",
        node_label="PolicyRule",
        decision="conditional",
        source_block_id=block.block_id,
        evidence_text="월세는 계좌이체 가능하나 임대차계약서와 이체확인증을 제출해야 합니다.",
        properties={"condition": "증빙서류 제출 시 인정"},
        relationships=[
            GraphRelationship(
                type="REQUIRES_EVIDENCE",
                target_label="EvidenceDocument",
                target_key="doc_lease",
                target_name="임대차계약서",
            ),
        ],
    )

    graph = build_graph_nodes_edges([block], [item])

    node_keys = {(node["label"], node["key"]) for node in graph["nodes"]}
    relationship_types = [relationship["type"] for relationship in graph["relationships"]]
    assert ("SourceChunk", block.block_id) in node_keys
    assert ("PolicyRule", "cash_transfer.rent.conditional") in node_keys
    assert ("EvidenceDocument", "doc_lease") in node_keys
    assert "SUPPORTED_BY" in relationship_types
    assert "REQUIRES_EVIDENCE" in relationship_types


def test_build_graph_nodes_edges_relinks_stale_source_block_to_evidence_chunk() -> None:
    pages = [
        OcrPage(
            page=12,
            text=clean_ocr_text(
                """
                # Card guide
                Youth allowance must be spent with the check card.
                Online card payments are allowed when they match program purpose.
                Simple payments such as KakaoPay and NaverPay are not allowed.
                Housing costs such as monthly rent, utilities, and management fees are allowed.
                """
            ),
            metadata={},
        )
    ]
    blocks = build_policy_blocks(pages, target_chars=110)
    evidence_text = "Simple payments such as KakaoPay and NaverPay are not allowed."
    evidence_block = next(block for block in blocks if evidence_text in block.text)
    item = ManualGraphItem(
        id="card.simple_payment.blocked",
        label="Simple payments blocked",
        node_label="PolicyRule",
        decision="blocked",
        source_block_id="ocr-page-12-block-0",
        evidence_text=evidence_text,
        relationships=[],
    )

    graph = build_graph_nodes_edges(blocks, [item])

    supported_by = next(
        relationship
        for relationship in graph["relationships"]
        if relationship["type"] == "SUPPORTED_BY"
    )
    assert evidence_block.block_id != item.source_block_id
    assert supported_by["end_key"] == evidence_block.block_id


def test_validate_graph_nodes_edges_rejects_relationship_with_missing_endpoint() -> None:
    graph = {
        "nodes": [{"label": "PolicyRule", "key": "rule.a", "properties": {}}],
        "relationships": [
            {
                "type": "SUPPORTED_BY",
                "start_label": "PolicyRule",
                "start_key": "rule.a",
                "end_label": "SourceChunk",
                "end_key": "missing",
                "properties": {},
            }
        ],
    }

    with pytest.raises(ValueError, match="missing relationship endpoint"):
        validate_graph_nodes_edges(graph)


def test_load_manual_graph_items_reads_yaml_relationships(tmp_path: Path) -> None:
    source = tmp_path / "graph_items.yaml"
    source.write_text(
        """
version: 1
items:
  - id: card.simple_payment.blocked
    label: 간편결제 불가
    node_label: PolicyRule
    decision: blocked
    source_block_id: ocr-page-12-block-1
    evidence_text: 카카오페이, 네이버페이 등 간편 결제 불가
    properties:
      condition: 전용 체크카드 직접 결제 필요
    relationships:
      - type: PROHIBITED_BY_METHOD
        target_label: PaymentMethod
        target_key: method_simple_payment
        target_name: 간편결제
        properties:
          reason: 간편결제 불가
""".strip(),
        encoding="utf-8",
    )

    items = load_manual_graph_items(source)

    assert items == [
        ManualGraphItem(
            id="card.simple_payment.blocked",
            label="간편결제 불가",
            node_label="PolicyRule",
            decision="blocked",
            source_block_id="ocr-page-12-block-1",
            evidence_text="카카오페이, 네이버페이 등 간편 결제 불가",
            properties={"condition": "전용 체크카드 직접 결제 필요"},
            relationships=[
                GraphRelationship(
                    type="PROHIBITED_BY_METHOD",
                    target_label="PaymentMethod",
                    target_key="method_simple_payment",
                    target_name="간편결제",
                    properties={"reason": "간편결제 불가"},
                )
            ],
        )
    ]


def test_neo4j_url_candidates_prefers_bolt_for_local_routing_uri() -> None:
    assert neo4j_url_candidates("neo4j://127.0.0.1:7687") == [
        "bolt://127.0.0.1:7687",
        "neo4j://127.0.0.1:7687",
    ]
    assert neo4j_url_candidates("neo4j://localhost:7687") == [
        "bolt://localhost:7687",
        "neo4j://localhost:7687",
    ]


def test_neo4j_url_candidates_keeps_remote_routing_uri() -> None:
    assert neo4j_url_candidates("neo4j+s://example.databases.neo4j.io") == [
        "neo4j+s://example.databases.neo4j.io"
    ]


def test_neo4j_reset_graph_queries_can_clear_entire_database() -> None:
    assert neo4j_reset_graph_queries(delete_all=True) == ["MATCH (n) DETACH DELETE n"]


def test_neo4j_reset_graph_queries_default_clears_known_graph_labels() -> None:
    queries = neo4j_reset_graph_queries()

    assert len(queries) == 1
    assert "MATCH (n)" in queries[0]
    assert "DETACH DELETE n" in queries[0]
    assert "SourceChunk" in queries[0]
    assert "PolicyRule" in queries[0]


def test_upsert_node_query_uses_cypher_map_literal_keys() -> None:
    query = upsert_node_query(
        {
            "label": "SourceChunk",
            "key": "ocr-page-1-block-0",
            "properties": {
                "page": 1,
                "section_title": "",
                "text": '서울시 청년수당\n"참여자" 안내책자',
            },
        }
    )

    assert 'SET n += {key: "ocr-page-1-block-0"' in query
    assert 'page: 1' in query
    assert 'section_title: ""' in query
    assert '"key":' not in query


def test_upsert_relationship_query_uses_cypher_map_literal_keys() -> None:
    query = upsert_relationship_query(
        {
            "type": "SUPPORTED_BY",
            "start_label": "PolicyRule",
            "start_key": "cash_transfer.rent.conditional",
            "end_label": "SourceChunk",
            "end_key": "ocr-page-1-block-0",
            "properties": {"evidence_text": "증빙서류 제출"},
        }
    )

    assert 'SET r += {evidence_text: "증빙서류 제출"}' in query
    assert '"evidence_text":' not in query
