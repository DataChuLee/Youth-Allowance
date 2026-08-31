from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_OCR_JSONL_PATH = (
    PROJECT_ROOT / "study" / "ocr_output" / "upstage_youth_allowance_parse.jsonl"
)
DEFAULT_ARTIFACT_DIR = BACKEND_DIR / "graph_rag_artifacts"

NODE_LABELS = [
    "Program",
    "EligibilityCriterion",
    "ExclusionCriterion",
    "Benefit",
    "PaymentRound",
    "SpendingCategory",
    "SpendingItem",
    "PaymentMethod",
    "EvidenceDocument",
    "RequiredField",
    "Obligation",
    "Report",
    "Deadline",
    "Penalty",
    "Restriction",
    "ExceptionRule",
    "PolicyRule",
    "SourceChunk",
]

RELATIONSHIP_TYPES = [
    "HAS_ELIGIBILITY",
    "HAS_EXCLUSION",
    "PROVIDES_BENEFIT",
    "HAS_PAYMENT_ROUND",
    "ALLOWS",
    "PROHIBITS",
    "RESTRICTS",
    "CONTAINS",
    "ALLOWED_BY_METHOD",
    "PROHIBITED_BY_METHOD",
    "REQUIRES_EVIDENCE",
    "MUST_INCLUDE",
    "HAS_DEADLINE",
    "VIOLATION_CAUSES",
    "HAS_EXCEPTION",
    "SUPPORTED_BY",
]

SECTION_PATTERN = re.compile(r"(?m)(^#{1,3}\s+.+$)")
TABLE_ROW_PATTERN = re.compile(r"(?m)(^\|.+\|$)")
BULLET_PATTERN = re.compile(r"(?m)(^\s*(?:[-*]|[0-9]+[.)]|[①-⑩]|◆|□|☑|▶)\s*.+$)")
IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
HTML_TAG_PATTERN = re.compile(r"</?(?:table|thead|tbody|tr|td|th)[^>]*>")
LOGO_PATTERN = re.compile(r"\b(?:SEOUL|SE♡UL|SOUL|M!)\b", re.IGNORECASE)
CYPHER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")
POLICY_BLOCK_ID_PATTERN = re.compile(r"^ocr-page-(?P<page>\d+)-block-\d+$")
INLINE_POLICY_MARKER_PATTERN = re.compile(
    r"(?<!^)(?=\s*(?:[◆■□☑○]\s+|Usage example\b|사용 예\b|유의 사항\b|제출 증빙서류\b|현금 사용 가능 항목\b))"
)
POLICY_BOUNDARY_START_PATTERN = re.compile(
    r"^(?:#{1,3}\s+|[◆■□☑○]\s+|Usage example\b|사용 예\b|유의 사항\b|제출 증빙서류\b|현금 사용 가능 항목\b)"
)
DEFAULT_POLICY_TARGET_CHARS = 650
DEFAULT_POLICY_MAX_CHARS = 900


@dataclass(frozen=True)
class OcrPage:
    page: int
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PolicyBlock:
    block_id: str
    page: int
    section_title: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class GraphRelationship:
    type: str
    target_label: str
    target_key: str
    target_name: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManualGraphItem:
    id: str
    label: str
    node_label: str
    decision: str
    source_block_id: str
    evidence_text: str
    properties: dict[str, Any] = field(default_factory=dict)
    relationships: list[GraphRelationship] = field(default_factory=list)


def resolve_project_path(value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_graph_rag_env() -> dict[str, str | None]:
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "OCR_JSONL_PATH": os.getenv("OCR_JSONL_PATH"),
        "GRAPH_RAG_ARTIFACT_DIR": os.getenv("GRAPH_RAG_ARTIFACT_DIR"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
    }


def artifact_paths(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Path]:
    return {
        "cleaned_pages": artifact_dir / "cleaned_pages.json",
        "policy_blocks": artifact_dir / "policy_blocks.json",
        "graph_items": artifact_dir / "graph_items.yaml",
        "graph_nodes_edges": artifact_dir / "graph_nodes_edges.json",
    }


def load_ocr_pages(path: Path = DEFAULT_OCR_JSONL_PATH) -> list[OcrPage]:
    pages: list[OcrPage] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata", {})
        page_number = int(metadata.get("page", row.get("page", line_number)))
        pages.append(
            OcrPage(
                page=page_number,
                text=str(row.get("page_content", "")),
                metadata=dict(metadata),
            )
        )
    return pages


def clean_ocr_text(text: str) -> str:
    cleaned = IMAGE_PATTERN.sub(" ", text)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = LOGO_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = cleaned.replace("\\", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def clean_ocr_pages(pages: list[OcrPage]) -> list[OcrPage]:
    return [
        OcrPage(page=page.page, text=clean_ocr_text(page.text), metadata=page.metadata)
        for page in pages
    ]


def latest_section_title(text: str, offset: int) -> str:
    title = ""
    for match in SECTION_PATTERN.finditer(text):
        if match.start() > offset:
            break
        title = match.group(1).lstrip("#").strip()
    return title


def split_text_by_policy_markers(text: str) -> list[tuple[int, int]]:
    starts = {0}
    for pattern in (
        SECTION_PATTERN,
        BULLET_PATTERN,
        TABLE_ROW_PATTERN,
        INLINE_POLICY_MARKER_PATTERN,
    ):
        for match in pattern.finditer(text):
            if pattern is INLINE_POLICY_MARKER_PATTERN and text[: match.start()].rstrip().endswith("|"):
                continue
            starts.add(match.start())
    sorted_starts = sorted(starts)
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(sorted_starts):
        end = sorted_starts[index + 1] if index + 1 < len(sorted_starts) else len(text)
        if text[start:end].strip():
            spans.append((start, end))
    return spans


def normalized_policy_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def trimmed_text_span(value: str, start: int) -> tuple[str, int, int]:
    leading_length = len(value) - len(value.lstrip())
    trailing_end = len(value.rstrip())
    return (
        value[leading_length:trailing_end],
        start + leading_length,
        start + trailing_end,
    )


def sentence_fragments(value: str, start: int) -> list[tuple[str, int, int]]:
    fragments: list[tuple[str, int, int]] = []
    cursor = 0
    for match in SENTENCE_BOUNDARY_PATTERN.finditer(value):
        fragment, fragment_start, fragment_end = trimmed_text_span(
            value[cursor:match.start()],
            start + cursor,
        )
        if fragment:
            fragments.append((fragment, fragment_start, fragment_end))
        cursor = match.end()
    fragment, fragment_start, fragment_end = trimmed_text_span(value[cursor:], start + cursor)
    if fragment:
        fragments.append((fragment, fragment_start, fragment_end))
    return fragments


def hard_split_fragment(value: str, start: int, max_chars: int) -> list[tuple[str, int, int]]:
    if len(value) <= max_chars:
        return [(value, start, start + len(value))]

    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    while cursor < len(value):
        hard_end = min(cursor + max_chars, len(value))
        if hard_end < len(value):
            soft_end = value.rfind(" ", cursor + max_chars // 2, hard_end + 1)
            if soft_end > cursor:
                hard_end = soft_end
        piece, piece_start, piece_end = trimmed_text_span(value[cursor:hard_end], start + cursor)
        if piece:
            pieces.append((piece, piece_start, piece_end))
        cursor = hard_end
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
    return pieces


def split_policy_span(
    value: str,
    start: int,
    *,
    max_chars: int,
) -> list[tuple[str, int, int]]:
    candidate, candidate_start, candidate_end = trimmed_text_span(value, start)
    if not candidate:
        return []
    if len(candidate) <= max_chars:
        return [(candidate, candidate_start, candidate_end)]

    # 긴 marker span은 문장/줄 단위로 다시 나눠 서로 다른 정책 근거가 한 chunk에 섞이지 않게 한다.
    chunks: list[tuple[str, int, int]] = []
    for line_match in re.finditer(r"[^\n]+", candidate):
        line, line_start, _ = trimmed_text_span(
            line_match.group(0),
            candidate_start + line_match.start(),
        )
        if not line:
            continue
        for fragment, fragment_start, _ in sentence_fragments(line, line_start):
            chunks.extend(hard_split_fragment(fragment, fragment_start, max_chars))
    return chunks


def should_merge_with_previous(previous: str, current: str, target_chars: int) -> bool:
    if len(previous) + len(current) + 1 > target_chars:
        return False
    # 표 행은 서로 붙어 있어야 증빙서류/항목 관계를 잃지 않는다.
    if POLICY_BOUNDARY_START_PATTERN.match(current):
        return False
    if previous.startswith("|") and current.startswith("|"):
        return True
    if current.startswith("|"):
        return False
    if re.match(r"^(?:[-*]|[0-9]+[.)]|[①-⑩]|◆|□|☑|▶)", current):
        return False
    return True


def build_policy_blocks(
    pages: list[OcrPage],
    *,
    target_chars: int = DEFAULT_POLICY_TARGET_CHARS,
    max_chars: int | None = None,
) -> list[PolicyBlock]:
    resolved_max_chars = (
        max_chars
        if max_chars is not None
        else max(target_chars, min(DEFAULT_POLICY_MAX_CHARS, int(target_chars * 1.4)))
    )
    blocks: list[PolicyBlock] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        spans = split_text_by_policy_markers(text)
        page_blocks: list[tuple[str, int, int]] = []
        for start, end in spans:
            for candidate, candidate_start, candidate_end in split_policy_span(
                text[start:end],
                start,
                max_chars=resolved_max_chars,
            ):
                if page_blocks and should_merge_with_previous(
                    page_blocks[-1][0],
                    candidate,
                    target_chars,
                ):
                    previous_text, previous_start, _ = page_blocks[-1]
                    page_blocks[-1] = (
                        f"{previous_text}\n{candidate}",
                        previous_start,
                        candidate_end,
                    )
                else:
                    page_blocks.append((candidate, candidate_start, candidate_end))

        for page_block_index, (block_text, start, end) in enumerate(page_blocks):
            blocks.append(
                PolicyBlock(
                    block_id=f"ocr-page-{page.page}-block-{page_block_index}",
                    page=page.page,
                    section_title=latest_section_title(text, start),
                    text=block_text,
                    char_start=start,
                    char_end=end,
                )
            )
    return blocks


def graph_schema() -> dict[str, list[str]]:
    return {"node_labels": NODE_LABELS, "relationship_types": RELATIONSHIP_TYPES}


def node(label: str, key: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"label": label, "key": key, "properties": properties}


def relationship(
    relationship_type: str,
    start_label: str,
    start_key: str,
    end_label: str,
    end_key: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": relationship_type,
        "start_label": start_label,
        "start_key": start_key,
        "end_label": end_label,
        "end_key": end_key,
        "properties": properties or {},
    }


def deduplicate_by_signature(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        signature = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique


def page_number_from_block_id(block_id: str) -> int | None:
    match = POLICY_BLOCK_ID_PATTERN.fullmatch(block_id)
    if not match:
        return None
    return int(match.group("page"))


def evidence_overlap_score(block_text: str, evidence_text: str) -> int:
    normalized_block = normalized_policy_text(block_text)
    normalized_evidence = normalized_policy_text(evidence_text)
    if not normalized_block or not normalized_evidence:
        return 0
    if normalized_evidence in normalized_block:
        return 10_000 + len(normalized_evidence)

    terms = {
        term
        for term in re.split(r"\W+", normalized_evidence)
        if len(term) >= 2
    }
    return sum(len(term) for term in terms if term in normalized_block)


def resolve_source_block_id(
    item: ManualGraphItem,
    blocks: list[PolicyBlock],
    blocks_by_id: dict[str, PolicyBlock],
) -> str:
    source_block = blocks_by_id.get(item.source_block_id)
    source_page = (
        source_block.page
        if source_block is not None
        else page_number_from_block_id(item.source_block_id)
    )
    candidate_blocks = [
        block
        for block in blocks
        if source_page is None or block.page == source_page
    ]

    # 재분할 후에는 기존 수동 source_block_id가 같은 page의 더 큰 옛 chunk를 가리킬 수 있다.
    scored_blocks = [
        (evidence_overlap_score(block.text, item.evidence_text), len(block.text), block)
        for block in candidate_blocks
    ]
    scored_blocks = [scored for scored in scored_blocks if scored[0] > 0]
    if scored_blocks:
        _, _, best_block = max(
            scored_blocks,
            key=lambda scored: (scored[0], -scored[1], -scored[2].char_start),
        )
        return best_block.block_id

    if source_block is None:
        raise ValueError(f"unknown source_block_id: {item.source_block_id}")
    return source_block.block_id


def build_graph_nodes_edges(
    blocks: list[PolicyBlock],
    items: list[ManualGraphItem],
) -> dict[str, list[dict[str, Any]]]:
    blocks_by_id = {block.block_id: block for block in blocks}
    nodes: list[dict[str, Any]] = [
        node(
            "SourceChunk",
            block.block_id,
            {
                "page": block.page,
                "section_title": block.section_title,
                "text": block.text,
                "char_start": block.char_start,
                "char_end": block.char_end,
            },
        )
        for block in blocks
    ]
    relationships: list[dict[str, Any]] = []

    for item in items:
        source_block_id = resolve_source_block_id(item, blocks, blocks_by_id)
        item_properties = {
            "id": item.id,
            "name": item.label,
            "decision": item.decision,
            "evidence_text": item.evidence_text,
            **item.properties,
        }
        nodes.append(node(item.node_label, item.id, item_properties))
        relationships.append(
            relationship(
                "SUPPORTED_BY",
                item.node_label,
                item.id,
                "SourceChunk",
                source_block_id,
                {"evidence_text": item.evidence_text},
            )
        )
        for item_relationship in item.relationships:
            nodes.append(
                node(
                    item_relationship.target_label,
                    item_relationship.target_key,
                    {"name": item_relationship.target_name},
                )
            )
            relationships.append(
                relationship(
                    item_relationship.type,
                    item.node_label,
                    item.id,
                    item_relationship.target_label,
                    item_relationship.target_key,
                    item_relationship.properties,
                )
            )

    graph = {
        "nodes": deduplicate_by_signature(nodes),
        "relationships": deduplicate_by_signature(relationships),
    }
    validate_graph_nodes_edges(graph)
    return graph


def validate_graph_nodes_edges(graph: dict[str, list[dict[str, Any]]]) -> None:
    node_keys = {(item["label"], item["key"]) for item in graph.get("nodes", [])}
    for item in graph.get("nodes", []):
        if item["label"] not in NODE_LABELS:
            raise ValueError(f"unknown node label: {item['label']}")
    for item in graph.get("relationships", []):
        if item["type"] not in RELATIONSHIP_TYPES:
            raise ValueError(f"unknown relationship type: {item['type']}")
        start_key = (item["start_label"], item["start_key"])
        end_key = (item["end_label"], item["end_key"])
        if start_key not in node_keys or end_key not in node_keys:
            raise ValueError(f"missing relationship endpoint: {item}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in value] if isinstance(value, list) else value
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def write_graph_items_yaml_template(
    path: Path,
    blocks: list[PolicyBlock],
    *,
    max_examples: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = "\n".join(
        (
            f"  # - id: policy.rule.{index}\n"
            f"  #   label: \"수동 추출 정책 항목\"\n"
            f"  #   node_label: PolicyRule\n"
            f"  #   decision: conditional\n"
            f"  #   source_block_id: {block.block_id}\n"
            f"  #   evidence_text: \"{block.text[:80].replace(chr(10), ' ')}\"\n"
            f"  #   properties: {{}}\n"
            f"  #   relationships: []"
        )
        for index, block in enumerate(blocks[:max_examples], start=1)
    )
    content = (
        "version: 1\n"
        "description: Manual Graph RAG policy items. Fill this file after reviewing policy_blocks.json.\n"
        "items: []\n\n"
        "# Examples copied from the first policy blocks:\n"
        f"{examples}\n"
    )
    path.write_text(content, encoding="utf-8")


def load_manual_graph_items(path: Path) -> list[ManualGraphItem]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML이 설치되어 있지 않아 graph_items.yaml을 읽을 수 없습니다.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = payload.get("items") or []
    manual_items: list[ManualGraphItem] = []
    for item in items:
        relationships = [
            GraphRelationship(
                type=relationship_item["type"],
                target_label=relationship_item["target_label"],
                target_key=relationship_item["target_key"],
                target_name=relationship_item["target_name"],
                properties=relationship_item.get("properties") or {},
            )
            for relationship_item in item.get("relationships", [])
        ]
        manual_items.append(
            ManualGraphItem(
                id=item["id"],
                label=item["label"],
                node_label=item.get("node_label", "PolicyRule"),
                decision=item["decision"],
                source_block_id=item["source_block_id"],
                evidence_text=item["evidence_text"],
                properties=item.get("properties") or {},
                relationships=relationships,
            )
        )
    return manual_items


def neo4j_url_candidates(uri: str) -> list[str]:
    parsed = urlparse(uri)
    if parsed.scheme == "neo4j" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        bolt_uri = urlunparse(parsed._replace(scheme="bolt"))
        return [bolt_uri, uri] if bolt_uri != uri else [uri]
    return [uri]


def create_neo4j_graph_from_env():
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        from langchain_neo4j import Neo4jGraph
    except ImportError as exc:
        raise RuntimeError(
            "langchain_neo4j가 설치되어 있지 않습니다. `pip install langchain-neo4j` 후 다시 실행하세요."
        ) from exc

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE") or None
    missing = [
        name
        for name, value in {
            "NEO4J_URI": uri,
            "NEO4J_USERNAME": username,
            "NEO4J_PASSWORD": password,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Neo4j 환경변수가 누락되었습니다: {', '.join(missing)}")
    errors: list[str] = []
    for candidate_uri in neo4j_url_candidates(uri):
        try:
            return Neo4jGraph(
                url=candidate_uri,
                username=username,
                password=password,
                database=database,
                refresh_schema=False,
            )
        except ValueError as exc:
            errors.append(f"{candidate_uri}: {exc}")
    raise RuntimeError(
        "Neo4j 연결에 실패했습니다. 로컬 Neo4j Desktop은 보통 "
        "`NEO4J_URI=bolt://127.0.0.1:7687`를 사용해야 합니다. "
        f"시도한 URI: {'; '.join(errors)}"
    )



def neo4j_constraint_queries() -> list[str]:
    return [
        "CREATE CONSTRAINT source_chunk_block_id IF NOT EXISTS FOR (n:SourceChunk) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT policy_rule_key IF NOT EXISTS FOR (n:PolicyRule) REQUIRE n.key IS UNIQUE",
        "CREATE INDEX spending_item_name IF NOT EXISTS FOR (n:SpendingItem) ON (n.name)",
        "CREATE INDEX payment_method_name IF NOT EXISTS FOR (n:PaymentMethod) ON (n.name)",
        "CREATE INDEX evidence_document_name IF NOT EXISTS FOR (n:EvidenceDocument) ON (n.name)",
    ]


def neo4j_reset_graph_queries(*, delete_all: bool = False) -> list[str]:
    if delete_all:
        return ["MATCH (n) DETACH DELETE n"]
    labels = ", ".join(cypher_literal(label) for label in NODE_LABELS)
    return [
        (
            "MATCH (n) "
            f"WHERE any(label IN labels(n) WHERE label IN [{labels}]) "
            "DETACH DELETE n"
        )
    ]


def cypher_property_key(key: Any) -> str:
    key_text = str(key)
    if CYPHER_IDENTIFIER_PATTERN.fullmatch(key_text):
        return key_text
    return f"`{key_text.replace('`', '``')}`"


def cypher_literal(value: Any) -> str:
    # Cypher map literal은 JSON과 달리 property key를 따옴표로 감싸지 않는다.
    if isinstance(value, dict):
        entries = [
            f"{cypher_property_key(key)}: {cypher_literal(item)}"
            for key, item in value.items()
        ]
        return "{" + ", ".join(entries) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cypher_literal(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def upsert_node_query(item: dict[str, Any]) -> str:
    properties = {"key": item["key"], **item.get("properties", {})}
    return (
        f"MERGE (n:{item['label']} {{key: {cypher_literal(item['key'])}}})\n"
        f"SET n += {cypher_literal(properties)}"
    )


def upsert_relationship_query(item: dict[str, Any]) -> str:
    properties = item.get("properties", {})
    return (
        f"MATCH (a:{item['start_label']} {{key: {cypher_literal(item['start_key'])}}})\n"
        f"MATCH (b:{item['end_label']} {{key: {cypher_literal(item['end_key'])}}})\n"
        f"MERGE (a)-[r:{item['type']}]->(b)\n"
        f"SET r += {cypher_literal(properties)}"
    )


def smoke_test_queries() -> dict[str, str]:
    return {
        "policy_by_keyword": (
            "MATCH (r:PolicyRule)-[:SUPPORTED_BY]->(s:SourceChunk) "
            "WHERE r.name CONTAINS $keyword OR s.text CONTAINS $keyword "
            "RETURN r.key AS rule_id, r.name AS rule_name, r.decision AS decision, "
            "s.page AS page, s.text AS evidence LIMIT 10"
        ),
        "evidence_required": (
            "MATCH (r:PolicyRule)-[:REQUIRES_EVIDENCE]->(d:EvidenceDocument) "
            "WHERE r.name CONTAINS $keyword "
            "RETURN r.name AS rule_name, d.name AS document_name LIMIT 10"
        ),
    }
