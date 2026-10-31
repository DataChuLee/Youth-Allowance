import re
from collections.abc import Iterator
from typing import Any

import pytest

from app.core.config import DEFAULT_OCR_JSONL_PATH
from app.indexing.index_pdf import load_ocr_pages
from app.rules.loader import load_rules

RULE_FILES = ("spending", "eligibility")
VERDICTS = {"allowed", "conditional", "prohibited", "unclear"}


def iter_sources(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        yield from node.get("sources", [])
        for value in node.values():
            yield from iter_sources(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_sources(value)


def compact(text: str) -> str:
    # OCR 줄바꿈과 표 구분자 위치가 원문 인용과 달라도 같은 문장으로 본다.
    return re.sub(r"[\s|]", "", text)


@pytest.mark.parametrize("name", RULE_FILES)
def test_rule_ids_are_unique_within_each_section(name: str) -> None:
    rules = load_rules(name)
    for section in rules.values():
        if isinstance(section, list):
            ids = [entry["id"] for entry in section]
            assert len(ids) == len(set(ids))


def test_spending_rules_use_known_verdicts_and_cite_sources() -> None:
    rules = load_rules("spending")
    for entry in [*rules["payment_methods"], *rules["items"]]:
        assert entry["names"] and entry["sources"], entry["id"]
        for key in ("verdict", "transfer", "reimbursement"):
            assert entry.get(key, "prohibited") in VERDICTS, (entry["id"], key)
    assert len(rules["blocked_merchants"]["names"]) == 48


@pytest.mark.skipif(not DEFAULT_OCR_JSONL_PATH.is_file(), reason="OCR 원문(Data/)이 없는 환경")
@pytest.mark.parametrize("name", RULE_FILES)
def test_rule_quotes_appear_on_cited_booklet_page(name: str) -> None:
    pages = {page.metadata["page"]: compact(page.page_content) for page in load_ocr_pages(DEFAULT_OCR_JSONL_PATH)}
    missing = [
        (source["page"], source["quote"])
        for source in iter_sources(load_rules(name))
        if compact(source["quote"]) not in pages.get(source["page"], "")
    ]
    assert missing == []


@pytest.mark.skipif(not DEFAULT_OCR_JSONL_PATH.is_file(), reason="OCR 원문(Data/)이 없는 환경")
def test_blocked_merchant_names_appear_in_page_12_table() -> None:
    page_12 = compact(load_ocr_pages(DEFAULT_OCR_JSONL_PATH)[11].page_content)
    missing = [name for name in load_rules("spending")["blocked_merchants"]["names"] if compact(name) not in page_12]
    assert missing == []
