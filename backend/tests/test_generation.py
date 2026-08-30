from types import SimpleNamespace

from langchain_core.documents import Document

from app.graph.state import GraphPolicyResult
from app.rag import generation


def test_format_context_includes_chunk_and_page_metadata() -> None:
    document = Document(
        page_content="청년수당 카드 사용처 기준 안내",
        metadata={"page": 3, "chunk_id": "pdf-page-3-chunk-0"},
    )

    context = generation.format_context([document])

    assert "[chunk_id=pdf-page-3-chunk-0 page=3]" in context
    assert "청년수당 카드 사용처 기준 안내" in context


def test_generate_pdf_answer_invokes_chat_model_with_pdf_context(monkeypatch) -> None:
    captured = {}
    document = Document(
        page_content="청년수당 카드는 구직활동 비용에 사용할 수 있습니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0"},
    )
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_chat_model="test-chat-model",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="PDF 근거 기반 답변")

    monkeypatch.setattr(generation, "ChatOpenAI", FakeChatOpenAI)

    answer = generation.generate_pdf_answer("카드는 어디에 써?", [document], settings)

    assert answer == "PDF 근거 기반 답변"
    assert captured["kwargs"] == {
        "api_key": "test-key",
        "model": "test-chat-model",
        "temperature": 0,
    }
    assert captured["messages"][1][0] == "human"
    assert "카드는 어디에 써?" in captured["messages"][1][1]
    assert "pdf-page-12-chunk-0" in captured["messages"][1][1]


def test_stream_pdf_answer_yields_chat_model_chunks(monkeypatch) -> None:
    captured = {}
    document = Document(
        page_content="청년수당 카드는 구직활동 비용에 사용할 수 있습니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0"},
    )
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_chat_model="test-chat-model",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def stream(self, messages):
            captured["messages"] = messages
            yield SimpleNamespace(content="PDF")
            yield SimpleNamespace(content=" 답변")

    monkeypatch.setattr(generation, "ChatOpenAI", FakeChatOpenAI)

    chunks = list(generation.stream_pdf_answer("카드는 어디에 써?", [document], settings))

    assert chunks == ["PDF", " 답변"]
    assert captured["kwargs"] == {
        "api_key": "test-key",
        "model": "test-chat-model",
        "temperature": 0,
    }
    assert "카드는 어디에 써?" in captured["messages"][1][1]
    assert "pdf-page-12-chunk-0" in captured["messages"][1][1]


def test_generate_policy_context_answer_injects_graph_and_pdf_context(monkeypatch) -> None:
    captured = {}
    document = Document(
        page_content="청년수당은 생활비에 사용할 수 있습니다.",
        metadata={"page": 12, "chunk_id": "pdf-page-12-chunk-0"},
    )
    graph_policy = GraphPolicyResult(
        rule_id="spending.asset_accumulation.blocked",
        rule_name="자산 운영·축적 및 투자 지출 불가",
        decision="blocked",
        page=12,
        chunk_id="ocr-page-12-block-7",
        evidence="예금, 적금, 상품권 구매 등",
        score=0.6,
    )
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_chat_model="test-chat-model",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="정책 context 답변")

    monkeypatch.setattr(generation, "ChatOpenAI", FakeChatOpenAI)

    answer = generation.generate_policy_context_answer(
        "쿠팡에서 구매해도 돼?",
        [document],
        [graph_policy],
        settings,
    )

    prompt_text = "\n".join(message[1] for message in captured["messages"])
    assert "Graph PolicyRule 근거" in prompt_text
    assert "PDF 원문 근거" in prompt_text
    assert "직접 적용될 때만" in prompt_text
    assert "플랫폼명만으로 금지 판단하지 않는다" in prompt_text
    assert "자산 운영·축적" in prompt_text
    assert "pdf-page-12-chunk-0" in prompt_text
    assert answer == "정책 context 답변"


def test_food_delivery_context_filters_broad_blockers_and_adds_guidance(monkeypatch) -> None:
    captured = {}
    graph_policies = [
        GraphPolicyResult(
            rule_id="spending.clean_card.blocked_categories",
            rule_name="클린카드 사용 불가 업종 제한",
            decision="blocked",
            page=12,
            chunk_id="ocr-page-12-block-4",
            evidence="호텔, 주점, 복권 판매, 피부미용, 마사지 등 48개 업종",
            score=1.0,
        ),
        GraphPolicyResult(
            rule_id="card.simple_payment.blocked",
            rule_name="간편결제 사용 불가",
            decision="blocked",
            page=12,
            chunk_id="ocr-page-12-block-1",
            evidence="카카오페이, 네이버페이 등 간편 결제 불가",
            score=0.8,
        ),
        GraphPolicyResult(
            rule_id="spending.living.allowed",
            rule_name="생활비 사용 가능",
            decision="allowed",
            page=12,
            chunk_id="ocr-page-12-block-3",
            evidence="생활비 | 식비, 통신비, 교통비 등",
            score=0.7,
        ),
    ]
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_chat_model="test-chat-model",
    )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="배민은 조건부 가능")

    monkeypatch.setattr(generation, "ChatOpenAI", FakeChatOpenAI)

    selected = generation.select_graph_policies_for_question(
        "배민에서 청년수당을 사용해도 돼?",
        graph_policies,
    )
    answer = generation.generate_policy_context_answer(
        "배민에서 청년수당을 사용해도 돼?",
        [],
        selected,
        settings,
    )

    prompt_text = "\n".join(message[1] for message in captured["messages"])
    assert [policy.rule_id for policy in selected] == [
        "card.simple_payment.blocked",
        "spending.living.allowed",
    ]
    assert "배민/배달앱 자체를 클린카드 사용 불가 업종이라고 단정하지 않는다" in prompt_text
    assert "식비 또는 생활비 목적" in prompt_text
    assert "전용 체크카드로 직접 결제" in prompt_text
    assert "간편결제는 불가" in prompt_text
    assert "클린카드 사용 불가 업종 제한" not in prompt_text
    assert answer == "배민은 조건부 가능"
