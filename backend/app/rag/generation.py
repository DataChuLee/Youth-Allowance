from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.graph.state import GraphPolicyResult, PolicyDecision
from app.rag.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT

FOOD_DELIVERY_PLATFORM_TERMS = ("배민", "배달", "배달의민족", "음식배달", "음식 배달")
PERSONAL_MONEY_MIX_TERMS = ("합산", "개인 돈", "개인돈", "다른 계좌", "제3자 계좌", "섞")


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        (
            f"[chunk_id={document.metadata.get('chunk_id')} "
            f"page={document.metadata.get('page')}]\n{document.page_content}"
        )
        for document in documents
    )


def is_food_delivery_platform_question(question: str) -> bool:
    return any(term in question for term in FOOD_DELIVERY_PLATFORM_TERMS)


def is_personal_money_mix_question(question: str) -> bool:
    return any(term in question for term in PERSONAL_MONEY_MIX_TERMS)


def is_broad_clean_card_blocker(policy: GraphPolicyResult) -> bool:
    text = f"{policy.rule_id} {policy.rule_name} {policy.evidence}"
    return (
        "클린카드" in text
        or "사용 불가 업종" in text
        or "사용불가 업종" in text
    )


def is_personal_money_mix_policy(policy: GraphPolicyResult) -> bool:
    text = f"{policy.rule_id} {policy.rule_name} {policy.evidence}"
    return "합산" in text or "개인 돈" in text or "개인돈" in text


def select_graph_policies_for_question(
    question: str,
    policies: list[GraphPolicyResult],
) -> list[GraphPolicyResult]:
    selected: list[GraphPolicyResult] = []
    food_delivery_question = is_food_delivery_platform_question(question)
    personal_money_question = is_personal_money_mix_question(question)

    for policy in policies:
        if food_delivery_question and is_broad_clean_card_blocker(policy):
            continue
        if not personal_money_question and is_personal_money_mix_policy(policy):
            continue
        selected.append(policy)
    return selected


def build_question_specific_guidance(question: str) -> str:
    if is_food_delivery_platform_question(question):
        return (
            "질문별 해석 지침:\n"
            "- 이 질문은 배민/배달앱/음식배달 플랫폼 사용 가능 여부다.\n"
            "- 배민/배달앱 자체를 클린카드 사용 불가 업종이라고 단정하지 않는다.\n"
            "- 식비 또는 생활비 목적이고 청년수당 전용 체크카드로 직접 결제하는 경우는 사용 가능하다고 답한다.\n"
            "- 카카오페이, 네이버페이 등 간편결제는 불가 조건으로 안내한다.\n"
            "- 결제수단이 확인되지 않으면 '전용 체크카드 직접 결제라면 가능, 간편결제는 불가'라고 조건부로 답한다."
        )
    return "질문별 해석 지침: 추가 지침 없음."


def format_graph_policy_context(policies: list[GraphPolicyResult]) -> str:
    if not policies:
        return "검색된 Graph PolicyRule 근거가 없습니다."
    return "\n\n".join(
        (
            f"[rule_id={policy.rule_id} decision={policy.decision} "
            f"page={policy.page} chunk_id={policy.chunk_id} score={policy.score:.4f}]\n"
            f"rule_name: {policy.rule_name}\n"
            f"condition: {policy.condition or ''}\n"
            f"evidence: {policy.evidence}\n"
            f"evidence_documents: {', '.join(policy.evidence_documents)}\n"
            f"penalties: {', '.join(policy.penalties)}\n"
            f"deadlines: {', '.join(policy.deadlines)}"
        )
        for policy in policies
    )


POLICY_CONTEXT_SYSTEM_PROMPT = """너는 서울시 청년수당 참여자 안내책자 기반 QA assistant다.

규칙:
- 제공된 Graph PolicyRule 근거와 PDF 원문 근거만 사용해서 답한다.
- 근거에 없는 정책 사실을 추측하지 않는다.
- blocked/restricted rule은 사용자 질문의 품목, 결제수단, 업종, 증빙 조건에 직접 적용될 때만 우선한다.
- 쿠팡, 배민 같은 플랫폼명만으로 금지 판단하지 않는다.
- 자산 운영·축적 및 투자 지출 불가 rule을 모든 물건 구매에 일반화하지 않는다.
- 플랫폼 질문은 구매 품목, 결제수단, 간편결제 여부가 부족하면 조건부로 답하고 추가 확인이 필요하다고 말한다.
- 카카오페이, 네이버페이 등 간편결제, 상품권, 포인트 충전처럼 직접 금지 근거가 있으면 불가 조건으로 말한다.
- 답변은 한국어로 간결하게 작성하고, 가능/불가/조건부/추가 확인 필요 중 하나가 드러나게 한다.
- 목록으로 답변할 때는 상위 항목을 `1.`, `2.`, `3.`처럼 순차 번호로 작성하고, 각 상위 항목의 세부 증빙이나 조건은 `-` 하위 항목으로 작성한다."""


POLICY_CONTEXT_USER_PROMPT = """질문:
{question}

Graph PolicyRule 근거:
{graph_context}

PDF 원문 근거:
{pdf_context}

{question_guidance}

위 근거만 사용해서 답변하세요."""


def generate_policy_context_answer(
    question: str,
    documents: list[Document],
    graph_policies: list[GraphPolicyResult],
    settings: Settings,
) -> str:
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )
    messages = build_policy_context_messages(question, documents, graph_policies)
    response = llm.invoke(messages)
    return str(response.content).strip()


def build_policy_context_messages(
    question: str,
    documents: list[Document],
    graph_policies: list[GraphPolicyResult],
) -> list[tuple[str, str]]:
    return [
        ("system", POLICY_CONTEXT_SYSTEM_PROMPT),
        (
            "human",
            POLICY_CONTEXT_USER_PROMPT.format(
                question=question,
                graph_context=format_graph_policy_context(graph_policies),
                pdf_context=format_context(documents) if documents else "검색된 PDF 근거가 없습니다.",
                question_guidance=build_question_specific_guidance(question),
            ),
        ),
    ]


def stream_policy_context_answer(
    question: str,
    documents: list[Document],
    graph_policies: list[GraphPolicyResult],
    settings: Settings,
) -> Iterator[str]:
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )
    yield from stream_chat_content(
        llm,
        build_policy_context_messages(question, documents, graph_policies),
    )


def generate_pdf_answer(question: str, documents: list[Document], settings: Settings) -> str:
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )
    messages = build_pdf_answer_messages(question, documents)
    response = llm.invoke(messages)
    return str(response.content).strip()


def build_pdf_answer_messages(question: str, documents: list[Document]) -> list[tuple[str, str]]:
    return [
        ("system", ANSWER_SYSTEM_PROMPT),
        (
            "human",
            ANSWER_USER_PROMPT.format(
                question=question,
                context=format_context(documents),
            ),
        ),
    ]


def stream_pdf_answer(
    question: str,
    documents: list[Document],
    settings: Settings,
) -> Iterator[str]:
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )
    yield from stream_chat_content(llm, build_pdf_answer_messages(question, documents))


def stream_chat_content(llm: ChatOpenAI, messages: list[tuple[str, str]]) -> Iterator[str]:
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            if content:
                yield content
            continue
        if content:
            yield str(content)


def build_food_delivery_answer() -> str:
    return (
        "안내책자에 배달의민족이라는 이름이 직접 언급되지는 않지만, "
        "안내책자에는 생활비 항목에 식비가 포함되고 오프라인은 물론 온라인에서도 카드 결제로 사용할 수 있다고 되어 있습니다. "
        "따라서 배달 음식 주문을 식비 목적으로 청년수당 전용 체크카드로 직접 결제하는 경우에는 사용 가능하다고 볼 수 있습니다. "
        "다만 카카오페이, 네이버페이 등 간편결제, 포인트 충전, 상품권·기프티콘 구매 방식은 불가 조건에 해당할 수 있으므로 "
        "배달앱 안에서도 전용 체크카드 직접 결제 방식인지 확인해야 합니다."
    )


def build_study_device_answer() -> str:
    return (
        "안내책자에 해당 물품명이 직접 언급되어 있지 않더라도, "
        "태블릿 PC, 노트북, 의자 등은 인터넷강의 청취, 공부, 면접 준비처럼 청년수당 사업 취지에 부합하는 경우 구입 가능하다고 되어 있습니다. "
        "따라서 질문한 물품이 코딩 학습, 인터넷강의 수강, 공부, 면접 준비 등 구직활동이나 자기계발 목적에 필요한 도구라면 조건부로 사용 가능하다고 볼 수 있습니다. "
        "다만 사회 정서에 반하는 고가의 사치품으로 볼 수 있는 구매는 제한되므로, 사용 목적과 금액이 사업 취지에 맞는지 설명 가능해야 합니다."
    )


def build_blocked_answer(decision: PolicyDecision) -> str:
    blocker_messages = [blocker.message for blocker in decision.matched_blockers]
    detail = " ".join(dict.fromkeys(blocker_messages))
    return (
        "안내책자 근거를 기준으로 보면 해당 방식은 사용하기 어렵습니다. "
        f"{detail} "
        "가능한 항목에 해당하더라도 질문에 포함된 결제 방식, 거래 방식, 사용 목적이 사용 불가 조건과 충돌하면 불가 조건을 우선 적용해야 합니다."
    ).strip()


def build_fallback_answer() -> str:
    return "안내책자에서 해당 내용을 확인하지 못했습니다. 최신 공식 안내 확인이 필요할 수 있습니다."


def build_general_answer(question: str) -> str:
    normalized = " ".join(question.split())
    identity_terms = ("누구", "뭐하는", "정체", "소개", "할 수 있어", "기능", "사용법")
    greeting_terms = ("안녕", "하이", "hello", "hi")

    if any(term in normalized for term in identity_terms):
        return (
            "저는 서울 청년수당 참여자 안내책자를 바탕으로 청년수당 사용처, 사용 제한, "
            "카드 결제 방식, 증빙 관련 질문을 도와드리는 챗봇입니다. "
            "안내책자 근거가 충분한 경우에는 출처와 함께 답변합니다."
        )

    if any(term.lower() in normalized.lower() for term in greeting_terms):
        return (
            "안녕하세요. 저는 서울 청년수당 참여자 안내책자를 바탕으로 청년수당 사용처, "
            "제한 항목, 카드 결제, 증빙 관련 질문을 도와드리는 챗봇입니다. "
            "궁금한 내용을 물어봐 주세요."
        )

    return (
        "저는 청년수당 안내를 돕는 챗봇이라 일반 정보는 답변하지 않습니다. "
        "청년수당 사용처, 제한 항목, 카드 결제, 증빙 관련 질문을 물어봐 주세요."
    )
