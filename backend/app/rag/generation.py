from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.graph.state import PolicyDecision
from app.rag.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT


def format_context(documents: list[Document]) -> str:
    return "\n\n".join(
        (
            f"[chunk_id={document.metadata.get('chunk_id')} "
            f"page={document.metadata.get('page')}]\n{document.page_content}"
        )
        for document in documents
    )


def generate_pdf_answer(question: str, documents: list[Document], settings: Settings) -> str:
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
        temperature=0,
    )
    messages = [
        ("system", ANSWER_SYSTEM_PROMPT),
        (
            "human",
            ANSWER_USER_PROMPT.format(
                question=question,
                context=format_context(documents),
            ),
        ),
    ]
    response = llm.invoke(messages)
    return str(response.content).strip()


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
