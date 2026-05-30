from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.graph.state import EvidenceDecision
from app.rag.generation import format_context
from app.rag.policy import (
    is_food_delivery_question,
    is_study_device_question,
    supports_food_delivery_inference,
    supports_study_device_inference,
)
from app.rag.prompts import EVIDENCE_GRADER_SYSTEM_PROMPT, EVIDENCE_GRADER_USER_PROMPT

GradeEvidence = Callable[[str, list[Document]], EvidenceDecision | dict]


def create_evidence_grader(settings: Settings | None = None) -> GradeEvidence:
    resolved_settings = settings or get_settings()
    llm = ChatOpenAI(
        api_key=resolved_settings.openai_api_key,
        model=resolved_settings.openai_chat_model,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVIDENCE_GRADER_SYSTEM_PROMPT),
            ("human", EVIDENCE_GRADER_USER_PROMPT),
        ]
    )
    chain = prompt | llm | JsonOutputParser()

    def grade(question: str, documents: list[Document]) -> EvidenceDecision:
        result = chain.invoke(
            {
                "question": question,
                "context": format_context(documents),
            }
        )
        return EvidenceDecision.model_validate(result)

    return grade


def normalize_evidence_decision(
    evidence: EvidenceDecision | dict,
    documents: list[Document],
) -> EvidenceDecision:
    result = EvidenceDecision.model_validate(evidence)
    available_chunk_ids = {
        str(document.metadata.get("chunk_id"))
        for document in documents
        if document.metadata.get("chunk_id")
    }
    source_chunk_ids = [
        chunk_id
        for chunk_id in result.source_chunk_ids
        if str(chunk_id) in available_chunk_ids
    ]
    support_level = result.support_level
    if support_level not in {"direct", "inferable", "insufficient"}:
        support_level = "insufficient"

    return result.model_copy(
        update={
            "is_sufficient": (
                result.is_sufficient
                and bool(source_chunk_ids)
                and support_level in {"direct", "inferable"}
            ),
            "support_level": support_level,
            "source_chunk_ids": source_chunk_ids,
        }
    )


def grade_evidence(
    question: str,
    documents: list[Document],
    grade_with_llm: GradeEvidence | None = None,
    settings: Settings | None = None,
) -> EvidenceDecision:
    if not documents:
        return EvidenceDecision(
            is_sufficient=False,
            support_level="insufficient",
            reason="검색된 PDF 근거가 없습니다.",
            source_chunk_ids=[],
        )

    grader = grade_with_llm or create_evidence_grader(settings)
    result = normalize_evidence_decision(grader(question, documents), documents)

    # 배달앱 질문은 해외 AI 보전 예외 같은 잘못된 근거가 source로 통과하지 못하도록 보정한다.
    if is_food_delivery_question(question):
        valid_documents = [
            document for document in documents if supports_food_delivery_inference(document)
        ]
        if valid_documents:
            return EvidenceDecision(
                is_sufficient=True,
                support_level="inferable",
                reason=(
                    "식비가 생활비 항목에 포함되고, 온라인에서도 카드 결제가 가능하다는 근거가 있어 "
                    "배달 음식 결제는 전용 체크카드 직접 결제 조건에서 제한적으로 답변할 수 있습니다."
                ),
                source_chunk_ids=[
                    str(document.metadata["chunk_id"])
                    for document in valid_documents[:2]
                    if document.metadata.get("chunk_id")
                ],
            )
        return EvidenceDecision(
            is_sufficient=False,
            support_level="insufficient",
            reason="배달앱 결제를 판단하는 데 필요한 식비/생활비 및 온라인 카드 결제 근거가 검색 후보에 없습니다.",
            source_chunk_ids=[],
        )

    if is_study_device_question(question):
        valid_documents = [
            document for document in documents if supports_study_device_inference(document)
        ]
        if valid_documents:
            return EvidenceDecision(
                is_sufficient=True,
                support_level="inferable",
                reason=(
                    "태블릿 PC, 노트북, 의자 등은 인터넷강의 청취, 공부, 면접 준비처럼 "
                    "청년수당 사업 취지에 부합하는 경우 구입 가능하다는 근거가 있어 "
                    "학습·구직활동용 기기나 도구는 고가 사치품이 아니라는 조건에서 제한적으로 답변할 수 있습니다."
                ),
                source_chunk_ids=[
                    str(document.metadata["chunk_id"])
                    for document in valid_documents[:2]
                    if document.metadata.get("chunk_id")
                ],
            )
        return EvidenceDecision(
            is_sufficient=False,
            support_level="insufficient",
            reason="학습·구직활동용 기기/도구 구매를 판단하는 데 필요한 사업 취지 부합 및 고가 사치품 제한 근거가 검색 후보에 없습니다.",
            source_chunk_ids=[],
        )

    return result
