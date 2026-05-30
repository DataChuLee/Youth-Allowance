from langchain_core.documents import Document

from app.graph.state import EvidenceDecision, PolicyBlocker, PolicyDecision

FOOD_DELIVERY_TERMS = ("배달", "배달앱", "배달의민족", "배민", "음식 주문", "음식 배달")
STUDY_DEVICE_TERMS = ("키보드", "아이패드", "iPad", "태블릿", "태블릿 PC", "노트북", "의자", "전자기기")
AI_REIMBURSEMENT_TERMS = ("해외 생성형 AI", "해외결제", "보전", "개인 카드", "개인 돈")

BLOCKING_SIGNALS = {
    "easy_payment": {
        "question_terms": ("카카오페이", "네이버페이", "간편결제", "간편 결제"),
        "evidence_terms": ("카카오페이", "네이버페이", "간편 결제 불가", "간편결제 불가"),
        "label": "간편결제 불가",
        "message": "카카오페이, 네이버페이 등 간편결제는 안내책자에서 불가하다고 되어 있습니다.",
    },
    "stored_value": {
        "question_terms": ("포인트 충전", "포인트", "충전식 결제", "충전"),
        "evidence_terms": ("결제 포인트 구매", "충전식 결제", "포인트"),
        "label": "포인트 충전·충전식 결제 불가",
        "message": "결제 포인트 구매나 충전식 결제는 사용 불가 항목으로 안내되어 있습니다.",
    },
    "voucher": {
        "question_terms": ("상품권", "기프티콘", "쿠폰"),
        "evidence_terms": ("상품권", "기프티콘"),
        "label": "상품권·기프티콘 구매 불가",
        "message": "상품권과 기프티콘 구매는 사용 불가 항목으로 안내되어 있습니다.",
    },
    "secondhand": {
        "question_terms": ("중고거래", "중고 거래", "중고"),
        "evidence_terms": ("중고거래", "증빙이 어려운 항목"),
        "label": "중고거래 불가",
        "message": "중고거래는 증빙이 어려운 항목으로 사용 불가하다고 안내되어 있습니다.",
    },
    "cash_withdrawal": {
        "question_terms": ("현금 인출", "현금인출", "ATM", "인출"),
        "evidence_terms": ("현금인출은 불가", "현금 인출은 불가", "현금인출 불가"),
        "label": "현금인출 불가",
        "message": "현금 사용은 계좌이체를 의미하며 현금인출은 불가하다고 안내되어 있습니다.",
    },
    "non_purpose": {
        "question_terms": ("게임", "유흥", "명품", "사치품", "사치", "도박"),
        "evidence_terms": ("도박", "게임", "사행성", "유흥비", "고가의 사치품", "구직활동과 무관"),
        "label": "사업 취지와 무관하거나 사회 정서상 부적절한 지출 제한",
        "message": "도박·게임 등 사행성 분야, 유흥비, 구직활동과 무관한 고가 사치품은 제한된다고 안내되어 있습니다.",
    },
}


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def is_food_delivery_question(question: str) -> bool:
    return contains_any(question, FOOD_DELIVERY_TERMS)


def is_study_device_question(question: str) -> bool:
    return contains_any(question, STUDY_DEVICE_TERMS)


def detect_question_blockers(question: str) -> list[PolicyBlocker]:
    blockers: list[PolicyBlocker] = []
    for key, rule in BLOCKING_SIGNALS.items():
        if contains_any(question, rule["question_terms"]):
            blockers.append(
                PolicyBlocker(
                    key=key,
                    label=rule["label"],
                    message=rule["message"],
                )
            )
    return blockers


def attach_evidence_to_blockers(
    blockers: list[PolicyBlocker],
    documents: list[Document],
) -> list[PolicyBlocker]:
    resolved: list[PolicyBlocker] = []
    for blocker in blockers:
        rule = BLOCKING_SIGNALS[blocker.key]
        matching_documents = [
            document
            for document in documents
            if contains_any(document.page_content, rule["evidence_terms"])
        ]
        if not matching_documents:
            continue

        resolved.append(
            blocker.model_copy(
                update={
                    "source_chunk_ids": [
                        str(document.metadata["chunk_id"])
                        for document in matching_documents[:2]
                        if document.metadata.get("chunk_id")
                    ]
                }
            )
        )
    return resolved


def score_food_delivery_document(document: Document) -> int:
    text = document.page_content
    score = 0
    if "식비" in text:
        score += 5
    if "생활비" in text:
        score += 4
    if "온라인" in text and "카드" in text:
        score += 4
    if "카드 결제" in text:
        score += 4
    if "체크카드" in text or "전용 카드" in text:
        score += 3
    if "사용 가능 항목" in text:
        score += 2
    if "간편 결제" in text or "간편결제" in text:
        score += 2
    if "카카오페이" in text or "네이버페이" in text:
        score += 2
    if "상품권" in text or "기프티콘" in text or "포인트" in text:
        score += 1
    if contains_any(text, AI_REIMBURSEMENT_TERMS):
        score -= 8
    return score


def supports_food_delivery_inference(document: Document) -> bool:
    text = document.page_content
    has_food_category = "식비" in text or "생활비" in text
    has_online_card = "온라인" in text and ("카드 결제" in text or "카드" in text)
    has_card_rule = "체크카드" in text or "전용 카드" in text or "카드로 사용" in text
    return has_food_category and has_online_card and has_card_rule


def score_study_device_document(document: Document) -> int:
    text = document.page_content
    score = 0
    if "태블릿" in text or "노트북" in text or "의자" in text:
        score += 6
    if "인터넷강의" in text or "공부" in text or "면접 준비" in text:
        score += 5
    if "청년수당 사업 취지" in text and "구입 가능" in text:
        score += 6
    if "구직활동" in text or "자기 계발" in text or "교육비" in text:
        score += 2
    if "고가의 사치품" in text or "구입 제한" in text:
        score += 3
    if "해외 생성형 AI" in text or "해외결제" in text or "보전" in text:
        score -= 6
    return score


def supports_study_device_inference(document: Document) -> bool:
    text = document.page_content
    has_device_example = "태블릿" in text or "노트북" in text or "의자" in text
    has_activity_purpose = "인터넷강의" in text or "공부" in text or "면접 준비" in text
    has_policy_condition = "청년수당 사업 취지" in text and "구입 가능" in text
    has_luxury_limit = "고가의 사치품" in text or "구입 제한" in text
    return has_device_example and has_activity_purpose and has_policy_condition and has_luxury_limit


def resolve_policy_decision(
    question: str,
    documents: list[Document],
    evidence: EvidenceDecision,
) -> PolicyDecision:
    matched_blockers = attach_evidence_to_blockers(
        detect_question_blockers(question),
        documents,
    )

    if matched_blockers:
        source_chunk_ids: list[str] = []
        for blocker in matched_blockers:
            source_chunk_ids.extend(blocker.source_chunk_ids)

        # 질문에 명시된 금지 조건이 PDF 근거에서도 확인되면 허용 템플릿보다 항상 우선한다.
        return PolicyDecision(
            decision="blocked",
            reason="질문에 포함된 결제 방식 또는 사용 목적이 안내책자의 사용 불가 조건과 충돌합니다.",
            matched_blockers=matched_blockers,
            source_chunk_ids=list(dict.fromkeys(source_chunk_ids)),
        )

    if evidence.is_sufficient:
        return PolicyDecision(
            decision="conditional",
            reason=evidence.reason,
            source_chunk_ids=evidence.source_chunk_ids,
        )

    return PolicyDecision(
        decision="insufficient",
        reason=evidence.reason or "검색된 PDF 근거만으로는 답변하기 어렵습니다.",
        source_chunk_ids=[],
    )
