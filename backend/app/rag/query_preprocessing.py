import re
from dataclasses import dataclass
from typing import Literal


RetrievalIntentName = Literal["conceptual", "identifier", "ambiguous", "policy"]


SYNONYM_DICT: dict[str, list[str]] = {
    "수당": ["지원금", "수급금"],
    "신청 조건": ["자격 요건", "지원 대상", "신청 자격", "참여 자격"],
    "활동 보고": ["자기성장기록서", "성장 기록"],
    "지급": ["수령", "지원"],
    "미취업": ["구직", "취업 준비"],
    "참여자": ["수혜자", "수급자"],
}

SPACING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"청년수당신청", "청년수당 신청"),
    (r"신청자격", "신청 자격"),
    (r"자격조건", "자격 조건"),
    (r"지급금액", "지급 금액"),
    (r"활동보고", "활동 보고"),
    (r"참여자격", "참여 자격"),
)

TYPO_DICT: dict[str, str] = {
    "신청조껀": "신청 조건",
    "지급긍액": "지급 금액",
    "쳥년수당": "청년수당",
    "쳥년": "청년",
}

INTENT_ALPHA: dict[RetrievalIntentName, float] = {
    "conceptual": 0.3,
    "identifier": 0.8,
    "ambiguous": 0.5,
    # 사용 불가 항목은 의미 유사도보다 규정에 쓰인 정확한 용어가 중요하다.
    "policy": 0.9,
}

POLICY_RULE_TERMS = (
    "카카오페이",
    "네이버페이",
    "간편결제",
    "간편 결제",
    "상품권",
    "기프티콘",
    "중고거래",
    "중고 거래",
    "현금인출",
    "현금 인출",
    "atm",
    "게임",
    "도박",
)


@dataclass(frozen=True)
class RetrievalIntent:
    intent: RetrievalIntentName
    alpha: float


def normalize_for_retrieval(query: object) -> str:
    normalized = " ".join(str(query).split())
    for pattern, replacement in SPACING_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized)
    for typo, replacement in TYPO_DICT.items():
        normalized = normalized.replace(typo, replacement)
    for pattern, replacement in SPACING_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized)
    return " ".join(normalized.split())


def expand_synonyms(query: object) -> list[str]:
    normalized = normalize_for_retrieval(query)
    expanded = [normalized]
    for key, synonyms in SYNONYM_DICT.items():
        if key in normalized:
            expanded.extend(normalized.replace(key, synonym) for synonym in synonyms)
    return list(dict.fromkeys(expanded))


def classify_retrieval_intent(query: object) -> RetrievalIntent:
    normalized = normalize_for_retrieval(query)
    if any(term.lower() in normalized.lower() for term in POLICY_RULE_TERMS):
        return RetrievalIntent(intent="policy", alpha=INTENT_ALPHA["policy"])
    if re.search(r"\d{3,4}-\d{3,4}", normalized) or re.fullmatch(
        r"[\d\s\-]+",
        normalized.strip(),
    ):
        return RetrievalIntent(intent="identifier", alpha=INTENT_ALPHA["identifier"])
    if len(normalized.strip()) < 6:
        return RetrievalIntent(intent="ambiguous", alpha=INTENT_ALPHA["ambiguous"])
    return RetrievalIntent(intent="conceptual", alpha=INTENT_ALPHA["conceptual"])
