from collections.abc import Callable

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings, get_settings
from app.graph.state import PolicyBlocker
from app.rag.prompts import BLOCKER_CHECK_SYSTEM_PROMPT, BLOCKER_CHECK_USER_PROMPT

ConfirmBlockers = Callable[[str, list[PolicyBlocker]], list[PolicyBlocker]]


def select_confirmed_blockers(blockers: list[PolicyBlocker], decision: object) -> list[PolicyBlocker]:
    """지출 대상과 결제 방식이 모두 제한 항목이 아니라고 명시된 후보만 뺀다.

    사용 불가 항목을 허용하는 실수가 과잉 차단보다 비싸므로, 출력이 깨졌거나
    후보 key가 빠져 있으면 키워드 차단을 그대로 유지한다.
    """
    checks = decision.get("checks") if isinstance(decision, dict) else None
    if not isinstance(checks, list):
        return blockers
    dismissed = {
        check.get("key")
        for check in checks
        if isinstance(check, dict)
        and check.get("spend_matches") is False
        and check.get("payment_matches") is False
    }
    return [blocker for blocker in blockers if blocker.key not in dismissed]


def format_blocker_candidates(blockers: list[PolicyBlocker]) -> str:
    return "\n".join(f"- {blocker.key}: {blocker.label} ({blocker.message})" for blocker in blockers)


def create_blocker_confirmer(settings: Settings | None = None) -> ConfirmBlockers:
    resolved_settings = settings or get_settings()
    llm = ChatOpenAI(
        api_key=resolved_settings.openai_api_key,
        model=resolved_settings.openai_blocker_check_model,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BLOCKER_CHECK_SYSTEM_PROMPT),
            ("human", BLOCKER_CHECK_USER_PROMPT),
        ]
    )
    chain = prompt | llm | JsonOutputParser()

    def confirm(question: str, blockers: list[PolicyBlocker]) -> list[PolicyBlocker]:
        if not blockers:
            return blockers
        try:
            decision = chain.invoke(
                {"question": question, "candidates": format_blocker_candidates(blockers)}
            )
        except Exception:
            # 확인 호출이 실패해도 차단을 풀지 않는다(안전 우선).
            return blockers
        return select_confirmed_blockers(blockers, decision)

    return confirm
