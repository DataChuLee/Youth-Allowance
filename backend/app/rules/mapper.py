"""질문을 선언된 판정 규칙에 연결한다.

LLM은 질문에서 "어느 규칙인가"와 판정에 필요한 사실(결제 경로, 나이, 예외)만 뽑는다.
결론은 handlers가 정한다. LLM이 없는 ID를 내도 validate_match가 걸러 RAG로 넘긴다.
"""

from collections.abc import Callable
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.rules.handlers import PAYMENT_PATHS, RuleVerdict, check_eligibility, find_rule, resolve_spending, unclear
from app.rules.loader import load_rules

ELIGIBILITY_SECTIONS = ("requirements", "exclusions", "loss_reasons")


class RuleMatch(BaseModel):
    # 아래 두 칸을 먼저 채우게 해서, 사용 목적보다 업종 표 확인이 앞서도록 한다(필드 순서대로 생성된다).
    purchase: str | None = Field(default=None, description="질문에서 돈을 쓰려는 대상(품목·서비스·가맹점). 없으면 null")
    blocked_merchant_entry: str | None = Field(
        default=None, description="purchase가 해당하는 카드 사용 불가 업종 표의 항목 이름 그대로. 해당 없으면 null"
    )
    domain: Literal["spending", "eligibility", "none"] = Field(
        description="spending=지출 가능 여부, eligibility=신청·자격 여부, none=판정 질문이 아니거나 맞는 규칙이 없음"
    )
    rule_id: str | None = Field(default=None, description="목록에 있는 규칙 ID. 없으면 null")
    section: Literal["requirements", "exclusions", "loss_reasons"] | None = Field(
        default=None, description="eligibility일 때 규칙이 속한 구역"
    )
    payment: str = Field(
        default="unknown",
        description="card | transfer | reimbursement | unknown 또는 결제 방식 규칙 ID(easy_pay 등)",
    )
    exception_id: str | None = Field(default=None, description="질문 속 사실이 해당하는 예외 ID")
    age: int | None = Field(default=None, description="질문에 나온 만 나이")
    violates_condition: bool = Field(
        default=False, description="질문에 규칙의 인정 조건을 어기는 사실이 명시돼 있으면 true"
    )
    unverified_alias: bool = Field(default=False, description="미확인 표현으로만 연결했으면 true")


def _entry_line(entry: dict[str, Any]) -> str:
    parts = [f"- {entry['id']}: {', '.join(entry.get('names', []) + entry.get('aliases', []))}"]
    if entry.get("condition"):
        parts.append(f"  (인정 조건: {entry['condition']})")
    if entry.get("unverified_aliases"):
        parts.append(f"  [미확인 표현: {', '.join(entry['unverified_aliases'])}]")
    return "".join(parts)


def build_rule_catalog(spending: dict[str, Any], eligibility: dict[str, Any]) -> str:
    """연결기 프롬프트에 넣을 규칙 목록. ID와 이름만 주고 결론(verdict)은 주지 않는다."""
    lines = ["[지출: 결제 방식] (rule_id 또는 payment에 쓸 수 있음)"]
    lines += [_entry_line(entry) for entry in spending["payment_methods"]]
    lines.append("[지출: 항목]")
    lines += [_entry_line(entry) for entry in spending["items"]]
    lines.append("[지출: 계좌 이용]")
    lines += [_entry_line(entry) for entry in spending["account_actions"]]
    lines.append(f"[지출: 카드 사용 불가 업종] rule_id=blocked_merchant: {', '.join(spending['blocked_merchants']['names'])}")
    for section in ELIGIBILITY_SECTIONS:
        lines.append(f"[자격: {section}]")
        for entry in eligibility[section]:
            lines.append(_entry_line(entry))
            for exception in entry.get("exceptions", []):
                lines.append(f"  · 예외 {exception['id']}: {', '.join(exception['names'])} → {exception['effect']}")
    return "\n".join(lines)


MAPPER_SYSTEM_PROMPT = """너는 서울시 청년수당 질문을 규칙 ID에 연결하는 분류기다. 결론(가능/불가)은 정하지 않는다.

연결 원칙:
- "~해도 되나요/돼요/가능한가요/받을 수 있나요"처럼 가능 여부를 묻는 질문만 연결한다.
- 방법·절차·기간·금액·서류 목록·연락처를 묻는 질문은 domain=none이다.
- 목록의 이름·표현과 뜻이 같은 경우만 연결한다. 비슷해 보이기만 하면 domain=none이다.
- 질문에 어떤 규칙의 이름·표현이 그대로 나오면, 사용자가 덧붙인 다른 범주 설명("~니까 ~비죠?")보다 그 규칙을 우선한다.
- 질문 속 항목이 [미확인 표현]에만 해당하면 그 규칙에 연결하고 unverified_alias=true로 둔다.
- 결제 방식: 청년수당 카드 결제=card, 청년수당 계좌에서 이체=transfer, 개인 돈으로 먼저 내고 청년수당에서 돌려받기=reimbursement, 알 수 없으면 unknown.
  - 대출 상환이나 증빙(고지서·이체확인증) 인정을 묻는 지출은 이체(transfer)로 본다.
  - 카카오페이·네이버페이 같은 결제 방식이 항목과 함께 나오면 rule_id는 항목, payment는 결제 방식 규칙 ID로 둔다.
- 돈의 방향을 구분한다. 청년수당 계좌에서 나가는 돈은 지출·이체 규칙이고, 청년수당 계좌로 들어오는 입금만 personal_deposit이다.
- 구매 대상이나 가맹점이 카드 사용 불가 업종 표의 업종·품목에 해당하면 사용 목적과 관계없이 rule_id=blocked_merchant다.
- violates_condition은 질문에 규칙의 인정 조건을 어기는 사실이 명시돼 지출 전체가 인정되지 않을 때만 true다(예: 본인 명의가 아닌 대출).
  일부 금액만 빠지는 경우(고지서의 일부 항목 등)는 false다.

자격 질문 원칙:
- 질문 속 사람이 제외 대상·상실 사유의 정의를 실제로 충족할 때만 exclusions/loss_reasons에 연결한다.
- 근로자는 고용보험 가입 3개월 초과 또는 주 30시간 초과일 때만 insured_worker다. 각각을 따로 넓혀 해석하지 않는다.
  그 기준에 못 미치는 아르바이트·단기근로는 requirements/employment에 exception_id=short_term_worker로 연결한다.
- 대학원생은 재학·휴학 중일 때만 제외 대상이다.
- 창업에는 취업의 근로시간·근로기간 조건을 적용하지 않는다.
- 유사 사업 제외는 26년 3월 1일부터 사업 참여 종료일까지의 참여에만 적용한다.
- 나이가 나오면 age에 만 나이를 넣고, 제대군인이면 exception_id=veteran이다.
- 요건이 무엇인지만 묻는 질문(예: 나이 조건은?)은 domain=none이다.

규칙 목록:
{catalog}"""


def validate_match(match: RuleMatch, question: str, spending: dict[str, Any], eligibility: dict[str, Any]) -> RuleMatch:
    """LLM 출력 중 선언되지 않은 ID·조합을 걸러 낸다. 걸러지면 none으로 바꿔 RAG로 넘긴다."""
    none = RuleMatch(domain="none")
    # 업종 표 항목이 실제로 있으면 연결 결과와 무관하게 불가 업종으로 고정한다.
    if match.blocked_merchant_entry in spending["blocked_merchants"]["names"]:
        return match.model_copy(update={"domain": "spending", "rule_id": "blocked_merchant", "unverified_alias": False})
    if match.domain == "none" or not match.rule_id:
        return none
    if match.domain == "eligibility":
        rule = find_rule(eligibility.get(match.section or "", []), match.rule_id)
        if rule is None:
            return none
        exception_ids = {exception["id"] for exception in rule.get("exceptions", [])}
        exception_id = match.exception_id if match.exception_id in exception_ids else None
        return match.model_copy(update={"exception_id": exception_id})

    entries = [*spending["payment_methods"], *spending["items"], *spending["account_actions"]]
    rule = find_rule(entries, match.rule_id)
    if rule is None and match.rule_id != "blocked_merchant":
        return none
    method_ids = {entry["id"] for entry in spending["payment_methods"]}
    payment = match.payment if match.payment in (*PAYMENT_PATHS, *method_ids) else "unknown"
    # 미확인 표현이 질문에 그대로 있으면 LLM 판단과 무관하게 RAG로 넘긴다.
    verified_terms = [*rule.get("names", []), *rule.get("aliases", [])] if rule else []
    unverified = match.unverified_alias or (
        rule is not None
        and any(term in question for term in rule.get("unverified_aliases", []))
        and not any(term in question for term in verified_terms)
    )
    checked = match.model_copy(update={"payment": payment, "unverified_alias": unverified})
    if has_prohibited_conflict(checked, question, spending) and resolve_match(checked, spending).verdict != "prohibited":
        return none
    return checked


def prohibited_terms(spending: dict[str, Any]) -> dict[str, str]:
    """금지 규칙의 이름·표현 → 규칙 ID. 질문에 그대로 나오는지 보는 용도라 두 글자 이상만 쓴다."""
    entries = [entry for key in ("payment_methods", "items") for entry in spending[key] if entry["verdict"] == "prohibited"]
    return {
        term: entry["id"]
        for entry in entries
        for term in [*entry.get("names", []), *entry.get("aliases", [])]
        if len(term) >= 2
    }


def has_prohibited_conflict(match: RuleMatch, question: str, spending: dict[str, Any]) -> bool:
    """질문에 금지 규칙 이름이 그대로 있는데 다른 규칙에 연결됐는가. 이때는 결론을 내지 않고 RAG로 넘긴다."""
    return any(
        term in question and rule_id not in (match.rule_id, match.payment)
        for term, rule_id in prohibited_terms(spending).items()
    )


def resolve_match(match: RuleMatch, spending: dict[str, Any] | None = None) -> RuleVerdict:
    """연결 결과를 핸들러로 보내 결론을 받는다. unclear면 호출 측이 RAG로 넘긴다."""
    if match.domain == "none" or not match.rule_id:
        return unclear()
    if match.domain == "eligibility":
        return check_eligibility(match.section or "", match.rule_id, match.exception_id, match.age)
    result = resolve_spending(match.rule_id, match.payment, match.unverified_alias, spending)
    # 질문에 인정 조건을 어기는 사실이 있으면(예: 부모님 명의 대출) 허용 결론을 불가로 바꾼다.
    if match.violates_condition and result.condition and result.verdict in ("allowed", "conditional"):
        return RuleVerdict("prohibited", result.rule_id, result.path, result.condition, (), result.sources)
    return result


def create_rule_mapper(settings: Settings | None = None, model: str | None = None) -> Callable[[str], RuleMatch]:
    resolved = settings or get_settings()
    spending, eligibility = load_rules("spending"), load_rules("eligibility")
    llm = ChatOpenAI(
        api_key=resolved.openai_api_key,
        model=model or resolved.openai_rule_mapper_model,
        temperature=0,
    ).with_structured_output(RuleMatch, method="function_calling")
    system = MAPPER_SYSTEM_PROMPT.format(catalog=build_rule_catalog(spending, eligibility))

    def map_question(question: str) -> RuleMatch:
        raw = llm.invoke([("system", system), ("human", question)])
        return validate_match(RuleMatch.model_validate(raw), question, spending, eligibility)

    return map_question
