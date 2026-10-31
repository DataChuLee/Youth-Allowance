from dataclasses import dataclass, field
from typing import Any

from app.rules.loader import load_rules

# 결제수단을 모를 때 가장 유리한 경로를 고르기 위한 순서. unclear는 불가보다 앞에 둔다(근거 부족 ≠ 금지).
VERDICT_RANK = {"allowed": 3, "conditional": 2, "unclear": 1, "prohibited": 0}
PAYMENT_PATHS = ("card", "transfer", "reimbursement")


@dataclass(frozen=True)
class RuleVerdict:
    """핸들러가 정한 결론. 답변 생성은 이 결론을 바꾸지 않고 문장으로만 옮긴다."""

    verdict: str
    rule_id: str
    path: str | None = None
    condition: str | None = None
    evidence: tuple[str, ...] = ()
    sources: tuple[dict[str, Any], ...] = field(default=())


def unclear(rule_id: str = "none") -> RuleVerdict:
    return RuleVerdict(verdict="unclear", rule_id=rule_id)


def find_rule(entries: list[dict[str, Any]], rule_id: str) -> dict[str, Any] | None:
    return next((entry for entry in entries if entry["id"] == rule_id), None)


def item_path_verdict(item: dict[str, Any], path: str, defaults: dict[str, Any]) -> RuleVerdict:
    if path == "card":
        return RuleVerdict(
            verdict=item["verdict"],
            rule_id=item["id"],
            path=path,
            condition=item.get("condition"),
            sources=tuple(item["sources"]),
        )
    # 계좌이체·보전은 항목에 적힌 경우만 열려 있고, 나머지는 13·18쪽 기본 규칙으로 불가다.
    explicit = path in item
    verdict = item[path] if explicit else defaults[path]
    sources = tuple(item["sources"]) if explicit else (*item["sources"], *defaults["sources"])
    return RuleVerdict(
        verdict=verdict,
        rule_id=item["id"],
        path=path,
        condition=item.get("condition"),
        evidence=tuple(item.get("evidence", [])) if verdict == "conditional" else (),
        sources=sources,
    )


def resolve_spending(
    rule_id: str | None,
    payment: str = "unknown",
    unverified_alias: bool = False,
    rules: dict[str, Any] | None = None,
) -> RuleVerdict:
    """지출 질문의 결론을 정한다.

    rule_id: 지출 항목, 계좌 이용 행위, 결제 방식 규칙 ID 또는 "blocked_merchant"
    payment: card | transfer | reimbursement | unknown | 결제 방식 규칙 ID(easy_pay 등)
    """
    rules = rules or load_rules("spending")
    if rule_id is None or unverified_alias:
        # 책자로 범주를 확정할 수 없는 표현은 규칙으로 결론 내지 않고 RAG로 넘긴다.
        return unclear(rule_id or "none")

    # 결제 방식 자체가 막히면 항목과 무관하게 불가다(예: 월세를 카카오페이로).
    method = find_rule(rules["payment_methods"], payment) or find_rule(rules["payment_methods"], rule_id)
    if method:
        return RuleVerdict(verdict=method["verdict"], rule_id=method["id"], sources=tuple(method["sources"]))

    if rule_id == "blocked_merchant":
        merchants = rules["blocked_merchants"]
        return RuleVerdict(verdict=merchants["verdict"], rule_id=rule_id, sources=tuple(merchants["sources"]))

    action = find_rule(rules["account_actions"], rule_id)
    if action:
        return RuleVerdict(
            verdict=action["verdict"],
            rule_id=action["id"],
            condition=action.get("condition"),
            sources=tuple(action["sources"]),
        )

    item = find_rule(rules["items"], rule_id)
    if item is None:
        return unclear(rule_id)
    if payment in PAYMENT_PATHS:
        return item_path_verdict(item, payment, rules["defaults"])
    # 결제수단을 모르면 가능한 경로 중 가장 유리한 결론을 쓴다(예: 해외 AI 구독은 보전으로 조건부).
    return max(
        (item_path_verdict(item, path, rules["defaults"]) for path in PAYMENT_PATHS),
        key=lambda result: VERDICT_RANK[result.verdict],
    )


def check_eligibility(
    section: str,
    rule_id: str,
    exception_id: str | None = None,
    age: int | None = None,
    rules: dict[str, Any] | None = None,
) -> RuleVerdict:
    """자격 질문의 결론을 정한다.

    section: requirements(신청 요건) | exclusions(신청 제외 대상) | loss_reasons(참여 중 자격 상실)
    verdict 의미: allowed=신청 가능/자격 유지, conditional=다른 요건까지 맞으면 가능, prohibited=신청 불가/자격 상실
    """
    rules = rules or load_rules("eligibility")
    rule = find_rule(rules.get(section, []), rule_id)
    if rule is None:
        return unclear(rule_id)
    sources = tuple(rule["sources"])
    exception = find_rule(rule.get("exceptions", []), exception_id) if exception_id else None

    if section == "loss_reasons":
        if exception:
            return RuleVerdict("allowed", rule_id, condition=exception["effect"], sources=sources)
        return RuleVerdict("prohibited", rule_id, condition=rule.get("definition"), sources=sources)

    if section == "exclusions":
        # 제외 사유를 벗어나도 다른 신청 요건은 따로 맞아야 하므로 "가능"이 아니라 조건부다.
        if exception:
            return RuleVerdict("conditional", rule_id, condition=exception["effect"], sources=sources)
        return RuleVerdict("prohibited", rule_id, sources=sources)

    # requirements
    if rule_id == "age" and age is not None:
        max_age = exception.get("max_age", rule["max_age"]) if exception else rule["max_age"]
        if not rule["min_age"] <= age <= max_age:
            return RuleVerdict("prohibited", rule_id, sources=sources)
    if exception:
        return RuleVerdict("conditional", rule_id, condition=exception["effect"], sources=sources)
    if rule_id == "age" and age is not None:
        return RuleVerdict("conditional", rule_id, condition="나이 요건 충족, 다른 요건도 맞아야 함", sources=sources)
    # 요건 내용만 묻는 정보성 질문은 RAG가 원문으로 답한다.
    return unclear(rule_id)
