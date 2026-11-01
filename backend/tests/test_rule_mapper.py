import pytest

from app.rules.loader import load_rules
from app.rules.mapper import RuleMatch, build_rule_catalog, resolve_match, validate_match

SPENDING = load_rules("spending")
ELIGIBILITY = load_rules("eligibility")


def check(match: RuleMatch, question: str = "질문") -> RuleMatch:
    return validate_match(match, question, SPENDING, ELIGIBILITY)


def test_catalog_lists_ids_but_not_verdicts() -> None:
    catalog = build_rule_catalog(SPENDING, ELIGIBILITY)
    assert "pension" in catalog and "blocked_merchant" in catalog and "veteran" in catalog
    assert "prohibited" not in catalog and "allowed" not in catalog


@pytest.mark.parametrize(
    "match",
    [
        RuleMatch(domain="spending", rule_id="not_a_rule"),
        RuleMatch(domain="eligibility", section="exclusions", rule_id="age"),
        RuleMatch(domain="spending", rule_id=None),
    ],
)
def test_undeclared_ids_fall_back_to_none(match: RuleMatch) -> None:
    assert check(match).domain == "none"
    assert resolve_match(check(match)).verdict == "unclear"


def test_unknown_payment_and_foreign_exception_are_dropped() -> None:
    assert check(RuleMatch(domain="spending", rule_id="rent", payment="bitcoin")).payment == "unknown"
    match = check(RuleMatch(domain="eligibility", section="requirements", rule_id="age", exception_id="open_university"))
    assert match.exception_id is None


def test_unverified_alias_in_question_forces_unclear_even_if_llm_missed_it() -> None:
    match = check(RuleMatch(domain="spending", rule_id="cosmetic"), "피부과 시술비 써도 돼요?")
    assert match.unverified_alias and resolve_match(match).verdict == "unclear"


def test_verified_name_beats_unverified_alias() -> None:
    match = check(RuleMatch(domain="spending", rule_id="prepaid_points"), "토스 포인트 충전 돼요?")
    assert resolve_match(match).verdict == "prohibited"


def test_condition_violation_turns_allowed_loan_into_prohibited() -> None:
    match = check(RuleMatch(domain="spending", rule_id="housing_loan", payment="transfer", violates_condition=True))
    assert resolve_match(match).verdict == "prohibited"
    without = check(RuleMatch(domain="spending", rule_id="housing_loan", payment="transfer"))
    assert resolve_match(without).verdict == "conditional"


def test_easy_pay_as_payment_overrides_item() -> None:
    match = check(RuleMatch(domain="spending", rule_id="rent", payment="easy_pay"))
    assert resolve_match(match).verdict == "prohibited"


@pytest.mark.parametrize(
    ("match", "expected"),
    [
        (RuleMatch(domain="eligibility", section="requirements", rule_id="age", exception_id="veteran", age=36), "conditional"),
        (RuleMatch(domain="eligibility", section="requirements", rule_id="employment", exception_id="short_term_worker"), "conditional"),
        (RuleMatch(domain="eligibility", section="exclusions", rule_id="student"), "prohibited"),
        (RuleMatch(domain="spending", rule_id="blocked_merchant"), "prohibited"),
        (RuleMatch(domain="spending", rule_id="domestic_ai_subscription", payment="reimbursement"), "prohibited"),
    ],
)
def test_resolve_match_dispatches_to_handlers(match: RuleMatch, expected: str) -> None:
    assert resolve_match(check(match)).verdict == expected


def test_blocked_merchant_entry_overrides_purpose_based_mapping() -> None:
    match = check(RuleMatch(domain="spending", rule_id="interview_clothes", blocked_merchant_entry="귀금속, 금, 은, 보석"))
    assert (match.rule_id, resolve_match(match).verdict) == ("blocked_merchant", "prohibited")
    invented = check(RuleMatch(domain="spending", rule_id="interview_clothes", blocked_merchant_entry="금은방"))
    assert invented.rule_id == "interview_clothes"


def test_prohibited_name_in_question_mapped_to_allowed_rule_falls_back() -> None:
    question = "휴대폰 소액결제로 쓴 금액도 통신비니까 내도 되죠?"
    assert check(RuleMatch(domain="spending", rule_id="telecom"), question).domain == "none"
    assert resolve_match(check(RuleMatch(domain="spending", rule_id="micropayment"), question)).verdict == "prohibited"
