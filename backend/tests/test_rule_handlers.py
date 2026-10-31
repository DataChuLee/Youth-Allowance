import pytest

from app.rules.handlers import VERDICT_RANK, check_eligibility, resolve_spending
from app.rules.loader import load_rules


def test_health_insurance_is_allowed_but_insurance_payment_is_prohibited() -> None:
    assert resolve_spending("health_insurance", "card").verdict == "allowed"
    assert resolve_spending("insurance_payment", "card").verdict == "prohibited"


def test_transfer_for_listed_item_is_conditional_with_evidence() -> None:
    result = resolve_spending("maintenance_fee", "transfer")
    assert result.verdict == "conditional"
    assert "이체확인증" in result.evidence


def test_unknown_payment_prefers_card_path_for_allowed_item() -> None:
    result = resolve_spending("maintenance_fee")
    assert (result.verdict, result.path) == ("allowed", "card")


def test_transfer_for_unlisted_item_is_prohibited_by_default_rule() -> None:
    result = resolve_spending("food", "transfer")
    assert result.verdict == "prohibited"
    assert any(source["page"] == 13 for source in result.sources)


def test_overseas_ai_subscription_is_conditional_only_through_reimbursement() -> None:
    assert resolve_spending("overseas_ai_subscription", "card").verdict == "prohibited"
    result = resolve_spending("overseas_ai_subscription")
    assert (result.verdict, result.path) == ("conditional", "reimbursement")
    assert resolve_spending("domestic_ai_subscription", "reimbursement").verdict == "prohibited"


def test_blocked_payment_method_overrides_allowed_item() -> None:
    result = resolve_spending("rent", "easy_pay")
    assert (result.verdict, result.rule_id) == ("prohibited", "easy_pay")


def test_blocked_merchant_is_prohibited() -> None:
    assert resolve_spending("blocked_merchant").verdict == "prohibited"


@pytest.mark.parametrize(
    ("rule_id", "kwargs"),
    [
        ("easy_pay", {"unverified_alias": True}),
        (None, {}),
        ("not_a_rule", {}),
        ("health_insurance", {"payment": "transfer"}),
    ],
)
def test_unresolvable_spending_questions_fall_back_as_unclear(rule_id, kwargs) -> None:
    assert resolve_spending(rule_id, **kwargs).verdict == "unclear"


def test_account_action_uses_its_own_verdict() -> None:
    result = resolve_spending("carry_over", "card")
    assert result.verdict == "conditional"
    assert "이월사유" in result.condition


def test_every_spending_rule_resolves_to_known_verdict_for_every_payment() -> None:
    rules = load_rules("spending")
    rule_ids = [entry["id"] for key in ("payment_methods", "items", "account_actions") for entry in rules[key]]
    for rule_id in [*rule_ids, "blocked_merchant"]:
        for payment in ("card", "transfer", "reimbursement", "unknown"):
            assert resolve_spending(rule_id, payment).verdict in VERDICT_RANK


@pytest.mark.parametrize(
    ("section", "rule_id", "exception_id", "age", "expected"),
    [
        ("exclusions", "student", None, None, "prohibited"),
        ("exclusions", "student", "open_university", None, "conditional"),
        ("loss_reasons", "further_education", None, None, "prohibited"),
        ("loss_reasons", "further_education", "open_university", None, "allowed"),
        ("requirements", "age", None, 25, "conditional"),
        ("requirements", "age", None, 36, "prohibited"),
        ("requirements", "age", "veteran", 36, "conditional"),
        ("requirements", "age", "veteran", 38, "prohibited"),
        ("requirements", "employment", "short_term_worker", None, "conditional"),
        ("requirements", "residence", None, None, "unclear"),
        ("exclusions", "not_a_rule", None, None, "unclear"),
    ],
)
def test_check_eligibility(section, rule_id, exception_id, age, expected) -> None:
    assert check_eligibility(section, rule_id, exception_id, age).verdict == expected
