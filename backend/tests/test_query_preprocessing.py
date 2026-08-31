from app.rag.query_preprocessing import (
    expand_synonyms,
    classify_retrieval_intent,
    normalize_for_retrieval,
)


def test_normalize_for_retrieval_fixes_spacing_and_common_typos() -> None:
    assert normalize_for_retrieval("  쳥년수당신청조껀  ") == "청년수당 신청 조건"


def test_expand_synonyms_keeps_original_and_adds_domain_variants() -> None:
    queries = expand_synonyms("청년수당 신청 조건")

    assert queries[0] == "청년수당 신청 조건"
    assert "청년지원금 신청 조건" in queries
    assert "청년수당 자격 요건" in queries
    assert len(queries) == len(set(queries))


def test_classify_retrieval_intent_sets_alpha_by_query_shape() -> None:
    assert classify_retrieval_intent("1566-3344").intent == "identifier"
    assert classify_retrieval_intent("1566-3344").alpha == 0.8

    assert classify_retrieval_intent("지급일").intent == "ambiguous"
    assert classify_retrieval_intent("지급일").alpha == 0.5

    assert classify_retrieval_intent("청년수당 신청 자격이 뭐야?").intent == "conceptual"
    assert classify_retrieval_intent("청년수당 신청 자격이 뭐야?").alpha == 0.3


def test_classify_retrieval_intent_prioritizes_explicit_policy_terms() -> None:
    result = classify_retrieval_intent("카카오페이로 결제해도 되나요?")

    assert result.intent == "policy"
    assert result.alpha == 0.9
