from benchmarks.eval_v1 import (
    EvalCase,
    cohen_kappa,
    parse_judge_label,
    quote_ranks,
    score_answers,
    score_retrieval,
    wilson_ci,
)


def make_case(case_id, category="info", verdict="info", quotes=("근거",), pair_of=None):
    return EvalCase(case_id, category, "질문", verdict, tuple(quotes), pair_of)


def test_quote_ranks_ignore_whitespace_and_html_tags() -> None:
    chunks = ["무관한 청크", "<td>청년수당 공고일(2026.</td> 3. 4.) 이후"]

    assert quote_ranks(["공고일(2026. 3. 4.) 이후", "없는 문장"], chunks) == [2, None]


def test_score_retrieval_counts_hits_mrr_and_full_evidence() -> None:
    cases = [
        make_case("a", quotes=("첫째",)),
        make_case("b", quotes=("둘째",)),
        make_case("c", quotes=("앞쪽", "뒤쪽")),
        make_case("d", category="not_in_booklet", quotes=()),
    ]
    retrieved = {"a": ["첫째 포함"], "b": ["x", "y", "z", "w", "v", "둘째"], "c": ["뒤쪽만"]}

    summary = score_retrieval(cases, retrieved, k=5)

    # b는 6위라 상위 5개 밖이고, d는 근거가 없어 분모에서 빠진다.
    assert summary["overall"]["hit@5"]["k"] == 2
    assert summary["overall"]["hit@5"]["n"] == 3
    assert summary["overall"]["mrr@5"] == round((1 + 0 + 1) / 3, 4)
    assert summary["full_evidence@5"]["k"] == 0
    assert summary["misses"] == ["b"]


def test_score_answers_separates_unsafe_allow_and_over_block() -> None:
    cases = [
        make_case("n1", "not_allowed", "no"),
        make_case("n2", "not_allowed", "no"),
        make_case("b1", "benign_keyword", "yes", pair_of="i1"),
        make_case("i1", "info", "yes"),
        make_case("x1", "not_in_booklet", "not_in_booklet", quotes=()),
    ]
    answers = {
        "n1": {"status": "answered_from_pdf"},
        "n2": {"status": "blocked_by_policy"},
        "b1": {"status": "blocked_by_policy"},
        "i1": {"status": "answered_from_pdf"},
        "x1": {"status": "answered_from_pdf"},
    }
    labels = {"n1": "CONDITIONAL", "n2": "DENY", "b1": "DENY", "i1": "ALLOW", "x1": "ALLOW"}

    result = score_answers(cases, answers, labels)

    assert result["unsafe_allow"]["k"] == 1
    assert result["over_block"]["k"] == 1
    assert result["fabrication"]["k"] == 1
    assert result["pair_splits"] == [{"id": "b1", "pair_of": "i1"}]
    assert result["verdict_accuracy"]["k"] == 2  # n2, i1


def test_wilson_ci_and_kappa_edge_cases() -> None:
    low, high = wilson_ci(0, 10)
    assert low == 0.0 and 0.27 < high < 0.29
    assert wilson_ci(0, 0) == (0.0, 0.0)
    assert cohen_kappa(["A", "B"], ["A", "B"]) == 1.0
    assert cohen_kappa([], []) is None


def test_parse_judge_label_rejects_unknown_labels() -> None:
    assert parse_judge_label('{"label": "deny", "reason": "r"}') == "DENY"
    assert parse_judge_label('{"label": "MAYBE"}') == "PARSE_ERROR"
    assert parse_judge_label("not json") == "PARSE_ERROR"
