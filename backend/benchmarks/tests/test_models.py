from benchmarks.models import (
    PolicyCase,
    RetrievalCase,
    score_policy_runs,
    score_retrieval_runs,
)


def test_score_retrieval_runs_calculates_rank_metrics() -> None:
    cases = [
        RetrievalCase("one", "keyword", "질문", ("gold-1",)),
        RetrievalCase("two", "keyword", "질문", ("gold-2",)),
    ]
    runs = [
        {"id": "one", "variant": "hybrid", "retrieved_chunk_ids": ["gold-1", "other"]},
        {"id": "two", "variant": "hybrid", "retrieved_chunk_ids": ["other", "gold-2"]},
    ]

    summary = score_retrieval_runs(cases, runs, k=5)

    assert summary["n_scored"] == 2
    assert summary["metrics"] == {"mrr": 0.75, "recall_at_k": 1.0, "ndcg_at_k": 0.8155}


def test_score_policy_runs_reports_unsafe_route_rate() -> None:
    cases = [
        PolicyCase("blocked", "policy_blocked", "상품권", "blocked_by_policy"),
        PolicyCase("answer", "answerable", "지급일", "answered_from_pdf", ("sem-1",)),
    ]
    runs = [
        {"id": "blocked", "status": "answered_from_pdf", "answer": "가능합니다.", "source_chunk_ids": []},
        {"id": "answer", "status": "answered_from_pdf", "answer": "29일", "source_chunk_ids": ["sem-1"]},
    ]

    summary = score_policy_runs(cases, runs)

    assert summary["metrics"]["status_accuracy"] == 0.5
    assert summary["metrics"]["source_hit_rate"] == 1.0
    assert summary["metrics"]["unsafe_route_rate"] == 1.0


def test_score_policy_runs_marks_unlabelled_optional_metrics_as_not_applicable() -> None:
    cases = [PolicyCase("answer", "answerable", "질문", "answered_from_pdf")]
    runs = [{"id": "answer", "status": "answered_from_pdf", "answer": "답변", "source_chunk_ids": []}]

    summary = score_policy_runs(cases, runs)

    assert summary["metrics"]["source_hit_rate"] is None
    assert summary["metrics"]["required_term_rate"] is None
