from pathlib import Path

from benchmarks.build_content_quality_v1 import (
    build_conditional_annotation_queue,
    build_content_cases,
    read_jsonl,
    read_seed_rows,
)
from benchmarks.models import ContentQualityCase, score_content_quality_runs
from benchmarks.analyze_safety_failures import analyze
from benchmarks.build_policy_alignment_v1 import build_alignment_cases
from benchmarks.score_policy_alignment import score as score_policy_alignment
from benchmarks.run_policy_alignment import insufficient_judgment


def test_build_content_cases_reuses_reference_for_paraphrases() -> None:
    project_root = Path(__file__).resolve().parents[3]
    policy_cases = read_jsonl(
        project_root / "backend" / "benchmarks" / "datasets" / "policy_safety_v2_150.jsonl"
    )
    seed_rows = read_seed_rows(project_root / "backend" / "evaluation_results" / "testset.csv")

    cases = build_content_cases(policy_cases, seed_rows)

    assert len(cases) == 54
    assert cases[0]["gold_chunk_ids"] == ["sem_0014"]
    assert cases[0]["reference_answer"] == cases[1]["reference_answer"]
    assert cases[0]["review_status"] == "seed_reference_and_manual_source_review"


def test_conditional_annotation_queue_does_not_treat_candidates_as_gold() -> None:
    policy_cases = [
        {"id": "conditional-001", "scenario": "conditional", "question": "조건 질문"},
    ]
    policy_runs = [{"id": "conditional-001", "source_chunk_ids": ["sem_0035"]}]

    queue = build_conditional_annotation_queue(policy_cases, policy_runs)

    assert queue == [
        {
            "id": "conditional-001",
            "question": "조건 질문",
            "candidate_chunk_ids": ["sem_0035"],
            "reference_answer": "",
            "gold_chunk_ids": [],
            "critical_facts": [],
            "review_status": "needs_human_adjudication",
        }
    ]


def test_score_content_quality_runs_separates_retrieval_and_generation_metrics() -> None:
    cases = [
        ContentQualityCase(
            "content-001",
            "질문",
            "참조 답변",
            ("sem-1",),
            ("핵심 사실",),
            "reviewed",
        )
    ]
    runs = [
        {
            "id": "content-001",
            "retrieved_chunk_ids": ["sem-1", "sem-2"],
            "source_chunk_ids": ["sem-1"],
            "faithfulness": 0.8,
            "answer_correctness": 0.6,
        }
    ]

    summary = score_content_quality_runs(cases, runs)

    assert summary["metrics"] == {
        "mrr": 1.0,
        "recall_at_k": 1.0,
        "ndcg_at_k": 1.0,
        "source_hit_rate": 1.0,
        "faithfulness": 0.8,
        "answer_correctness": 0.6,
    }
    assert summary["coverage"] == {"faithfulness_scored": 1, "answer_correctness_scored": 1}


def test_analyze_safety_failures_distinguishes_policy_and_evidence_gaps() -> None:
    cases = [
        {
            "id": "blocked",
            "scenario": "policy_blocked",
            "question": "중고거래로 구매해도 되나요?",
            "expected_status": "blocked_by_policy",
        },
        {
            "id": "insufficient",
            "scenario": "insufficient",
            "question": "2027년 신청 일정은 언제인가요?",
            "expected_status": "insufficient_pdf_evidence",
        },
    ]
    runs = [
        {"id": "blocked", "status": "insufficient_pdf_evidence"},
        {"id": "insufficient", "status": "answered_from_pdf"},
    ]

    report = analyze(cases, runs)

    assert report["by_cause_hypothesis"] == {
        "policy_evidence_retrieval_gap": 1,
        "insufficient_unsafe_pdf_answer": 1,
    }


def test_policy_alignment_scores_conditional_answer_without_penalizing_card_restriction() -> None:
    cases = [
        {
            "id": "answerable-018",
            "expected_resolution": "conditional",
            "required_facts": ["카드 직접 결제는 불가", "개인 카드 후 보전 가능"],
        }
    ]
    runs = [
        {
            "id": "answerable-018",
            "resolution_correct": True,
            "fact_results": [
                {"fact": "카드 직접 결제는 불가", "label": "supported"},
                {"fact": "개인 카드 후 보전 가능", "label": "supported"},
            ],
            "unsupported_claims": [],
        }
    ]

    summary = score_policy_alignment(runs, cases)

    assert summary["metrics"] == {
        "resolution_accuracy": 1.0,
        "required_fact_coverage": 1.0,
        "contradiction_free_rate": 1.0,
        "unsupported_claim_free_rate": 1.0,
        "actionable_policy_pass_rate": 1.0,
    }


def test_build_alignment_cases_reuses_atomic_facts_for_variants() -> None:
    project_root = Path(__file__).resolve().parents[3]
    policy_cases = read_jsonl(
        project_root / "backend" / "benchmarks" / "datasets" / "policy_safety_v2_150.jsonl"
    )

    rows = build_alignment_cases(policy_cases)

    matched = [row for row in rows if row["id"] in {"answerable-018", "answerable-048"}]
    assert [row["expected_resolution"] for row in matched] == ["conditional", "conditional"]
    assert matched[0]["required_facts"] == matched[1]["required_facts"]


def test_insufficient_response_never_receives_credit_from_gold_evidence() -> None:
    judgment = insufficient_judgment(
        {"required_facts": ["허용된다", "증빙이 필요하다"]}
    )

    assert judgment["observed_resolution"] == "insufficient"
    assert judgment["resolution_correct"] is False
    assert [item["label"] for item in judgment["fact_results"]] == ["omitted", "omitted"]
