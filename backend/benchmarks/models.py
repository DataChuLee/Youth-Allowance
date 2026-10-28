"""벤치마크 사례 로딩과 점수 계산을 위한 순수 함수 모음."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    category: str
    question: str
    gold_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class PolicyCase:
    case_id: str
    scenario: str
    question: str
    expected_status: str
    acceptable_chunk_ids: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    prohibited_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContentQualityCase:
    """PDF 근거와 참조 답변이 검토된 문서 기반 답변 평가 문항."""

    case_id: str
    question: str
    reference_answer: str
    gold_chunk_ids: tuple[str, ...]
    critical_facts: tuple[str, ...]
    review_status: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    return [
        RetrievalCase(
            case_id=str(row["id"]),
            category=str(row["category"]),
            question=str(row["question"]),
            gold_chunk_ids=tuple(str(value) for value in row["gold_chunk_ids"]),
        )
        for row in _read_jsonl(path)
    ]


def load_policy_cases(path: Path) -> list[PolicyCase]:
    return [
        PolicyCase(
            case_id=str(row["id"]),
            scenario=str(row["scenario"]),
            question=str(row["question"]),
            expected_status=str(row["expected_status"]),
            acceptable_chunk_ids=tuple(str(value) for value in row.get("acceptable_chunk_ids", [])),
            required_terms=tuple(str(value) for value in row.get("required_terms", [])),
            prohibited_terms=tuple(str(value) for value in row.get("prohibited_terms", [])),
        )
        for row in _read_jsonl(path)
    ]


def load_content_quality_cases(path: Path) -> list[ContentQualityCase]:
    return [
        ContentQualityCase(
            case_id=str(row["id"]),
            question=str(row["question"]),
            reference_answer=str(row["reference_answer"]),
            gold_chunk_ids=tuple(str(value) for value in row["gold_chunk_ids"]),
            critical_facts=tuple(str(value) for value in row.get("critical_facts", [])),
            review_status=str(row.get("review_status", "unknown")),
        )
        for row in _read_jsonl(path)
    ]


def reciprocal_rank(retrieved_chunk_ids: list[str], gold_chunk_ids: tuple[str, ...], k: int) -> float:
    gold = set(gold_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in gold:
            return 1 / rank
    return 0.0


def recall_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: tuple[str, ...], k: int) -> float:
    return float(bool(set(retrieved_chunk_ids[:k]) & set(gold_chunk_ids)))


def ndcg_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: tuple[str, ...], k: int) -> float:
    gold = set(gold_chunk_ids)
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1)
        if chunk_id in gold
    )
    ideal_count = min(len(gold), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def score_retrieval_runs(
    cases: list[RetrievalCase],
    runs: list[dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    cases_by_id = {case.case_id: case for case in cases}
    scored: list[dict[str, Any]] = []
    for run in runs:
        case = cases_by_id.get(str(run.get("id")))
        if case is None:
            continue
        chunk_ids = [str(value) for value in run.get("retrieved_chunk_ids", [])]
        scored.append(
            {
                "id": case.case_id,
                "category": case.category,
                "variant": str(run.get("variant", "unknown")),
                "mrr": reciprocal_rank(chunk_ids, case.gold_chunk_ids, k),
                "recall_at_k": recall_at_k(chunk_ids, case.gold_chunk_ids, k),
                "ndcg_at_k": ndcg_at_k(chunk_ids, case.gold_chunk_ids, k),
                "elapsed_ms": run.get("elapsed_ms"),
            }
        )
    if not scored:
        return {"n_scored": 0, "k": k, "metrics": {}, "by_category": {}}

    def average(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mrr": round(sum(float(row["mrr"]) for row in rows) / len(rows), 4),
            "recall_at_k": round(sum(float(row["recall_at_k"]) for row in rows) / len(rows), 4),
            "ndcg_at_k": round(sum(float(row["ndcg_at_k"]) for row in rows) / len(rows), 4),
        }

    categories = sorted({str(row["category"]) for row in scored})
    return {
        "n_scored": len(scored),
        "k": k,
        "metrics": average(scored),
        "latency_ms": summarize_latency(scored),
        "by_category": {
            category: average([row for row in scored if row["category"] == category])
            for category in categories
        },
        "details": scored,
    }


def score_policy_runs(cases: list[PolicyCase], runs: list[dict[str, Any]]) -> dict[str, Any]:
    cases_by_id = {case.case_id: case for case in cases}
    details: list[dict[str, Any]] = []
    for run in runs:
        case = cases_by_id.get(str(run.get("id")))
        if case is None:
            continue
        status = str(run.get("status", ""))
        answer = str(run.get("answer", "")).lower()
        source_ids = {str(value) for value in run.get("source_chunk_ids", [])}
        source_hit = (
            bool(source_ids & set(case.acceptable_chunk_ids))
            if case.acceptable_chunk_ids
            else None
        )
        required_terms_present = all(term.lower() in answer for term in case.required_terms)
        prohibited_terms_absent = all(term.lower() not in answer for term in case.prohibited_terms)
        expected_safe_path = case.expected_status in {"blocked_by_policy", "insufficient_pdf_evidence"}
        # 응답 문구가 우연히 올바르더라도, 정책 제한·근거 부족 질문이 일반 답변
        # 경로를 탔다면 API 계약 관점에서는 안전하지 않은 라우팅이다.
        unsafe_route = expected_safe_path and status == "answered_from_pdf"
        details.append(
            {
                "id": case.case_id,
                "scenario": case.scenario,
                "status_correct": status == case.expected_status,
                "source_hit": source_hit,
                "required_terms_present": required_terms_present,
                "prohibited_terms_absent": prohibited_terms_absent,
                "unsafe_route": unsafe_route,
                "elapsed_ms": run.get("elapsed_ms"),
            }
        )

    if not details:
        return {"n_scored": 0, "metrics": {}, "by_scenario": {}}

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        applicable_source_rows = [row for row in rows if row["source_hit"] is not None]
        applicable_required_term_rows = [
            row
            for row in rows
            if cases_by_id[row["id"]].required_terms
        ]
        safe_rows = [row for row in rows if row["scenario"] in {"policy_blocked", "insufficient"}]
        return {
            "status_accuracy": round(sum(row["status_correct"] for row in rows) / len(rows), 4),
            "source_hit_rate": round(
                sum(bool(row["source_hit"]) for row in applicable_source_rows) / len(applicable_source_rows), 4
            ) if applicable_source_rows else None,
            "required_term_rate": round(
                sum(row["required_terms_present"] for row in applicable_required_term_rows)
                / len(applicable_required_term_rows),
                4,
            ) if applicable_required_term_rows else None,
            "unsafe_route_rate": round(sum(row["unsafe_route"] for row in safe_rows) / len(safe_rows), 4)
            if safe_rows else 0.0,
        }

    scenarios = sorted({str(row["scenario"]) for row in details})
    return {
        "n_scored": len(details),
        "metrics": summarize(details),
        "latency_ms": summarize_latency(details),
        "by_scenario": {
            scenario: summarize([row for row in details if row["scenario"] == scenario])
            for scenario in scenarios
        },
        "details": details,
    }


def score_content_quality_runs(
    cases: list[ContentQualityCase],
    runs: list[dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """검색·출처·생성 품질을 서로 다른 지표로 집계한다."""

    cases_by_id = {case.case_id: case for case in cases}
    details: list[dict[str, Any]] = []
    for run in runs:
        case = cases_by_id.get(str(run.get("id")))
        if case is None:
            continue
        retrieved_chunk_ids = [str(value) for value in run.get("retrieved_chunk_ids", [])]
        cited_chunk_ids = {str(value) for value in run.get("source_chunk_ids", [])}
        faithfulness = run.get("faithfulness")
        answer_correctness = run.get("answer_correctness")
        details.append(
            {
                "id": case.case_id,
                "review_status": case.review_status,
                "status": str(run.get("status", "unknown")),
                "mrr": reciprocal_rank(retrieved_chunk_ids, case.gold_chunk_ids, k),
                "recall_at_k": recall_at_k(retrieved_chunk_ids, case.gold_chunk_ids, k),
                "ndcg_at_k": ndcg_at_k(retrieved_chunk_ids, case.gold_chunk_ids, k),
                "source_hit": bool(cited_chunk_ids & set(case.gold_chunk_ids)),
                "faithfulness": float(faithfulness) if faithfulness is not None else None,
                "answer_correctness": (
                    float(answer_correctness) if answer_correctness is not None else None
                ),
                "elapsed_ms": run.get("elapsed_ms"),
            }
        )

    if not details:
        return {"n_scored": 0, "k": k, "metrics": {}, "details": []}

    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    def summarize(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        return {
            "mrr": mean(rows, "mrr"),
            "recall_at_k": mean(rows, "recall_at_k"),
            "ndcg_at_k": mean(rows, "ndcg_at_k"),
            "source_hit_rate": mean(rows, "source_hit"),
            "faithfulness": mean(rows, "faithfulness"),
            "answer_correctness": mean(rows, "answer_correctness"),
        }

    statuses = sorted({str(row["status"]) for row in details})
    return {
        "n_scored": len(details),
        "k": k,
        "metrics": summarize(details),
        "by_status": {
            status: summarize([row for row in details if row["status"] == status])
            for status in statuses
        },
        "latency_ms": summarize_latency(details),
        "coverage": {
            "faithfulness_scored": sum(row["faithfulness"] is not None for row in details),
            "answer_correctness_scored": sum(
                row["answer_correctness"] is not None for row in details
            ),
        },
        "details": details,
    }


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(row["elapsed_ms"]) for row in rows if row.get("elapsed_ms") is not None)
    if not values:
        return {}

    def percentile(percent: float) -> float:
        index = max(0, math.ceil(len(values) * percent) - 1)
        return round(values[index], 2)

    return {
        "avg": round(sum(values) / len(values), 2),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }
