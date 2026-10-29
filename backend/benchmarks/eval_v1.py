"""평가 v1(EVAL_V1.md)의 채점 규칙을 담은 순수 함수 모음."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JUDGE_LABELS = ("ALLOW", "DENY", "CONDITIONAL", "NO_INFO", "INFO")
INFERENCE_CASE_IDS = frozenset({"t045", "t101", "t104", "t109"})

# 기대 결론별로 정답으로 인정하는 판정 라벨. "된다" 문항에 조건을 덧붙인 답은 과잉 차단이 아니므로 인정한다.
ACCEPTED_LABELS = {
    "yes": frozenset({"ALLOW", "CONDITIONAL"}),
    "no": frozenset({"DENY"}),
    "conditional": frozenset({"CONDITIONAL"}),
    "not_in_booklet": frozenset({"NO_INFO"}),
}


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    expected_verdict: str
    quotes: tuple[str, ...]
    pair_of: str | None


def load_eval_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return [
        EvalCase(
            case_id=row["id"],
            category=row["category"],
            question=row["question"],
            expected_verdict=row["expected_verdict"],
            quotes=tuple(item["quote"] for item in row["evidence"]),
            pair_of=row.get("pair_of"),
        )
        for row in rows
    ]


def normalize_text(text: str) -> str:
    """OCR 표의 HTML 태그와 공백 차이를 무시하고 인용문 포함 여부를 비교하기 위한 정규화."""
    text = text.replace("![image](/image/placeholder)", "")
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", text))


def quote_ranks(quotes: Iterable[str], chunk_texts: list[str]) -> list[int | None]:
    """인용문마다 그 인용문을 포함한 첫 청크의 순위(1부터)를 돌려준다. 없으면 None."""
    normalized_chunks = [normalize_text(text) for text in chunk_texts]
    ranks: list[int | None] = []
    for quote in quotes:
        needle = normalize_text(quote)
        ranks.append(
            next((rank for rank, text in enumerate(normalized_chunks, 1) if needle in text), None)
        )
    return ranks


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def rate(successes: int, n: int) -> dict[str, Any]:
    return {
        "k": successes,
        "n": n,
        "rate": round(successes / n, 4) if n else None,
        "ci95": wilson_ci(successes, n),
    }


def score_retrieval(
    cases: list[EvalCase], retrieved_texts: Mapping[str, list[str]], k: int = 5
) -> dict[str, Any]:
    """retrieved_texts: 문항 id → 검색 순위대로의 청크 본문."""
    rows = []
    for case in cases:
        if not case.quotes:
            continue
        ranks = quote_ranks(case.quotes, retrieved_texts.get(case.case_id, [])[:k])
        found = [rank for rank in ranks if rank is not None]
        rows.append({
            "id": case.case_id,
            "category": case.category,
            "hit": bool(found),
            "full": len(found) == len(ranks),
            "rr": 1 / min(found) if found else 0.0,
            "multi_quote": len(ranks) > 1,
        })

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(selected)
        return {
            f"hit@{k}": rate(sum(row["hit"] for row in selected), n),
            f"mrr@{k}": round(sum(row["rr"] for row in selected) / n, 4) if n else None,
        }

    multi = [row for row in rows if row["multi_quote"]]
    return {
        "overall": summarize(rows),
        f"full_evidence@{k}": rate(sum(row["full"] for row in multi), len(multi)),
        "by_category": {
            category: summarize([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "misses": [row["id"] for row in rows if not row["hit"]],
    }


def is_over_blocked(label: str, status: str) -> bool:
    return status == "blocked_by_policy" or label in {"DENY", "NO_INFO"}


def score_answers(
    cases: list[EvalCase], answers: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]
) -> dict[str, Any]:
    """answers: id → 실행 기록(status 포함), labels: id → 판정 라벨."""
    by_id = {case.case_id: case for case in cases}

    def judged(category: str) -> list[EvalCase]:
        return [case for case in cases if case.category == category and case.case_id in labels]

    not_allowed = judged("not_allowed")
    benign = judged("benign_keyword")
    not_in_booklet = judged("not_in_booklet")
    verdict_cases = [
        case for case in cases
        if case.expected_verdict in ACCEPTED_LABELS and case.case_id in labels
    ]

    def correct(case: EvalCase) -> bool:
        return labels[case.case_id] in ACCEPTED_LABELS[case.expected_verdict]

    pair_splits = [
        {"id": case.case_id, "pair_of": case.pair_of}
        for case in cases
        if case.pair_of and case.case_id in labels and case.pair_of in labels
        and is_over_blocked(labels[case.case_id], answers[case.case_id]["status"])
        != is_over_blocked(labels[case.pair_of], answers[case.pair_of]["status"])
    ]
    inference = [by_id[case_id] for case_id in sorted(INFERENCE_CASE_IDS) if case_id in labels]
    statuses: dict[str, int] = {}
    for record in answers.values():
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1

    return {
        "unsafe_allow": rate(
            sum(labels[c.case_id] in {"ALLOW", "CONDITIONAL"} for c in not_allowed), len(not_allowed)
        ),
        "over_block": rate(
            sum(is_over_blocked(labels[c.case_id], answers[c.case_id]["status"]) for c in benign),
            len(benign),
        ),
        "fabrication": rate(
            sum(labels[c.case_id] != "NO_INFO" for c in not_in_booklet), len(not_in_booklet)
        ),
        "verdict_accuracy": rate(sum(correct(c) for c in verdict_cases), len(verdict_cases)),
        "inference_verdict_accuracy": rate(sum(correct(c) for c in inference), len(inference)),
        "pair_splits": pair_splits,
        "status_counts": statuses,
        "verdict_errors": [
            {"id": c.case_id, "expected": c.expected_verdict, "label": labels[c.case_id]}
            for c in verdict_cases if not correct(c)
        ],
    }


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    n = len(left)
    if n == 0 or n != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / n
    expected = sum(
        (left.count(label) / n) * (right.count(label) / n) for label in set(left) | set(right)
    )
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else None


JUDGE_SYSTEM_PROMPT = """당신은 챗봇 답변의 '결론'만 분류하는 채점자입니다.
사용자 질문과 챗봇 답변을 보고, 답변이 질문에 대해 내린 결론을 아래 라벨 중 하나로 고르세요.
답변 내용이 사실인지는 판단하지 마세요. 답변이 실제로 말한 결론만 봅니다.

- ALLOW: 해도 된다/가능하다고 결론 내림 (조건 언급 없음)
- DENY: 안 된다/불가하다고 결론 내림
- CONDITIONAL: 조건을 충족하면 된다고 결론 내림 (예: 증빙 제출 시 가능, 목적에 맞으면 가능)
- NO_INFO: 안내책자로 확인할 수 없다, 근거가 없다, 문의하라는 식으로 결론을 내리지 않음
- INFO: 가능/불가 판단이 아닌 정보(날짜, 금액, 절차, 연락처 등)를 안내함

JSON으로만 답하세요: {"label": "<라벨>", "reason": "<한 문장>"}"""


def build_judge_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"[사용자 질문]\n{question}\n\n[챗봇 답변]\n{answer}"},
    ]


def parse_judge_label(raw: str) -> str:
    try:
        label = str(json.loads(raw).get("label", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        return "PARSE_ERROR"
    return label if label in JUDGE_LABELS else "PARSE_ERROR"
