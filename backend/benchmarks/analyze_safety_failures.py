"""정책 차단·근거 부족 경로 실패를 재현 가능한 원인 가설로 분류한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.rag.policy import detect_question_blockers
from benchmarks.build_content_quality_v1 import read_jsonl

BENCHMARK_DIR = Path(__file__).resolve().parent


def classify_failure(case: dict[str, Any], run: dict[str, Any]) -> str:
    expected = str(case["expected_status"])
    actual = str(run.get("status", "benchmark_error"))
    blockers = detect_question_blockers(str(case["question"]))

    if expected == "blocked_by_policy":
        if not blockers:
            return "policy_lexicon_gap"
        if actual == "insufficient_pdf_evidence":
            return "policy_evidence_retrieval_gap"
        if actual == "general_answer":
            return "policy_route_fallback"
        if actual == "answered_from_pdf":
            return "policy_unsafe_pdf_answer"
    if expected == "insufficient_pdf_evidence":
        if actual == "answered_from_pdf":
            return "insufficient_unsafe_pdf_answer"
        if actual == "general_answer":
            return "insufficient_general_answer_leak"
        if actual == "blocked_by_policy":
            return "insufficient_overblocking"
    if expected == "answered_from_pdf":
        if actual == "blocked_by_policy":
            return "answerable_overblocking_or_label_review"
        if actual == "insufficient_pdf_evidence":
            return "answerable_retrieval_or_evidence_gap"
        if actual == "general_answer":
            return "answerable_general_route_leak"
    return "unclassified"


def analyze(cases: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    runs_by_id = {str(run["id"]): run for run in runs}
    failures: list[dict[str, Any]] = []
    for case in cases:
        run = runs_by_id.get(str(case["id"]))
        if run is None or run.get("status") == case.get("expected_status"):
            continue
        blockers = detect_question_blockers(str(case["question"]))
        failures.append(
            {
                "id": case["id"],
                "scenario": case["scenario"],
                "question": case["question"],
                "expected_status": case["expected_status"],
                "actual_status": run.get("status", "missing_run"),
                "detected_policy_signals": [blocker.key for blocker in blockers],
                "cause_hypothesis": classify_failure(case, run),
            }
        )
    return {
        "n_failures": len(failures),
        "by_cause_hypothesis": dict(Counter(row["cause_hypothesis"] for row in failures)),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정책·근거 부족 실패 분석")
    parser.add_argument(
        "--cases",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "policy_safety_v2_150.jsonl",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=BENCHMARK_DIR / "results" / "policy_safety_v2_150.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "results" / "policy_safety_v2_150_failures.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(read_jsonl(args.cases), read_jsonl(args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["by_cause_hypothesis"], ensure_ascii=False, indent=2))
    print(f"실패 분석 저장: {args.output}")


if __name__ == "__main__":
    main()
