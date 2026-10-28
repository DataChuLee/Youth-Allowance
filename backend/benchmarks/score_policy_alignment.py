"""정책 결론과 원자적 사실 판정 결과를 집계한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.build_content_quality_v1 import read_jsonl

BENCHMARK_DIR = Path(__file__).resolve().parent


def score(rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_by_id = {str(case["id"]): case for case in cases}
    details: list[dict[str, Any]] = []
    for row in rows:
        case = case_by_id.get(str(row.get("id")))
        if case is None or row.get("error"):
            continue
        fact_results = list(row.get("fact_results", []))
        labels = [str(item.get("label")) for item in fact_results if isinstance(item, dict)]
        supported = labels.count("supported")
        details.append({
            "id": case["id"],
            "expected_resolution": case["expected_resolution"],
            "resolution_correct": bool(row.get("resolution_correct")),
            "fact_coverage": supported / len(case["required_facts"]) if case["required_facts"] else 1.0,
            "has_contradiction": "contradicted" in labels,
            "has_unsupported_claim": bool(row.get("unsupported_claims")),
        })
    if not details:
        return {"n_scored": 0, "metrics": {}, "details": []}

    def mean(key: str) -> float:
        return round(sum(float(row[key]) for row in details) / len(details), 4)

    actionable = [row for row in details if row["expected_resolution"] != "informational"]
    policy_pass = [
        row["resolution_correct"] and not row["has_contradiction"]
        for row in actionable
    ]
    return {
        "n_scored": len(details),
        "metrics": {
            "resolution_accuracy": mean("resolution_correct"),
            "required_fact_coverage": mean("fact_coverage"),
            "contradiction_free_rate": round(sum(not row["has_contradiction"] for row in details) / len(details), 4),
            "unsupported_claim_free_rate": round(sum(not row["has_unsupported_claim"] for row in details) / len(details), 4),
            "actionable_policy_pass_rate": round(sum(policy_pass) / len(actionable), 4) if actionable else None,
        },
        "by_resolution": {
            label: {
                "n": len(group),
                "resolution_accuracy": round(sum(row["resolution_correct"] for row in group) / len(group), 4),
                "required_fact_coverage": round(sum(row["fact_coverage"] for row in group) / len(group), 4),
            }
            for label in sorted({str(row["expected_resolution"]) for row in details})
            for group in [[row for row in details if row["expected_resolution"] == label]]
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="정책 정합성 평가 결과 집계")
    parser.add_argument("--cases", type=Path, default=BENCHMARK_DIR / "datasets" / "policy_alignment_v1.jsonl")
    parser.add_argument("--runs", type=Path, default=BENCHMARK_DIR / "results" / "policy_alignment_v1.jsonl")
    args = parser.parse_args()
    print(json.dumps(score(read_jsonl(args.runs), read_jsonl(args.cases)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
