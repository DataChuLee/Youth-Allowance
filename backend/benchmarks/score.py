"""벤치마크 실행 결과를 포트폴리오용 요약 JSON으로 집계한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.models import (
    load_content_quality_cases,
    load_policy_cases,
    load_retrieval_cases,
    score_content_quality_runs,
    score_policy_runs,
    score_retrieval_runs,
)

BENCHMARK_DIR = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="벤치마크 결과 채점")
    parser.add_argument("kind", choices=["retrieval", "policy", "content"])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_case_name = {
        "retrieval": "retrieval_challenge_v1.jsonl",
        "policy": "policy_safety_v1.jsonl",
        "content": "content_quality_v1.jsonl",
    }[args.kind]
    cases_path = args.cases or BENCHMARK_DIR / "datasets" / default_case_name
    output_path = args.output or BENCHMARK_DIR / "results" / f"{args.kind}_summary.json"
    runs = read_jsonl(args.runs)
    if args.kind == "retrieval":
        summary = score_retrieval_runs(load_retrieval_cases(cases_path), runs)
    elif args.kind == "policy":
        summary = score_policy_runs(load_policy_cases(cases_path), runs)
    else:
        summary = score_content_quality_runs(load_content_quality_cases(cases_path), runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": summary["metrics"], "latency_ms": summary.get("latency_ms", {})}, ensure_ascii=False, indent=2))
    print(f"요약 저장: {output_path}")


if __name__ == "__main__":
    main()
