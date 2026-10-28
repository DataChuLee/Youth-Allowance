"""실제 LangGraph 응답 경로와 정책 안전성 벤치마크를 실행한다."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.graph.workflow import run_chat_graph

from benchmarks.models import load_policy_cases

BENCHMARK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="청년수당 RAG 정책·안전성 벤치마크")
    parser.add_argument(
        "--cases",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "policy_safety_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "results" / "policy_safety_v1.jsonl",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=["answerable", "conditional", "policy_blocked", "insufficient"],
        help="지정한 시나리오만 실행 (여러 번 지정 가능)",
    )
    parser.add_argument("--execute", action="store_true", help="실제 LLM 호출을 허용하고 실행")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_policy_cases(args.cases)
    if args.scenario:
        cases = [case for case in cases if case.scenario in set(args.scenario)]
    if not args.execute:
        print(f"{len(cases)}개 사례를 준비했습니다. 실제 LLM 실행은 --execute 옵션이 필요합니다.")
        print(f"결과 저장 경로: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for case in cases:
            print(f"실행 중: {case.case_id}", flush=True)
            started = time.perf_counter()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            try:
                response = run_chat_graph(case.question)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                record = {
                    "id": case.case_id,
                    "question": case.question,
                    "status": response.status,
                    "answer": response.answer,
                    "source_chunk_ids": [source.chunk_id for source in response.sources],
                    "needs_external_search": response.needs_external_search,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as error:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                print(f"실패: {case.case_id} ({type(error).__name__})", flush=True)
                record = {
                    "id": case.case_id,
                    "question": case.question,
                    "status": "benchmark_error",
                    "answer": "",
                    "source_chunk_ids": [],
                    "needs_external_search": True,
                    "elapsed_ms": elapsed_ms,
                    # API 키·원문 컨텍스트가 로그에 남지 않도록 예외 타입과 메시지만 저장한다.
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    print(f"정책·안전성 벤치마크 완료: {args.output}")


if __name__ == "__main__":
    main()
