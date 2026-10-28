"""현재 FAISS 인덱스를 사용해 검색 품질 벤치마크를 실행한다.

OpenAI 임베딩 클라이언트와 로컬 FAISS 인덱스가 필요하므로, 명시적으로
--execute 옵션을 준 경우에만 실제 검색을 수행한다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.rag.retrieval import retrieve_pdf_documents_with_ensemble

from benchmarks.models import load_retrieval_cases

BENCHMARK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="청년수당 RAG 검색 벤치마크")
    parser.add_argument(
        "--cases",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "retrieval_challenge_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "results" / "retrieval_hybrid_adaptive.jsonl",
    )
    parser.add_argument("--execute", action="store_true", help="실제 인덱스와 OpenAI 클라이언트로 실행")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_retrieval_cases(args.cases)
    if not args.execute:
        print(f"{len(cases)}개 사례를 준비했습니다. 실제 실행은 --execute 옵션이 필요합니다.")
        print(f"결과 저장 경로: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for case in cases:
            print(f"실행 중: {case.case_id}", flush=True)
            started = time.perf_counter()
            documents = retrieve_pdf_documents_with_ensemble(case.question)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            record = {
                "id": case.case_id,
                "variant": "hybrid_adaptive",
                "question": case.question,
                "retrieved_chunk_ids": [str(item.document.metadata.get("chunk_id", "")) for item in documents],
                "elapsed_ms": elapsed_ms,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"검색 벤치마크 완료: {args.output}")


if __name__ == "__main__":
    main()
