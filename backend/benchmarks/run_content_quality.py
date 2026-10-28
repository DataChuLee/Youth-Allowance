"""문서 기반 답변의 검색·출처·생성 일치도를 평가한다.

이 Runner는 이미 저장된 LangGraph 응답을 재사용한다. 따라서 답변 생성은 다시
실행하지 않고, 현재 검색기 결과와 RAGAS Faithfulness/AnswerCorrectness만 측정한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings, get_settings
from app.rag.retrieval import (
    load_trusted_faiss_store,
    retrieve_pdf_documents_with_ensemble,
)
from benchmarks.build_content_quality_v1 import read_jsonl
from benchmarks.models import load_content_quality_cases

BENCHMARK_DIR = Path(__file__).resolve().parent


class NoopEmbeddings(Embeddings):
    """FAISS docstore에서 원문 청크만 읽기 위한 임베딩 placeholder."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def load_chunk_texts(settings: Settings) -> dict[str, str]:
    """검색 호출 없이 로컬 FAISS docstore에서 chunk_id별 원문을 읽는다."""

    store = load_trusted_faiss_store(
        settings.faiss_index_dir,
        NoopEmbeddings(),
    )
    documents = getattr(store.docstore, "_dict", {}).values()
    return {
        str(document.metadata["chunk_id"]): str(document.page_content)
        for document in documents
        if getattr(document, "metadata", {}).get("chunk_id") and getattr(document, "page_content", "")
    }


async def create_ragas_metrics(settings: Settings):
    """RAGAS 0.4의 구조화된 judge를 지연 로드한다.

    AnswerCorrectness는 의미 유사도 대신 사실 주장 F1만 사용한다. 정책 문서에서는
    핵심 조건 누락을 더 강하게 반영하기 위해 임베딩 유사도와 섞지 않는다.
    """

    from openai import AsyncOpenAI
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, Faithfulness

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.openai_chat_model, client=client)
    return (
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, weights=[1.0, 0.0], beta=1.0),
    )


async def judge_answer(
    *,
    faithfulness_metric: Any,
    correctness_metric: Any,
    question: str,
    answer: str,
    reference_answer: str,
    source_contexts: list[str],
) -> tuple[float | None, float | None, str | None]:
    """응답이 인용한 PDF 원문과 참조 답변에 각각 부합하는지 평가한다."""

    try:
        correctness_result = await correctness_metric.ascore(
            user_input=question,
            response=answer,
            reference=reference_answer,
        )
        faithfulness_result = (
            await faithfulness_metric.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=source_contexts,
            )
            if source_contexts
            else None
        )
        return (
            float(faithfulness_result.value) if faithfulness_result is not None else None,
            float(correctness_result.value),
            None,
        )
    except Exception as error:  # 외부 judge 실패는 전체 평가를 중단시키지 않는다.
        return None, None, f"{type(error).__name__}: {error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF 답변 품질 벤치마크 실행")
    parser.add_argument(
        "--cases",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "content_quality_v1.jsonl",
    )
    parser.add_argument(
        "--policy-runs",
        type=Path,
        default=BENCHMARK_DIR / "results" / "policy_safety_v2_150.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "results" / "content_quality_v1.jsonl",
    )
    parser.add_argument("--limit", type=int, help="연결 점검용 최대 실행 문항 수")
    parser.add_argument("--execute", action="store_true", help="OpenAI 임베딩·RAGAS judge 호출 허용")
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    cases = load_content_quality_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not args.execute:
        print(f"{len(cases)}개 문서 기반 답변을 평가합니다. 실제 호출은 --execute가 필요합니다.")
        return

    settings = get_settings()
    policy_runs = {str(row["id"]): row for row in read_jsonl(args.policy_runs)}
    chunk_texts = load_chunk_texts(settings)
    faithfulness_metric, correctness_metric = await create_ragas_metrics(settings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            print(f"문서 품질 평가 중: {index}/{len(cases)} {case.case_id}", flush=True)
            policy_run = policy_runs.get(case.case_id)
            if policy_run is None:
                record = {"id": case.case_id, "error": "missing_policy_run"}
            else:
                retrieved = retrieve_pdf_documents_with_ensemble(case.question, settings)
                retrieved_chunk_ids = [
                    str(item.document.metadata["chunk_id"])
                    for item in retrieved
                    if item.document.metadata.get("chunk_id")
                ]
                source_chunk_ids = [str(value) for value in policy_run.get("source_chunk_ids", [])]
                source_contexts = [chunk_texts[chunk_id] for chunk_id in source_chunk_ids if chunk_id in chunk_texts]
                faithfulness, answer_correctness, judge_error = await judge_answer(
                    faithfulness_metric=faithfulness_metric,
                    correctness_metric=correctness_metric,
                    question=case.question,
                    answer=str(policy_run.get("answer", "")),
                    reference_answer=case.reference_answer,
                    source_contexts=source_contexts,
                )
                record = {
                    "id": case.case_id,
                    "question": case.question,
                    "status": policy_run.get("status"),
                    "answer": policy_run.get("answer", ""),
                    "source_chunk_ids": source_chunk_ids,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "faithfulness": faithfulness,
                    "answer_correctness": answer_correctness,
                    "judge_error": judge_error,
                    "elapsed_ms": policy_run.get("elapsed_ms"),
                }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    print(f"문서 기반 답변 품질 평가 완료: {args.output}")


if __name__ == "__main__":
    asyncio.run(run())
