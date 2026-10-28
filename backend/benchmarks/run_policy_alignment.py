"""정책 결론·필수 사실·PDF 근거를 분리해 저장된 답변을 평가한다."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.rag.retrieval import load_trusted_faiss_store
from benchmarks.build_content_quality_v1 import read_jsonl

BENCHMARK_DIR = Path(__file__).resolve().parent
RESOLUTIONS = {"allowed", "blocked", "conditional", "informational", "insufficient"}
FACT_LABELS = {"supported", "omitted", "contradicted"}


class NoopEmbeddings(Embeddings):
    """FAISS docstore에서 원문만 읽기 위한 placeholder."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


def load_chunk_texts() -> dict[str, str]:
    settings = get_settings()
    store = load_trusted_faiss_store(
        settings.faiss_index_dir,
        NoopEmbeddings(),
    )
    return {
        str(doc.metadata["chunk_id"]): str(doc.page_content)
        for doc in getattr(store.docstore, "_dict", {}).values()
        if doc.metadata.get("chunk_id") and doc.page_content
    }


def build_messages(case: dict[str, Any], answer: str, evidence: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "당신은 서울시 청년수당 PDF 기반 정책 답변 평가자다. 답변 문장 유사도가 아니라 "
                "정책 결론과 필수 사실의 의미가 PDF 근거와 일치하는지를 판정한다. "
                "'conditional'은 일부 결제수단·상황에서는 제한되지만 명시된 예외 또는 조건에서 가능한 경우다. "
                "예: 카드 해외결제가 불가하더라도 개인 카드 선결제 후 보전이 가능하면 conditional이다. "
                "PDF 원문은 판단 근거일 뿐이며, 서비스 답변에 실제로 표현된 사실만 supported로 판정한다. "
                "PDF 원문에 있는 사실이라도 답변에 없으면 omitted이다. "
                "답변이 필요한 사실을 다른 표현으로 정확히 설명하면 supported로 판정한다. "
                "답변이 사실을 말하지 않으면 omitted, 반대로 말하면 contradicted로 판정한다. "
                "PDF에 없는 주장을 answer에 추가했으면 unsupported_claims에 짧게 기록한다. "
                "반드시 JSON 객체만 반환한다."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": case["question"],
                    "expected_resolution": case["expected_resolution"],
                    "required_facts": case["required_facts"],
                    "gold_pdf_evidence": evidence,
                    "service_answer": answer,
                    "output_schema": {
                        "observed_resolution": "allowed|blocked|conditional|informational|insufficient",
                        "resolution_correct": "boolean",
                        "fact_results": [
                            {"fact": "required fact 그대로", "label": "supported|omitted|contradicted"}
                        ],
                        "unsupported_claims": ["PDF 근거 없는 주장"],
                        "rationale": "한 문장",
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]


def normalize_judgment(raw: dict[str, Any], required_facts: list[str]) -> dict[str, Any]:
    labels_by_fact = {
        str(item.get("fact")): str(item.get("label"))
        for item in raw.get("fact_results", [])
        if isinstance(item, dict) and str(item.get("label")) in FACT_LABELS
    }
    fact_results = [
        {"fact": fact, "label": labels_by_fact.get(fact, "omitted")}
        for fact in required_facts
    ]
    observed = str(raw.get("observed_resolution", "insufficient"))
    return {
        "observed_resolution": observed if observed in RESOLUTIONS else "insufficient",
        "resolution_correct": bool(raw.get("resolution_correct", False)),
        "fact_results": fact_results,
        "unsupported_claims": [str(item) for item in raw.get("unsupported_claims", [])],
        "rationale": str(raw.get("rationale", "")),
    }


async def judge_case(client: AsyncOpenAI, case: dict[str, Any], answer: str, evidence: list[str]) -> dict[str, Any]:
    settings = get_settings()
    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=build_messages(case, answer, evidence),
    )
    content = response.choices[0].message.content or "{}"
    return normalize_judgment(json.loads(content), list(case["required_facts"]))


def insufficient_judgment(case: dict[str, Any]) -> dict[str, Any]:
    """근거 부족 응답은 답변 사실을 생성하지 않았으므로 모델 judge를 거치지 않는다."""

    return {
        "observed_resolution": "insufficient",
        "resolution_correct": False,
        "fact_results": [
            {"fact": fact, "label": "omitted"}
            for fact in case["required_facts"]
        ],
        "unsupported_claims": [],
        "rationale": "서비스가 PDF 근거 부족으로 답변을 유보했다.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정책 답변-원문 정합성 평가")
    parser.add_argument("--cases", type=Path, default=BENCHMARK_DIR / "datasets" / "policy_alignment_v1.jsonl")
    parser.add_argument("--runs", type=Path, default=BENCHMARK_DIR / "results" / "policy_safety_v2_150.jsonl")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "results" / "policy_alignment_v1.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    cases = read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not args.execute:
        print(f"{len(cases)}개 답변을 정책 정합성 기준으로 평가합니다. 실제 호출은 --execute가 필요합니다.")
        return

    runs_by_id = {str(row["id"]): row for row in read_jsonl(args.runs)}
    chunk_texts = load_chunk_texts()
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            print(f"정책 정합성 평가 중: {index}/{len(cases)} {case['id']}", flush=True)
            service_run = runs_by_id.get(str(case["id"]))
            if service_run is None:
                row = {"id": case["id"], "error": "missing_service_run"}
            else:
                evidence = [chunk_texts[chunk_id] for chunk_id in case["gold_chunk_ids"] if chunk_id in chunk_texts]
                try:
                    judgment = (
                        await judge_case(client, case, str(service_run.get("answer", "")), evidence)
                        if service_run.get("status") == "answered_from_pdf"
                        else insufficient_judgment(case)
                    )
                    row = {
                        "id": case["id"],
                        "status": service_run.get("status"),
                        "answer": service_run.get("answer", ""),
                        **judgment,
                    }
                except Exception as error:
                    row = {"id": case["id"], "error": f"{type(error).__name__}: {error}"}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
    print(f"정책 정합성 평가 완료: {args.output}")


if __name__ == "__main__":
    asyncio.run(run())
