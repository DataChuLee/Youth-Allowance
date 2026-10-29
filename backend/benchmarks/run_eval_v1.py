"""평가 v1 실행기. 단계: retrieval → answer → judge → sample → score.

실제 API를 호출하는 단계(retrieval, answer, judge)는 --execute가 있어야 실행한다.
모든 명령은 backend/ 디렉터리에서 `python -m benchmarks.run_eval_v1 <단계>`로 실행한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.eval_v1 import (
    build_judge_messages,
    cohen_kappa,
    load_eval_cases,
    parse_judge_label,
    score_answers,
    score_retrieval,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
CASES_PATH = BENCHMARK_DIR / "datasets" / "eval_test_v1.jsonl"
RESULTS_DIR = BENCHMARK_DIR / "results"
BASELINE_CHUNKS_SHA256 = "8fc01b77bf3db6254657f6bf176922aee4130d1913854a21eaaad2cc7768a49d"
JUDGE_MODEL = "gpt-4.1-2025-04-14"
SAMPLE_SIZE = 30


def result_path(name: str, tag: str) -> Path:
    return RESULTS_DIR / f"eval_v1_{tag}_{name}"


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_index_manifest() -> dict[str, Any]:
    from app.core.config import get_settings
    from app.rag.retrieval import resolve_trusted_faiss_index_dir

    index_dir = resolve_trusted_faiss_index_dir(get_settings().faiss_index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    # 재생성한 인덱스는 청크 경계가 달라지므로, 고정한 기준선이 아니면 측정을 거부한다.
    if manifest["chunks_sha256"] != BASELINE_CHUNKS_SHA256:
        raise SystemExit(
            "현재 FAISS 인덱스가 기준선과 다릅니다. Data/faiss_rag_baseline_18fec50을 복원하세요."
        )
    return manifest


def disable_query_cache() -> None:
    # 캐시 key에 인덱스 버전이 없어 이전 인덱스 결과가 섞일 수 있으므로 측정 중에는 끈다.
    from app.core import cache

    cache._instance = cache.DisabledQueryCache()


def run_meta(stage: str, **extra: Any) -> dict[str, Any]:
    from app.core.config import get_settings

    settings = get_settings()
    return {
        "stage": stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "cases": CASES_PATH.name,
        "app_env": settings.app_env,
        "chat_model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
        **extra,
    }


def write_jsonl(path: Path, meta: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_meta": meta}, ensure_ascii=False)]
    lines += [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[0]["_meta"], {row["id"]: row for row in rows[1:]}


def stage_retrieval(tag: str) -> None:
    from app.rag.retrieval import retrieve_pdf_documents_with_ensemble

    manifest = load_index_manifest()
    disable_query_cache()
    records = []
    for case in load_eval_cases(CASES_PATH):
        started = time.perf_counter()
        documents = retrieve_pdf_documents_with_ensemble(case.question)
        records.append({
            "id": case.case_id,
            "chunk_ids": [str(item.document.metadata["chunk_id"]) for item in documents],
            "chunk_texts": [item.document.page_content for item in documents],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        print(f"retrieval {case.case_id}", flush=True)
    write_jsonl(result_path("retrieval.jsonl", tag), run_meta("retrieval", index=manifest), records)


def stage_answer(tag: str) -> None:
    from app.graph.workflow import run_chat_graph

    manifest = load_index_manifest()
    disable_query_cache()
    records = []
    for case in load_eval_cases(CASES_PATH):
        started = time.perf_counter()
        try:
            response = run_chat_graph(case.question)
            record = {
                "id": case.case_id,
                "status": response.status,
                "answer": response.answer,
                "source_chunk_ids": [source.chunk_id for source in response.sources],
            }
        except Exception as error:
            # 원문 컨텍스트나 키가 남지 않도록 예외 타입만 기록한다.
            record = {"id": case.case_id, "status": "benchmark_error", "answer": "",
                      "source_chunk_ids": [], "error": type(error).__name__}
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        records.append(record)
        print(f"answer {case.case_id} {record['status']}", flush=True)
    write_jsonl(result_path("answers.jsonl", tag), run_meta("answer", index=manifest), records)


def stage_judge(tag: str) -> None:
    from openai import OpenAI

    from app.core.config import get_settings

    client = OpenAI(api_key=get_settings().openai_api_key)
    _, answers = read_jsonl(result_path("answers.jsonl", tag))
    records = []
    for case in load_eval_cases(CASES_PATH):
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=build_judge_messages(case.question, answers[case.case_id]["answer"]),
        )
        raw = response.choices[0].message.content or ""
        records.append({"id": case.case_id, "label": parse_judge_label(raw), "raw": raw})
        print(f"judge {case.case_id} {records[-1]['label']}", flush=True)
    write_jsonl(result_path("judge.jsonl", tag), run_meta("judge", judge_model=JUDGE_MODEL), records)


def stage_sample(tag: str) -> None:
    """판정자 검증용: 범주별로 고르게 30개를 뽑아 사람이 라벨을 다는 CSV를 만든다(판정 결과는 숨김)."""
    _, answers = read_jsonl(result_path("answers.jsonl", tag))
    cases = load_eval_cases(CASES_PATH)
    rng = random.Random(20261029)
    by_category: dict[str, list] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)
    picked = []
    per_category = SAMPLE_SIZE // len(by_category)
    for category in sorted(by_category):
        picked += rng.sample(by_category[category], min(per_category, len(by_category[category])))
    rng.shuffle(picked)
    path = result_path("judge_sample.csv", tag)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "question", "answer", "human_label(ALLOW/DENY/CONDITIONAL/NO_INFO/INFO)"])
        for case in picked:
            writer.writerow([case.case_id, case.question, answers[case.case_id]["answer"], ""])
    print(f"{len(picked)}개 표본 저장: {path}")


def read_human_labels(path: Path) -> dict[str, str]:
    # 엑셀이 CP949로 다시 저장하는 경우가 있어 두 인코딩을 모두 시도한다.
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=encoding, newline="") as file:
                rows = list(csv.reader(file))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"CSV 인코딩을 읽을 수 없습니다: {path}")
    return {row[0]: row[3].strip().upper() for row in rows[1:] if len(row) > 3 and row[3].strip()}


def stage_score(tag: str) -> None:
    cases = load_eval_cases(CASES_PATH)
    retrieval_meta, retrieval = read_jsonl(result_path("retrieval.jsonl", tag))
    answer_meta, answers = read_jsonl(result_path("answers.jsonl", tag))
    judge_meta, judged = read_jsonl(result_path("judge.jsonl", tag))
    labels = {case_id: row["label"] for case_id, row in judged.items()}

    summary: dict[str, Any] = {
        "meta": {"retrieval": retrieval_meta, "answer": answer_meta, "judge": judge_meta},
        "retrieval": score_retrieval(
            cases, {case_id: row["chunk_texts"] for case_id, row in retrieval.items()}
        ),
        "answers": score_answers(cases, answers, labels),
        "judge_parse_errors": [case_id for case_id, label in labels.items() if label == "PARSE_ERROR"],
        "latency_ms": latency_summary([row["elapsed_ms"] for row in answers.values()]),
    }
    human_path = result_path("judge_sample.csv", tag)
    human = read_human_labels(human_path) if human_path.exists() else {}
    if human:
        ids = sorted(case_id for case_id in human if case_id in labels)
        summary["judge_agreement"] = {
            "n": len(ids),
            "agreement": round(sum(human[i] == labels[i] for i in ids) / len(ids), 4) if ids else None,
            "cohen_kappa": cohen_kappa([human[i] for i in ids], [labels[i] for i in ids]),
            "disagreements": [{"id": i, "human": human[i], "judge": labels[i]} for i in ids
                              if human[i] != labels[i]],
        }
    path = result_path("summary.json", tag)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, ensure_ascii=False, indent=2))
    print(f"요약 저장: {path}")


def latency_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"p50": pick(0.5), "p95": pick(0.95), "n": len(ordered)}


STAGES = {
    "retrieval": stage_retrieval,
    "answer": stage_answer,
    "judge": stage_judge,
    "sample": stage_sample,
    "score": stage_score,
}
API_STAGES = {"retrieval", "answer", "judge"}


def main() -> None:
    parser = argparse.ArgumentParser(description="청년수당 RAG 평가 v1")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--tag", default="baseline", help="결과 파일 구분용 이름")
    parser.add_argument("--execute", action="store_true", help="외부 API 호출 단계를 실제로 실행")
    args = parser.parse_args()
    if args.stage in API_STAGES and not args.execute:
        print(f"{args.stage} 단계는 API를 호출합니다. 실행하려면 --execute를 붙이세요.")
        return
    STAGES[args.stage](args.tag)


if __name__ == "__main__":
    main()
