"""키워드 차단 확인기 개발셋 비교. 키워드만 쓴 경우(keyword)와 확인기를 붙인 경우(confirm)를 같은 질문으로 돌린다.

backend/에서 `python -m benchmarks.run_keyword_dev --execute`로 실행한다. 시험셋은 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.eval_v1 import rate
from benchmarks.run_eval_v1 import disable_query_cache, load_index_manifest

BENCHMARK_DIR = Path(__file__).resolve().parent
CASES_PATH = BENCHMARK_DIR / "datasets" / "policy_keyword_dev_v1.jsonl"
OUTPUT_PATH = BENCHMARK_DIR / "results" / "policy_keyword_dev_v1.json"


def summarize(cases: list[dict], statuses: dict[str, str]) -> dict:
    benign = [case for case in cases if not case["expected_blocked"]]
    blocked = [case for case in cases if case["expected_blocked"]]
    return {
        "benign_blocked": rate(sum(statuses[c["id"]] == "blocked_by_policy" for c in benign), len(benign)),
        "blocked_kept": rate(sum(statuses[c["id"]] == "blocked_by_policy" for c in blocked), len(blocked)),
        "released_blocked_ids": [c["id"] for c in blocked if statuses[c["id"]] != "blocked_by_policy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="키워드 차단 확인기 개발셋 비교")
    parser.add_argument("--execute", action="store_true", help="실제 LLM 호출을 허용하고 실행")
    args = parser.parse_args()
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line]
    if not args.execute:
        print(f"{len(cases)}개 사례 x 2개 조건. 실행하려면 --execute를 붙이세요.")
        return

    from app.graph.workflow import run_chat_graph

    load_index_manifest()
    disable_query_cache()
    # keyword: 확인기 없이 기존 키워드 차단, confirm: 서비스 기본 확인기
    conditions = {"keyword": lambda question, blockers: blockers, "confirm": None}
    runs: dict[str, dict[str, dict]] = {}
    for name, confirm in conditions.items():
        runs[name] = {}
        for case in cases:
            response = run_chat_graph(case["question"], confirm_blockers=confirm)
            runs[name][case["id"]] = {"status": response.status, "answer": response.answer}
            print(f"{name} {case['id']} {response.status}", flush=True)
    summary = {
        name: summarize(cases, {case_id: row["status"] for case_id, row in rows.items()})
        for name, rows in runs.items()
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
