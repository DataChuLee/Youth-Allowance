"""규칙 연결기 개발셋 측정. 시험셋에는 쓰지 않는다.

python -m benchmarks.run_rule_mapping_dev --execute [--model gpt-4.1-mini-2025-04-14] [--tag name]

지표
- 결론 일치: 연결기 + 핸들러 결론이 gold_verdicts 안에 있는가
- 잘못 끼어듦: gold가 unclear(규칙 대상 아님)뿐인데 규칙이 결론을 낸 수. 서비스에서 RAG 대신 틀린 결론을 내는 경우라 따로 센다
- 규칙 ID 일치: gold_rule_ids가 있는 문항에서 연결한 ID가 그 안에 있는가
- 규칙 처리율: 결론이 unclear가 아닌 비율
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.run_eval_v1 import RESULTS_DIR

DEV_PATH = Path(__file__).resolve().parent / "datasets" / "rule_mapping_dev_v1.jsonl"


def score(rows: list[dict]) -> dict:
    n = len(rows)
    with_rule = [row for row in rows if row["gold_rule_ids"]]
    none_only = [row for row in rows if row["gold_verdicts"] == ["unclear"]]
    return {
        "n": n,
        "verdict_match": sum(row["verdict"] in row["gold_verdicts"] for row in rows),
        "wrong_intrusion": sum(row["verdict"] != "unclear" for row in none_only),
        "n_none_only": len(none_only),
        "rule_id_match": sum(row["rule_id"] in row["gold_rule_ids"] for row in with_rule),
        "n_with_rule": len(with_rule),
        "handled": sum(row["verdict"] != "unclear" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="규칙 연결기 개발셋 측정")
    parser.add_argument("--execute", action="store_true", help="LLM을 실제로 호출")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tag", default="mapper")
    args = parser.parse_args()
    if not args.execute:
        print("연결기는 LLM을 호출합니다. 실행하려면 --execute를 붙이세요.")
        return

    from app.rules.mapper import create_rule_mapper, resolve_match

    mapper = create_rule_mapper(model=args.model)
    rows = []
    for line in DEV_PATH.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        match = mapper(case["question"])
        verdict = resolve_match(match)
        rows.append({**case, "verdict": verdict.verdict, "rule_id": match.rule_id if match.domain != "none" else None,
                     "match": match.model_dump()})
        flag = "" if verdict.verdict in case["gold_verdicts"] else "  <-- 불일치"
        print(f"{case['id']} {verdict.verdict} {match.rule_id} {match.payment} {match.exception_id}{flag}", flush=True)

    summary = score(rows)
    summary["model"] = args.model or "default(openai_rule_mapper_model)"
    out = RESULTS_DIR / f"rule_mapping_dev_{args.tag}.jsonl"
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
