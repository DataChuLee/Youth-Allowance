"""평가 v1 반복 실행 집계. 조건마다 여러 태그의 결과를 모아 실행 간 편차를 본다.

예: python -m benchmarks.aggregate_eval_v1 baseline=baseline,baseline_r2,baseline_r3 \
        blocker_check=blocker_check,blocker_check_r2,blocker_check_r3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.eval_v1 import EvalCase, load_eval_cases, rate, score_answers
from benchmarks.run_eval_v1 import CASES_PATH, read_jsonl, result_path

METRICS = ("unsafe_allow", "over_block", "fabrication", "verdict_accuracy")


def load_run(tag: str) -> tuple[dict[str, dict], dict[str, str]]:
    _, answers = read_jsonl(result_path("answers.jsonl", tag))
    _, judged = read_jsonl(result_path("judge.jsonl", tag))
    return answers, {case_id: row["label"] for case_id, row in judged.items()}


def aggregate_runs(
    cases: list[EvalCase], runs: list[tuple[dict[str, dict], dict[str, str]]]
) -> dict[str, Any]:
    """runs: 실행마다 (answers, labels)."""
    scores = [score_answers(cases, answers, labels) for answers, labels in runs]
    metrics = {}
    for name in METRICS:
        values = [score[name]["k"] for score in scores]
        metrics[name] = {"k_per_run": values, "n": scores[0][name]["n"], "mean_k": round(mean(values), 2)}
    # 한 번이라도 "된다"고 답한 사용 불가 문항: 가끔만 위험한 답을 놓치지 않기 위한 최악 기준
    not_allowed = [case for case in cases if case.category == "not_allowed"]
    unsafe_any = [
        case.case_id for case in not_allowed
        if any(labels[case.case_id] in {"ALLOW", "CONDITIONAL"} for _, labels in runs)
    ]
    flipped = [case.case_id for case in cases if len({labels[case.case_id] for _, labels in runs}) > 1]
    return {
        "metrics": metrics,
        "unsafe_allow_any_run": {**rate(len(unsafe_any), len(not_allowed)), "ids": unsafe_any},
        "label_flip": {**rate(len(flipped), len(cases)), "ids": flipped},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 v1 반복 실행 집계")
    parser.add_argument("conditions", nargs="+", help="조건=태그1,태그2,...")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="문항 파일(기본: 시험셋)")
    args = parser.parse_args()
    summary = {}
    for item in args.conditions:
        name, tags = item.split("=", 1)
        tag_list = tags.split(",")
        runs = [load_run(tag) for tag in tag_list]
        summary[name] = {"tags": tag_list, **aggregate_runs(load_eval_cases(args.cases), runs)}
    path = result_path("aggregate.json", "_".join(summary))
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
