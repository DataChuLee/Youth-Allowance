"""문서 기반 답변 품질 평가용 골드셋과 검토 큐를 생성한다.

60개 answerable 문항은 기존 testset의 참조 답변과 수동 확인한 PDF chunk를
사용한다. conditional 40개는 실제 실행 결과의 출처를 후보로만 제공하며,
사람이 참조 답변과 gold chunk를 확정하기 전에는 점수 산출에 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCHMARK_DIR.parent

# 안내책자 원문을 읽어 각 질문에 필요한 최소 근거 chunk를 수동으로 확인했다.
# 변형 질문은 동일한 의미의 원문 질문과 같은 근거를 공유한다.
ANSWERABLE_GOLD_CHUNKS: dict[int, tuple[str, ...]] = {
    1: ("sem_0014",),
    2: ("sem_0014",),
    3: ("sem_0014",),
    4: ("sem_0015",),
    5: ("sem_0014",),
    6: ("sem_0015",),
    7: ("sem_0016",),
    8: ("sem_0085",),
    9: ("sem_0014",),
    10: ("sem_0014",),
    11: ("sem_0020",),
    12: ("sem_0026", "sem_0097"),
    13: ("sem_0089",),
    14: ("sem_0032",),
    15: ("sem_0032",),
    16: ("sem_0032", "sem_0090"),
    17: ("sem_0087",),
    18: ("sem_0049",),
    19: ("sem_0034",),
    20: ("sem_0033", "sem_0035"),
    21: ("sem_0048",),
    22: ("sem_0031",),
    23: ("sem_0032",),
    24: ("sem_0029",),
    25: ("sem_0052", "sem_0065"),
    26: ("sem_0052", "sem_0093"),
    27: ("sem_0022", "sem_0096"),
    28: ("sem_0035", "sem_0063"),
    29: ("sem_0014", "sem_0016"),
    30: ("sem_0016", "sem_0098"),
}

# 이 세 질문은 seed testset에 참조 문장이 있더라도 현재 서비스 계약상
# `blocked_by_policy`로 처리하는 항목이다. PDF 답변 품질 평균에는 섞지 않고,
# 정책 차단 품질 평가로 분리한다.
POLICY_ONLY_SEED_IDS = {14, 15, 19}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_seed_rows(testset_path: Path) -> list[dict[str, str]]:
    with testset_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 30:
        raise ValueError(f"참조 답변이 있는 seed 문항은 30개여야 합니다. 현재: {len(rows)}")
    return rows


def build_content_cases(
    policy_cases: list[dict[str, Any]],
    seed_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    policy_by_id = {str(case["id"]): case for case in policy_cases}
    cases: list[dict[str, Any]] = []
    for index, seed in enumerate(seed_rows, start=1):
        if index in POLICY_ONLY_SEED_IDS:
            continue
        for case_id in (f"answerable-{index:03d}", f"answerable-{index + 30:03d}"):
            policy_case = policy_by_id.get(case_id)
            if policy_case is None:
                raise ValueError(f"정책 평가셋에 {case_id} 문항이 없습니다.")
            cases.append(
                {
                    "id": case_id,
                    "question": policy_case["question"],
                    "reference_answer": seed["ground_truth"].strip(),
                    "gold_chunk_ids": list(ANSWERABLE_GOLD_CHUNKS[index]),
                    # 현 단계에서는 참조 답변 전체를 최소 충족 사실으로 취급한다.
                    # 세부 사실 분해는 conditional 검토와 함께 다음 버전에서 진행한다.
                    "critical_facts": [seed["ground_truth"].strip()],
                    "review_status": "seed_reference_and_manual_source_review",
                }
            )
    return cases


def build_conditional_annotation_queue(
    policy_cases: list[dict[str, Any]],
    policy_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs_by_id = {str(run["id"]): run for run in policy_runs}
    queue: list[dict[str, Any]] = []
    for case in policy_cases:
        if case.get("scenario") != "conditional":
            continue
        run = runs_by_id.get(str(case["id"]), {})
        queue.append(
            {
                "id": case["id"],
                "question": case["question"],
                "candidate_chunk_ids": run.get("source_chunk_ids", []),
                "reference_answer": "",
                "gold_chunk_ids": [],
                "critical_facts": [],
                "review_status": "needs_human_adjudication",
            }
        )
    return queue


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="문서 기반 답변 품질 평가셋 생성")
    parser.add_argument(
        "--policy-cases",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "policy_safety_v2_150.jsonl",
    )
    parser.add_argument(
        "--policy-runs",
        type=Path,
        default=BENCHMARK_DIR / "results" / "policy_safety_v2_150.jsonl",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=BACKEND_DIR / "evaluation_results" / "testset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "content_quality_v1.jsonl",
    )
    parser.add_argument(
        "--conditional-queue-output",
        type=Path,
        default=BENCHMARK_DIR / "datasets" / "conditional_annotation_queue_v1.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_cases = read_jsonl(args.policy_cases)
    content_cases = build_content_cases(policy_cases, read_seed_rows(args.testset))
    write_jsonl(args.output, content_cases)

    policy_runs = read_jsonl(args.policy_runs)
    conditional_queue = build_conditional_annotation_queue(policy_cases, policy_runs)
    write_jsonl(args.conditional_queue_output, conditional_queue)
    print(f"문서 기반 평가 문항 {len(content_cases)}개 생성: {args.output}")
    print(f"조건부 문항 검토 큐 {len(conditional_queue)}개 생성: {args.conditional_queue_output}")


if __name__ == "__main__":
    main()
