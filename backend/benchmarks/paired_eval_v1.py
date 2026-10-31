"""조건 간 문항별 쌍대 비교. 3회 다수결로 문항별 정답 여부를 정하고 정확 McNemar 검정을 한다.

예: python -m benchmarks.paired_eval_v1 baseline bm25_bigram restriction_query
(각 태그의 _r2, _r3 결과가 있어야 한다)
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

from benchmarks.aggregate_eval_v1 import load_run
from benchmarks.eval_v1 import ACCEPTED_LABELS, load_eval_cases
from benchmarks.run_eval_v1 import CASES_PATH


def majority_correct(cases, tag: str) -> dict[str, bool]:
    runs = [load_run(name)[1] for name in (tag, f"{tag}_r2", f"{tag}_r3")]
    return {
        case.case_id: sum(labels[case.case_id] in ACCEPTED_LABELS[case.expected_verdict] for labels in runs) >= 2
        for case in cases
    }


def exact_mcnemar(fixed: int, broke: int) -> float:
    """바뀐 문항만 보는 양측 정확 검정(이항 분포, p=0.5)."""
    n = fixed + broke
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, broke) + 1)) / 2**n)


def compare(before: dict[str, bool], after: dict[str, bool]) -> tuple[int, int, float]:
    fixed = sum(1 for key in before if not before[key] and after[key])
    broke = sum(1 for key in before if before[key] and not after[key])
    return fixed, broke, exact_mcnemar(fixed, broke)


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 v1 조건 간 쌍대 비교")
    parser.add_argument("tags", nargs="+", help="비교할 조건 태그(순서대로 인접 비교 + 처음↔끝)")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()
    cases = [case for case in load_eval_cases(args.cases) if case.expected_verdict in ACCEPTED_LABELS]
    correct = {tag: majority_correct(cases, tag) for tag in args.tags}
    print(f"n={len(cases)}")
    for tag in args.tags:
        print(f"{tag}: 다수결 정답 {sum(correct[tag].values())}")
    pairs = list(zip(args.tags, args.tags[1:]))
    if len(args.tags) > 2:
        pairs.append((args.tags[0], args.tags[-1]))
    for before, after in pairs:
        fixed, broke, p = compare(correct[before], correct[after])
        print(f"{before} -> {after}: 고침 {fixed}, 망가짐 {broke}, 정확 McNemar p={p:.4f}")


if __name__ == "__main__":
    main()
