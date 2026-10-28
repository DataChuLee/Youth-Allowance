"""정책 답변을 결론과 원자적 필수 사실 단위로 평가할 골드셋을 생성한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.build_content_quality_v1 import (
    ANSWERABLE_GOLD_CHUNKS,
    POLICY_ONLY_SEED_IDS,
    read_jsonl,
    write_jsonl,
)

BENCHMARK_DIR = Path(__file__).resolve().parent

# 각 사양은 안내책자 원문과 seed 참조답안을 함께 검토해 수동 분해했다.
# ``resolution``은 단일 가능/불가가 아닌 정책적 결론의 형태를 나타낸다.
SEED_ALIGNMENT_SPECS: dict[int, dict[str, Any]] = {
    1: {"resolution": "informational", "facts": ["신청 연령은 만 19세 이상 34세 이하이다.", "제대군인은 군 복무기간에 따라 최대 만 37세까지 신청할 수 있다."]},
    2: {"resolution": "informational", "facts": ["신청일 기준 주민등록상 서울시 거주자여야 한다."]},
    3: {"resolution": "informational", "facts": ["2026년 2월 기준 중위소득 50% 초과 150% 이하가 기준이다.", "건강보험료 월 부과액을 기준으로 판단한다."]},
    4: {"resolution": "conditional", "facts": ["일반 재학생과 휴학생은 신청할 수 없다.", "방송통신대학·사이버대학·학점은행제 학생은 신청할 수 있다."]},
    5: {"resolution": "conditional", "facts": ["원칙적으로 미취업자여야 한다.", "주 30시간 이하 또는 3개월 이하 단기근로자는 근로계약서 제출 시 신청할 수 있다."]},
    6: {"resolution": "blocked", "facts": ["기초생활수급자와 차상위계층은 신청 제외 대상이다.", "중위소득 50% 이하가 해당 기준이다."]},
    7: {"resolution": "blocked", "facts": ["실업급여 수급 중인 사람은 신청할 수 없다.", "신청 시점부터 사업 참여 종료일까지 제한된다."]},
    8: {"resolution": "blocked", "facts": ["청년수당은 생애 1회만 지원된다.", "2017년부터 2025년 사업 참여자는 재신청할 수 없다."]},
    9: {"resolution": "informational", "facts": ["월 지원 금액은 50만원이다."]},
    10: {"resolution": "informational", "facts": ["최대 지원 기간은 6개월이다."]},
    11: {"resolution": "informational", "facts": ["2026년 사업 참여 기간은 4월 29일부터 10월 28일까지이다."]},
    12: {"resolution": "conditional", "facts": ["취업성공금은 남은 청년수당 지급분의 50%이다.", "마지막 월에 취업하면 잔여 1회분 50만원 전액을 받는다."]},
    13: {"resolution": "conditional", "facts": ["청년수당은 당월 사용이 원칙이다.", "개인 활동 계획에 따라 이월할 수 있다.", "이월 시 자기성장기록서 이월사유란에 사유를 작성해야 한다."]},
    16: {"resolution": "blocked", "facts": ["노래방에서는 청년수당 카드 결제가 불가하다.", "노래방은 클린카드 사용 불가 업종에 포함된다."]},
    17: {"resolution": "conditional", "facts": ["학습·구직활동 등 사업 취지에 부합하면 태블릿 PC나 노트북을 구입할 수 있다.", "사회 정서에 반하는 고가 사치품은 제한된다."]},
    18: {"resolution": "conditional", "facts": ["청년수당 카드로 해외 생성형 AI 프로그램을 직접 결제할 수 없다.", "개인 카드로 결제한 뒤 실제 지출액을 청년수당 계좌에서 보전할 수 있다.", "해당 예외에는 거래명세서 또는 영수증과 이체확인증 등 최소 2종의 증빙이 필요하다."]},
    20: {"resolution": "conditional", "facts": ["부득이한 경우 월세를 계좌이체로 납부할 수 있다.", "임대차계약서와 이체확인증 등 최소 2종의 증빙을 제출해야 한다."]},
    21: {"resolution": "blocked", "facts": ["청년수당을 본인의 다른 개인 계좌로 이체해 사용하는 것은 금지된다."]},
    22: {"resolution": "allowed", "facts": ["학원비는 교육비 또는 구직활동비로 청년수당 사용이 가능하다."]},
    23: {"resolution": "allowed", "facts": ["국민건강보험료는 생활비 항목으로 청년수당으로 납부할 수 있다."]},
    24: {"resolution": "conditional", "facts": ["전용 계좌는 신한은행 신한 청년 DREAM 통장(서울시) 상품이다.", "신한 슈퍼SOL 모바일 앱으로만 개설할 수 있다.", "영업점 방문 개설은 불가하다."]},
    25: {"resolution": "informational", "facts": ["자기성장기록서는 매월 11일부터 다음 달 10일까지 작성·제출할 수 있다.", "마감일이 공휴일이면 하루 연장된다."]},
    26: {"resolution": "blocked", "facts": ["기한 내 자기성장기록서를 제출하지 않으면 해당 월부터 지급이 중단된다.", "제출 기한의 예외나 연장은 없다."]},
    27: {"resolution": "informational", "facts": ["청년몽땅정보통에 로그인해야 한다.", "마이페이지에서 청년수당 메뉴의 자격상실신고 제출로 온라인 신고한다."]},
    28: {"resolution": "conditional", "facts": ["계좌이체 월세는 임대차계약서와 이체확인증을 포함해 최소 2종을 제출해야 한다.", "계약자가 가족 또는 동거인이면 주민등록등본도 제출해야 한다."]},
    29: {"resolution": "blocked", "facts": ["군복무자와 사회복무요원은 청년수당 제외 대상이다."]},
    30: {"resolution": "informational", "facts": ["청년수당 전담 콜센터 번호는 1566-3344이다."]},
}


def build_alignment_cases(policy_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(case["id"]): case for case in policy_cases}
    rows: list[dict[str, Any]] = []
    for seed_id, spec in SEED_ALIGNMENT_SPECS.items():
        if seed_id in POLICY_ONLY_SEED_IDS:
            continue
        for case_id in (f"answerable-{seed_id:03d}", f"answerable-{seed_id + 30:03d}"):
            case = by_id[case_id]
            rows.append({
                "id": case_id,
                "question": case["question"],
                "expected_resolution": spec["resolution"],
                "required_facts": spec["facts"],
                "gold_chunk_ids": list(ANSWERABLE_GOLD_CHUNKS[seed_id]),
                "review_status": "manual_policy_resolution_and_atomic_facts_v1",
            })
    return rows


def main() -> None:
    cases = read_jsonl(BENCHMARK_DIR / "datasets" / "policy_safety_v2_150.jsonl")
    output = BENCHMARK_DIR / "datasets" / "policy_alignment_v1.jsonl"
    rows = build_alignment_cases(cases)
    write_jsonl(output, rows)
    print(f"정책 정합성 평가 문항 {len(rows)}개 생성: {output}")


if __name__ == "__main__":
    main()
