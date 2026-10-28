# RAG·Agent 벤치마크

이 폴더는 포트폴리오용 수치를 만들기 위한 도구가 아니라, 청년수당 RAG의 검색·응답 안전성·워크플로 제어 품질을 같은 기준으로 반복 측정하기 위한 평가 패키지입니다.

## 구성

- `datasets/retrieval_challenge_v1.jsonl`: 정답 `chunk_id`가 있는 검색 평가 시드 10문항
- `datasets/policy_safety_v1.jsonl`: 답변 가능·조건부·정책 제한·근거 부족을 포함한 응답 제어 시드 13문항
- `datasets/policy_safety_v2_150.jsonl`: v1을 확장한 150문항 정책·안전성 회귀 평가셋
- `run_retrieval.py`: 현재 배포 경로인 adaptive hybrid 검색 결과를 저장
- `run_policy.py`: 실제 LangGraph 응답 상태와 출처를 저장
- `score.py`: 실행 결과를 정량 지표 JSON으로 집계

시드 데이터는 기존 25문항 challenge set에서 출처가 검증된 문항을 옮긴 시작점입니다. 포트폴리오의 대표 수치로 사용하기 전에 `TODO.md`의 기준에 따라 최소 100/150문항까지 늘려야 합니다.

## 실행

모든 명령은 `backend/` 디렉터리에서 실행합니다.

```powershell
python -m pytest benchmarks/tests -q
python -m benchmarks.build_policy_safety_v2
python -m benchmarks.run_retrieval
python -m benchmarks.run_policy
```

마지막 두 명령은 기본적으로 사례 수와 결과 경로만 보여 주며, 외부 API를 호출하지 않습니다.

실제 인덱스·OpenAI 설정이 준비된 뒤에만 다음 명령으로 실행합니다.

```powershell
python -m benchmarks.run_retrieval --execute
python -m benchmarks.score retrieval --runs benchmarks/results/retrieval_hybrid_adaptive.jsonl

python -m benchmarks.run_policy --execute
python -m benchmarks.score policy --runs benchmarks/results/policy_safety_v1.jsonl
```

## 읽는 법

검색 평가는 `MRR`, `Recall@5`, `NDCG@5`를 제공합니다. MRR은 정답 근거가 얼마나 상위에 왔는지, Recall@5는 상위 5개 안에 정답 근거가 있었는지, NDCG@5는 여러 정답 근거의 순위 품질을 의미합니다.

정책·안전성 평가는 다음을 분리합니다.

- `status_accuracy`: 기대한 응답 상태로 분기한 비율
- `source_hit_rate`: 지정한 핵심 근거 chunk를 출처로 반환한 비율
- `unsafe_route_rate`: 정책 제한 또는 근거 부족 질문이 `answered_from_pdf` 경로로 잘못 분기한 비율

`required_term_rate`는 문자열 기반 보조 점검입니다. 답변의 실제 사실성 평가는 사람이 별도 리뷰하거나 LLM-as-a-judge를 도입한 뒤에 추가합니다.

## 보고 원칙

결과에는 항상 평가셋 버전·문항 수·실행 모델·인덱스 버전을 함께 기록합니다. 13개 시드 문항의 실행 결과를 성능 성과로 홍보하지 않으며, 수치가 나쁘더라도 삭제하지 않고 실패 사례와 개선안을 함께 남깁니다.
