# RAG TODO

## 결정 사항

- 채팅 그래프는 FAISS + BM25(RRF) 검색 결과를 먼저 가져온 뒤, 최소 similarity 기준을 통과한 PDF 청크만 LLM 답변 생성에 전달한다.
- 인덱스가 없거나 PDF 근거가 약하면 기존 fallback 응답을 유지한다.
- LLM 답변 생성 프롬프트는 PDF 근거만 사용하도록 제한한다.
- 키워드 사전 차단은 `blocker_check.py`의 LLM 확인기를 거친다. 실제 지출이 제한 항목이 아니라고 명시한 후보만 풀고,
  호출 실패·출력 오류면 차단을 유지한다. 기준선에서 정상 질문 17개 중 10개가 키워드만으로 차단됐다(`benchmarks/BASELINE_V1.md`).
- 키워드 차단은 사전 점검에서만 판단하고, 최종 정책 판단은 근거 평가 결과만 쓴다.

## 남은 일

- LLM 기반 evidence grader를 추가해 단순 score 기준보다 보수적으로 근거 충분성을 판단한다.
- 실제 안내책자 인덱스와 OpenAI API로 수동 질문 세트를 검증한다.
- similarity threshold와 top-k 값을 수동 검증 결과에 맞춰 조정한다.

## 검증 항목

- `venv\Scripts\python.exe -m pytest backend\tests`
- `venv\Scripts\python.exe -m app.indexing.index_pdf`
- 인덱싱 후 `/chat`에 대표 질문을 보내 `answered_from_pdf`와 출처 렌더링을 확인한다.
