# RAG TODO

## 결정 사항

- 채팅 그래프는 Chroma 검색 결과를 먼저 가져온 뒤, 최소 similarity 기준을 통과한 PDF 청크만 LLM 답변 생성에 전달한다.
- 인덱스가 없거나 PDF 근거가 약하면 기존 fallback 응답을 유지한다.
- LLM 답변 생성 프롬프트는 PDF 근거만 사용하도록 제한한다.

## 남은 일

- LLM 기반 evidence grader를 추가해 단순 score 기준보다 보수적으로 근거 충분성을 판단한다.
- 실제 안내책자 인덱스와 OpenAI API로 수동 질문 세트를 검증한다.
- similarity threshold와 top-k 값을 수동 검증 결과에 맞춰 조정한다.

## 검증 항목

- `venv\Scripts\python.exe -m pytest backend\tests`
- `venv\Scripts\python.exe -m app.indexing.index_pdf`
- 인덱싱 후 `/chat`에 대표 질문을 보내 `answered_from_pdf`와 출처 렌더링을 확인한다.
