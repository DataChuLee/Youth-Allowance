# Graph RAG TODO

## 결정 사항

- Neo4j 지식 그래프는 RDF가 아니라 LPG 모델로 설계한다.
- 기존 Chroma/BM25 Vector RAG는 유지하고, Neo4j는 정책 판단 레이어로 추가한다.
- LLM 자동 추출 결과를 바로 적재하지 않고, 수동 큐레이션한 `graph_items.yaml`을 중간 산출물로 둔다.

## 남은 일

- OCR 원문을 읽어 `graph_items.yaml`에 승인된 정책 항목을 채운다.
- Neo4j 접속 환경변수(`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`)를 `.env`에 정리한다.
- Graph retriever 결과를 기존 LangGraph workflow의 evidence 단계와 병합한다.

## 검증 항목

- `venv\Scripts\python.exe -m pytest backend\tests\test_graph_rag_pipeline.py`
- `backend/GraphRAG.ipynb`에서 OCR 로드, block 생성, artifact 저장 셀이 순서대로 실행되는지 확인한다.
- Neo4j 적재 후 label별 node count와 relationship type별 count를 확인한다.
