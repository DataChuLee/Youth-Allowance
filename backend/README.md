# Youth Allowance RAG Backend

청년수당 참여자 안내책자 OCR 결과를 FAISS로 인덱싱하고, FastAPI로 채팅 API를 제공하는 백엔드입니다.

## Development

```powershell
python -m pytest -v
python -m app.indexing.index_pdf
python -m uvicorn app.main:app --reload --port 8000
```

루트 `.env`의 `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL` 값이 필요합니다.
인덱싱 입력은 `Data/upstage_youth_allowance_parse.jsonl`(`OCR_JSONL_PATH`)이고, 결과는 `backend/faiss_rag_final/`(`FAISS_INDEX_DIR`)에 저장됩니다.
