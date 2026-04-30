# Youth Allowance RAG Backend

청년수당 참여자 안내책자 PDF를 인덱싱하고, FastAPI로 채팅 API를 제공하는 백엔드입니다.

## Development

```powershell
python -m pytest -v
python -m app.indexing.index_pdf
python -m uvicorn app.main:app --reload --port 8000
```

루트 `.env`의 `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL` 값이 필요합니다.
