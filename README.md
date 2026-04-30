# 청년수당 RAG 챗봇

청년수당 참여자가 안내책자를 직접 뒤지지 않고도 자주 묻는 내용을 확인할 수 있도록 만든 PDF 기반 RAG 웹앱입니다.

## 구성

- `backend`: FastAPI, LangChain, LangGraph, Chroma 기반 RAG API
- `frontend`: Next.js 기반 참여자용 채팅 UI
- `docs`: PRD, 아키텍처, 구현 계획 문서
- `Data/청년수당 참여자 안내책자.pdf`: 인덱싱 대상 원본 PDF

## 환경 변수

프로젝트 루트의 `.env`에 아래 값을 설정합니다.

```env
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=youth-allowance-rag
PDF_PATH=Data/청년수당 참여자 안내책자.pdf
CHROMA_DIR=backend/storage/chroma
RETRIEVAL_TOP_K=5
MIN_SIMILARITY_SCORE=0.2
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## 백엔드

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m app.indexing.index_pdf
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

## 프론트엔드

```powershell
cd frontend
npm install
npm test
npm run build
npm run dev
```

## 현재 MVP 범위

- PDF 텍스트 추출 품질 검사
- 페이지/청크 단위 출처 메타데이터 보존
- Chroma 벡터 인덱스 생성
- `/chat` API 계약 및 에러 응답
- 안내책자 근거 부족 시 `needs_external_search=true` 신호 반환
- 참여자용 채팅 UI와 출처 표시

외부 검색 보완은 MVP 이후 단계로 분리되어 있으며, 현재는 API 응답에서 보완 필요 여부만 명시합니다.
