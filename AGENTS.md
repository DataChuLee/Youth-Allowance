# Repository Guidelines

## 프로젝트 구조 및 모듈 구성

이 저장소는 현재 로컬 Youth Allowance RAG 챗봇 MVP를 위한 기획 문서를 중심으로 구성되어 있습니다.

- `README.md`는 프로젝트 진입 문서입니다.
- `docs/PRD.md`는 제품 범위, API 동작, 성공 기준, 수동 검증 질문을 정의합니다.
- `docs/architecture.md`는 예정된 FastAPI 백엔드, Next.js 프론트엔드, LangChain/LangGraph 워크플로, Chroma 저장소, 계약을 설명합니다.
- `Data/`에는 `Data/청년수당 참여자 안내책자.pdf` 같은 로컬 원본 PDF가 들어갑니다. Git에서 무시되며 커밋하면 안 됩니다.
- `.env`와 `venv/`는 로컬 전용이며 Git에서 무시됩니다.

구현을 시작하면 문서화된 구조를 따르세요.

```text
backend/app/{api,graph,indexing,rag,core}/
backend/storage/chroma/
frontend/
docs/
```

## 빌드, 테스트, 개발 명령

아직 실행 가능한 앱 코드, 패키지 매니페스트, 테스트 설정은 체크인되어 있지 않습니다. 앱 코드가 추가되면 이 섹션을 실제 명령으로 교체하고 `README.md`에도 동일하게 반영하세요.

아키텍처 기준 예상 명령은 다음과 같습니다.

- `python -m venv venv`: 로컬 Python 환경을 생성합니다.
- `pip install -r requirements.txt`: `requirements.txt`가 생기면 백엔드 의존성을 설치합니다.
- `pytest`: 테스트가 추가되면 백엔드 테스트를 실행합니다.
- `npm install`, `npm run dev`: `frontend/package.json`이 생기면 프론트엔드 의존성을 설치하고 개발 서버를 실행합니다.

## 코딩 스타일 및 명명 규칙

`api`, `graph`, `indexing`, `rag`, `core`처럼 아키텍처 경계에 맞춘 작은 모듈을 사용하세요. 문서화된 LangGraph 워크플로와 맞도록 `retrieve_pdf`, `assess_evidence`, `generate_answer` 같은 이름을 선호합니다. API 스키마는 `question`, `answer`, `sources`, `status`, `needs_external_search`를 중심으로 안정적으로 유지하세요.

Python은 4칸 들여쓰기, 공개 함수 타입 힌트, 요청/응답 계약을 위한 Pydantic 모델을 사용합니다. TypeScript/React는 `ChatPage`, `SourceList` 같은 PascalCase 컴포넌트명과 camelCase 변수를 사용합니다.

코드 작성 시 복잡한 로직, 비즈니스 규칙, 예외 처리에는 이해하기 쉬운 한글 주석을 작성하세요. 특히 RAG 판단 기준, 프롬프트 제약, PDF 근거 부족 처리처럼 프로젝트 도메인 맥락이 필요한 부분은 의도를 남기세요. 단순한 대입이나 함수명만으로 의미가 명확한 코드는 주석 처리하지 마세요.

데이터 변환, 검증, RAG 처리 로직은 함수형 패턴을 우선합니다. 순수 함수, 명시적 입력/출력, 작은 조합 가능한 함수로 나누고, 공유 상태 변경은 필요한 경우로 제한하세요. FastAPI 라우터, LangGraph 상태 흐름, React 상태 관리처럼 부수효과가 자연스러운 영역은 해당 프레임워크 관례를 따릅니다.

새 기능 단위가 별도 모듈로 추가될 때는 기능별 폴더를 만들고 해당 폴더에 `TODO.md`를 생성하세요. 작은 수정이나 기존 기능 변경에는 기존 문서를 업데이트하는 것으로 충분합니다. `TODO.md`에는 결정 사항, 남은 일, 검증 항목을 계속 기록해 이후 기여자가 맥락을 빠르게 파악할 수 있게 하세요.

## 테스트 지침

테스트는 구현 옆이나 최상위 `tests/` 디렉터리에 추가하세요. 백엔드 테스트는 PDF 인덱싱 실패, `/health`, `/chat`, `index_missing`, `answered_from_pdf`, `insufficient_pdf_evidence`를 다뤄야 합니다. 프론트엔드는 빠른 질문 버튼, 로딩 상태, 출처 렌더링, 백엔드 오류 상태를 확인하세요.

테스트 이름은 동작을 설명하도록 작성합니다. 예: `test_chat_returns_insufficient_evidence_when_no_sources`.

## 커밋 및 풀 리퀘스트 지침

커밋은 추가한 파일을 기준으로 가능한 한 분리하세요. 새 파일이 여러 개 추가되면 각 파일 또는 밀접하게 연결된 파일 묶음별로 별도 커밋을 만듭니다. 커밋 메시지는 파일명만 쓰지 말고 “무엇을 추가했는지”가 드러나는 짧은 명령형 문장을 사용하세요. 예: `Add PRD for RAG chatbot scope`, `Add chat API endpoint`, `Add PDF indexing workflow`.

기본 흐름은 `git add` → `git commit` → `git push` → PR 생성 → 리뷰/수정 → merge입니다. PR은 GitHub에 브랜치를 push한 뒤 생성합니다.

가능하면 에이전트를 활용해 다음 작업을 자동화하세요.

- 변경 파일을 확인하고 파일/기능 단위로 분류합니다.
- 각 단위별로 `git add`와 `git commit`을 실행합니다.
- 현재 작업 브랜치를 원격 저장소에 push합니다.
- GitHub CLI(`gh`)가 설정되어 있으면 PR을 생성하고 제목/본문을 작성합니다.
- PR 생성 전후로 `/review` 또는 코드 리뷰 요청을 실행해 버그, 리스크, 누락 테스트를 점검합니다.
- 리뷰 결과나 PR 코멘트를 반영하고 추가 커밋 후 다시 push합니다.

풀 리퀘스트에는 요약, 변경 영역, 수행한 검증, UI 변경 시 스크린샷을 포함하세요. 관련 이슈가 있으면 연결하세요. merge는 가능하더라도 테스트와 리뷰 상태를 확인한 뒤 사용자 확인 후 진행하세요. `.env`, `venv/`, Chroma 저장소, `Data/` 파일은 절대 포함하지 마세요.

## 보안 및 설정

비밀 값은 `.env`에만 보관하세요. OpenAI 또는 LangSmith 키를 로그에 남기면 안 됩니다. 사용자 질문과 LangSmith 추적 정보는 민감할 수 있으므로, MVP에서는 채팅 기록을 서버에 저장하지 마세요.
