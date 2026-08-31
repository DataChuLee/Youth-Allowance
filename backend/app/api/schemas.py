from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="대화 메시지 역할")
    content: str = Field(min_length=1, max_length=4000, description="대화 메시지 내용")


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="사용자 질문",
        examples=["청년수당 신청 자격이 뭐야?"],
    )
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="브라우저 탭 단위 요청 식별자. 서버는 이 값으로 대화를 저장하지 않습니다.",
        examples=["user-session-abc123"],
    )
    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=4,
        description="프론트엔드가 전달하는 최근 대화 문맥. 최대 4개 메시지이며 서버에 저장되지 않습니다.",
    )


class Source(BaseModel):
    type: str = Field(
        description="출처 유형 (pdf: 안내책자 원문 | graph: 정책 규칙 그래프)",
        examples=["pdf"],
    )
    title: str = Field(
        description="출처 문서 제목",
        examples=["청년수당 참여자 안내책자"],
    )
    page: int = Field(
        description="출처 페이지 번호",
        examples=[7],
    )
    excerpt: str = Field(
        description="답변 근거가 된 원문 발췌 (최대 280자)",
        examples=["청년수당 정책과 관련된 안내 문단 예시입니다."],
    )
    chunk_id: str = Field(
        description="검색된 청크의 고유 식별자",
        examples=["sem_0014"],
    )
    score: float = Field(
        default=0.0,
        description="검색 유사도 점수 (0.0~1.0, 높을수록 질문과 관련성이 높음)",
        examples=[0.92],
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="안내책자 근거 기반 AI 생성 답변")
    sources: list[Source] = Field(description="답변 생성에 사용된 출처 목록 (유사도 순 정렬)")
    status: Literal[
        "general_answer",
        "answered_from_pdf",
        "insufficient_pdf_evidence",
        "blocked_by_policy",
    ] = Field(
        description=(
            "응답 상태: "
            "general_answer=인사·챗봇 소개 등 일반 응답, "
            "answered_from_pdf=안내책자 근거로 답변, "
            "insufficient_pdf_evidence=근거 불충분(외부 확인 권장), "
            "blocked_by_policy=사용 불가 항목"
        )
    )
    needs_external_search: bool = Field(
        description="True이면 안내책자 외 공식 채널(콜센터 등) 추가 확인 권장"
    )
    intent: Literal["general_answer", "rag"] = Field(
        default="rag",
        description="질문 의도 분류: general_answer=일반 대화, rag=청년수당 정책 질문",
    )


class ErrorResponse(BaseModel):
    code: str = Field(description="클라이언트가 분기 처리할 안정적인 오류 코드")
    message: str = Field(description="사용자에게 표시할 수 있는 오류 설명")


# ── 헬스체크 ─────────────────────────────────────────────────────────────────

class HealthDependencies(BaseModel):
    faiss_index: Literal["ok", "missing"] = Field(
        description="FAISS 벡터 인덱스 상태 (missing이면 로컬 PDF 기반 BM25 fallback 시도)"
    )
    redis_cache: Literal["ok", "unavailable"] = Field(
        description="Redis 쿼리 캐시 상태 (unavailable이어도 서비스 동작, 응답 속도 저하)"
    )


class HealthConfig(BaseModel):
    chat_model: str = Field(description="답변 생성에 사용 중인 LLM 모델명")
    embedding_model: str = Field(description="문서 임베딩 모델명")
    retrieval_top_k: int = Field(description="질문당 검색할 최대 청크 수")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(
        description="전체 서비스 상태: ok=정상, degraded=일부 의존성 비정상(서비스는 동작)"
    )
    dependencies: HealthDependencies
    config: HealthConfig
