import json
from collections.abc import Iterator

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthConfig,
    HealthDependencies,
    HealthResponse,
)
from app.core.cache import get_query_cache
from app.core.config import get_settings
from app.core.errors import InvalidRequestError
from app.graph.workflow import (
    PreparedChatStream,
    prepare_chat_stream,
    run_chat_graph,
)

router = APIRouter()
health_router = APIRouter()
legacy_router = APIRouter()


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="서비스 헬스체크",
    description="FAISS 인덱스 존재 여부와 Redis 캐시 연결 상태를 포함한 의존성 상태를 반환합니다.",
)
def health() -> HealthResponse:
    settings = get_settings()

    faiss_ok = (settings.faiss_index_dir / "index.faiss").exists()
    redis_ok = get_query_cache().enabled

    faiss_status = "ok" if faiss_ok else "missing"
    redis_status = "ok" if redis_ok else "unavailable"
    overall = "ok" if faiss_ok and redis_ok else "degraded"

    return HealthResponse(
        status=overall,
        dependencies=HealthDependencies(
            faiss_index=faiss_status,
            redis_cache=redis_status,
        ),
        config=HealthConfig(
            chat_model=settings.openai_chat_model,
            embedding_model=settings.openai_embedding_model,
            retrieval_top_k=settings.retrieval_top_k,
        ),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}},
    summary="단발 질의응답",
    description=(
        "청년수당 관련 질문에 대해 안내책자 근거 기반 답변을 반환합니다. "
        "`thread_id`는 탭 식별용이며, 멀티턴 문맥은 `history`로 전달합니다. 서버는 대화를 저장하지 않습니다."
    ),
)
def chat(request: ChatRequest) -> ChatResponse:
    return execute_chat(request)


@legacy_router.post(
    "/chat",
    response_model=ChatResponse,
    deprecated=True,
    responses={400: {"model": ErrorResponse}},
    summary="단발 질의응답 (deprecated)",
    description="호환성을 위한 임시 경로입니다. 새 클라이언트는 POST /api/v1/chat을 사용하세요.",
)
def legacy_chat(request: ChatRequest, response: Response) -> ChatResponse:
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</api/v1/chat>; rel="successor-version"'
    return execute_chat(request)


def execute_chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise InvalidRequestError("질문을 입력하세요.")
    return run_chat_graph(
        question,
        thread_id=request.thread_id,
        history=[message.model_dump() for message in request.history],
    )


@router.post(
    "/chat/stream",
    summary="스트리밍 질의응답 (SSE)",
    responses={400: {"model": ErrorResponse}},
    description=(
        "답변을 Server-Sent Events 형식으로 토큰 단위 스트리밍합니다.\n\n"
        "이벤트 순서: `token` (반복) → `done` (최종 응답 + 출처)\n\n"
        "오류 발생 시 `error` 이벤트가 전송됩니다."
    ),
)
def chat_stream(request: ChatRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise InvalidRequestError("질문을 입력하세요.")
    prepared = prepare_chat_stream(
        question,
        thread_id=request.thread_id,
        history=[message.model_dump() for message in request.history],
    )
    return StreamingResponse(
        stream_chat_events(prepared),
        media_type="text/event-stream",
    )


def stream_chat_events(prepared: PreparedChatStream) -> Iterator[str]:
    answer_parts: list[str] = []

    try:
        for token in prepared.chunks:
            if not token:
                continue
            answer_parts.append(token)
            yield format_stream_event("token", {"text": token})
    except Exception:
        yield format_stream_event(
            "error",
            {"code": "generation_error", "message": "답변 생성 중 오류가 발생했습니다."},
        )
        return

    full_answer = "".join(answer_parts)

    response = ChatResponse(
        answer=full_answer,
        sources=prepared.sources,
        status=prepared.status,
        needs_external_search=prepared.needs_external_search,
        intent=prepared.intent,
    )
    yield format_stream_event("done", response.model_dump())


def format_stream_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
