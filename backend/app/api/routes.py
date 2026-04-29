from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.core.errors import InvalidRequestError

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise InvalidRequestError("질문을 입력하세요.")

    return ChatResponse(
        answer="아직 RAG 그래프가 연결되지 않았습니다.",
        sources=[],
        status="insufficient_pdf_evidence",
        needs_external_search=True,
    )
