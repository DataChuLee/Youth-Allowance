from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.core.errors import InvalidRequestError
from app.graph.workflow import run_chat_graph

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise InvalidRequestError("질문을 입력하세요.")

    return run_chat_graph(question)
