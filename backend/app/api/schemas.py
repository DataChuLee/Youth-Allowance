from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    type: str
    title: str
    page: int
    excerpt: str
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    needs_external_search: bool
