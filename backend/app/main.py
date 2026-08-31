from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import health_router, legacy_router, router
from app.core.errors import AppError

app = FastAPI(
    title="청년수당 안내 챗봇 API",
    description=(
        "서울시 청년수당 안내책자를 기반으로 하이브리드 RAG(FAISS + BM25) + LangGraph 워크플로우로 "
        "질문에 답변하는 챗봇 API입니다. 멀티턴 대화 및 출처 score 반환을 지원합니다."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(router, prefix="/api/v1")
app.include_router(legacy_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.error_code, "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"code": "invalid_request", "message": "요청 형식이 올바르지 않습니다."},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "not_found" if exc.status_code == 404 else "http_error"
    message = str(exc.detail) if isinstance(exc.detail, str) else "HTTP 요청 처리에 실패했습니다."
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": code, "message": message},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": "서버 오류가 발생했습니다."},
    )
