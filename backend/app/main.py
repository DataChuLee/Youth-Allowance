from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.errors import AppError

app = FastAPI(title="Youth Allowance RAG Chatbot API")
app.include_router(router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": exc.message},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
