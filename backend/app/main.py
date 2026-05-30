from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.errors import AppError

app = FastAPI(title="Youth Allowance RAG Chatbot API")
app.add_middleware(
      CORSMiddleware,
      allow_origins=[
          "http://localhost:3000",
          "http://127.0.0.1:3000",
      ],
      allow_credentials=False,
      allow_methods=["*"],
      allow_headers=["*"],
)
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
