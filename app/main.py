# app/main.py
import uvicorn
from fastapi import FastAPI

from app.core.logging import setup_logging
from app.core.request_id import RequestIdMiddleware
from app.common.exceptions import register_exception_handlers
from app.api.v1.router import router as v1_router
from app.core.model_registry import init_models, close_models
from app.core.config import settings  # <--- 이 줄이 꼭 필요합니다!

def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="AI Inference Server")

    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    app.include_router(v1_router, prefix="/api/v1")

    @app.on_event("startup")
    async def _startup():
        await init_models(app)

    @app.on_event("shutdown")
    async def _shutdown():
        await close_models(app)

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )