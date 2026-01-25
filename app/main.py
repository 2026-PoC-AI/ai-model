import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- [추가] NumPy 2.0 호환성 패치 시작 ---
import numpy as np
if int(np.__version__.split('.')[0]) >= 2:
    if not hasattr(np, "float_"): np.float_ = float
    if not hasattr(np, "bool_"): np.bool_ = bool
    if not hasattr(np, "int_"): np.int_ = int
# --- [추가] 패치 끝 ---


import uvicorn
from fastapi import FastAPI

from app.core.logging import setup_logging
from app.core.request_id import RequestIdMiddleware
from app.common.exceptions import register_exception_handlers
from app.api.v1.router import router as v1_router
from app.core.model_registry import init_models, close_models
from app.core.config import settings

def create_app() -> FastAPI:
    setup_logging()
    
    app = FastAPI(
        title="Deepfake Detection API",
        description="Video, Audio, Image, Text 딥페이크 탐지 통합 API",
        version="1.0.0"
    )

    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    app.include_router(v1_router, prefix="/api/v1")

    @app.on_event("startup")
    async def _startup():
        await init_models(app)

    @app.on_event("shutdown")
    async def _shutdown():
        await close_models(app)

    @app.get("/")
    async def root():
        return {
            "message": "Deepfake Detection API",
            "version": "1.0.0",
            "endpoints": {
                "video": "/api/v1/video",
                "audio": "/api/v1/audio",
                "image": "/api/v1/image",
                "text": "/api/v1/text"
            }
        }

    return app

app = create_app()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Deepfake Detection API")
    print("="*50)
    print("Audio: Mel-spectrogram CNN + LFCC CNN")
    print("Expected Accuracy: 99.6-99.8%")
    print("="*50 + "\n")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )


# app/main.py의 startup 이벤트에 추가
@app.on_event("startup")
async def _startup():
    await init_models(app)
    
    # Redis 연결 테스트
    try:
        from app.core.redis_client import redis_client
        redis_client.client.ping()
        print("✓ Redis 연결 성공!")
    except Exception as e:
        print(f"✗ Redis 연결 실패: {e}")