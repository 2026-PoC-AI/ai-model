# app/main.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# NumPy 2.0 호환성 패치
import numpy as np
if int(np.__version__.split('.')[0]) >= 2:
    if not hasattr(np, "float_"): np.float_ = float
    if not hasattr(np, "bool_"): np.bool_ = bool
    if not hasattr(np, "int_"): np.int_ = int

import uvicorn
from fastapi import FastAPI

from app.core.logging import setup_logging
from app.core.request_id import RequestIdMiddleware
from app.common.exceptions import register_exception_handlers
from app.api.v1.router import router as v1_router
from app.core.model_registry import init_models, close_models
from app.domains.text.fake_news.predictor import KlueBertFakeNewsPredictor
from app.domains.text.service import TextService
from app.core.config import settings


def create_app() -> FastAPI:
    setup_logging()
    
    app = FastAPI(
        title="Deepfake Detection API",
        description="Video, Audio, Image, Text 딥페이크 탐지 통합 API",
        version="2.0.0"  # 3-model ensemble 반영
    )

    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    app.include_router(v1_router, prefix="/api/v1")

    @app.on_event("startup")
    async def _startup():
        # 모델 로드
        await init_models(app)

        # Text 모델 추가 설정
        fake_news_predictor = KlueBertFakeNewsPredictor(
            state_dict_path="app/domains/text/fake_news/weights/model_state_dict.pth", 
            num_labels=2,
            id2label={0: "FAKE", 1: "REAL"},
            max_length=256,
        )

        text_service = TextService(fake_news_predictor=fake_news_predictor)
        app.state.text_service = text_service

        print("✓ Text FakeNews model loaded (KLUE-BERT)")
        
        # Redis 연결 테스트
        try:
            from app.core.redis_client import redis_client
            redis_client.client.ping()
            print("✓ Redis 연결 성공!")
        except Exception as e:
            print(f"✗ Redis 연결 실패: {e}")

    @app.on_event("shutdown")
    async def _shutdown():
        await close_models(app)

    @app.get("/")
    async def root():
        return {
            "message": "Deepfake Detection API",
            "version": "2.0.0",
            "endpoints": {
                "video": "/api/v1/video/analyze",
                "audio": "/api/v1/audio/analyze",
                "image": "/api/v1/image/analyze",
                "text": "/api/v1/text/analyze"
            },
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    @app.get("/api/v1/models/info")
    async def models_info():
        return {
            "video": {
                "type": "3-Model Ensemble",
                "models": ["XceptionNet", "EfficientNet-B4", "CNN-LSTM"],
                "capabilities": [
                    "공간적 아티팩트 탐지",
                    "구조적 불일치 분석",
                    "시간적 일관성 검증"
                ],
                "expected_accuracy": "93-94%"
            },
            "audio": {
                "type": "Dual-Model",
                "models": ["Mel-spectrogram CNN", "LFCC CNN"],
                "expected_accuracy": "99.6-99.8%"
            },
            "image": {
                "type": "Single Model",
                "expected_accuracy": "TBD"
            },
            "text": {
                "type": "KLUE-BERT Fine-tuned",
                "task": "Fake News Detection"
            }
        }

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )