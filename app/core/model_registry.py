# app/core/model_registry.py
import logging
from fastapi import FastAPI

from app.core.config import settings
from app.domains.audio.model import load_audio_model
from app.domains.video.model import load_video_model
from app.domains.image.model import load_image_model
from app.domains.text.model import load_text_model

logger = logging.getLogger("app")

async def init_models(app: FastAPI) -> None:
    app.state.models = {}
    logger.info("Loading models...")

    # Audio 모델 로드
    app.state.models["audio"] = load_audio_model()

    # Video 모델 로드 (3-Model Ensemble)
    try:
        logger.info("Loading Video Ensemble Models (Xception + EfficientNet + CNN-LSTM)...")
        app.state.models["video"] = load_video_model(
            xception_path=settings.XCEPTION_MODEL_PATH,
            efficientnet_path=settings.EFFICIENTNET_MODEL_PATH,
            cnn_lstm_path=settings.CNN_LSTM_MODEL_PATH,  # 새로 추가
            device='cuda' if settings.USE_GPU else 'cpu'
        )
        logger.info("✓ Video 3-model ensemble loaded successfully!")
        logger.info("  - XceptionNet: 공간적 아티팩트 탐지")
        logger.info("  - EfficientNet-B4: 구조적 불일치 탐지")
        logger.info("  - CNN-LSTM: 시간적 일관성 분석")
    except Exception as e:
        logger.error(f"Critical error loading video models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        app.state.models["video"] = None

    # Image 모델 로드
    app.state.models["image"] = load_image_model()

    # Text 모델 로드
    app.state.models["text"] = load_text_model()

    app.state.text_service = app.state.models["text"]

    logger.info("Models loaded: %s", list(app.state.models.keys()))

async def close_models(app: FastAPI) -> None:
    models = getattr(app.state, "models", None)
    if models:
        logger.info("Shutting down models...")

    if hasattr(app.state, "text_service"):
        app.state.text_service = None