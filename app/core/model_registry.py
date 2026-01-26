# // app/core/model_registry.py
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

    # // 1. Audio 모델 로드
    app.state.models["audio"] = load_audio_model()

    # // 2. Video 모델 로드 (주석 해제 및 예외 처리 강화)
    try:
        logger.info("Loading Video Ensemble Models (Xception + EfficientNet)...")
        app.state.models["video"] = load_video_model(
            xception_path=settings.XCEPTION_MODEL_PATH,
            efficientnet_path=settings.EFFICIENTNET_MODEL_PATH,
            ensemble_method='soft_voting',
            weights=[0.5, 0.5]
        )
        logger.info("✓ Video models loaded successfully!")
    except Exception as e:
        # // 로드 실패 시 서버가 죽지 않게 하고 에러 내용을 상세히 출력
        logger.error(f"Critical error loading video models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        app.state.models["video"] = None

    # // 3. Image & Text 모델 로드
    app.state.models["image"] = load_image_model()
    app.state.models["text"] = load_text_model()

    app.state.text_service = app.state.models["text"]

    logger.info("Models loaded: %s", list(app.state.models.keys()))

async def close_models(app: FastAPI) -> None:
    models = getattr(app.state, "models", None)
    if models:
        logger.info("Shutting down models...")

    if hasattr(app.state, "text_service"):
        app.state.text_service = None