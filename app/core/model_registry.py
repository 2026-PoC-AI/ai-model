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

    app.state.models["audio"] = load_audio_model()
    # app.state.models["video"] = load_video_model(
    #     xception_path=settings.XCEPTION_MODEL_PATH,
    #     efficientnet_path=settings.EFFICIENTNET_MODEL_PATH,
    #     ensemble_method='soft_voting',
    #     weights=[0.5, 0.5]
    # )

    # Video 모델 로드 (임시 비활성화)
    logger.warning("Video model loading temporarily disabled due to NumPy compatibility")
    app.state.models["video"] = None

    app.state.models["image"] = load_image_model()
    app.state.models["text"] = load_text_model()

    logger.info("Models loaded: %s", list(app.state.models.keys()))

async def close_models(app: FastAPI) -> None:
    models = getattr(app.state, "models", None)
    if models:
        logger.info("Shutting down models...")