# app/core/model_registry.py
import logging
from fastapi import FastAPI

from app.domains.audio.model import load_audio_model
from app.domains.video.model import load_video_model
from app.domains.image.model import load_image_model
from app.domains.text.model import load_text_model

logger = logging.getLogger("app")

async def init_models(app: FastAPI) -> None:
    app.state.models = {}
    logger.info("Loading models...")

    app.state.models["audio"] = load_audio_model()
    app.state.models["video"] = load_video_model()
    app.state.models["image"] = load_image_model()
    app.state.models["text"] = load_text_model()

    logger.info("Models loaded: %s", list(app.state.models.keys()))

async def close_models(app: FastAPI) -> None:
    models = getattr(app.state, "models", None)
    if models:
        logger.info("Shutting down models...")
