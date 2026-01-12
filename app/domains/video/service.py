# app/domains/video/service.py
from fastapi import Request
from app.common.exceptions import AppError
from app.common.errors import Errors
from app.domains.video.schemas import VideoAnalyzeResponse

def analyze_video(request: Request, video_bytes: bytes) -> VideoAnalyzeResponse:
    model = request.app.state.models.get("video")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        return model.predict(video_bytes)
    except Exception as e:
        raise AppError(Errors.INFERENCE_FAILED, details=str(e))
