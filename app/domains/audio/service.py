# app/domains/audio/service.py
from fastapi import Request
from app.common.exceptions import AppError
from app.common.errors import Errors
from app.domains.audio.schemas import AudioAnalyzeResponse

def analyze_audio(request: Request, audio_bytes: bytes) -> AudioAnalyzeResponse:
    model = request.app.state.models.get("audio")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        return model.predict(audio_bytes)
    except Exception as e:
        raise AppError(Errors.INFERENCE_FAILED, details=str(e))
