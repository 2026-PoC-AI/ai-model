# app/domains/image/service.py
from fastapi import Request
from app.common.exceptions import AppError
from app.common.errors import Errors
from app.domains.image.schemas import ImageAnalyzeResponse

def analyze_image(request: Request, image_bytes: bytes) -> ImageAnalyzeResponse:
    model = request.app.state.models.get("image")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        return model.predict(image_bytes)
    except Exception as e:
        raise AppError(Errors.INFERENCE_FAILED, details=str(e))
