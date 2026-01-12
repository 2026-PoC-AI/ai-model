# app/domains/text/service.py
from fastapi import Request
from app.common.exceptions import AppError
from app.common.errors import Errors
from app.domains.text.schemas import TextAnalyzeResponse

def analyze_text(request: Request, text: str) -> TextAnalyzeResponse:
    model = request.app.state.models.get("text")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        return model.predict(text)
    except Exception as e:
        raise AppError(Errors.INFERENCE_FAILED, details=str(e))
