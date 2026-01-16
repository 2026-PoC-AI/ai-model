# app/domains/image/service.py
from fastapi import Request
from app.common.exceptions import AppError
from app.common.errors import Errors
from app.domains.image.schemas import ImageAnalyzeResponse
from app.core.s3 import s3_client

# 딥페이크 판단 문구
def interpret_deepfake(label: str, confidence: float) -> dict:
    if label == "FAKE" and confidence >= 0.8:
        return {
            "risk_level": "HIGH",
            "message": "AI 생성 또는 얼굴 합성 가능성이 매우 높습니다."
        }
    if label == "FAKE":
        return {
            "risk_level": "MEDIUM",
            "message": "조작 가능성이 있어 주의가 필요합니다."
        }
    if label == "REAL" and confidence >= 0.8:
        return {
            "risk_level": "LOW",
            "message": "조작 흔적이 뚜렷하지 않습니다."
        }
    return {
        "risk_level": "MEDIUM",
        "message": "판단이 불확실하여 주의가 필요합니다."
    }

# 전용서비스함수
def analyze_image(
    request: Request,
    s3_key: str
) -> ImageAnalyzeResponse:
    model = request.app.state.models.get("image")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        image_bytes = s3_client.download_bytes(s3_key)
        
        label, confidence = model.predict_deepfake(image_bytes)
        interpretation = interpret_deepfake(label, confidence)

        return ImageAnalyzeResponse(
            task="deepfake_image",
            label=label,
            confidence=confidence,
            interpretation=interpretation,
            evidence=[],
            warnings=[]
        )

    except Exception as e:
        raise AppError(Errors.INFERENCE_FAILED, details=str(e))
    
    
