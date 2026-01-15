# app/domains/audio/service.py
from fastapi import Request
from app.common.exceptions import AppError
from app.common.errors import Errors
from app.domains.audio.schemas import AudioAnalyzeResponse
from app.core.s3 import s3_client # 추가

def analyze_audio(request: Request, s3_key: str) -> AudioAnalyzeResponse: # 파라미터 변경
    model = request.app.state.models.get("audio")
    if model is None:
        raise AppError(Errors.MODEL_NOT_LOADED)

    try:
        # 1. S3에서 파일 다운로드
        audio_bytes = s3_client.download_bytes(s3_key)
        
        # 2. 모델 예측
        return model.predict(audio_bytes)
    except Exception as e:
        # S3 에러나 모델 에러 처리
        raise AppError(Errors.INFERENCE_FAILED, details=str(e))