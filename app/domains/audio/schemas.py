from pydantic import BaseModel, Field
from typing import Optional

class AudioAnalysisRequest(BaseModel):
    """
    오디오 분석 요청 스키마
    """
    analysis_id: int = Field(..., description="Spring Boot에서 생성한 분석 ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": 123
            }
        }

class AudioAnalysisResponse(BaseModel):
    """
    오디오 분석 응답 스키마
    """
    analysis_id: int = Field(..., description="분석 ID")
    prediction: str = Field(..., description="예측 결과 (real/fake)")
    confidence: float = Field(..., ge=0, le=1, description="확신도 (0~1)")
    real_probability: float = Field(..., ge=0, le=1, description="진짜일 확률")
    fake_probability: float = Field(..., ge=0, le=1, description="가짜일 확률")
    model_version: str = Field(..., description="모델 버전")
    processing_time: float = Field(..., description="처리 시간 (초)")
    file_name: str = Field(..., description="파일명")
    file_size: int = Field(..., description="파일 크기 (바이트)")
    status: str = Field(default="completed", description="처리 상태")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": 123,
                "prediction": "fake",
                "confidence": 0.9234,
                "real_probability": 0.0766,
                "fake_probability": 0.9234,
                "model_version": "lightweight_cnn",
                "processing_time": 1.23,
                "file_name": "sample_audio.wav",
                "file_size": 524288,
                "status": "completed"
            }
        }

class ErrorResponse(BaseModel):
    """
    에러 응답 스키마
    """
    detail: str = Field(..., description="에러 메시지")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "지원하지 않는 파일 형식: .txt"
            }
        }

class HealthCheckResponse(BaseModel):
    """
    헬스체크 응답 스키마
    """
    status: str = Field(..., description="서비스 상태")
    service: str = Field(..., description="서비스 이름")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "service": "audio-deepfake-detection"
            }
        }