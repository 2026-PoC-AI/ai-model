from pydantic import BaseModel, Field
from typing import Optional, Dict

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
    오디오 분석 응답 스키마 (Ensemble)
    """
    analysis_id: int = Field(..., description="분석 ID")
    prediction: str = Field(..., description="예측 결과 (real/fake)")
    confidence: float = Field(..., ge=0, le=1, description="확신도 (0~1)")
    probabilities: Dict[str, float] = Field(..., description="각 클래스별 확률")
    model_outputs: Dict[str, Dict[str, float]] = Field(..., description="각 모델의 개별 예측")
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
                "confidence": 0.9876,
                "probabilities": {
                    "real": 0.0124,
                    "fake": 0.9876
                },
                "model_outputs": {
                    "mel": {
                        "real": 0.0132,
                        "fake": 0.9868
                    },
                    "lfcc": {
                        "real": 0.0117,
                        "fake": 0.9883
                    }
                },
                "model_version": "ensemble_v1.0",
                "processing_time": 0.34,
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
    models: Optional[Dict[str, str]] = Field(None, description="로드된 모델 정보")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "service": "audio-deepfake-detection",
                "models": {
                    "mel_cnn": "loaded (99.15%)",
                    "lfcc_cnn": "loaded (99.57%)",
                    "ensemble": "active"
                }
            }
        }