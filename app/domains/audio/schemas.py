from pydantic import BaseModel, Field
from typing import Optional, Dict, List

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

class TimeSegment(BaseModel):
    """
    시간대별 위험도 분석 결과
    """
    start: float = Field(..., description="시작 시간 (초)")
    end: float = Field(..., description="종료 시간 (초)")
    risk: str = Field(..., description="위험도 (high/medium/low)")
    reason: str = Field(..., description="판단 근거")

class DetailedAnalysis(BaseModel):
    """
    딥페이크 생성 기술 확률 분석
    """
    voice_synthesis_probability: float = Field(..., ge=0, le=1, description="TTS 확률")
    voice_conversion_probability: float = Field(..., ge=0, le=1, description="Voice Conversion 확률")
    replay_attack_probability: float = Field(..., ge=0, le=1, description="Replay Attack 확률")

class AudioAnalysisResponse(BaseModel):
    """
    오디오 분석 응답 스키마 (Ensemble + 딥페이크 기술 분류)
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
    
    # 3단계: 딥페이크 생성 기술 분류 필드
    suspected_method: Optional[str] = Field(None, description="의심되는 생성 기술")
    method_confidence: Optional[float] = Field(None, ge=0, le=1, description="생성 기술 판단 신뢰도")
    detailed_analysis: Optional[DetailedAnalysis] = Field(None, description="생성 기술별 확률")
    suspicious_patterns: Optional[List[str]] = Field(None, description="탐지된 의심스러운 패턴")
    time_segments: Optional[List[TimeSegment]] = Field(None, description="시간대별 위험도 분석")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": 123,
                "prediction": "fake",
                "confidence": 0.96,
                "probabilities": {
                    "real": 0.04,
                    "fake": 0.96
                },
                "model_outputs": {
                    "mel": {
                        "real": 0.08,
                        "fake": 0.92
                    },
                    "lfcc": {
                        "real": 0.12,
                        "fake": 0.88
                    }
                },
                "model_version": "ensemble_v1.0",
                "processing_time": 0.34,
                "file_name": "sample_audio.wav",
                "file_size": 524288,
                "status": "completed",
                "suspected_method": "TTS (Text-to-Speech)",
                "method_confidence": 0.78,
                "detailed_analysis": {
                    "voice_synthesis_probability": 0.85,
                    "voice_conversion_probability": 0.12,
                    "replay_attack_probability": 0.03
                },
                "suspicious_patterns": [
                    "TTS 특유의 운율 패턴",
                    "부자연스러운 포먼트 전환",
                    "일정한 피치 변화"
                ],
                "time_segments": [
                    {
                        "start": 0.0,
                        "end": 0.8,
                        "risk": "high",
                        "reason": "합성 음성 특징"
                    },
                    {
                        "start": 0.8,
                        "end": 2.0,
                        "risk": "medium",
                        "reason": "자연스러운 구간"
                    },
                    {
                        "start": 2.0,
                        "end": 3.5,
                        "risk": "high",
                        "reason": "voice conversion 의심"
                    }
                ]
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