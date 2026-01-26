from pydantic import BaseModel, ConfigDict
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict

class VideoFileData(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat() if v else None
    })
    
    fileId: Optional[int] = None
    analysisId: int
    originalFilename: str
    storedFilename: str
    filePath: str
    webFilePath: Optional[str] = None
    fileSize: int
    durationSeconds: Optional[Decimal] = None
    resolution: Optional[str] = None
    format: Optional[str] = None
    fps: Optional[Decimal] = None
    uploadedAt: datetime

# ========================================
# 새로 추가되는 클래스들
# ========================================

class ModelPredictionData(BaseModel):
    """개별 모델 예측 결과"""
    modelName: str
    fakeProbability: Decimal
    prediction: str  # 'fake' or 'real'
    confidence: Decimal
    detectedPatterns: List[str]
    suspiciousFrames: Optional[List[int]] = None  # CNN-LSTM 전용

class ArtifactCategoryData(BaseModel):
    """아티팩트 카테고리별 정보"""
    detected: bool
    sources: List[str]  # 탐지한 모델 목록
    patterns: List[str]  # 탐지된 패턴 설명

class DetectedArtifactsData(BaseModel):
    """탐지된 아티팩트 전체"""
    spatial: ArtifactCategoryData
    temporal: ArtifactCategoryData
    structural: ArtifactCategoryData

# ========================================
# 기존 클래스 확장
# ========================================

class AnalysisResultData(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat() if v else None
    })
    
    resultId: Optional[int] = None
    analysisId: int
    createdAt: datetime
    confidenceScore: Decimal
    isDeepfake: bool
    modelVersion: str
    processingTimeMs: int
    detectedTechniques: Optional[str] = None
    summary: Optional[str] = None
    analyzedAt: datetime
    
    # ========================================
    # 새로 추가되는 필드들
    # ========================================
    ensembleFakeProbability: Optional[Decimal] = None  # 앙상블 종합 확률
    modelAgreement: Optional[Decimal] = None  # 모델 합의도 (0~1)
    riskLevel: Optional[str] = None  # 'HIGH', 'MEDIUM', 'LOW', 'SAFE'
    individualModels: Optional[Dict[str, ModelPredictionData]] = None  # 개별 모델 결과
    detectedArtifacts: Optional[DetectedArtifactsData] = None  # 탐지된 아티팩트

class FrameAnalysisData(BaseModel):
    frameId: Optional[int] = None
    frameNumber: int
    timestampSeconds: Decimal
    isDeepfake: bool
    confidenceScore: Decimal
    anomalyType: Optional[str] = None
    features: Optional[str] = None

class VideoAnalysisResponse(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat() if v else None
    })
    
    analysisId: int
    title: str
    status: str
    createdAt: datetime
    completedAt: Optional[datetime] = None
    videoFile: Optional[VideoFileData] = None
    analysisResult: Optional[AnalysisResultData] = None
    frameAnalyses: Optional[List[FrameAnalysisData]] = None