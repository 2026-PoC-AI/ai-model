from pydantic import BaseModel, ConfigDict, field_serializer
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
    
    @field_serializer('uploadedAt')
    def serialize_datetime(self, dt: datetime, _info):
        if dt is None:
            return None
        # 타임존 정보 포함
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + dt.strftime('%z')

class ModelPredictionData(BaseModel):
    """개별 모델 예측 결과"""
    modelName: str
    fakeProbability: Decimal
    prediction: str
    confidence: Decimal
    detectedPatterns: List[str]
    suspiciousFrames: Optional[List[int]] = None

class ArtifactCategoryData(BaseModel):
    """아티팩트 카테고리별 정보"""
    detected: bool
    sources: List[str]
    patterns: List[str]

class DetectedArtifactsData(BaseModel):
    """탐지된 아티팩트 전체"""
    spatial: ArtifactCategoryData
    temporal: ArtifactCategoryData
    structural: ArtifactCategoryData

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
    ensembleFakeProbability: Optional[Decimal] = None
    modelAgreement: Optional[Decimal] = None
    riskLevel: Optional[str] = None
    individualModels: Optional[Dict[str, ModelPredictionData]] = None
    detectedArtifacts: Optional[DetectedArtifactsData] = None
    
    @field_serializer('createdAt', 'analyzedAt')
    def serialize_datetime(self, dt: datetime, _info):
        if dt is None:
            return None
        # 타임존 정보 포함
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + dt.strftime('%z')

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
    
    @field_serializer('createdAt', 'completedAt')
    def serialize_datetime(self, dt: datetime, _info):
        if dt is None:
            return None
        # 밀리초 3자리 + 타임존
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + dt.strftime('%z')