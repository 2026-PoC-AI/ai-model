from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from decimal import Decimal

class VideoFileData(BaseModel):
    originalFilename: str
    storedFilename: Optional[str] = None
    filePath: Optional[str] = None
    fileSize: int
    durationSeconds: Optional[Decimal] = None
    resolution: Optional[str] = None
    format: str
    fps: Optional[Decimal] = None
    uploadedAt: Optional[datetime] = None
    analysisId: int # 어떤 분석에 대한 파일인지 식별용

class AnalysisResultData(BaseModel):
    analysisId: int # 식별용
    createdAt: datetime
    confidenceScore: Decimal
    isDeepfake: bool
    modelVersion: str
    processingTimeMs: int
    detectedTechniques: str
    summary: str
    analyzedAt: datetime

class FrameAnalysisData(BaseModel):
    frameNumber: int
    timestampSeconds: Decimal
    isDeepfake: bool
    confidenceScore: Decimal
    anomalyType: str
    features: str

class VideoAnalysisResponse(BaseModel):
    analysisId: int
    title: str
    status: str
    createdAt: datetime
    completedAt: Optional[datetime] = None
    videoFile: VideoFileData
    analysisResult: AnalysisResultData
    frameAnalyses: List[FrameAnalysisData]