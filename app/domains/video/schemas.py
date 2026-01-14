from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from decimal import Decimal

class VideoFileData(BaseModel):
    fileId: str
    originalFilename: str
    storedFilename: Optional[str] = None
    filePath: Optional[str] = None
    fileSize: int
    durationSeconds: Optional[Decimal] = None
    resolution: Optional[str] = None
    format: str
    fps: Optional[Decimal] = None
    uploadedAt: Optional[datetime] = None
    analysisId: str

class AnalysisResultData(BaseModel):
    resultId: str
    analysisId: str
    createdAt: datetime
    confidenceScore: Decimal
    isDeepfake: bool
    modelVersion: str
    processingTimeMs: int
    detectedTechniques: str
    summary: str
    analyzedAt: datetime

class FrameAnalysisData(BaseModel):
    frameId: str
    frameNumber: int
    timestampSeconds: Decimal
    isDeepfake: bool
    confidenceScore: Decimal
    anomalyType: str
    features: str

class VideoAnalysisResponse(BaseModel):
    """ERD 기반 전체 응답 - Spring Boot와 완전히 일치"""
    analysisId: str
    title: str
    status: str
    createdAt: datetime
    completedAt: Optional[datetime] = None
    videoFile: VideoFileData
    analysisResult: AnalysisResultData
    frameAnalyses: List[FrameAnalysisData]