from pydantic import BaseModel, ConfigDict
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

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