# app/domains/video/schemas.py
from pydantic import BaseModel, Field
from typing import List

class VideoEvidence(BaseModel):
    frame_index: int
    score: float
    reason: str

class VideoAnalyzeResponse(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    grade: str  # LOW/MEDIUM/HIGH
    evidence: List[VideoEvidence] = []
    warnings: List[str] = []
