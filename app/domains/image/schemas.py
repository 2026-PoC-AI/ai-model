# app/domains/image/schemas.py
from pydantic import BaseModel, Field
from typing import List

class ImageEvidence(BaseModel):
    region: str       # e.g., "face_1"
    score: float
    reason: str

class ImageAnalyzeResponse(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    grade: str  # LOW/MEDIUM/HIGH
    evidence: List[ImageEvidence] = []
    warnings: List[str] = []
