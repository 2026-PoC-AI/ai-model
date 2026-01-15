# app/domains/audio/schemas.py
from pydantic import BaseModel, Field
from typing import List
from pydantic import BaseModel

class AudioAnalyzeRequest(BaseModel):
    audio_s3_key: str
class AudioEvidence(BaseModel):
    score: float
    reason: str

class AudioAnalyzeResponse(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    grade: str  # LOW/MEDIUM/HIGH
    evidence: List[AudioEvidence] = []
    warnings: List[str] = []
