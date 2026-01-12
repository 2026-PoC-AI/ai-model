# app/domains/text/schemas.py
from pydantic import BaseModel, Field
from typing import List

class TextEvidence(BaseModel):
    score: float
    reason: str

class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)

class TextAnalyzeResponse(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    grade: str  # LOW/MEDIUM/HIGH
    evidence: List[TextEvidence] = []
    warnings: List[str] = []
