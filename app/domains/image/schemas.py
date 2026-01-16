# app/domains/image/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict

class ImageEvidence(BaseModel):
    region: str
    score: float
    reason: str

# 딥페이크 전용 응답
class ImageAnalyzeResponse(BaseModel):
    task: str = "deepfake_image"
    label: str  # FAKE | REAL
    confidence: float = Field(..., ge=0.0, le=1.0)

    interpretation: Dict[str, str]

    evidence: List[ImageEvidence] = []
    warnings: List[str] = []

