# app/domains/image/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict

class ImageEvidence(BaseModel):
    region: str
    score: float
    reason: str

# S3입력 Request
class ImageAnalyzeRequest(BaseModel):
    s3_key: str = Field(..., description="S3 object key for image file")

# 딥페이크 전용 응답
class ImageAnalyzeResponse(BaseModel):
    task: str = "deepfake_image"
    
    risk_score: int = Field(..., ge=0, le=100)
    grade: str  # LOW | MEDIUM | HIGH
    
    label: str
    confidence: float

    interpretation: Dict[str, str]

    evidence: List[ImageEvidence] = []
    warnings: List[str] = []

