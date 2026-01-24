# app/domains/text/schemas.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# =========================
# ✅ (기존 코드 호환) 스키마
# =========================
class TextEvidence(BaseModel):
    text: str = Field(..., description="근거 텍스트")
    score: Optional[float] = Field(default=None, description="근거 점수")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="추가 메타데이터")


class TextAnalyzeResponse(BaseModel):
    label: str
    score: float
    evidences: Optional[List[TextEvidence]] = None
    probabilities: Optional[Dict[str, float]] = None


# =========================
# ✅ (추가 기능) FakeNews 스키마
# =========================
class FakeNewsPredictRequest(BaseModel):
    text: str = Field(..., description="분류할 텍스트")


class FakeNewsPredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="분류할 텍스트 목록")


class FakeNewsPredictResponse(BaseModel):
    label: str
    score: float
    probabilities: Dict[str, float]


class FakeNewsPredictBatchResponse(BaseModel):
    results: List[FakeNewsPredictResponse]
