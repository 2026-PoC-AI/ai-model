from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# =========================
# ✅ Text Analyze (고정 API)
# =========================
class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., description="검증할 텍스트 본문")
    evidence_k: int = Field(3, ge=1, le=10, description="근거 문장 개수 (기본 3)")
    include_references: bool = Field(True, description="추가 참고자료(네이버 등) 포함 여부")


class TextEvidence(BaseModel):
    text: str = Field(..., description="근거 텍스트")
    score: Optional[float] = Field(default=None, description="근거 점수")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="추가 메타데이터")


class TextHighlight(BaseModel):
    start: int = Field(..., ge=0, description="원문 기준 시작 인덱스")
    end: int = Field(..., ge=0, description="원문 기준 끝 인덱스(파이썬 슬라이스 end 미포함)")
    text: str = Field(..., description="하이라이트된 원문 substring")
    weight: float = Field(..., ge=0, description="강조 가중치(0~1 권장)")


class TextReference(BaseModel):
    title: str = Field(..., description="참고 자료 제목")
    url: str = Field(..., description="참고 자료 URL")
    snippet: Optional[str] = Field(default=None, description="요약/설명(있으면)")


class TextAnalyzeResponse(BaseModel):
    label: str
    score: float
    evidences: List[TextEvidence] = Field(default_factory=list)
    highlights: List[TextHighlight] = Field(default_factory=list)
    references: List[TextReference] = Field(default_factory=list)


# =========================
# ✅ (기존) FakeNews 스키마 유지
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
