# app/domains/text/fake_news/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class NewsRequest(BaseModel):
    text: str

class Evidence(BaseModel):
    keywords: List[str]  # 의심 단어들
    sentences: List[str] # 의심 문장들

class NewsResponse(BaseModel):
    score: float         # 0~100 점수
    label: int           # 0: 정상, 1: 가짜
    level: str           # LOW, MID, HIGH
    evidence: Evidence   # 증거 데이터
    message: str         # 사용자용 요약 메시지