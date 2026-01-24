# app/domains/text/fake_news/router.py
from fastapi import APIRouter, HTTPException
from .schemas import NewsRequest, NewsResponse
from .service import analyze_fake_news
from .predictor import FakeNewsPredictor
import os

router = APIRouter(prefix="/text/fake-news", tags=["Fake News"])

# 모델 전역 로드 (서버 시작 시 로드)
predictor = FakeNewsPredictor()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cpu_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "cpu_vectorizer.pkl")

try:
    predictor.load(MODEL_PATH, VEC_PATH)
except Exception as e:
    print(f"⚠️ 모델 로드 실패: {e}")

@router.post("/analyze", response_model=NewsResponse)
async def analyze(req: NewsRequest):
    if predictor.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    result = analyze_fake_news(req.text, predictor)
    return result