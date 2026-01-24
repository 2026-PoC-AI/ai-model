# app/domains/text/router.py

from fastapi import APIRouter, Depends, Request
from .schemas import FakeNewsPredictRequest, FakeNewsPredictResponse

from .schemas import (
    FakeNewsPredictRequest,
    FakeNewsPredictBatchRequest,
    FakeNewsPredictResponse,
    FakeNewsPredictBatchResponse,
)
from .service import TextService
from .fake_news.predictor import FakeNewsPrediction

router = APIRouter(prefix="/text", tags=["text"])


# ---- DI (간단 버전) ----
# 실제 프로젝트에서는 core/model_registry.py 같은 곳에서 싱글톤으로 꺼내는게 더 깔끔함
_text_service: TextService | None = None

def get_text_service() -> TextService:
    assert _text_service is not None, "TextService not initialized"
    return _text_service

def init_text_service(service: TextService) -> None:
    global _text_service
    _text_service = service


@router.post("/fake-news/predict", response_model=FakeNewsPredictResponse)
async def predict(req: FakeNewsPredictRequest, request: Request):
    svc = request.app.state.text_service
    pred = svc.predict_fake_news(req.text)

    return FakeNewsPredictResponse(
        label=pred.label,
        score=pred.score,
        probabilities=pred.probabilities,
    )

@router.post("/fake-news/predict-batch", response_model=FakeNewsPredictBatchResponse)
def predict_batch(req: FakeNewsPredictBatchRequest, svc: TextService = Depends(get_text_service)):
    preds = svc.predict_fake_news_batch(req.texts)
    return FakeNewsPredictBatchResponse(
        results=[FakeNewsPredictResponse(label=p.label, score=p.score, probabilities=p.probabilities) for p in preds]
    )
