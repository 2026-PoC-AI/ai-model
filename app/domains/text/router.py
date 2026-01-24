from fastapi import APIRouter, Request, HTTPException

from .schemas import (
    TextAnalyzeRequest,
    TextAnalyzeResponse,
    FakeNewsPredictRequest,
    FakeNewsPredictBatchRequest,
    FakeNewsPredictResponse,
    FakeNewsPredictBatchResponse,
)

router = APIRouter(prefix="/text", tags=["text"])


def _get_text_service(request: Request):
    svc = getattr(request.app.state, "text_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Text model not loaded.")
    return svc


@router.post("/analyze", response_model=TextAnalyzeResponse)
async def analyze(req: TextAnalyzeRequest, request: Request):
    svc = _get_text_service(request)
    return svc.analyze(
        text=req.text,
        evidence_k=req.evidence_k,
        include_references=req.include_references,
    )


@router.post("/fake-news/predict", response_model=FakeNewsPredictResponse)
async def predict(req: FakeNewsPredictRequest, request: Request):
    svc = _get_text_service(request)
    pred = svc.predict_fake_news(req.text)
    return FakeNewsPredictResponse(label=pred.label, score=pred.score, probabilities=pred.probabilities)


@router.post("/fake-news/predict-batch", response_model=FakeNewsPredictBatchResponse)
async def predict_batch(req: FakeNewsPredictBatchRequest, request: Request):
    svc = _get_text_service(request)
    preds = svc.predict_fake_news_batch(req.texts)
    return FakeNewsPredictBatchResponse(
        results=[FakeNewsPredictResponse(label=p.label, score=p.score, probabilities=p.probabilities) for p in preds]
    )
