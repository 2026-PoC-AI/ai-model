# app/domains/text/router.py
from fastapi import APIRouter, Request
from app.common.responses import ok, ApiResponse
from app.domains.text.schemas import TextAnalyzeRequest, TextAnalyzeResponse
from app.domains.text.service import analyze_text

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[TextAnalyzeResponse])
async def analyze(request: Request, body: TextAnalyzeRequest):
    rid = request.state.request_id
    result = analyze_text(request, body.text)
    return ok(result, request_id=rid)
