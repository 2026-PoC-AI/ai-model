# app/domains/image/router.py
from fastapi import APIRouter, Request
from app.common.responses import ok, ApiResponse
from app.domains.image.schemas import (
    ImageAnalyzeRequest,
    ImageAnalyzeResponse,
)
from app.domains.image.service import analyze_image

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[ImageAnalyzeResponse])
async def analyze(request: Request, body: ImageAnalyzeRequest):
    rid = request.state.request_id
    
    result = analyze_image(
        request=request,
        s3_key=body.s3_key,
    )
    
    return ok(result, request_id=rid)