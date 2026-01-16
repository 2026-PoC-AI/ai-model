# app/domains/image/router.py
from fastapi import APIRouter, UploadFile, File, Request
from app.common.responses import ok, ApiResponse
from app.domains.image.schemas import ImageAnalyzeResponse
from app.domains.image.service import analyze_image

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[ImageAnalyzeResponse])
async def analyze(request: Request, file: UploadFile = File(...)):
    rid = request.state.request_id
    content = await file.read()

    result = analyze_image(request, content)
    return ok(result, request_id=rid)