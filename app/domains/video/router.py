# app/domains/video/router.py
from fastapi import APIRouter, UploadFile, File, Request
from app.common.responses import ok, ApiResponse
from app.domains.video.schemas import VideoAnalyzeResponse
from app.domains.video.service import analyze_video

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[VideoAnalyzeResponse])
async def analyze(request: Request, file: UploadFile = File(...)):
    rid = request.state.request_id
    content = await file.read()

    result = analyze_video(request, content)
    return ok(result, request_id=rid)
