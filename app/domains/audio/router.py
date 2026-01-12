# app/domains/audio/router.py
from fastapi import APIRouter, UploadFile, File, Request
from app.common.responses import ok, ApiResponse
from app.domains.audio.schemas import AudioAnalyzeResponse
from app.domains.audio.service import analyze_audio

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[AudioAnalyzeResponse])
async def analyze(request: Request, file: UploadFile = File(...)):
    rid = request.state.request_id
    content = await file.read()

    result = analyze_audio(request, content)
    return ok(result, request_id=rid)
