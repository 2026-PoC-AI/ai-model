# app/domains/audio/router.py
from fastapi import APIRouter, Request
from app.common.responses import ok, ApiResponse
from app.domains.audio.schemas import AudioAnalyzeResponse, AudioAnalyzeRequest # Request 추가
from app.domains.audio.service import analyze_audio

router = APIRouter()

@router.post("/analyze", response_model=ApiResponse[AudioAnalyzeResponse])
async def analyze(request: Request, body: AudioAnalyzeRequest): # Body로 받음
    rid = request.state.request_id
    
    # 서비스에 키 전달
    result = analyze_audio(request, body.audio_s3_key)
    
    return ok(result, request_id=rid)