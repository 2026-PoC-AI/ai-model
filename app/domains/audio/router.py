from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Request
import logging

from .service import AudioAnalysisService
from .schemas import AudioAnalysisResponse, HealthCheckResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    request: Request,  # ⭐ 이게 핵심
    file: UploadFile = File(...),
    analysis_id: int = Form(...)
):
    """
    오디오 딥페이크 분석 엔드포인트
    """
    logger.info(f"Audio analysis started: {file.filename} (ID: {analysis_id})")
    
    try:
        # app.state에서 미리 로드된 모델 가져오기
        predictor = request.app.state.models.get("audio")
        
        if predictor is None:
            raise HTTPException(status_code=500, detail="Audio model not loaded")
        
        # 서비스 생성 (모델을 파라미터로 전달)
        service = AudioAnalysisService(predictor=predictor)
        result = await service.analyze_audio(file.file, file.filename, analysis_id)
        
        logger.info(f"Analysis completed - ID: {analysis_id}")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed - ID: {analysis_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {"status": "ok", "service": "audio-deepfake-detection"}