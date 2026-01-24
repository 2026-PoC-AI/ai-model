from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Request
import logging

from .service import VideoAnalysisService 

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze")
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    analysis_id: int = Form(...)
):
    """
    비디오 딥페이크 분석 엔드포인트
    """
    logger.info(f"Video analysis started: {file.filename} (ID: {analysis_id})")
    
    try:
        # app.state에서 미리 로드된 모델 가져오기
        predictor = request.app.state.models.get("video")
        
        if predictor is None:
            raise HTTPException(status_code=500, detail="Video model not loaded")
        
        # 서비스 생성 (모델 전달)
        service = VideoAnalysisService(predictor=predictor)
        result = await service.analyze_video(file.file, file.filename, analysis_id)
        
        logger.info(f"Analysis completed successfully - ID: {analysis_id}")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed - ID: {analysis_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

@router.get("/health")
async def health_check():
    """Video 분석 서비스 상태 확인"""
    return {"status": "ok", "service": "video-deepfake-detection"}