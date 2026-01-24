from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Request
from .service import VideoAnalysisService
from .schemas import VideoAnalysisResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    analysis_id: int = Form(...)
):
    """
    영상 딥페이크 분석 - Spring Boot 연동 엔드포인트
    """
    logger.info(f"Video analysis started: {file.filename} (ID: {analysis_id})")
    
    try:
        # app.state에서 미리 로드된 모델 가져오기
        predictor = request.app.state.models.get("video")
        
        if predictor is None:
            raise HTTPException(status_code=500, detail="Video model not loaded")
        
        # 파일 내용 읽기
        content = await file.read()
        
        # 형식 검증
        allowed_formats = ['mp4', 'avi', 'mov']
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 허용 형식: {', '.join(allowed_formats)}"
            )
        
        # 크기 검증 (100MB)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기가 100MB를 초과합니다")
        
        # 서비스 생성 및 분석
        service = VideoAnalysisService(predictor=predictor)
        result = await service.analyze_video(content, file.filename, analysis_id)
        
        logger.info(f"Analysis completed successfully - ID: {analysis_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed - ID: {analysis_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI 분석 중 서버 내부 오류 발생: {str(e)}")

@router.get("/health")
async def health_check():
    """Video 분석 서비스 상태 확인"""
    return {"status": "ok", "service": "video-deepfake-detection"}