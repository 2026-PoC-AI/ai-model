# // app/domains/video/router.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Request
from .service import VideoAnalysisService
from .schemas import VideoAnalysisResponse
import logging
import traceback  # // 상세 에러 추적을 위해 추가
import numpy as np # // 호환성 체크용
from app.core.redis_client import redis_client

router = APIRouter()
logger = logging.getLogger(__name__)

# // NumPy 2.0 호환성 패치 (모듈 로드 시점에 한 번 더 확실히 체크)
if int(np.__version__.split('.')[0]) >= 2:
    if not hasattr(np, "float_"): np.float_ = float
    if not hasattr(np, "bool_"): np.bool_ = bool
    if not hasattr(np, "int_"): np.int_ = int

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
        # // 1. 모델 로드 상태 확인
        # // model_registry.py에서 정상 로드되었는지 체크
        predictor = request.app.state.models.get("video")
        
        if predictor is None:
            logger.error(f"Analysis failed - ID: {analysis_id}, Reason: Video model not loaded in app.state")
            raise HTTPException(status_code=503, detail="Video model is currently unavailable (loading failed)")
        
        # // 2. 파일 내용 읽기
        content = await file.read()
        
        # // 3. 형식 검증
        allowed_formats = ['mp4', 'avi', 'mov']
        file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 허용 형식: {', '.join(allowed_formats)}"
            )
        
        # // 4. 크기 검증 (100MB)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기가 100MB를 초과합니다")
        
        # // 5. 서비스 생성 및 분석 수행
        # // 여기서 실제 NumPy 연산이 포함된 service.analyze_video가 실행됩니다.
        service = VideoAnalysisService(predictor=predictor)
        result = await service.analyze_video(content, file.filename, analysis_id)
        
        logger.info(f"Analysis completed successfully - ID: {analysis_id}")
        return result

    except HTTPException as he:
        # // FastAPI에서 의도적으로 발생시킨 예외는 그대로 전달
        raise he
    except Exception as e:
        # // [핵심 수정] 예상치 못한 에러(NumPy 충돌 등) 발생 시 상세 로그 출력
        error_traceback = traceback.format_exc()
        logger.error(f"Critical analysis failure - ID: {analysis_id}")
        logger.error(f"Error Message: {str(e)}")
        logger.error(f"Traceback:\n{error_traceback}") # // 터미널에서 이 부분을 확인해야 합니다.
        
        # // 클라이언트(Spring Boot)에게 구체적인 에러 메시지 전달
        raise HTTPException(
            status_code=500, 
            detail=f"AI 분석 중 내부 오류 발생 (NumPy/Model Error): {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Video 분석 서비스 상태 확인"""
    return {"status": "ok", "service": "video-deepfake-detection"}


@router.get("/progress/{analysis_id}")
async def get_analysis_progress(analysis_id: int):
    """비디오 분석 진행률 조회"""
    progress = redis_client.get_progress(analysis_id)
    
    if progress is None:
        raise HTTPException(status_code=404, detail="진행률 정보를 찾을 수 없습니다")
    
    return progress