from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import logging
# 실제 서비스 클래스 임포트 (경로를 도메인 구조에 맞게 설정)
from .service import VideoAnalysisService 

# 라우터 설정
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    analysis_id: int = Form(...) # ★ Spring Boot의 Long id를 Form 데이터로 수신
):
    """
    비디오 딥페이크 분석 엔드포인트
    """
    # 로그에 수신한 숫자 ID가 정상적으로 찍히는지 확인하는 용도
    logger.info(f"Video analysis started: {file.filename} (ID: {analysis_id})")
    
    try:
        # 서비스 객체 생성
        service = VideoAnalysisService()
        
        # ★ 수정: 서비스의 analyze_video 함수 호출 시 수신한 analysis_id를 인자로 전달
        # file.file은 바이너리 데이터를 읽기 위해 전달합니다.
        result = await service.analyze_video(file.file, file.filename, analysis_id)
        
        logger.info(f"Analysis completed successfully - ID: {analysis_id}")
        return result
        
    except Exception as e:
        # 에러 발생 시에도 어떤 ID에서 문제가 생겼는지 로그를 남깁니다.
        logger.error(f"Analysis failed - ID: {analysis_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

@router.get("/health")
async def health_check():
    """Video 분석 서비스 상태 확인"""
    return {"status": "ok", "service": "video-deepfake-detection"}