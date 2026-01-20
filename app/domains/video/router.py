from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from .service import VideoAnalysisService
from .schemas import VideoAnalysisResponse
import logging

# 라우터 및 서비스 객체 초기화
router = APIRouter()
service = VideoAnalysisService()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    analysis_id: int = Form(...) # Spring Boot의 Long ID 수신
):
    """
    영상 딥페이크 분석 - Spring Boot 연동 엔드포인트
    """
    logger.info(f"Video analysis started: {file.filename} (ID: {analysis_id})")
    
    # 1. 파일 내용 읽기 (메모리에 bytes로 저장)
    content = await file.read()
    
    # 2. 형식 검증
    allowed_formats = ['mp4', 'avi', 'mov']
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"지원하지 않는 파일 형식입니다. 허용 형식: {', '.join(allowed_formats)}"
        )
    
    # 3. 크기 검증 (100MB)
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기가 100MB를 초과합니다")
    
    # 📝 참고: 이미 content(bytes)를 만들었으므로 file.seek(0)은 더 이상 필요 없습니다.
    
    try:
        # 4. 서비스 호출 (이미 읽은 content 바이트 데이터를 직접 전달)
        # service.py의 analyze_video(self, content: bytes, ...) 형식을 따릅니다.
        result = await service.analyze_video(content, file.filename, analysis_id)
        
        logger.info(f"Analysis completed successfully - ID: {analysis_id}")
        return result

    except Exception as e:
        logger.error(f"Analysis failed - ID: {analysis_id}, Error: {str(e)}")
        # 에러 메시지를 명확하게 반환하여 Spring Boot에서 확인할 수 있게 함
        raise HTTPException(status_code=500, detail=f"AI 분석 중 서버 내부 오류 발생: {str(e)}")

@router.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile):
    """단순 딥페이크 탐지용 (Spring Boot 연동 외 별도 사용 시)"""
    # 업로드 폴더가 없을 경우 대비
    upload_path = f"uploads/{file.filename}"
    result = await service.detect_deepfake(upload_path) 
    return result