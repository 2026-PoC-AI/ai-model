from fastapi import APIRouter, UploadFile, File, HTTPException
from .service import VideoAnalysisService
from .schemas import VideoAnalysisResponse

router = APIRouter()
service = VideoAnalysisService()

@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """영상 딥페이크 분석 - Spring Boot 연동"""
    
    # 파일 형식 검증
    allowed_formats = ['mp4', 'avi', 'mov']
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 허용 형식: {', '.join(allowed_formats)}"
        )
    
    # 파일 크기 검증 (100MB)
    max_size = 100 * 1024 * 1024
    
    # 수정: await 사용
    content = await file.read()
    
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail="파일 크기가 100MB를 초과합니다"
        )
    
    # 수정: 파일 포인터 리셋
    await file.seek(0)
    
    return await service.analyze_video(file.file, file.filename)

@router.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile):
    # 파일 저장
    video_path = f"uploads/{file.filename}"
    
    # 딥페이크 탐지
    result = await video_service.detect_deepfake(video_path)
    
    return result