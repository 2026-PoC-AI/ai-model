import os
import cv2
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import List
from app.core.config import settings
from .schemas import (
    VideoAnalysisResponse,
    VideoFileData,
    AnalysisResultData,
    FrameAnalysisData
)

class VideoAnalysisService:
    def __init__(self, predictor):
        """
        Args:
            predictor: app.state.models["video"]에서 전달받은 앙상블 모델
        """
        # 업로드 디렉토리 설정
        self.upload_dir = Path("uploads/video")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 전달받은 모델 사용
        self.model = predictor
    
    async def analyze_video(self, content: bytes, filename: str, analysis_id: int) -> VideoAnalysisResponse:
        """영상 딥페이크 분석 메인 로직"""
        start_time = datetime.now()
    
        # 1. 파일 저장 경로 설정 (ID 기반 파일명)
        file_extension = filename.split('.')[-1]
        stored_filename = f"{analysis_id}_{int(datetime.now().timestamp())}.{file_extension}"
        file_path = os.path.join(self.upload_dir, stored_filename)
        
        # 2. 바이너리 데이터 파일로 저장
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # 3. 영상 메타데이터 추출 (OpenCV 사용)
        video_info = self._extract_video_info(file_path)
        
        # 4. 프레임 분석 (AI 모델 호출)
        frame_analyses = self._analyze_frames(file_path, video_info['fps'])
        
        # 5. 최종 딥페이크 판정
        is_deepfake, confidence_score = self._detect_deepfake(frame_analyses)
        
        # 6. 처리 시간 계산
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # 7. 응답 생성
        return VideoAnalysisResponse(
            analysisId=analysis_id,
            title=filename,
            status="COMPLETED",
            createdAt=start_time,
            completedAt=datetime.now(),
            videoFile=VideoFileData(
                originalFilename=filename,
                storedFilename=stored_filename,
                filePath=file_path,
                fileSize=len(content),
                durationSeconds=Decimal(str(round(video_info['duration'], 2))),
                resolution=video_info['resolution'],
                format=file_extension,
                fps=Decimal(str(round(video_info['fps'], 2))),
                uploadedAt=start_time,
                analysisId=analysis_id
            ),
            analysisResult=AnalysisResultData(
                analysisId=analysis_id,
                createdAt=start_time,
                confidenceScore=Decimal(str(round(confidence_score, 4))),
                isDeepfake=is_deepfake,
                modelVersion="v1.0.0",
                processingTimeMs=processing_time_ms,
                detectedTechniques=self._get_detected_techniques(frame_analyses),
                summary=self._generate_summary(is_deepfake, confidence_score, frame_analyses),
                analyzedAt=datetime.now()
            ),
            frameAnalyses=frame_analyses
        )
    
    def _extract_video_info(self, file_path: str) -> dict:
        """영상 메타데이터 추출"""
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {'fps': fps, 'duration': duration, 'resolution': f"{width}X{height}", 'frame_count': frame_count}
    
    def _analyze_frames(self, file_path: str, fps: float) -> List[FrameAnalysisData]:
        """프레임별 분석"""
        cap = cv2.VideoCapture(file_path)
        frame_analyses = []
        frame_number = 0
        sample_interval = int(fps) if fps > 0 else 30
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 지정된 간격마다 프레임 분석 수행
            if frame_number % sample_interval == 0:
                is_suspicious, confidence, anomaly_type = self.model.analyze_frame(frame)
                frame_analyses.append(FrameAnalysisData(
                    frameNumber=frame_number,
                    timestampSeconds=Decimal(str(round(frame_number / fps, 2))),
                    isDeepfake=is_suspicious,
                    confidenceScore=Decimal(str(round(confidence, 4))),
                    anomalyType=anomaly_type,
                    features="{}"
                ))
            frame_number += 1
            
        cap.release()
        return frame_analyses
    
    def _detect_deepfake(self, frame_analyses):
        """전체 영상 딥페이크 판정 논리"""
        if not frame_analyses: 
            return False, 0.0
        
        # 각 프레임의 confidence 평균 계산
        avg_confidence = sum(float(f.confidenceScore) for f in frame_analyses) / len(frame_analyses)
        
        # 평균 confidence가 0.5 이상이면 딥페이크
        is_deepfake = avg_confidence > 0.5
        
        return is_deepfake, avg_confidence
    
    def _get_detected_techniques(self, frames: List[FrameAnalysisData]) -> str:
        """탐지된 딥페이크 기법 목록 정리"""
        techniques = {frame.anomalyType for frame in frames if frame.isDeepfake and frame.anomalyType != "normal"}
        return ", ".join(sorted(techniques)) if techniques else "none"
    
    def _generate_summary(self, is_deepfake, confidence, frames):
        """분석 결과 요약 텍스트 생성"""
        if is_deepfake:
            return f"딥페이크 가능성 높음 (평균 신뢰도 {confidence*100:.1f}%)"
        return f"정상 영상으로 판단 (평균 신뢰도 {(1-confidence)*100:.1f}%)"