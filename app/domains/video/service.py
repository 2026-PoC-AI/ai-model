import os
import cv2
import logging
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import List
from app.core.config import settings
from app.core.redis_client import redis_client
from .schemas import (
    VideoAnalysisResponse,
    VideoFileData,
    AnalysisResultData,
    FrameAnalysisData
)

logger = logging.getLogger(__name__)

class VideoAnalysisService:
    def __init__(self, predictor):
        """
        Args:
            predictor: app.state.models["video"]에서 전달받은 앙상블 모델
        """
        self.upload_dir = Path("uploads/video")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model = predictor
    
    def _update_progress(self, analysis_id: int, progress: int, stage: str, detail: str):
        """Redis에 진행률 업데이트"""
        try:
            logger.info(f"[REDIS] 저장 시도 - ID:{analysis_id} {progress}% - {stage}: {detail}")
            redis_client.set_progress(analysis_id, progress, stage, detail)
            logger.info(f"[REDIS] 저장 성공 - ID:{analysis_id}")
            
            saved = redis_client.get_progress(analysis_id)
            logger.info(f"[REDIS] 저장 확인 - ID:{analysis_id}: {saved}")
        except Exception as e:
            logger.error(f"[REDIS] 저장 실패 - ID:{analysis_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def analyze_video(self, content: bytes, filename: str, analysis_id: int) -> VideoAnalysisResponse:
        """영상 딥페이크 분석 메인 로직"""
        start_time = datetime.now(timezone.utc)
        
        self._update_progress(analysis_id, 5, "video_upload", "영상 업로드 완료")
        
        file_extension = filename.split('.')[-1]
        stored_filename = f"{analysis_id}_{int(datetime.now().timestamp())}.{file_extension}"
        file_path = os.path.join(self.upload_dir, stored_filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        self._update_progress(analysis_id, 10, "video_upload", "파일 저장 완료")
        
        video_info = self._extract_video_info(file_path)
        
        self._update_progress(analysis_id, 15, "frame_extraction", "영상 메타데이터 추출 완료")
        
        frame_analyses = self._analyze_frames(file_path, video_info['fps'], analysis_id)
        
        self._update_progress(analysis_id, 90, "result_generation", "결과 생성 중")
        
        is_deepfake, confidence_score = self._detect_deepfake(frame_analyses)
        
        processing_time_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        
        self._update_progress(analysis_id, 100, "completed", "분석 완료")
        
        response = VideoAnalysisResponse(
            analysisId=analysis_id,
            title=filename,
            status="COMPLETED",
            createdAt=start_time,
            completedAt=datetime.now(timezone.utc),
            videoFile=VideoFileData(
                originalFilename=filename,
                storedFilename=stored_filename,
                filePath=file_path,
                fileSize=len(content),
                durationSeconds=Decimal(str(round(video_info['duration'], 2))),
                resolution=video_info['resolution'],
                format=file_extension,
                fps=Decimal(str(round(video_info['fps'], 2))),
                uploadedAt=datetime.now(timezone.utc),
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
                analyzedAt=datetime.now(timezone.utc)
            ),
            frameAnalyses=frame_analyses
        )
        
        return response
    
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
    
    def _analyze_frames(self, file_path: str, fps: float, analysis_id: int) -> List[FrameAnalysisData]:
        """프레임별 분석"""
        cap = cv2.VideoCapture(file_path)
        frame_analyses = []
        frame_number = 0
        sample_interval = int(fps) if fps > 0 else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            if frame_number % sample_interval == 0:
                progress = int(15 + (frame_number / total_frames) * 70)
                
                if processed_count % 5 == 0:
                    self._update_progress(
                        analysis_id, 
                        progress, 
                        "frame_extraction", 
                        f"프레임 추출 중 ({frame_number}/{total_frames})"
                    )
                
                if processed_count % 5 == 0:
                    self._update_progress(
                        analysis_id, 
                        progress, 
                        "face_detection", 
                        f"얼굴 검출 중 (처리 중: {processed_count + 1}번째)"
                    )
                
                is_suspicious, confidence, anomaly_type = self.model.analyze_frame(frame)
                
                if processed_count % 5 == 0:
                    self._update_progress(
                        analysis_id, 
                        progress, 
                        "ai_analysis", 
                        f"AI 모델 분석 중 ({processed_count + 1}개 프레임 완료)"
                    )
                
                frame_analyses.append(FrameAnalysisData(
                    frameNumber=frame_number,
                    timestampSeconds=Decimal(str(round(frame_number / fps, 2))),
                    isDeepfake=is_suspicious,
                    confidenceScore=Decimal(str(round(confidence, 4))),
                    anomalyType=anomaly_type,
                    features="{}"
                ))
                
                processed_count += 1
                
            frame_number += 1
            
        cap.release()
        return frame_analyses
    
    def _detect_deepfake(self, frame_analyses):
        """전체 영상 딥페이크 판정 논리"""
        if not frame_analyses: 
            return False, 0.0
        
        avg_confidence = sum(float(f.confidenceScore) for f in frame_analyses) / len(frame_analyses)
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