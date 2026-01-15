import os
import cv2
import uuid
from datetime import datetime
from decimal import Decimal
from typing import List
from .schemas import (
    VideoAnalysisResponse,
    VideoFileData,
    AnalysisResultData,
    FrameAnalysisData
)
from .model import load_video_model
# from app.domains.video.deepfake_detection.inference.predictor import DeepfakePredictor

class VideoService:
    def __init__(self):
        self.deepfake_predictor = DeepfakePredictor(
            model_path='video/deepfake_detection/weights/best_model.pth'
        )
    
    async def detect_deepfake(self, video_path: str):
        result, error = self.deepfake_predictor.predict_video(video_path)
        return result

class VideoAnalysisService:
    
    def __init__(self):
        self.upload_dir = "uploads/videos"
        os.makedirs(self.upload_dir, exist_ok=True)
        self.model = load_video_model()  # 모델 로드
    
    async def analyze_video(self, file, filename: str) -> VideoAnalysisResponse:
        """영상 딥페이크 분석 메인 로직"""
        start_time = datetime.now()
    
        # 1. ID 생성
        analysis_id = str(uuid.uuid4())
        file_id = str(uuid.uuid4())
        result_id = str(uuid.uuid4())
        
        # 2. 파일 저장
        file_extension = filename.split('.')[-1]
        stored_filename = f"{analysis_id}_{int(datetime.now().timestamp())}.{file_extension}"
        file_path = os.path.join(self.upload_dir, stored_filename)
        
        # 수정: await 제거
        content = file.read()  # await 제거!
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # 3. 영상 메타데이터 추출
        video_info = self._extract_video_info(file_path)
        
        # 4. 프레임 분석 (모델 사용)
        frame_analyses = self._analyze_frames(file_path, video_info['fps'])
        
        # 5. 딥페이크 판정
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
                fileId=file_id,
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
                resultId=result_id,
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
        
        return {
            'fps': fps,
            'duration': duration,
            'resolution': f"{width}X{height}",
            'frame_count': frame_count
        }
    
    def _analyze_frames(self, file_path: str, fps: float) -> List[FrameAnalysisData]:
        """프레임별 분석"""
        cap = cv2.VideoCapture(file_path)
        frame_analyses = []
        frame_number = 0
        
        # MVP: 1초에 1프레임만 샘플링
        sample_interval = int(fps) if fps > 0 else 30
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % sample_interval == 0:
                # 모델을 사용한 프레임 분석
                is_suspicious, confidence, anomaly_type = self.model.analyze_frame(frame)
                
                frame_analyses.append(FrameAnalysisData(
                    frameId=str(uuid.uuid4()),
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
    
    def _detect_deepfake(self, frame_analyses: List[FrameAnalysisData]) -> tuple:
        """전체 영상 딥페이크 판정"""
        if not frame_analyses:
            return False, 0.0
        
        # 의심 프레임 비율 계산
        suspicious_count = sum(1 for f in frame_analyses if f.isDeepfake)
        confidence = suspicious_count / len(frame_analyses)
        
        # 30% 이상 의심 프레임이면 딥페이크로 판정
        is_deepfake = confidence > 0.3
        
        return is_deepfake, confidence
    
    def _get_detected_techniques(self, frames: List[FrameAnalysisData]) -> str:
        """탐지된 딥페이크 기법 목록"""
        techniques = set()
        for frame in frames:
            if frame.isDeepfake and frame.anomalyType != "normal":
                techniques.add(frame.anomalyType)
        
        return ", ".join(sorted(techniques)) if techniques else "none"
    
    def _generate_summary(self, is_deepfake: bool, confidence: float, frames: List[FrameAnalysisData]) -> str:
        """분석 결과 요약 생성"""
        if is_deepfake:
            suspicious_frames = [f for f in frames if f.isDeepfake]
            techniques = self._get_detected_techniques(frames)
            return f"딥페이크 가능성 {confidence*100:.1f}% - {len(suspicious_frames)}개 프레임에서 의심 패턴 발견 ({techniques})"
        else:
            return f"정상 영상으로 판단 (신뢰도 {(1-confidence)*100:.1f}%)"