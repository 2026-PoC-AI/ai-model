# app/domains/video/service.py
import os
import cv2
import logging
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import List
import numpy as np
from app.core.config import settings
from app.core.redis_client import redis_client
from .schemas import (
    VideoAnalysisResponse,
    VideoFileData,
    AnalysisResultData,
    FrameAnalysisData,
    ModelPredictionData,
    DetectedArtifactsData,
    ArtifactCategoryData
)

logger = logging.getLogger(__name__)

class VideoAnalysisService:
    def __init__(self, predictor):
        """
        Args:
            predictor: DeepfakeEnsemble 인스턴스
        """
        self.upload_dir = Path("uploads/video")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble = predictor
    
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
        """영상 딥페이크 분석 메인 로직 (3-Model Ensemble)"""
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
        
        # 16프레임 시퀀스 추출 (CNN-LSTM용)
        self._update_progress(analysis_id, 20, "frame_extraction", "프레임 시퀀스 추출 중")
        frame_sequence = self._extract_frame_sequence(file_path, num_frames=16)
        
        # 중간 프레임 추출 (XceptionNet, EfficientNet용)
        self._update_progress(analysis_id, 30, "frame_extraction", "대표 프레임 추출 중")
        representative_frame = self._extract_representative_frame(file_path)
        
        # 앙상블 예측
        self._update_progress(analysis_id, 40, "ai_analysis", "XceptionNet 분석 중")
        self._update_progress(analysis_id, 55, "ai_analysis", "EfficientNet-B4 분석 중")
        self._update_progress(analysis_id, 70, "ai_analysis", "CNN-LSTM 시간적 분석 중")
        
        ensemble_result = self.ensemble.predict_ensemble(
            representative_frame,
            frame_sequence
        )
        
        self._update_progress(analysis_id, 85, "ai_analysis", "앙상블 결과 통합 중")
        
        # 프레임별 상세 분석 (UI 표시용)
        frame_analyses = self._create_frame_analyses(
            video_info,
            ensemble_result
        )
        
        self._update_progress(analysis_id, 90, "result_generation", "결과 생성 중")
        
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
                confidenceScore=Decimal(str(round(ensemble_result['ensemble_confidence'], 4))),
                isDeepfake=(ensemble_result['final_prediction'] == 'fake'),
                modelVersion="ensemble-v1.0.0",
                processingTimeMs=processing_time_ms,
                detectedTechniques=self._format_detected_techniques(ensemble_result['detected_artifacts']),
                summary=self._generate_summary(ensemble_result),
                analyzedAt=datetime.now(timezone.utc),
                # 새로운 필드 추가
                ensembleFakeProbability=Decimal(str(round(ensemble_result['ensemble_fake_probability'], 4))),
                modelAgreement=Decimal(str(round(ensemble_result['model_agreement'], 4))),
                riskLevel=ensemble_result['risk_level'],
                individualModels=self._format_individual_models(ensemble_result['individual_models']),
                detectedArtifacts=self._format_detected_artifacts(ensemble_result['detected_artifacts'])
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
        return {
            'fps': fps, 
            'duration': duration, 
            'resolution': f"{width}X{height}", 
            'frame_count': frame_count
        }
    
    def _extract_frame_sequence(self, file_path: str, num_frames: int = 16) -> np.ndarray:
        """16프레임 균등 샘플링 (CNN-LSTM용)"""
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 균등 간격으로 프레임 인덱스 계산
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 112x112로 리사이즈 (CNN-LSTM 입력 크기)
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # (16, 112, 112, 3) 형태로 반환
        return np.array(frames)
    
    def _extract_representative_frame(self, file_path: str) -> np.ndarray:
        """중간 프레임 추출 (XceptionNet, EfficientNet용)"""
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 중간 프레임
        mid_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # 299x299로 리사이즈 (XceptionNet, EfficientNet 입력 크기)
            frame = cv2.resize(frame, (299, 299))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        raise ValueError("프레임 추출 실패")
    
    def _create_frame_analyses(self, video_info: dict, ensemble_result: dict) -> List[FrameAnalysisData]:
        """프레임별 분석 데이터 생성 (UI 표시용)"""
        frame_analyses = []
        
        # CNN-LSTM의 의심 프레임 정보 추출
        suspicious_frames = ensemble_result['individual_models']['cnn_lstm'].get('suspicious_frames', [])
        attention_weights = ensemble_result['individual_models']['cnn_lstm'].get('attention_weights', [])
        
        fps = video_info['fps']
        
        # 16프레임에 대한 정보 생성
        for i in range(16):
            is_suspicious = i in suspicious_frames
            confidence = attention_weights[i] if i < len(attention_weights) else 0.5
            
            # 실제 프레임 번호 계산 (균등 샘플링 기준)
            actual_frame_number = int((i / 16) * video_info['frame_count'])
            
            frame_analyses.append(FrameAnalysisData(
                frameNumber=actual_frame_number,
                timestampSeconds=Decimal(str(round(actual_frame_number / fps, 2))),
                isDeepfake=is_suspicious,
                confidenceScore=Decimal(str(round(confidence, 4))),
                anomalyType=self._determine_anomaly_type(is_suspicious, ensemble_result),
                features="{}"
            ))
        
        return frame_analyses
    
    def _determine_anomaly_type(self, is_suspicious: bool, ensemble_result: dict) -> str:
        """이상 유형 결정"""
        if not is_suspicious:
            return "normal"
        
        artifacts = ensemble_result['detected_artifacts']
        
        if artifacts['temporal']['detected']:
            return "temporal_inconsistency"
        elif artifacts['spatial']['detected']:
            return "spatial_artifact"
        elif artifacts['structural']['detected']:
            return "structural_anomaly"
        
        return "unknown"
    
    def _format_detected_techniques(self, artifacts: dict) -> str:
        """탐지된 기법 목록 (기존 포맷 유지)"""
        techniques = []
        
        if artifacts['spatial']['detected']:
            techniques.append("공간적 아티팩트")
        if artifacts['temporal']['detected']:
            techniques.append("시간적 불일치")
        if artifacts['structural']['detected']:
            techniques.append("구조적 이상")
        
        return ", ".join(techniques) if techniques else "none"
    
    def _generate_summary(self, ensemble_result: dict) -> str:
        """분석 결과 요약 생성"""
        prediction = ensemble_result['final_prediction']
        confidence = ensemble_result['ensemble_confidence'] * 100
        agreement = ensemble_result['model_agreement'] * 100
        risk = ensemble_result['risk_level']
        
        if prediction == 'fake':
            return (
                f"딥페이크로 판정 (신뢰도 {confidence:.1f}%, "
                f"모델 합의도 {agreement:.0f}%, 위험도 {risk})"
            )
        else:
            return (
                f"정상 영상으로 판정 (신뢰도 {confidence:.1f}%, "
                f"모델 합의도 {agreement:.0f}%)"
            )
    
    def _format_individual_models(self, models: dict) -> dict:
        """개별 모델 예측 결과 포맷팅"""
        return {
            'xception': ModelPredictionData(
                modelName='XceptionNet',
                fakeProbability=Decimal(str(round(models['xception']['fake_probability'], 4))),
                prediction=models['xception']['prediction'],
                confidence=Decimal(str(round(models['xception']['confidence'], 4))),
                detectedPatterns=models['xception']['detected_patterns']
            ),
            'efficientnet': ModelPredictionData(
                modelName='EfficientNet-B4',
                fakeProbability=Decimal(str(round(models['efficientnet']['fake_probability'], 4))),
                prediction=models['efficientnet']['prediction'],
                confidence=Decimal(str(round(models['efficientnet']['confidence'], 4))),
                detectedPatterns=models['efficientnet']['detected_patterns']
            ),
            'cnn_lstm': ModelPredictionData(
                modelName='CNN-LSTM',
                fakeProbability=Decimal(str(round(models['cnn_lstm']['fake_probability'], 4))),
                prediction=models['cnn_lstm']['prediction'],
                confidence=Decimal(str(round(models['cnn_lstm']['confidence'], 4))),
                detectedPatterns=models['cnn_lstm']['detected_patterns'],
                suspiciousFrames=models['cnn_lstm']['suspicious_frames']
            )
        }
    
    def _format_detected_artifacts(self, artifacts: dict) -> DetectedArtifactsData:
        """탐지된 아티팩트 포맷팅"""
        return DetectedArtifactsData(
            spatial=ArtifactCategoryData(
                detected=artifacts['spatial']['detected'],
                sources=artifacts['spatial']['sources'],
                patterns=artifacts['spatial']['patterns']
            ),
            temporal=ArtifactCategoryData(
                detected=artifacts['temporal']['detected'],
                sources=artifacts['temporal']['sources'],
                patterns=artifacts['temporal']['patterns']
            ),
            structural=ArtifactCategoryData(
                detected=artifacts['structural']['detected'],
                sources=artifacts['structural']['sources'],
                patterns=artifacts['structural']['patterns']
            )
        )