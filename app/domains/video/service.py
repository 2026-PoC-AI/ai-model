import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import pytz 
from typing import List, Dict
import logging
from decimal import Decimal
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

# 한국 타임존 정의
KST = pytz.timezone('Asia/Seoul')

class VideoAnalysisService:
    def __init__(self, predictor):
        """
        Args:
            predictor: app.state.models["video"]에서 전달받은 EnsemblePredictor
        """
        self.upload_dir = Path("uploads/video")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictor = predictor
    
    async def analyze_video(self, file_content: bytes, filename: str, analysis_id: int) -> VideoAnalysisResponse:
        """영상 딥페이크 분석 메인 로직"""
        start_time = datetime.now(KST)  # 변경!
    
        temp_path = self.upload_dir / f"temp_{analysis_id}_{filename}"
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Video saved to {temp_path} (ID: {analysis_id})")

            def update_progress(progress: int, stage: str, detail: str, aid: int):
                from app.core.redis_client import redis_client
                redis_client.set_progress(aid, progress, stage, detail)
                logger.info(f"Progress updated - ID: {aid}, {progress}%, {stage}")
            
            result, error = self.predictor.predict_video(
                video_path=str(temp_path),
                sample_rate=5,
                aggregation='mean',
                progress_callback=update_progress,
                analysis_id=analysis_id
            )
            
            if error:
                raise Exception(f"Video prediction failed: {error}")
            
            logger.info(f"Analysis completed - {result['processed_frames']} frames analyzed (ID: {analysis_id})")
            
            frame_analyses = self._generate_frame_analyses_from_result(result)
            individual_models = self._convert_individual_models(result['individual_models'])
            detected_artifacts = self._convert_detected_artifacts(result['detected_artifacts'])
            
            processing_time = int((datetime.now(KST) - start_time).total_seconds() * 1000)  # 변경!
            
            summary = self._generate_summary(
                is_fake=result['is_fake'],
                fake_prob=result['fake_probability'],
                risk_level=result['risk_level'],
                model_agreement=result['model_agreement']
            )
            
            return VideoAnalysisResponse(
                analysisId=analysis_id,
                title=filename,
                status="COMPLETED",
                createdAt=start_time,  # 이미 KST 포함
                completedAt=datetime.now(KST),  # 변경!
                analysisResult=AnalysisResultData(
                    analysisId=analysis_id,
                    createdAt=start_time,  # 이미 KST 포함
                    confidenceScore=Decimal(str(round(result['fake_probability'], 4))),
                    isDeepfake=result['is_fake'],
                    modelVersion="ensemble-v1.0.0",
                    processingTimeMs=processing_time,
                    detectedTechniques=self._get_detected_techniques(result),
                    summary=summary,
                    analyzedAt=datetime.now(KST),  # 변경!
                    ensembleFakeProbability=Decimal(str(round(result['fake_probability'], 4))),
                    modelAgreement=Decimal(str(round(result['model_agreement'], 4))),
                    riskLevel=result['risk_level'],
                    individualModels=individual_models,
                    detectedArtifacts=detected_artifacts
                ),
                frameAnalyses=frame_analyses
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for ID {analysis_id}: {str(e)}")
            raise
        finally:
            if temp_path.exists():
                temp_path.unlink()
                logger.info(f"Temporary file deleted: {temp_path}")
    
    def _generate_frame_analyses_from_result(self, result: Dict) -> List[FrameAnalysisData]:
        """
        predictor 결과에서 프레임별 분석 데이터 생성
        """
        frame_analyses = []
        
        frame_details = result.get('frame_details', [])
        
        for frame_detail in frame_details:
            # 개별 모델 중 최대값 계산
            xception_prob = frame_detail['xception']['fake_probability']
            efficientnet_prob = frame_detail['efficientnet']['fake_probability']
            cnn_lstm_prob = frame_detail['cnn_lstm']['fake_probability'] if frame_detail['cnn_lstm'] else 0
            
            # 앙상블 확률과 최대 개별 확률 중 더 높은 값 사용
            ensemble_prob = frame_detail['ensemble_prob']
            max_individual_prob = max(xception_prob, efficientnet_prob, cnn_lstm_prob)
            display_prob = max(ensemble_prob, max_individual_prob)  # 더 보수적인 값
            
            # 이상 유형 판단
            anomaly_type = self._determine_anomaly_type_from_detail(frame_detail)
            
            frame_analysis = FrameAnalysisData(
                frameNumber=frame_detail['frame_number'],
                timestampSeconds=Decimal(str(round(frame_detail['timestamp'], 3))),
                isDeepfake=display_prob > 0.5,  # 표시 확률 기준
                confidenceScore=Decimal(str(round(display_prob, 4))),  # 더 높은 값 사용
                anomalyType=anomaly_type,
                features=None
            )
            
            frame_analyses.append(frame_analysis)
        
        logger.info(f"Generated {len(frame_analyses)} frame analyses")
        return frame_analyses
    
    def _determine_anomaly_type_from_detail(self, frame_detail: Dict) -> str:
        """프레임 상세 정보로부터 이상 유형 판단"""
        xception_prob = frame_detail['xception']['fake_probability']
        efficientnet_prob = frame_detail['efficientnet']['fake_probability']
        
        # CNN-LSTM은 시퀀스 단위라 None일 수 있음
        cnn_lstm_prob = frame_detail['cnn_lstm']['fake_probability'] if frame_detail['cnn_lstm'] else 0.5
        
        ensemble_prob = frame_detail['ensemble_prob']
        
        if ensemble_prob < 0.4:
            return "normal"
        
        # 가장 높은 점수를 준 모델 기준
        max_prob = max(xception_prob, efficientnet_prob, cnn_lstm_prob)
        
        if max_prob == xception_prob and xception_prob > 0.6:
            return "spatial"
        elif max_prob == cnn_lstm_prob and cnn_lstm_prob > 0.6:
            return "temporal"
        elif max_prob == efficientnet_prob and efficientnet_prob > 0.6:
            return "structural"
        elif ensemble_prob > 0.7:
            return "multiple"
        else:
            return "suspicious"
    
    def _convert_individual_models(self, individual_models: Dict) -> Dict[str, ModelPredictionData]:
        """
        predictor의 individual_models를 DTO로 변환
        """
        return {
            'xception': ModelPredictionData(
                modelName="XceptionNet",
                fakeProbability=Decimal(str(round(individual_models['xception']['fake_probability'], 4))),
                prediction=individual_models['xception']['prediction'],
                confidence=Decimal(str(round(individual_models['xception']['confidence'], 4))),
                detectedPatterns=individual_models['xception']['detected_patterns']
            ),
            'efficientnet': ModelPredictionData(
                modelName="EfficientNet-B4",
                fakeProbability=Decimal(str(round(individual_models['efficientnet']['fake_probability'], 4))),
                prediction=individual_models['efficientnet']['prediction'],
                confidence=Decimal(str(round(individual_models['efficientnet']['confidence'], 4))),
                detectedPatterns=individual_models['efficientnet']['detected_patterns']
            ),
            'cnn_lstm': ModelPredictionData(
                modelName="CNN-LSTM",
                fakeProbability=Decimal(str(round(individual_models['cnn_lstm']['fake_probability'], 4))),
                prediction=individual_models['cnn_lstm']['prediction'],
                confidence=Decimal(str(round(individual_models['cnn_lstm']['confidence'], 4))),
                detectedPatterns=individual_models['cnn_lstm']['detected_patterns'],
                suspiciousFrames=individual_models['cnn_lstm'].get('suspicious_frames', [])
            )
        }
    
    def _convert_detected_artifacts(self, detected_artifacts: Dict) -> DetectedArtifactsData:
        """
        predictor의 detected_artifacts를 DTO로 변환
        """
        return DetectedArtifactsData(
            spatial=ArtifactCategoryData(
                detected=detected_artifacts['spatial']['detected'],
                sources=detected_artifacts['spatial']['sources'],
                patterns=detected_artifacts['spatial']['patterns']
            ),
            temporal=ArtifactCategoryData(
                detected=detected_artifacts['temporal']['detected'],
                sources=detected_artifacts['temporal']['sources'],
                patterns=detected_artifacts['temporal']['patterns']
            ),
            structural=ArtifactCategoryData(
                detected=detected_artifacts['structural']['detected'],
                sources=detected_artifacts['structural']['sources'],
                patterns=detected_artifacts['structural']['patterns']
            )
        )
    
    def _get_detected_techniques(self, result: Dict) -> str:
        """탐지된 기법 문자열 생성"""
        artifacts = result['detected_artifacts']
        techniques = []
        
        if artifacts['spatial']['detected']:
            techniques.append("공간적 아티팩트")
        if artifacts['temporal']['detected']:
            techniques.append("시간적 불일치")
        if artifacts['structural']['detected']:
            techniques.append("구조적 이상")
        
        return ", ".join(techniques) if techniques else "none"
    
    def _generate_summary(self, is_fake: bool, fake_prob: float, risk_level: str, model_agreement: float) -> str:
        """분석 요약 생성"""
        if is_fake:
            return f"딥페이크로 판정 (신뢰도 {fake_prob*100:.1f}%, 모델 합의도 {model_agreement*100:.0f}%, 위험도 {risk_level})"
        else:
            return f"진짜 영상으로 판정 (신뢰도 {(1-fake_prob)*100:.1f}%, 모델 합의도 {model_agreement*100:.0f}%)"