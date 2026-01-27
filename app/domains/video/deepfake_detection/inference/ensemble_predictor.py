# app/domains/video/deepfake_detection/inference/ensemble_predictor.py

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from PIL import Image

from ..models.ensemble import DeepfakeEnsemble
from ..preprocessing.face_detector import FaceDetector
from ..preprocessing.dataset import get_transforms

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """
    3-모델 앙상블 기반 딥페이크 탐지 추론 클래스
    """
    def __init__(self, weights_dir: str, device: str = 'cuda'):
        """
        Args:
            weights_dir: 가중치 파일 디렉토리 경로
            device: 사용할 디바이스
        """
        self.device = torch.device(device)
        
        logger.info(f"Loading DeepfakeEnsemble from {weights_dir}")
        
        # DeepfakeEnsemble 로드
        self.ensemble = DeepfakeEnsemble(weights_dir=weights_dir, device=device)
        
        # Face Detector 및 Transform
        self.face_detector = FaceDetector(device=device)
        self.transform = get_transforms('val')
        
        logger.info("Ensemble model loaded successfully (XceptionNet + EfficientNet + CNN-LSTM)")
        
        # 모델 로드 검증 로그
        logger.info(f"Device: {self.device}")
        logger.info(f"Weights directory: {weights_dir}")
    
    def predict_video(
        self, 
        video_path: str, 
        sample_rate: int = 5, 
        aggregation: str = 'mean',
        progress_callback=None, 
        analysis_id=None
    ) -> Tuple[Dict, str]:
        """
        비디오에 대한 3-모델 앙상블 예측
        
        Args:
            video_path: 비디오 파일 경로
            sample_rate: 프레임 샘플링 비율
            aggregation: 집계 방식
            progress_callback: 진행률 콜백
            analysis_id: 분석 ID
        
        Returns:
            (result_dict, error_message)
        """
        logger.info(f"Starting video prediction: {video_path}")
        logger.info(f"Sample rate: {sample_rate}, Aggregation: {aggregation}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, f"Failed to open video: {video_path}"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps}")
        
        if progress_callback:
            progress_callback(0, "video_upload", "영상 업로드 완료", analysis_id)
        
        # 프레임별 예측 결과 저장
        frame_predictions = []
        frame_details = []
        
        # CNN-LSTM용 시퀀스 버퍼 (16프레임)
        sequence_buffer = []
        sequence_predictions = []
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # 샘플링
            if frame_count % sample_rate != 0:
                continue
            
            current_progress = int((frame_count / total_frames) * 100)
            
            # 얼굴 검출
            faces = self.face_detector.detect_and_align(frame)
            
            if len(faces) == 0:
                logger.debug(f"No face detected in frame {frame_count}")
                continue
            
            face = faces[0]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # 전처리된 텐서 정보 로그
            if processed_count == 0:
                logger.info(f"First face tensor shape: {face_tensor.shape}")
                logger.info(f"First face tensor dtype: {face_tensor.dtype}")
                logger.info(f"First face tensor range: [{face_tensor.min():.4f}, {face_tensor.max():.4f}]")
                logger.info(f"First face tensor mean: {face_tensor.mean():.4f}, std: {face_tensor.std():.4f}")
            
            # 단일 프레임 예측 (XceptionNet, EfficientNet)
            single_result = self.ensemble.predict_single_frame(face_tensor)
            
            # 첫 프레임 예측 결과 상세 로그
            if processed_count == 0:
                logger.info("=== First Frame Prediction Results ===")
                logger.info(f"XceptionNet - fake_prob: {single_result['xception']['fake_probability']:.4f}")
                logger.info(f"EfficientNet - fake_prob: {single_result['efficientnet']['fake_probability']:.4f}")
            
            # 5프레임마다 예측 결과 로그
            if processed_count % 5 == 0:
                logger.debug(f"Frame {frame_count} - Xception: {single_result['xception']['fake_probability']:.4f}, "
                           f"EfficientNet: {single_result['efficientnet']['fake_probability']:.4f}")
            
            # 시퀀스 버퍼에 추가 (CNN-LSTM용)
            sequence_buffer.append(face_tensor)
            if len(sequence_buffer) > 16:
                sequence_buffer.pop(0)
            
            # 16프레임이 모이면 CNN-LSTM 예측
            cnn_lstm_result = None
            if len(sequence_buffer) == 16:
                sequence_tensor = torch.cat(sequence_buffer, dim=0).unsqueeze(0)
                
                if len(sequence_predictions) == 0:
                    logger.info(f"First sequence tensor shape: {sequence_tensor.shape}")
                
                cnn_lstm_result = self.ensemble.predict_sequence(sequence_tensor)
                
                if len(sequence_predictions) == 0:
                    logger.info(f"CNN-LSTM first prediction - fake_prob: {cnn_lstm_result['cnn_lstm']['fake_probability']:.4f}")
                
                sequence_predictions.append(cnn_lstm_result['cnn_lstm'])
            
            # 프레임별 앙상블 점수 계산
            if cnn_lstm_result:
                frame_ensemble_prob = (
                    single_result['xception']['fake_probability'] * 0.4 +
                    single_result['efficientnet']['fake_probability'] * 0.3 +
                    cnn_lstm_result['cnn_lstm']['fake_probability'] * 0.3
                )
                
                if processed_count == 0 or processed_count % 10 == 0:
                    logger.debug(f"Frame {frame_count} ensemble (3 models): {frame_ensemble_prob:.4f}")
            else:
                # CNN-LSTM 결과 없으면 2개 모델만 사용
                frame_ensemble_prob = (
                    single_result['xception']['fake_probability'] * 0.57 +
                    single_result['efficientnet']['fake_probability'] * 0.43
                )
                
                if processed_count == 0:
                    logger.debug(f"Frame {frame_count} ensemble (2 models): {frame_ensemble_prob:.4f}")
            
            frame_predictions.append(frame_ensemble_prob)
            
            # 프레임 상세 정보 저장
            frame_details.append({
                'frame_number': frame_count - 1,
                'timestamp': (frame_count - 1) / fps if fps > 0 else 0,
                'xception': single_result['xception'],
                'efficientnet': single_result['efficientnet'],
                'cnn_lstm': cnn_lstm_result['cnn_lstm'] if cnn_lstm_result else None,
                'ensemble_prob': frame_ensemble_prob
            })
            
            processed_count += 1
            
            if progress_callback and processed_count % 5 == 0:
                progress_callback(
                    current_progress, 
                    "ai_analysis", 
                    f"AI 모델 분석 중 ({processed_count}개 프레임 완료)",
                    analysis_id
                )
        
        cap.release()
        
        logger.info(f"Video processing completed - Processed {processed_count} frames out of {frame_count} total frames")
        
        if len(frame_predictions) == 0:
            return None, "No faces detected in video frames"
        
        if progress_callback:
            progress_callback(95, "result_generation", "결과 생성 중", analysis_id)
        
        # 전체 비디오 수준 집계
        aggregated_prob = self._aggregate_predictions(frame_predictions, method=aggregation)
        
        logger.info(f"Aggregated probability ({aggregation}): {aggregated_prob:.4f}")
        
        # 개별 모델 평균 계산
        xception_avg = np.mean([f['xception']['fake_probability'] for f in frame_details])
        efficientnet_avg = np.mean([f['efficientnet']['fake_probability'] for f in frame_details])
        cnn_lstm_avg = np.mean([sp['fake_probability'] for sp in sequence_predictions]) if sequence_predictions else 0.5
        
        logger.info("=== Individual Model Averages ===")
        logger.info(f"XceptionNet average: {xception_avg:.4f}")
        logger.info(f"EfficientNet average: {efficientnet_avg:.4f}")
        logger.info(f"CNN-LSTM average: {cnn_lstm_avg:.4f}")
        
        # 모델 합의도 계산
        predictions = [
            1 if xception_avg > 0.5 else 0,
            1 if efficientnet_avg > 0.5 else 0,
            1 if cnn_lstm_avg > 0.5 else 0
        ]
        model_agreement = sum(predictions) / 3.0
        
        logger.info(f"Model predictions: {predictions}")
        logger.info(f"Model agreement: {model_agreement:.4f}")
        
        # 의심 프레임 추출 (CNN-LSTM attention 기반)
        suspicious_frames = []
        if sequence_predictions:
            for sp in sequence_predictions:
                if 'suspicious_frames' in sp:
                    suspicious_frames.extend(sp['suspicious_frames'])
        
        logger.info(f"Total suspicious frames: {len(suspicious_frames)}")
        
        result = {
            'is_fake': aggregated_prob > 0.5,
            'fake_probability': aggregated_prob,
            'real_probability': 1 - aggregated_prob,
            'confidence': max(aggregated_prob, 1 - aggregated_prob),
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'frame_predictions': frame_predictions,
            'frame_details': frame_details,
            
            # 개별 모델 결과
            'individual_models': {
                'xception': {
                    'prediction': 'fake' if xception_avg > 0.5 else 'real',
                    'confidence': float(xception_avg),
                    'fake_probability': float(xception_avg),
                    'detected_patterns': self._get_patterns('xception', xception_avg)
                },
                'efficientnet': {
                    'prediction': 'fake' if efficientnet_avg > 0.5 else 'real',
                    'confidence': float(efficientnet_avg),
                    'fake_probability': float(efficientnet_avg),
                    'detected_patterns': self._get_patterns('efficientnet', efficientnet_avg)
                },
                'cnn_lstm': {
                    'prediction': 'fake' if cnn_lstm_avg > 0.5 else 'real',
                    'confidence': float(cnn_lstm_avg),
                    'fake_probability': float(cnn_lstm_avg),
                    'detected_patterns': self._get_patterns('cnn_lstm', cnn_lstm_avg),
                    'suspicious_frames': suspicious_frames
                }
            },
            
            'model_agreement': model_agreement,
            'risk_level': self._calculate_risk_level(aggregated_prob, model_agreement),
            'detected_artifacts': self._get_artifacts(xception_avg, efficientnet_avg, cnn_lstm_avg)
        }
        
        logger.info(f"Final prediction - is_fake: {result['is_fake']}, confidence: {result['confidence']:.4f}")
        logger.info(f"Risk level: {result['risk_level']}")
        
        if progress_callback:
            progress_callback(100, "completed", "분석 완료", analysis_id)
        
        return result, None
    
    def _aggregate_predictions(self, predictions: List[float], method: str = 'mean') -> float:
        """프레임별 예측 집계"""
        if method == 'mean':
            return float(np.mean(predictions))
        elif method == 'max':
            return float(np.max(predictions))
        elif method == 'median':
            return float(np.median(predictions))
        else:
            return float(np.mean(predictions))
    
    def _get_patterns(self, model_name: str, prob: float) -> List[str]:
        """모델별 탐지 패턴"""
        if model_name == 'xception':
            if prob > 0.7:
                return ["얼굴 경계의 부자연스러운 블렌딩 감지", "피부 텍스처의 비정상적 매끄러움"]
            elif prob > 0.5:
                return ["미세한 공간적 아티팩트 감지"]
        elif model_name == 'efficientnet':
            if prob > 0.7:
                return ["다층 스케일에서 구조적 불일치 감지", "조명과 그림자의 불일치"]
            elif prob > 0.5:
                return ["전체적 특징 분포 이상"]
        elif model_name == 'cnn_lstm':
            if prob > 0.7:
                return ["프레임 간 불연속적 변화 감지", "시간적 일관성 결여", "비정상적인 움직임 패턴"]
            elif prob > 0.5:
                return ["시간적 아티팩트 감지"]
        return []
    
    def _calculate_risk_level(self, prob: float, agreement: float) -> str:
        """위험도 레벨 계산"""
        if prob > 0.8 and agreement >= 0.67:
            return 'HIGH'
        elif prob > 0.6 or (prob > 0.5 and agreement >= 0.67):
            return 'MEDIUM'
        elif prob > 0.5:
            return 'LOW'
        else:
            return 'SAFE'
    
    def _get_artifacts(self, xception: float, efficientnet: float, cnn_lstm: float) -> Dict:
        """탐지된 아티팩트"""
        return {
            'spatial': {
                'detected': xception > 0.5,
                'sources': ['XceptionNet'] if xception > 0.5 else [],
                'patterns': self._get_patterns('xception', xception)
            },
            'structural': {
                'detected': efficientnet > 0.5,
                'sources': ['EfficientNet-B4'] if efficientnet > 0.5 else [],
                'patterns': self._get_patterns('efficientnet', efficientnet)
            },
            'temporal': {
                'detected': cnn_lstm > 0.5,
                'sources': ['CNN-LSTM'] if cnn_lstm > 0.5 else [],
                'patterns': self._get_patterns('cnn_lstm', cnn_lstm)
            }
        }