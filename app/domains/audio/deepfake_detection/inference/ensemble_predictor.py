import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import time

from .predictor import AudioPredictor
from .deepfake_classifier import DeepfakeMethodClassifier

class EnsemblePredictor:
    """
    Mel-spectrogram CNN + LFCC CNN 앙상블
    """
    
    def __init__(
        self,
        mel_predictor: AudioPredictor,
        lfcc_predictor: AudioPredictor,
        ensemble_method: str = 'weighted_avg',
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            mel_predictor: Mel-spectrogram CNN 예측기
            lfcc_predictor: LFCC CNN 예측기
            ensemble_method: 앙상블 방법 ('avg', 'weighted_avg', 'max')
            weights: weighted_avg 사용 시 가중치 [mel_weight, lfcc_weight]
        """
        self.mel_predictor = mel_predictor
        self.lfcc_predictor = lfcc_predictor
        self.ensemble_method = ensemble_method
        
        # 기본 가중치: 검증 정확도 기반
        if weights is None:
            self.weights = [0.49, 0.51]  # mel: 99.15%, lfcc: 99.57%
        else:
            self.weights = weights
        
        # 딥페이크 생성 기술 분류기
        self.classifier = DeepfakeMethodClassifier()
        
        self.model_version = "ensemble_v1.0"
    
    def predict(self, audio_path: str, detailed: bool = False) -> Dict:
        """
        앙상블 예측
        """
        # 각 모델 예측
        mel_result = self.mel_predictor.predict(audio_path)
        lfcc_result = self.lfcc_predictor.predict(audio_path)
        
        # 앙상블
        if self.ensemble_method == 'weighted_avg':
            fake_prob = (
                mel_result['probabilities']['fake'] * self.weights[0] +
                lfcc_result['probabilities']['fake'] * self.weights[1]
            )
        elif self.ensemble_method == 'avg':
            fake_prob = (
                mel_result['probabilities']['fake'] +
                lfcc_result['probabilities']['fake']
            ) / 2
        elif self.ensemble_method == 'max':
            fake_prob = max(
                mel_result['probabilities']['fake'],
                lfcc_result['probabilities']['fake']
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        real_prob = 1.0 - fake_prob
        prediction = 'fake' if fake_prob > 0.5 else 'real'
        confidence = fake_prob if prediction == 'fake' else real_prob
        
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                'real': float(real_prob),
                'fake': float(fake_prob)
            },
            'model_outputs': {
                'mel': mel_result['probabilities'],
                'lfcc': lfcc_result['probabilities']
            },
            'model_version': self.model_version
        }
        
        # 딥페이크 생성 기술 분류 추가
        print(f"[DEBUG] detailed={detailed}, prediction={prediction}")  # 디버그 로그
        
        if detailed and prediction == 'fake':
            try:
                print(f"[DEBUG] Starting method analysis...")  # 디버그 로그
                method_analysis = self.classifier.analyze(
                    audio_path,
                    mel_pred=mel_result['probabilities']['fake'],
                    lfcc_pred=lfcc_result['probabilities']['fake']
                )
                print(f"[DEBUG] Method analysis result: {method_analysis}")  # 디버그 로그
                result.update(method_analysis)
            except Exception as e:
                print(f"[ERROR] Failed to analyze deepfake method: {e}")  # 에러 로그
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG] Skipping detailed analysis: detailed={detailed}, prediction={prediction}")
        
        return result
    
    def batch_predict(
        self,
        audio_paths: List[str],
        detailed: bool = False
    ) -> List[Dict]:
        """
        배치 예측
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, detailed=detailed)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'file': audio_path
                })
        
        return results


def get_ensemble_predictor(
    weights_dir: str = 'weights',
    device: str = 'cpu',
    ensemble_method: str = 'weighted_avg'
) -> EnsemblePredictor:
    """
    앙상블 예측기 생성
    """
    from ..models.spectrogram_cnn import LightweightAudioCNN
    from ..models.lfcc_resnet import LightweightLFCCCNN
    
    weights_dir = Path(weights_dir)
    
    # Mel-spectrogram CNN
    mel_model = LightweightAudioCNN(num_classes=2)
    mel_weights = weights_dir / 'audio_cnn' / 'best_model_latest.pth'
    
    if not mel_weights.exists():
        raise FileNotFoundError(f"Mel CNN weights not found: {mel_weights}")
    
    # 체크포인트에서 model_state_dict 추출
    checkpoint = torch.load(mel_weights, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        mel_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        mel_model.load_state_dict(checkpoint)
    
    print(f"Loaded Mel CNN: {mel_weights}")
    
    mel_predictor = AudioPredictor(
        model=mel_model,
        device=device,
        feature_type='mel'
    )
    
    # LFCC CNN
    lfcc_model = LightweightLFCCCNN(num_classes=2)
    lfcc_weights = weights_dir / 'lfcc_cnn' / 'best_lfcc_model_20260124_101858.pth'
    
    if not lfcc_weights.exists():
        lfcc_dir = weights_dir / 'lfcc_cnn'
        if lfcc_dir.exists():
            lfcc_files = sorted(lfcc_dir.glob('best_lfcc_model_*.pth'))
            if lfcc_files:
                lfcc_weights = lfcc_files[-1]
                print(f"Using latest LFCC weights: {lfcc_weights}")
            else:
                raise FileNotFoundError(f"No LFCC weights found in: {lfcc_dir}")
        else:
            raise FileNotFoundError(f"LFCC weights directory not found: {lfcc_dir}")
    
    # 체크포인트에서 model_state_dict 추출
    checkpoint = torch.load(lfcc_weights, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        lfcc_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        lfcc_model.load_state_dict(checkpoint)
    
    print(f"Loaded LFCC CNN: {lfcc_weights}")
    
    lfcc_predictor = AudioPredictor(
        model=lfcc_model,
        device=device,
        feature_type='lfcc'
    )
    
    # 앙상블
    ensemble = EnsemblePredictor(
        mel_predictor=mel_predictor,
        lfcc_predictor=lfcc_predictor,
        ensemble_method=ensemble_method
    )
    
    return ensemble