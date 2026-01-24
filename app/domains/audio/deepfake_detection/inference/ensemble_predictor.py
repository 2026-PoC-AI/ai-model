import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from models.spectrogram_cnn import get_model as get_mel_model
from models.lfcc_resnet import get_lfcc_model
from preprocessing.audio_preprocessor import AudioPreprocessor
from preprocessing.lfcc_preprocessor import LFCCPreprocessor
from utils.audio_analyzer import AudioAnalyzer


class AudioEnsemblePredictor:
    """
    Mel-spectrogram + LFCC 앙상블 예측기
    
    두 모델의 예측을 결합하여 최종 판단
    """
    def __init__(
        self,
        mel_model_path: str,
        lfcc_model_path: str,
        device: str = 'cpu',
        ensemble_method: str = 'weighted_avg'
    ):
        self.device = torch.device(device)
        self.ensemble_method = ensemble_method
        
        # 전처리기
        self.mel_preprocessor = AudioPreprocessor()
        self.lfcc_preprocessor = LFCCPreprocessor()
        
        # 분석기
        self.analyzer = AudioAnalyzer(sample_rate=16000)
        
        # 모델 로드
        print("Loading models...")
        self.mel_model = self._load_mel_model(mel_model_path)
        self.lfcc_model = self._load_lfcc_model(lfcc_model_path)
        print("Models loaded successfully!")
        
        # 모델별 가중치 (검증 정확도 기반)
        self.weights = {
            'mel': 0.49,    # 99.15%
            'lfcc': 0.51    # 99.57%
        }
    
    def _load_mel_model(self, model_path: str):
        """
        Mel-spectrogram CNN 모델 로드
        """
        model = get_mel_model('lightweight_cnn', num_classes=2, dropout=0.3)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        model = model.to(self.device)
        
        print(f"  ✓ Mel-spectrogram CNN loaded (Val Acc: 99.15%)")
        return model
    
    def _load_lfcc_model(self, model_path: str):
        """
        LFCC CNN 모델 로드
        """
        model = get_lfcc_model('lightweight_lfcc', num_classes=2, dropout=0.3)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        model = model.to(self.device)
        
        print(f"  ✓ LFCC CNN loaded (Val Acc: 99.57%)")
        return model
    
    def _predict_single_model(self, model, input_tensor) -> Dict[str, float]:
        """
        단일 모델 예측
        
        Returns:
            {'real': float, 'fake': float}
        """
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            
            real_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()
            
            return {
                'real': real_prob,
                'fake': fake_prob
            }
    
    def predict(self, audio_path: str, detailed: bool = True) -> Dict:
        """
        앙상블 예측
        
        Args:
            audio_path: 오디오 파일 경로
            detailed: 상세 분석 포함 여부
            
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float,
                'probabilities': {
                    'real': float,
                    'fake': float
                },
                'model_outputs': {
                    'mel': {'real': float, 'fake': float},
                    'lfcc': {'real': float, 'fake': float}
                },
                'analysis': {...},  # detailed=True일 때만
                'deepfake_indicators': [...]  # detailed=True일 때만
            }
        """
        # Mel-spectrogram 예측
        mel_spec = self.mel_preprocessor.preprocess(audio_path)
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        mel_output = self._predict_single_model(self.mel_model, mel_tensor)
        
        # LFCC 예측
        lfcc_spec = self.lfcc_preprocessor.preprocess(audio_path)
        lfcc_tensor = torch.from_numpy(lfcc_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        lfcc_output = self._predict_single_model(self.lfcc_model, lfcc_tensor)
        
        # 앙상블
        if self.ensemble_method == 'weighted_avg':
            ensemble_real = (
                mel_output['real'] * self.weights['mel'] +
                lfcc_output['real'] * self.weights['lfcc']
            )
            ensemble_fake = (
                mel_output['fake'] * self.weights['mel'] +
                lfcc_output['fake'] * self.weights['lfcc']
            )
        elif self.ensemble_method == 'average':
            ensemble_real = (mel_output['real'] + lfcc_output['real']) / 2
            ensemble_fake = (mel_output['fake'] + lfcc_output['fake']) / 2
        elif self.ensemble_method == 'voting':
            mel_pred = 'fake' if mel_output['fake'] > 0.5 else 'real'
            lfcc_pred = 'fake' if lfcc_output['fake'] > 0.5 else 'real'
            
            if mel_pred == lfcc_pred:
                ensemble_real = (mel_output['real'] + lfcc_output['real']) / 2
                ensemble_fake = (mel_output['fake'] + lfcc_output['fake']) / 2
            else:
                if max(mel_output.values()) > max(lfcc_output.values()):
                    ensemble_real = mel_output['real']
                    ensemble_fake = mel_output['fake']
                else:
                    ensemble_real = lfcc_output['real']
                    ensemble_fake = lfcc_output['fake']
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # 최종 판단
        prediction = 'fake' if ensemble_fake > 0.5 else 'real'
        confidence = max(ensemble_fake, ensemble_real)
        
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                'real': float(ensemble_real),
                'fake': float(ensemble_fake)
            },
            'model_outputs': {
                'mel': {
                    'real': float(mel_output['real']),
                    'fake': float(mel_output['fake'])
                },
                'lfcc': {
                    'real': float(lfcc_output['real']),
                    'fake': float(lfcc_output['fake'])
                }
            }
        }
        
        # 상세 분석 추가
        if detailed:
            detailed_analysis = self.analyzer.full_analysis(
                audio_path,
                mel_probs=mel_output,
                lfcc_probs=lfcc_output
            )
            
            result['analysis'] = {
                'suspicious_frequency_bands': detailed_analysis['frequency_analysis']['suspicious_frequency_bands'],
                'band_energies': detailed_analysis['frequency_analysis']['band_energies'],
                'spectral_flux': detailed_analysis['frequency_analysis']['spectral_flux'],
                'phase_coherence': detailed_analysis['phase_analysis']['phase_coherence'],
                'phase_stability': detailed_analysis['phase_analysis']['phase_stability'],
                'suspicious_time_segments': detailed_analysis['suspicious_time_segments']
            }
            
            result['deepfake_indicators'] = detailed_analysis['deepfake_indicators']
        
        return result
        """
        앙상블 예측
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float,
                'probabilities': {
                    'real': float,
                    'fake': float
                },
                'model_outputs': {
                    'mel': {'real': float, 'fake': float},
                    'lfcc': {'real': float, 'fake': float}
                }
            }
        """
        # Mel-spectrogram 예측
        mel_spec = self.mel_preprocessor.preprocess(audio_path)
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        mel_output = self._predict_single_model(self.mel_model, mel_tensor)
        
        # LFCC 예측
        lfcc_spec = self.lfcc_preprocessor.preprocess(audio_path)
        lfcc_tensor = torch.from_numpy(lfcc_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
        lfcc_output = self._predict_single_model(self.lfcc_model, lfcc_tensor)
        
        # 앙상블
        if self.ensemble_method == 'weighted_avg':
            ensemble_real = (
                mel_output['real'] * self.weights['mel'] +
                lfcc_output['real'] * self.weights['lfcc']
            )
            ensemble_fake = (
                mel_output['fake'] * self.weights['mel'] +
                lfcc_output['fake'] * self.weights['lfcc']
            )
        elif self.ensemble_method == 'average':
            ensemble_real = (mel_output['real'] + lfcc_output['real']) / 2
            ensemble_fake = (mel_output['fake'] + lfcc_output['fake']) / 2
        elif self.ensemble_method == 'voting':
            mel_pred = 'fake' if mel_output['fake'] > 0.5 else 'real'
            lfcc_pred = 'fake' if lfcc_output['fake'] > 0.5 else 'real'
            
            if mel_pred == lfcc_pred:
                ensemble_real = (mel_output['real'] + lfcc_output['real']) / 2
                ensemble_fake = (mel_output['fake'] + lfcc_output['fake']) / 2
            else:
                # 의견이 다르면 더 확신하는 쪽 선택
                if max(mel_output.values()) > max(lfcc_output.values()):
                    ensemble_real = mel_output['real']
                    ensemble_fake = mel_output['fake']
                else:
                    ensemble_real = lfcc_output['real']
                    ensemble_fake = lfcc_output['fake']
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # 최종 판단
        prediction = 'fake' if ensemble_fake > 0.5 else 'real'
        confidence = max(ensemble_fake, ensemble_real)
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                'real': float(ensemble_real),
                'fake': float(ensemble_fake)
            },
            'model_outputs': {
                'mel': {
                    'real': float(mel_output['real']),
                    'fake': float(mel_output['fake'])
                },
                'lfcc': {
                    'real': float(lfcc_output['real']),
                    'fake': float(lfcc_output['fake'])
                }
            }
        }
    
    def batch_predict(self, audio_paths: list) -> list:
        """
        배치 예측
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'audio_path': audio_path
                })
        
        return results


def get_ensemble_predictor(
    weights_dir: str = '../weights',
    device: str = 'cpu',
    ensemble_method: str = 'weighted_avg'
) -> AudioEnsemblePredictor:
    """
    앙상블 예측기 생성 헬퍼 함수
    """
    weights_path = Path(weights_dir)
    
    mel_model_path = weights_path / 'audio_cnn' / 'best_model_latest.pth'
    lfcc_model_path = weights_path / 'lfcc_cnn' / 'best_lfcc_model_latest.pth'
    
    return AudioEnsemblePredictor(
        mel_model_path=str(mel_model_path),
        lfcc_model_path=str(lfcc_model_path),
        device=device,
        ensemble_method=ensemble_method
    )