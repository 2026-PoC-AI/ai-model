import torch
import torch.nn as nn
import librosa
import numpy as np
from typing import Dict, Literal

class AudioPredictor:
    """
    단일 모델 예측기
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        feature_type: Literal['mel', 'lfcc'] = 'mel',
        sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 256
    ):
        """
        Args:
            model: PyTorch 모델
            device: 'cpu' or 'cuda'
            feature_type: 'mel' or 'lfcc'
            sr: 샘플링 레이트
            n_mels: Mel 필터 수
            n_fft: FFT 크기
            hop_length: Hop 길이
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_type = feature_type
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        오디오 파일에서 특징 추출
        """
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        if self.feature_type == 'mel':
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # dB 스케일 변환
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            features = mel_db
        
        elif self.feature_type == 'lfcc':
            # LFCC (Linear Frequency Cepstral Coefficients)
            # STFT
            D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            S = np.abs(D) ** 2
            
            # Linear frequency scale
            # DCT 적용
            from scipy.fftpack import dct
            lfcc = dct(S, axis=0, norm='ortho')[:self.n_mels, :]
            features = lfcc
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        return features
    
    def predict(self, audio_path: str) -> Dict:
        """
        예측 수행
        
        Returns:
            prediction: 'real' or 'fake'
            probabilities: {'real': float, 'fake': float}
        """
        # 특징 추출
        features = self.extract_features(audio_path)
        
        # 텐서 변환 (batch_size=1, channels=1, height, width)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            probs_numpy = probs.cpu().numpy()[0]
        
        # 결과
        real_prob = float(probs_numpy[0])
        fake_prob = float(probs_numpy[1])
        prediction = 'fake' if fake_prob > 0.5 else 'real'
        
        return {
            'prediction': prediction,
            'probabilities': {
                'real': real_prob,
                'fake': fake_prob
            }
        }