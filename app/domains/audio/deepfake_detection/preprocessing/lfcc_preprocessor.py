import librosa
import numpy as np
import torch
from pathlib import Path
from scipy.fftpack import dct

class LFCCPreprocessor:
    """
    오디오를 LFCC (Linear Frequency Cepstral Coefficients)로 변환하는 전처리 클래스
    Mel-spectrogram과 다른 주파수 표현 방식으로 앙상블 성능 향상
    """
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_linear=128,
        n_lfcc=40,
        duration=4.0,
        spec_height=40,
        spec_width=256
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_linear = n_linear
        self.n_lfcc = n_lfcc
        self.duration = duration
        self.spec_height = spec_height
        self.spec_width = spec_width
        
        self.n_samples = int(sample_rate * duration)
    
    def load_audio(self, audio_path):
        """
        오디오 파일 로드
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # 길이 조정
        if len(audio) > self.n_samples:
            start = np.random.randint(0, len(audio) - self.n_samples)
            audio = audio[start:start + self.n_samples]
        else:
            audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
        
        return audio
    
    def audio_to_lfcc(self, audio):
        """
        오디오를 LFCC로 변환
        
        LFCC는 선형 주파수 스케일을 사용하여 Mel-spectrogram과 다른 특징 추출
        딥페이크 음성의 고주파 아티팩트 탐지에 효과적
        """
        # STFT 계산
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Power spectrum
        power_spec = np.abs(stft) ** 2
        
        # Linear frequency filterbank
        linear_filter = self._create_linear_filterbank()
        
        # Filterbank 적용
        linear_spec = np.dot(linear_filter, power_spec)
        
        # Log compression
        linear_spec_db = librosa.power_to_db(linear_spec, ref=np.max)
        
        # DCT (Discrete Cosine Transform)
        lfcc = dct(linear_spec_db, type=2, axis=0, norm='ortho')[:self.n_lfcc]
        
        # 정규화
        lfcc = (lfcc - lfcc.min()) / (lfcc.max() - lfcc.min() + 1e-8)
        lfcc = lfcc * 2 - 1
        
        return lfcc
    
    def _create_linear_filterbank(self):
        """
        선형 주파수 필터뱅크 생성
        """
        # 주파수 bins
        freqs = np.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        
        # Linear spacing
        linear_freqs = np.linspace(0, self.sample_rate / 2, self.n_linear + 2)
        
        # 필터뱅크 행렬
        filterbank = np.zeros((self.n_linear, self.n_fft // 2 + 1))
        
        for i in range(self.n_linear):
            # 삼각 필터
            left = linear_freqs[i]
            center = linear_freqs[i + 1]
            right = linear_freqs[i + 2]
            
            for j, freq in enumerate(freqs):
                if left <= freq <= center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        
        return filterbank
    
    def resize_spectrogram(self, spec):
        """
        스펙트로그램 크기 조정
        """
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        
        spec_resized = torch.nn.functional.interpolate(
            spec_tensor,
            size=(self.spec_height, self.spec_width),
            mode='bilinear',
            align_corners=False
        )
        
        return spec_resized.squeeze().numpy()
    
    def preprocess(self, audio_path):
        """
        전체 전처리 파이프라인
        """
        audio = self.load_audio(audio_path)
        lfcc = self.audio_to_lfcc(audio)
        lfcc_resized = self.resize_spectrogram(lfcc)
        
        return lfcc_resized


class DeltaFeatureExtractor:
    """
    Delta 및 Delta-Delta 특징 추출 (옵션)
    시간적 변화를 포착하여 음성 합성 아티팩트 탐지 향상
    """
    @staticmethod
    def compute_delta(features, N=2):
        """
        Delta 특징 계산
        
        Args:
            features: (n_features, n_frames)
            N: Delta 계산 윈도우 크기
        """
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta = np.zeros_like(features)
        
        padded = np.pad(features, ((0, 0), (N, N)), mode='edge')
        
        for t in range(features.shape[1]):
            delta[:, t] = np.dot(
                np.arange(-N, N+1),
                padded[:, t:t+2*N+1].T
            ) / denominator
        
        return delta
    
    @staticmethod
    def add_delta_features(features):
        """
        원본 특징에 Delta, Delta-Delta 추가
        
        Returns:
            (3, n_features, n_frames) - [원본, Delta, Delta-Delta]
        """
        delta = DeltaFeatureExtractor.compute_delta(features)
        delta_delta = DeltaFeatureExtractor.compute_delta(delta)
        
        return np.stack([features, delta, delta_delta], axis=0)