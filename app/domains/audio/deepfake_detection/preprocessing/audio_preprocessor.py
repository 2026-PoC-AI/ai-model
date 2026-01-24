import librosa
import numpy as np
import torch
from pathlib import Path

class AudioPreprocessor:
    """
    오디오를 스펙트로그램으로 변환하는 전처리 클래스
    """
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=64,
        duration=4.0,
        spec_height=64,
        spec_width=256
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration
        self.spec_height = spec_height
        self.spec_width = spec_width
        
        self.n_samples = int(sample_rate * duration)
    
    def load_audio(self, audio_path):
        """
        오디오 파일 로드
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        if len(audio) > self.n_samples:
            start = np.random.randint(0, len(audio) - self.n_samples)
            audio = audio[start:start + self.n_samples]
        else:
            audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
        
        return audio
    
    def audio_to_melspectrogram(self, audio):
        """
        오디오를 Mel-Spectrogram으로 변환
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_db = mel_spec_db * 2 - 1
        
        return mel_spec_db
    
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
        mel_spec = self.audio_to_melspectrogram(audio)
        mel_spec_resized = self.resize_spectrogram(mel_spec)
        
        return mel_spec_resized