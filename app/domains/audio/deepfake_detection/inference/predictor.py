import torch
import numpy as np
import librosa
from pathlib import Path
import sys

# 프로젝트 루트 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from models.spectrogram_cnn import get_model
from preprocessing.audio_preprocessor import AudioPreprocessor

class AudioDeepfakePredictor:
    """
    학습된 모델을 사용한 오디오 딥페이크 예측 클래스
    """
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # 모델 로드
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = get_model(
            model_name='lightweight_cnn',
            num_classes=2,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 전처리기
        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_fft=512,
            hop_length=256,
            n_mels=64,
            duration=4.0,
            spec_height=64,
            spec_width=256
        )
        
        # 모델 메타 정보
        self.model_version = checkpoint.get('config', {}).get('model_name', 'v1.0.0')
        
        print("Model loaded successfully!")
    
    def predict(self, audio_path):
        """
        오디오 파일에 대한 딥페이크 예측
        
        Args:
            audio_path: 오디오 파일 경로
        
        Returns:
            dict: {
                'prediction': 'real' or 'fake',
                'confidence': 0.9234,
                'real_probability': 0.0766,
                'fake_probability': 0.9234,
                'model_version': 'v1.0.0'
            }
        """
        # 전처리
        spec = self.preprocessor.preprocess(audio_path)
        
        # Tensor로 변환
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        spec_tensor = spec_tensor.to(self.device)
        
        # 예측
        with torch.no_grad():
            output = self.model(spec_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        # 결과 추출
        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()
        prediction = 'real' if real_prob > fake_prob else 'fake'
        confidence = max(real_prob, fake_prob)
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'real_probability': round(real_prob, 4),
            'fake_probability': round(fake_prob, 4),
            'model_version': self.model_version
        }
    
    def predict_from_bytes(self, audio_bytes, filename='temp.wav'):
        """
        바이트 데이터에서 직접 예측 (API 업로드용)
        
        Args:
            audio_bytes: 오디오 바이트 데이터
            filename: 임시 파일명
        
        Returns:
            dict: 예측 결과
        """
        import tempfile
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = self.predict(tmp_path)
        finally:
            # 임시 파일 삭제
            Path(tmp_path).unlink(missing_ok=True)
        
        return result