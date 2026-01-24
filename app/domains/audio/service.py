import logging
import tempfile
import time
from pathlib import Path
from typing import BinaryIO

from app.core.config import settings

logger = logging.getLogger(__name__)

class AudioAnalysisService:
    def __init__(self, predictor):
        """
        Args:
            predictor: app.state.models["audio"]에서 전달받은 모델
        """
        self.predictor = predictor  # 이미 로드된 모델 사용
    
    async def analyze_audio(self, file: BinaryIO, filename: str, analysis_id: int) -> dict:
        start_time = time.time()
        
        # 파일 검증
        allowed_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
        
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # 크기 검증
            file_size = len(content)
            max_size = settings.MAX_AUDIO_MB * 1024 * 1024
            
            if file_size > max_size:
                raise ValueError(f"파일 크기 초과: {file_size / 1024 / 1024:.2f}MB")
            
            # 예측 (이미 로드된 모델 사용 - 빠름!)
            result = self.predictor.predict(tmp_path)
            
            processing_time = time.time() - start_time
            
            return {
                "analysis_id": analysis_id,
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "real_probability": result['real_probability'],
                "fake_probability": result['fake_probability'],
                "model_version": result['model_version'],
                "processing_time": round(processing_time, 2),
                "file_name": filename,
                "file_size": file_size,
                "status": "completed"
            }
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)