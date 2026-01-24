import logging
from pathlib import Path
import torch

from app.core.config import settings
from app.domains.audio.deepfake_detection.inference.predictor import AudioDeepfakePredictor

logger = logging.getLogger(__name__)

def load_audio_model():
    """
    오디오 딥페이크 탐지 모델 로드
    
    Returns:
        AudioDeepfakePredictor: 로드된 모델 인스턴스
    """
    try:
        model_path = Path(settings.AUDIO_MODEL_PATH)
        
        if not model_path.exists():
            logger.error(f"Audio model not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # GPU 설정
        device = 'cuda' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading audio model from {model_path}")
        logger.info(f"Using device: {device}")
        
        # 모델 로드
        predictor = AudioDeepfakePredictor(
            model_path=str(model_path),
            device=device
        )
        
        logger.info("Audio model loaded successfully")
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to load audio model: {e}")
        raise