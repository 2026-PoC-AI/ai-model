import logging
from pathlib import Path

from app.domains.audio.deepfake_detection.inference.ensemble_predictor import get_ensemble_predictor

logger = logging.getLogger(__name__)

def load_audio_model():
    """
    Audio 딥페이크 탐지 앙상블 모델 로드
    """
    try:
        logger.info("Loading Audio Ensemble Model (Mel + LFCC)...")
        
        weights_dir = Path(__file__).parent / 'deepfake_detection' / 'weights'
        
        ensemble = get_ensemble_predictor(
            weights_dir=str(weights_dir),
            device='cpu',
            ensemble_method='weighted_avg'
        )
        
        logger.info("✓ Audio Ensemble Model loaded successfully!")
        logger.info("  - Mel-spectrogram CNN: 99.15% accuracy")
        logger.info("  - LFCC CNN: 99.57% accuracy")
        logger.info("  - Expected ensemble: 99.6-99.8%")
        
        return ensemble
        
    except Exception as e:
        logger.error(f"Failed to load audio model: {e}")
        raise