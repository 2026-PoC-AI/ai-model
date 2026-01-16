# app/domains/image/model.py
from typing import Tuple
from app.domains.image.deepfake.predictor import predict_xception

class ImageModel:
    def predict_deepfake(self, image_bytes: bytes) -> Tuple[str, float]:
        """
        Returns:
            label: FAKE | REAL
            confidence: 0.0 ~ 1.0
        """
        return predict_xception(image_bytes)

def load_image_model() -> ImageModel:
    return ImageModel()