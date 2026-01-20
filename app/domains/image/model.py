# app/domains/image/model.py
from app.domains.image.deepfake.predictor import predict_xception

class ImageModel:
    def predict(self, image_bytes: bytes) -> float:
        """
        Returns:
            confidence: 0.0 ~ 1.0
        """
        return predict_xception(image_bytes)

def load_image_model() -> ImageModel:
    return ImageModel()
