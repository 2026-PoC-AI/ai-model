# app/domains/image/model.py
from app.domains.image.schemas import ImageAnalyzeResponse, ImageEvidence

class ImageModel:
    def predict_deepfake(self, image_bytes: bytes) -> tuple[str, float]:
        # 더미 추론
        label = "FAKE"
        confidence = 0.82
        return label, confidence


def load_image_model() -> ImageModel:
    return ImageModel()