# app/domains/image/model.py
from app.domains.image.schemas import ImageAnalyzeResponse, ImageEvidence

class ImageModel:
    def predict(self, image_bytes: bytes) -> ImageAnalyzeResponse:
        return ImageAnalyzeResponse(
            risk_score=34,
            grade="MEDIUM",
            evidence=[
                ImageEvidence(region="face_1", score=0.42, reason="GAN-like texture artifact (dummy)"),
            ],
            warnings=[],
        )

def load_image_model() -> ImageModel:
    return ImageModel()
