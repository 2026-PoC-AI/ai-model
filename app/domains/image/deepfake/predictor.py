# app/domains/image/deepfake/predictor.py
import torch
from app.domains.image.deepfake.preprocess import preprocess_image
from app.domains.image.deepfake.xception_model import XceptionNet

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

def _load_model():
    global _model
    if _model is None:
        model = XceptionNet(pretrained=True)
        # 자체 학습 가중치 있을 경우
        # model.load_state_dict(torch.load("weights/xception.pth", map_location=_DEVICE))
        model.eval()
        model.to(_DEVICE)
        _model = model
    return _model


def predict_xception(image_bytes: bytes):
    model = _load_model()

    tensor = preprocess_image(image_bytes).to(_DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if label == "FAKE" else 1 - prob

    return label, round(confidence, 4)
