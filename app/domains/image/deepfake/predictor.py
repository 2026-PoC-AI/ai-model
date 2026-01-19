# app/domains/image/deepfake/predictor.py
import torch
from app.core.config import settings
from app.domains.image.deepfake.preprocess import preprocess_image
from app.domains.image.deepfake.xception_model import XceptionNet

def _get_device() -> torch.device:
    if settings.USE_GPU and torch.cuda.is_available():
        return torch.device(f"cuda:{settings.GPU_DEVICE_ID}")
    return torch.device("cpu")

_DEVICE = _get_device()
_model = None

def load_xception_model():
    global _model
    if _model is not None:
        return _model

    model = XceptionNet(pretrained=False)
    state = torch.load(settings.IMAGE_WEIGHT_PATH, map_location=_DEVICE)
    model.load_state_dict(state)
    model.eval().to(_DEVICE)
    _model = model
    return _model

def predict_xception(image_bytes: bytes) -> dict:
    model = load_xception_model()
    x = preprocess_image(image_bytes).to(_DEVICE)

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    # label rule (너 정책 유지)
    if prob >= 0.7:
        label = "FAKE"
    elif prob <= 0.3:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return {
        "label": label,
        "confidence": round(prob, 4),
    }
