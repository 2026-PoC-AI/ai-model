# app/domains/image/deepfake/predictor.py
import torch
from app.domains.image.deepfake.preprocess import preprocess_image
from app.domains.image.deepfake.xception_model import XceptionNet

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_WEIGHT_PATH = "app/domains/image/deepfake/weights/custom_xception.pth"

_model = None

def _load_model():
    global _model
    if _model is None:
        print("[XCEPTION] loading custom trained model...")
        model = XceptionNet(pretrained=False)
        state = torch.load(_WEIGHT_PATH, map_location=_DEVICE)
        model.load_state_dict(state)
        model.eval().to(_DEVICE)
        _model = model
    return _model



def predict_xception(image_bytes: bytes):
    model = _load_model()
    tensor = preprocess_image(image_bytes).to(_DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    if prob >= 0.7:
        label = "FAKE"
    elif prob <= 0.3:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return label, round(prob, 4)
