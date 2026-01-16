# app/domains/image/deepfake/predictor.py
import torch
from app.domains.image.deepfake.preprocess import preprocess_image
from app.domains.image.deepfake.xception_model import XceptionNet

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

_WEIGHT_PATH = "app/domains/image/deepfake/weights/deepfake_c0_xception.pkl"

def _load_model():
    global _model
    if _model is None:
        print("[XCEPTION] loading FF++ pretrained weight...")
        model = XceptionNet(pretrained=False)
        
        state = torch.load(_WEIGHT_PATH, map_location=_DEVICE)
        # 대부분의 FF++ weight는 state_dict 키를 가짐
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # key 이름 정리 (module. 제거)
        clean_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            clean_state[k] = v


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
