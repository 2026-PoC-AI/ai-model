from dataclasses import dataclass
from typing import List


@dataclass
class PredictResult:
    label: str                 # "fake" | "real"
    confidence: float          # 0.0 ~ 1.0
    suspicious_keywords: List[str]


class FakeNewsPredictor:
    """
    TODO:
    - Load KPF-BERT fine-tuned weights (.pth)
    - Load baseline TF-IDF+LogReg model (.pkl/.joblib)
    - Return ensemble score + evidence
    """
    def __init__(self) -> None:
        pass

    def predict(self, text: str) -> PredictResult:
        raise NotImplementedError("TODO: implement fake news inference pipeline")
