# app/domains/text/fake_news/baseline_cpu.py
from __future__ import annotations
import joblib
import numpy as np

class BaselineCpuModel:
    """
    TF-IDF + Logistic Regression baseline model
    """
    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None

    def load(self, model_dir: str) -> None:
        # model_dir: "app/models" 같은 폴더 경로
        self.model = joblib.load(f"{model_dir}/cpu_model.pkl")
        self.vectorizer = joblib.load(f"{model_dir}/cpu_vectorizer.pkl")

    def predict_proba(self, text: str) -> float:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("CPU model not loaded")

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]  # [p0, p1]
        return float(proba[1])  # label=1 확률
