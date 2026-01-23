class BaselineCpuModel:
    """
    TF-IDF + Logistic Regression baseline model wrapper (joblib/pkl).
    """
    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None

    def load(self, model_path: str) -> None:
        # TODO: joblib.load(model_path)
        raise NotImplementedError

    def predict_proba(self, text: str) -> float:
        # TODO: vectorize -> model.predict_proba -> float
        raise NotImplementedError
