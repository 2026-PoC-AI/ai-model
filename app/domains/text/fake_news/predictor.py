# app/domains/text/fake_news/predictor.py
import joblib
import numpy as np

class FakeNewsPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def load(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict_proba(self, text: str) -> float:
        # 가짜뉴스(1)일 확률 반환
        X = self.vectorizer.transform([text])
        return float(self.model.predict_proba(X)[0][1])

    def get_top_features(self, text: str, top_n: int = 3):
        # 텍스트 내 단어 중 가짜뉴스 가중치가 높은 단어 추출
        X = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.model.coef_[0]
        
        present_indices = X.nonzero()[1]
        word_scores = [(feature_names[i], coefs[i]) for i in present_indices if coefs[i] > 0]
        
        # 가중치 순 정렬
        sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:top_n]]