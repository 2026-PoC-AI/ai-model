# app/domains/text/service.py
from __future__ import annotations
from typing import List, Optional

from .fake_news.predictor import KlueBertFakeNewsPredictor, FakeNewsPrediction


class TextService:
    def __init__(self, fake_news_predictor: KlueBertFakeNewsPredictor):
        self.fake_news_predictor = fake_news_predictor

    def predict_fake_news(self, text: str) -> FakeNewsPrediction:
        return self.fake_news_predictor.predict_one(text)

    def predict_fake_news_batch(self, texts: List[str]) -> List[FakeNewsPrediction]:
        return self.fake_news_predictor.predict_batch(texts)
