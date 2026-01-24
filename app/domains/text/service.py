from __future__ import annotations
from typing import List, Optional

from .fake_news.predictor import KlueBertFakeNewsPredictor, FakeNewsPrediction
from .schemas import TextAnalyzeResponse, TextEvidence, TextHighlight, TextReference


class TextService:
    def __init__(self, fake_news_predictor: KlueBertFakeNewsPredictor, naver_client: Optional[object] = None):
        self.fake_news_predictor = fake_news_predictor
        self.naver_client = naver_client  # None이면 references는 빈 리스트로 내려감

    def predict_fake_news(self, text: str) -> FakeNewsPrediction:
        return self.fake_news_predictor.predict_one(text)

    def predict_fake_news_batch(self, texts: List[str]) -> List[FakeNewsPrediction]:
        return self.fake_news_predictor.predict_batch(texts)

    def analyze(self, text: str, evidence_k: int = 3, include_references: bool = True) -> TextAnalyzeResponse:
        out = self.fake_news_predictor.analyze(text=text, evidence_k=evidence_k)

        # label 매핑: REAL -> TRUE (너 UI 기준)
        label = out["label"]
        if label == "REAL":
            label = "TRUE"

        evidences = [TextEvidence(**e) for e in out.get("evidences", [])]
        highlights = [TextHighlight(**h) for h in out.get("highlights", [])]

        references: List[TextReference] = []
        if include_references and self.naver_client is not None:
            query = out.get("reference_query") or ""
            if query.strip():
                try:
                    refs = self.naver_client.search_news(query=query)
                    references = [TextReference(**r) for r in refs]
                except Exception:
                    # 참고자료 실패는 서비스 전체 실패로 만들지 않음
                    references = []

        return TextAnalyzeResponse(
            label=label,
            score=float(out["score"]),
            evidences=evidences,
            highlights=highlights,
            references=references,
        )
