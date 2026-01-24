from __future__ import annotations

import os
from typing import List, Optional

from app.domains.text.service import TextService
from app.domains.text.fake_news.predictor import KlueBertFakeNewsPredictor, FakeNewsPrediction

# references를 쓰려면(네이버) 아래 2줄 사용
from app.domains.text.naver_client import NaverNewsClient


class TextModel:
    """
    ✅ 기존 TextModel/ load_text_model() 구조는 유지하면서
    - /api/v1/text/analyze 에 필요한 analyze()
    - 기존 fake-news 엔드포인트에 필요한 predict_fake_news(), predict_fake_news_batch()
    를 제공하는 Facade 역할.

    즉, app.state.models["text"] 로도 쓰고
        app.state.text_service 로도 같은 객체를 쓰게 만들 수 있음.
    """

    def __init__(self, service: TextService):
        self._service = service

    # -------------------------
    # ✅ 고정 API: /text/analyze
    # -------------------------
    def analyze(self, text: str, evidence_k: int = 3, include_references: bool = True):
        return self._service.analyze(text=text, evidence_k=evidence_k, include_references=include_references)

    # -------------------------
    # ✅ 기존 fake-news 엔드포인트 지원
    # -------------------------
    def predict_fake_news(self, text: str) -> FakeNewsPrediction:
        return self._service.predict_fake_news(text)

    def predict_fake_news_batch(self, texts: List[str]) -> List[FakeNewsPrediction]:
        return self._service.predict_fake_news_batch(texts)

    # -------------------------
    # ✅ (호환용) 기존 predict 이름 유지
    # - 예전에는 risk_score/grade 같은 더미였는데,
    #   이제는 고정 스펙(TextAnalyzeResponse: label/score/evidences/highlights/references)에 맞게 반환
    # -------------------------
    def predict(self, text: str):
        # include_references는 기본 false로 두고 싶으면 여기서 False로 바꿔도 됨
        return self.analyze(text=text, evidence_k=3, include_references=False)


def load_text_model() -> TextModel:
    """
    ✅ 기존 함수명/호출부 유지
    - settings.TEXT_MODEL_PATH 같은 걸 쓰고 싶으면 여기에 연결하면 됨
    """
    # ---- 모델 가중치 경로 (상대경로 안전하게) ----
    # 환경변수/설정으로 받는 걸 추천하지만, 일단 프로젝트 루트 기준 상대경로도 안전하게 처리
    # 예: app/domains/text/fake_news/weights/model_state_dict.pth 같은 위치라면:
    base_dir = os.path.dirname(__file__)  # app/domains/text
    default_sd = os.path.join(base_dir, "fake_news", "weights", "model_state_dict.pth")

    state_dict_path: Optional[str] = None
    if os.path.exists(default_sd):
        state_dict_path = default_sd
    else:
        # 없으면 None (그러면 predictor가 베이스 모델로 올라갈 수 있음: 비추지만 서버는 살림)
        state_dict_path = None

    predictor = KlueBertFakeNewsPredictor(
        model_name="klue/bert-base",
        state_dict_path=state_dict_path,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        max_length=256,
    )

    # ---- references(네이버) 사용하려면 아래 주석 해제 ----
    naver_client = NaverNewsClient(display=3)  # 키는 환경변수 NAVER_CLIENT_ID/SECRET에서 읽게 구성
    service = TextService(fake_news_predictor=predictor, naver_client=naver_client)

    # ---- references 없이 MVP로만 갈 때 ----
    #service = TextService(fake_news_predictor=predictor, naver_client=None)

    return TextModel(service)
