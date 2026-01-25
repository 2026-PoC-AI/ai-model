from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from .preprocess import normalize_texts


@dataclass
class FakeNewsPrediction:
    label: str
    score: float
    probabilities: Dict[str, float]


class KlueBertFakeNewsPredictor:
    """
    klue/bert-base 기반 텍스트 이진/다중 분류 추론기.
    - model_path가 HF 디렉토리면 from_pretrained로 로드
    - state_dict_path가 있으면 state_dict 로드 (config/num_labels는 명시 필요)
    """

    def __init__(
        self,
        model_name: str = "klue/bert-base",
        model_path: Optional[str] = None,
        state_dict_path: Optional[str] = None,
        num_labels: int = 2,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.state_dict_path = state_dict_path
        self.num_labels = num_labels
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 라벨 매핑
        if id2label is None:
            # 기본: 0=FAKE, 1=REAL (학습 기준에 맞게 수정 가능)
            id2label = {0: "FAKE", 1: "REAL"} if num_labels == 2 else {i: str(i) for i in range(num_labels)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        self.id2label = id2label
        self.label2id = label2id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        # 1) HF 포맷 디렉토리 제공 시
        if self.model_path:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )

        # 2) state_dict 제공 시
        if self.state_dict_path:
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            model = AutoModelForSequenceClassification.from_config(config)
            sd = torch.load(self.state_dict_path, map_location="cpu")

            # 저장 형태에 따라 키가 다를 수 있음
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]

            # DataParallel 학습 시 module. prefix 제거
            cleaned = {}
            for k, v in sd.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                cleaned[nk] = v

            model.load_state_dict(cleaned, strict=False)
            return model

        # 3) 아무 것도 없으면 베이스 모델
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

    @torch.inference_mode()
    def predict_one(self, text: str) -> FakeNewsPrediction:
        return self.predict_batch([text])[0]

    @torch.inference_mode()
    def predict_batch(self, texts: List[str]) -> List[FakeNewsPrediction]:
        texts = normalize_texts(texts)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        logits = out.logits
        probs = F.softmax(logits, dim=-1)

        results: List[FakeNewsPrediction] = []
        for p in probs:
            p_list = p.detach().float().cpu().tolist()
            best_id = int(torch.argmax(p).item())
            label = self.id2label.get(best_id, str(best_id))
            score = float(p_list[best_id])
            prob_map = {self.id2label[i]: float(p_list[i]) for i in range(self.num_labels)}
            results.append(FakeNewsPrediction(label=label, score=score, probabilities=prob_map))

        return results

    # =========================
    # ✅ /text/analyze 지원 로직
    # =========================
    def _split_sentences(self, text: str) -> List[str]:
        """
        MVP용 간단 문장 분리.
        - 줄바꿈/마침표/물음표/느낌표 기준으로 분리
        """
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        parts = [p.strip() for p in parts if p and p.strip()]
        return parts

    @torch.inference_mode()
    def analyze(self, text: str, evidence_k: int = 3) -> Dict[str, Any]:
        """
        고정 API 스펙을 만들기 위한 '재료'를 반환.
        - label/score: 전체 텍스트 기준
        - evidences: 문장별 FAKE 확률 기준 Top-k
        - highlights: Top-k 문장 원문 위치(start/end) 기반 하이라이트
        - reference_query: 네이버 검색에 사용할 짧은 쿼리(근거 1문장 일부)
        """
        text = text or ""
        evidence_k = max(1, int(evidence_k or 3))

        # 1) 전체 텍스트 예측
        main_pred = self.predict_one(text)

        # 2) 문장 분리 및 문장별 예측
        sents = self._split_sentences(text)
        if not sents:
            sents = [text] if text.strip() else [""]

        sent_preds = self.predict_batch(sents)

        # 3) 문장별 'FAKE 확률' 스코어 계산
        scored = []
        for sent, pred in zip(sents, sent_preds):
            fake_p = pred.probabilities.get("FAKE")
            if fake_p is None:
                # fallback: 현재 pred.label이 FAKE면 pred.score, 아니면 1-pred.score
                fake_p = pred.score if pred.label == "FAKE" else (1.0 - pred.score)
            scored.append((sent, float(fake_p)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: evidence_k]

        evidences = [{"text": s, "score": sc} for s, sc in top if s.strip()]

        # 4) highlights 생성: 근거 문장을 원문에서 찾아 표시 (가장 안정적인 MVP)
        highlights = []
        used_ranges = set()
        for s, sc in top:
            s = s.strip()
            if not s:
                continue
            idx = text.find(s)
            if idx == -1:
                continue
            start, end = idx, idx + len(s)
            if (start, end) in used_ranges:
                continue
            used_ranges.add((start, end))
            highlights.append({"start": start, "end": end, "text": s, "weight": float(sc)})

        # 5) 네이버 검색용 쿼리: Top-1 근거 문장 일부(너무 길면 자름)
        reference_query = ""
        if top and top[0][0].strip():
            reference_query = top[0][0].strip()[:80]
        else:
            reference_query = text.strip()[:80]

        return {
            "label": main_pred.label,
            "score": float(main_pred.score),
            "evidences": evidences,
            "highlights": highlights,
            "reference_query": reference_query,
        }
