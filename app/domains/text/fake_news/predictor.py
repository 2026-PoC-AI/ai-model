# app/domains/text/fake_news/predictor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

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
            # 기본: 0=FAKE, 1=REAL (너 학습 기준에 맞게 바꿔)
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
            # 학습 저장 형태에 따라 key가 'model_state_dict'일 수 있음
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]

            # DataParallel로 학습했을 때 'module.' prefix 제거
            cleaned = {}
            for k, v in sd.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                cleaned[nk] = v
            model.load_state_dict(cleaned, strict=False)
            return model

        # 3) 아무 것도 없으면 베이스 모델(실전에서는 비추)
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
        logits = out.logits  # (B, num_labels)
        probs = F.softmax(logits, dim=-1)  # (B, num_labels)

        results: List[FakeNewsPrediction] = []
        for p in probs:
            p_list = p.detach().float().cpu().tolist()
            best_id = int(torch.argmax(p).item())
            label = self.id2label.get(best_id, str(best_id))
            score = float(p_list[best_id])

            prob_map = {self.id2label[i]: float(p_list[i]) for i in range(self.num_labels)}
            results.append(FakeNewsPrediction(label=label, score=score, probabilities=prob_map))

        return results
