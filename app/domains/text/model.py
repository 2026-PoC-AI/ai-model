# app/domains/text/model.py
from app.domains.text.schemas import TextAnalyzeResponse, TextEvidence

class TextModel:
    def predict(self, text: str) -> TextAnalyzeResponse:
        return TextAnalyzeResponse(
            risk_score=22,
            grade="LOW",
            evidence=[TextEvidence(score=0.19, reason="No synthetic style cues (dummy)")],
            warnings=[],
        )

def load_text_model() -> TextModel:
    return TextModel()
