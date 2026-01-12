# app/domains/audio/model.py
from app.domains.audio.schemas import AudioAnalyzeResponse, AudioEvidence

class AudioModel:
    def predict(self, audio_bytes: bytes) -> AudioAnalyzeResponse:
        # dummy output (schema-aligned)
        return AudioAnalyzeResponse(
            risk_score=12,
            grade="LOW",
            evidence=[AudioEvidence(score=0.12, reason="Low synthetic likelihood (dummy)")],
            warnings=[],
        )

def load_audio_model() -> AudioModel:
    return AudioModel()
